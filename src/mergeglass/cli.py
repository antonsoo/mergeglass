from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from . import __version__
from .audit import audit
from .storage import AuditError


def _range(layers: list[dict[str, Any]], field: str) -> str:
    values = [layer[field] for layer in layers if layer[field] is not None]
    return f"{min(values):.4g}..{max(values):.4g}" if values else "n/a"


def render(report: dict[str, Any], *, summary_only: bool = False) -> str:
    variants = Counter(layer.get("adapter_variant", "lora") for layer in report["layers"])
    kind = "LoRA"
    if variants.get("dora"):
        kind = "LoRA/DoRA" if variants.get("lora") else "DoRA"
    lines = [f"Mergeglass | saved {kind} merge audit", ""]
    if variants.get("dora"):
        lines.extend(
            [
                "DoRA reference includes learned magnitudes and row-normalized directions.",
                "Counts, gain and residual use the complete weight update, not B @ A alone.",
                "",
            ]
        )
    summary = report["summary"]
    if summary_only:
        layers = report["layers"]
        dtypes = Counter(layer["merged_dtype"] for layer in layers)
        storage = ", ".join(f"{dtype}: {count}" for dtype, count in sorted(dtypes.items()))
        observed = sum(layer["status"] == "observed-update" for layer in layers)
        changed_zero = sum(layer["status"] == "change-with-zero-reference" for layer in layers)
        lines.extend(
            [
                f"Adapted matrices: {summary['adapted_tensors']} ({storage}).",
                f"Matrix states: {observed} observed; "
                f"{summary['no_observed_update_tensors']} with no observed update.",
                f"Zero-reference matrices: {summary['zero_update_tensors']} unchanged; "
                f"{changed_zero} changed.",
                f"Gain range: {_range(layers, 'projection_gain')}; "
                f"relative residual range: {_range(layers, 'relative_residual')}.",
            ]
        )
    else:
        lines.append(
            "MODULE                       DTYPE   CHANGED    UNCHANGED*     GAIN  REL.RESID"
        )
        for layer in report["layers"]:
            fraction = layer["unchanged_nonzero_update_fraction"]
            residual = layer["relative_residual"]
            projection = layer["projection_gain"]
            unchanged = "n/a" if fraction is None else f"{fraction:.1%}"
            error = "n/a" if residual is None else f"{residual:.4g}"
            gain = "n/a" if projection is None else f"{projection:.4g}"
            # Escape control sequences in names; JSON retains full identities.
            name = ascii(layer["module"])[1:-1]
            if len(name) > 27:
                name = "..." + name[-24:]
            lines.append(
                f"{name:<28} {layer['merged_dtype']:<6} "
                f"{layer['changed_elements']:>8,} {unchanged:>12} {gain:>8} {error:>10}"
            )
            if layer["status"] != "observed-update":
                lines.append("  " + layer["status"])
    total = summary["intended_nonzero_elements"]
    unchanged_total = summary["unchanged_nonzero_update_elements"]
    aggregate = (
        f"Of {total:,} nonzero intended updates, {unchanged_total:,} are unchanged "
        f"({summary['unchanged_nonzero_update_fraction']:.2%})."
        if total
        else "The reference update is zero across all adapted weights."
    )
    lines.extend(["", aggregate])
    energy = summary["unchanged_reference_energy_fraction"]
    if energy is not None:
        lines.append(
            f"Those unchanged entries contain {100 * energy:.4g}% of the reference squared norm."
        )
    if not summary_only:
        lines.append("* Nonzero intended update elements equal to the dtype-cast base.")
    lines.extend(
        [
            "Relative error = ||observed change - intended update|| / ||intended update||.",
            "Gain = projection of the observed change onto the intended update (ideal 1).",
            f"Checked {summary['checked_tensors']} tensors; "
            f"{summary['unadapted_changed_tensors']} unadapted tensor(s) changed.",
            "Equal weights alone cannot tell rounding from an omitted adapter.",
            "This measures saved weights, not model quality.",
        ]
    )
    return "\n".join(lines)


def _write(path: Path, text: str) -> None:
    # Replacing only after serialization succeeds preserves an earlier report
    # if the audit refuses or the destination runs out of space mid-write.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".mergeglass-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a saved LoRA or DoRA merge, locally and read-only."
    )
    parser.add_argument("base", type=Path, help="original safetensors file or HF directory")
    parser.add_argument(
        "adapter", type=Path, help="PEFT adapter directory with adapter_config.json"
    )
    parser.add_argument("merged", type=Path, help="saved merged safetensors file or HF directory")
    parser.add_argument("--json", metavar="PATH", help="write a report; '-' writes JSON to stdout")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="show totals and metric ranges without per-layer rows",
    )
    parser.add_argument(
        "--block-mib", type=int, default=16, help="block workspace target (default 16)"
    )
    parser.add_argument(
        "--max-adapter-mib",
        type=int,
        default=512,
        help="maximum decoded A matrix size (default 512)",
    )
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    try:
        if args.json and args.json != "-":
            destination = Path(args.json).resolve()
            for source in (args.base, args.adapter, args.merged):
                resolved = source.resolve()
                if destination == resolved or (
                    resolved.is_dir() and destination.is_relative_to(resolved)
                ):
                    raise AuditError("Write the report outside the input checkpoint paths")
        report = audit(
            args.base,
            args.adapter,
            args.merged,
            block_mib=args.block_mib,
            max_adapter_mib=args.max_adapter_mib,
        )
        if args.json and args.json != "-":
            # Snapshot files can be symlinks into a cache outside the supplied
            # directory. Protect those actual input files too, after inventory.
            for role, source in (
                ("base", args.base),
                ("adapter", args.adapter),
                ("merged", args.merged),
            ):
                for item in report["inputs"][role]:
                    input_file = source / item["file"] if source.is_dir() else source
                    if destination == input_file.resolve():
                        raise AuditError("The report destination resolves to an input file")
        serialized = json.dumps(report, indent=2, allow_nan=False)
        if args.json == "-":
            print(serialized)
        else:
            if args.json:
                _write(Path(args.json), serialized)
            print(render(report, summary_only=args.summary))
    except (AuditError, OSError, ValueError, OverflowError) as exc:
        message = ascii(str(exc))[1:-1]
        print(f"mergeglass: {message}", file=sys.stderr)
        return 2
    return 0
