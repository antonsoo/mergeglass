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
from .dora_norms import audit_dora_norms
from .storage import AuditError


def _factor_text(log2: float) -> str:
    try:
        value = 2.0**log2
    except OverflowError:
        return f"2^({log2:.4g})"
    return f"{value:.8g}" if value else f"2^({log2:.4g})"


def _dora_row_lines(layers: list[dict[str, Any]], *, include_direction: bool = True) -> list[str]:
    rows = [layer["dora_rows"] for layer in layers if "dora_rows" in layer]
    if not rows:
        return []
    norms = [
        row["log2_row_norm_factor_range"]
        for row in rows
        if row["log2_row_norm_factor_range"] is not None
    ]
    directions = [
        row["direction_cosine_range"] for row in rows if row.get("direction_cosine_range")
    ]
    factors = (
        f"{_factor_text(min(pair[0] for pair in norms))}.."
        f"{_factor_text(max(pair[1] for pair in norms))}"
        if norms
        else "n/a"
    )
    cosines = (
        f"{min(pair[0] for pair in directions):.8g}..{max(pair[1] for pair in directions):.8g}"
        if directions
        else "n/a"
    )
    lines = [
        f"DoRA full-row norm factor (saved / |m|): {factors}; "
        f"{sum(row['norm_comparable_rows'] for row in rows):,} nonzero row pairs.",
        f"Nonzero magnitudes saved as zero rows: "
        f"{sum(row['nonzero_magnitude_zero_saved_rows'] for row in rows):,}; "
        f"zero magnitudes saved nonzero: "
        f"{sum(row['zero_magnitude_nonzero_saved_rows'] for row in rows):,}.",
    ]
    if include_direction:
        lines.insert(
            1,
            f"Full-row direction cosine: {cosines}; "
            f"{sum(row['direction_comparable_rows'] for row in rows):,} nonzero row pairs.",
        )
    return lines


def _range(layers: list[dict[str, Any]], field: str) -> str:
    values = [layer[field] for layer in layers if layer[field] is not None]
    return f"{min(values):.4g}..{max(values):.4g}" if values else "n/a"


def render(report: dict[str, Any], *, summary_only: bool = False) -> str:
    if report.get("schema") == "mergeglass.dora-norms":
        return _render_norms(report, summary_only=summary_only)
    variants = Counter(layer.get("adapter_variant", "lora") for layer in report["layers"])
    kind = "LoRA"
    if variants.get("dora"):
        kind = "LoRA/DoRA" if variants.get("lora") else "DoRA"
    lines = [f"Mergeglass | saved {kind} merge audit", ""]
    if any(layer.get("weight_layout") == "conv1d" for layer in report["layers"]):
        lines.extend(
            [
                "Conv1D reference: (B @ A).T; saved rows are inputs, columns are outputs.",
                "Layouts follow standard GPT module paths and declared model metadata (see JSON).",
                "",
            ]
        )
    if any(layer.get("weight_layout") == "embedding" for layer in report["layers"]):
        lines.extend(
            [
                "Embedding reference: (B @ A).T; rows are token indices, columns are dimensions.",
                "",
            ]
        )
    if any("expert_parameter" in layer for layer in report["layers"]):
        summary = report["summary"]
        expert_rows = [row for row in report["layers"] if "expert_parameter" in row]
        lines.extend(
            [
                "Qwen3 MoE expert parameters: A groups expert rows; B interleaves expert columns.",
                "Native saved gate/up/down matrices keep their exact keys and coordinates.",
                f"{summary['logical_expert_parameters']} logical parameters / "
                f"{summary['expert_weight_tensors']} saved expert matrices; "
                f"{summary['expert_no_observed_update_tensors']} with no observed update; "
                f"{summary['expert_zero_update_tensors']} zero-reference matrices.",
                f"Expert matrix gain range: {_range(expert_rows, 'projection_gain')}; "
                f"relative residual range: {_range(expert_rows, 'relative_residual')}.",
                "",
            ]
        )
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
    update_kind = "matrix updates" if report.get("lora_biases") else "updates"
    aggregate = (
        f"Of {total:,} nonzero intended {update_kind}, {unchanged_total:,} are unchanged "
        f"({summary['unchanged_nonzero_update_fraction']:.2%})."
        if total
        else "The reference update is zero across all adapted matrices."
        if report.get("lora_biases")
        else "The reference update is zero across all adapted weights."
    )
    lines.extend(["", aggregate])
    energy = summary["unchanged_reference_energy_fraction"]
    if energy is not None:
        lines.append(
            f"Those unchanged entries contain {100 * energy:.4g}% of the reference squared norm."
        )
    lines.extend(_dora_row_lines(report["layers"]))
    if not summary_only:
        lines.append("* Nonzero intended update elements equal to the dtype-cast base.")
    if report.get("lora_biases"):
        biases = report["lora_biases"]
        lines.extend(
            [
                "",
                f"LoRA B biases: {summary['lora_bias_tensors']:,} vectors / "
                f"{summary['lora_bias_elements']:,} entries (separate from matrix counts).",
                "Reference: base bias + saved lora_B.bias * scale, rounded to merged dtype.",
                f"Nonzero bias updates unchanged: "
                f"{summary['lora_bias_unchanged_nonzero_update_elements']:,} / "
                f"{summary['lora_bias_intended_nonzero_elements']:,}; "
                f"reference-merge mismatches: "
                f"{summary['lora_bias_reference_merge_mismatches']:,}.",
                f"Bias gain range: {_range(biases, 'projection_gain')}; "
                f"relative residual range: {_range(biases, 'relative_residual')}.",
            ]
        )
        if not summary_only:
            lines.append("BIAS TENSOR                             DTYPE       GAIN  REL.RESID")
            for row in biases:
                name = ascii(row["weight"])[1:-1]
                if len(name) > 38:
                    name = "..." + name[-35:]
                gain = "n/a" if row["projection_gain"] is None else f"{row['projection_gain']:.4g}"
                residual = (
                    "n/a" if row["relative_residual"] is None else f"{row['relative_residual']:.4g}"
                )
                lines.append(f"{name:<39} {row['merged_dtype']:<6} {gain:>9} {residual:>10}")
    if report.get("saved_modules"):
        saved = report["saved_modules"]
        lines.extend(
            [
                "",
                f"Full saved modules: {summary['saved_module_target_mismatches']:,} target "
                f"mismatches across {summary['saved_module_elements']:,} elements in "
                f"{summary['saved_module_tensors']:,} tensors.",
                "Target = absolute adapter state rounded to merged dtype; "
                "integer/bool values are compared exactly.",
            ]
        )
        if not summary_only:
            lines.append("FULL SAVED TENSOR                       DTYPE    MISMATCHES / ELEMENTS")
            for row in saved:
                name = ascii(row["weight"])[1:-1]
                if len(name) > 38:
                    name = "..." + name[-35:]
                lines.append(
                    f"{name:<39} {row['merged_dtype']:<6} "
                    f"{row['target_mismatches']:>10,} / {row['elements']:,}"
                )
        missing = sum(row["unchanged_expected_change_elements"] or 0 for row in saved)
        unavailable = sum(row["base_comparison_unavailable"] is not None for row in saved)
        lines.extend(
            [
                f"Comparable full-state target changes equal to the original base: {missing:,}.",
                f"Original-base comparison unavailable for {unavailable:,} full-state tensor(s); "
                "no missing base aliases or resized values are inferred.",
            ]
        )
        if report["unmatched_saved_module_selectors"]:
            lines.append(
                "Saved-module selectors with no observed tensor prefix: "
                + ", ".join(ascii(s) for s in report["unmatched_saved_module_selectors"])
                + "."
            )
        lines.append("")
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


def _render_norms(report: dict[str, Any], *, summary_only: bool) -> str:
    summary = report["summary"]
    lines = [
        "Mergeglass | DoRA norms without a base",
        "",
        "Partial observations: reference directions and complete updates are unmeasured.",
        f"Observed {summary['norm_observed_rows']:,} rows in "
        f"{summary['norm_observed_tensors']:,} DoRA matrices.",
    ]
    if not summary_only:
        lines.extend(["", "MODULE                       DTYPE      NORM FACTOR RANGE"])
        for layer in report["layers"]:
            name = ascii(layer["module"])[1:-1]
            if len(name) > 27:
                name = "..." + name[-24:]
            pair = layer["dora_rows"]["log2_row_norm_factor_range"]
            factors = "n/a" if pair is None else f"{_factor_text(pair[0])}..{_factor_text(pair[1])}"
            lines.append(f"{name:<28} {layer['merged_dtype']:<6} {factors:>21}")
    lines.extend(["", *_dora_row_lines(report["layers"], include_direction=False)])
    lines.extend(
        [
            f"Checkpoint: {summary['checkpoint_tensors']:,} tensors; "
            f"{summary['unmapped_tensors']:,} unmapped tensor(s) have no base comparison.",
            "A matching norm does not establish a correct merge or direction.",
            "No pass/fail threshold is applied. This measures saved weights, not model quality.",
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
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Respect the usual '--' separator: a checkpoint can literally have the
    # flag's name without changing the existing three-path invocation.
    options = arguments[: arguments.index("--")] if "--" in arguments else arguments
    norms_only = "--dora-norms" in options
    parser = argparse.ArgumentParser(
        description=(
            "Observe saved DoRA row norms without an original base."
            if norms_only
            else "Inspect a saved LoRA or DoRA merge, locally and read-only."
        )
    )
    if not norms_only:
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
        "--dora-norms", action="store_true", help="omit BASE and observe DoRA row norms only"
    )
    if not norms_only:
        parser.add_argument(
            "--max-adapter-mib",
            type=int,
            default=512,
            help="maximum decoded fixed factor: Linear A or Embedding B (default 512 MiB)",
        )
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(arguments)
    try:
        sources = {"adapter": args.adapter, "merged": args.merged}
        if not norms_only:
            sources["base"] = args.base
        if args.json and args.json != "-":
            destination = Path(args.json).resolve()
            for source in sources.values():
                resolved = source.resolve()
                if destination == resolved or (
                    resolved.is_dir() and destination.is_relative_to(resolved)
                ):
                    raise AuditError("Write the report outside the input checkpoint paths")
        report = (
            audit_dora_norms(args.adapter, args.merged, block_mib=args.block_mib)
            if norms_only
            else audit(
                args.base,
                args.adapter,
                args.merged,
                block_mib=args.block_mib,
                max_adapter_mib=args.max_adapter_mib,
            )
        )
        if args.json and args.json != "-":
            # Snapshot files can be symlinks into a cache outside the supplied
            # directory. Protect those actual input files too, after inventory.
            for role, source in sources.items():
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
