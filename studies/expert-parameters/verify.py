"""Independent byte decoding and 80-digit expert products; no producer imports.

Reuse only the earlier study's stdlib safetensors decoder, not its auditor or
array arithmetic. Object-identity records bind native factors to parameters.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import runpy
import struct
from bisect import bisect_left
from decimal import Decimal, localcontext
from functools import lru_cache
from pathlib import Path
from typing import Any

from mergeglass import AuditError, audit

HELPERS = runpy.run_path(str(Path(__file__).resolve().parents[1] / "lora-bias/verify.py"))
values, payload = (HELPERS[name] for name in ("values", "payload"))
D = Decimal.from_float


@lru_cache(maxsize=1)
def bf16_values() -> list[float]:
    return [struct.unpack("<f", struct.pack("<I", bits << 16))[0] for bits in range(0x7F80)]


def rounded(value: float, dtype: str) -> float:
    if dtype != "BF16":
        return HELPERS["rounded"](value, dtype)
    table = bf16_values()
    at = bisect_left(table, abs(value))
    choices = [i for i in (at - 1, at) if 0 <= i < len(table)]
    # Choose between adjacent exact BF16 values, with even-bit ties. Do not
    # round through float32: that can manufacture a BF16 midpoint tie.
    best = min(choices, key=lambda i: (abs(D(table[i]) - D(abs(value))), i & 1))
    return math.copysign(table[best], value)


def close(actual: float | None, expected: Decimal) -> None:
    assert actual is not None and math.isclose(
        actual, float(expected), rel_tol=4e-12, abs_tol=1e-14
    ), (actual, str(expected))


def measure(
    root: Path, case: dict[str, Any], merge: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    directory = root / case["profile"]
    matrices = {row["weight"]: row for row in report["layers"] if "expert_parameter" in row}
    entries, rows, uncertain_total = 0, [], 0
    for parameter, identity in case["parameters"].items():
        _, a = values(directory / "adapter", identity["A"])
        _, b = values(directory / "adapter", identity["B"])
        experts, rank = identity["experts"], identity["rank"]
        _, output, input_ = identity["runtime_shape"]
        scale = D(identity["scale"])
        parts = (
            [("down_proj", 0, output)]
            if parameter.endswith("down_proj")
            else [("gate_proj", 0, output // 2), ("up_proj", output // 2, output // 2)]
        )
        module = parameter.rsplit(".", 1)[0]
        for expert in range(experts):
            for part, offset, height in parts:
                name = f"{module}.{expert}.{part}.weight"
                row = matrices.pop(name)
                assert row["expert_parameter"] == parameter and row["expert_index"] == expert
                assert row["factor_tensors"] == {"A": identity["A"], "B": identity["B"]}
                assert row["rank"] == rank and row["scale"] == identity["scale"]
                assert row["parameter_output_rows"] == [offset, offset + height]
                _, original = values(directory / "base", name)
                dtype, saved = values(directory / merge["name"], name)
                delta, observed, residual = [], [], []
                changed = erased = nonzero = minimum = uncertain = 0
                for index, (base, after) in enumerate(zip(original, saved, strict=True)):
                    output_row, input_column = divmod(index, input_)
                    products = [
                        D(b[(offset + output_row) * rank * experts + k * experts + expert])
                        * D(a[(expert * rank + k) * input_ + input_column])
                        for k in range(rank)
                    ]
                    intended = sum(products, Decimal(0)) * scale
                    cast_base = rounded(base, dtype)
                    seen = D(after) - D(cast_base)
                    delta.append(intended)
                    observed.append(seen)
                    residual.append(seen - intended)
                    changed += after != cast_base
                    nonzero += intended != 0
                    erased += intended != 0 and after == cast_base
                    # A conservative error interval covers float64 dot, scale and
                    # addition rounding. Near a saved-dtype midpoint the exact
                    # Decimal oracle reports ambiguity instead of selecting a tie.
                    bound = (
                        Decimal(32)
                        * D(2.0**-52)
                        * (
                            sum((abs(v) for v in products), Decimal(0)) * abs(scale)
                            + abs(D(base))
                            + abs(intended)
                        )
                    )
                    target = D(base) + intended
                    low = rounded(float(target - bound), dtype)
                    high = rounded(float(target + bound), dtype)
                    if low == high:
                        minimum += after != low
                    else:
                        uncertain += 1
                assert row["elements"] == len(delta)
                assert row["changed_elements"] == changed
                assert row["intended_nonzero_elements"] == nonzero
                assert row["unchanged_nonzero_update_elements"] == erased
                assert minimum <= row["reference_merge_mismatches"] <= minimum + uncertain
                denominator = sum((v * v for v in delta), Decimal(0))
                reference_norm = denominator.sqrt()
                residual_norm = sum((v * v for v in residual), Decimal(0)).sqrt()
                close(row["norms"]["delta"], reference_norm)
                close(row["norms"]["residual"], residual_norm)
                gain = None
                if denominator:
                    gain = (
                        sum((x * y for x, y in zip(observed, delta, strict=True)), Decimal(0))
                        / denominator
                    )
                    close(row["projection_gain"], gain)
                    close(row["relative_residual"], residual_norm / reference_norm)
                else:
                    assert row["projection_gain"] is None and row["relative_residual"] is None
                    assert row["status"] == (
                        "change-with-zero-reference" if changed else "zero-update"
                    )
                worst = row["largest_residual"]
                assert worst["expert"] == expert
                assert worst["parameter_output_row"] == offset + worst["row"]
                at = worst["row"] * input_ + worst["column"]
                close(abs(worst["residual"]), max(abs(v) for v in residual))
                close(worst["intended_delta"], delta[at])
                if (
                    merge["name"] == "omitted-expert"
                    and parameter == merge["parameter"]
                    and expert == merge["expert"]
                ):
                    assert row["status"] == "no-observed-update" and gain == 0
                if (
                    merge["name"] == "unscaled-parameter"
                    and parameter == merge["parameter"]
                    and denominator
                ):
                    assert math.isclose(float(gain), 1 / identity["scale"], rel_tol=2e-6)
                if merge["name"].startswith("correct-") and denominator:
                    assert math.isclose(float(gain), 1, rel_tol=0.005)
                entries += len(delta)
                uncertain_total += uncertain
                rows.append(
                    {
                        "weight": name,
                        "entries": len(delta),
                        "gain": float(gain) if gain is not None else None,
                        "relative_residual": float(residual_norm / reference_norm)
                        if denominator
                        else None,
                        "unchanged": erased,
                        "reference_mismatch_min": minimum,
                        "reference_mismatch_max": minimum + uncertain,
                    }
                )
    assert not matrices
    assert report["unadapted_changes"] == []
    if not merge["name"].startswith("correct-"):
        # An authored fault must preserve all unrelated native saved matrices.
        affected = {
            row["weight"]
            for row in report["layers"]
            if row.get("expert_parameter") == merge["parameter"]
            and (merge["name"] != "omitted-expert" or row["expert_index"] == merge["expert"])
        }
        index = directory / "correct-f32-safe-false/model.safetensors.index.json"
        names = (
            json.loads(index.read_text())["weight_map"]
            if index.exists()
            else HELPERS["file_data"](index.parent / "model.safetensors")[0]
        )
        for name in names:
            if name == "__metadata__" or name in affected:
                continue
            assert payload(directory / merge["name"], name) == payload(
                directory / "correct-f32-safe-false", name
            ), name
    return {"entries": entries, "ambiguous_rounding_entries": uncertain_total, "matrices": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--before", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        raw = (args.root / artifact["file"]).read_bytes()
        assert (
            len(raw) == artifact["bytes"] and hashlib.sha256(raw).hexdigest() == artifact["sha256"]
        )
    results, reports = [], []
    with localcontext() as context:
        context.prec = 80
        for case in manifest["cases"]:
            for merge in case["merges"]:
                directory = args.root / case["profile"]
                try:
                    report = audit(
                        directory / "base",
                        directory / "adapter",
                        directory / merge["name"],
                        block_mib=1,
                        max_adapter_mib=1,
                    )
                except AuditError as error:
                    assert args.before and "target_parameters" in str(error), str(error)
                    results.append(
                        {"profile": case["profile"], "merge": merge["name"], "refused": str(error)}
                    )
                    continue
                assert not args.before
                oracle = measure(args.root, case, merge, report)
                results.append({"profile": case["profile"], "merge": merge["name"], **oracle})
                reports.append(
                    {"profile": case["profile"], "merge": merge["name"], "report": report}
                )
    receipt = {
        "oracle": (
            "stdlib safetensors byte reader, 80-digit Decimal expert products, "
            "captured native parameter identity"
        ),
        "native_manifest_sha256": hashlib.sha256(
            (args.root / "manifest.json").read_bytes()
        ).hexdigest(),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": len(results),
        "entries": sum(r.get("entries", 0) for r in results),
        "results": results,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    if reports:
        args.output.with_suffix(".reports.json.gz").write_bytes(
            gzip.compress((json.dumps(reports, separators=(",", ":")) + "\n").encode(), mtime=0)
        )
    print(f"Verified {len(results)} merges / {receipt['entries']:,} expert entries.")


if __name__ == "__main__":
    main()
