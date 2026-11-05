"""Independent source-change, reference-cast and observed-equality bias oracle."""

from __future__ import annotations

import gzip
import json
import runpy
import struct
import sys
from pathlib import Path
from typing import Any

from mergeglass import AuditError, audit

READER = runpy.run_path(str(Path(__file__).with_name("verify.py")))
check, identities, tensors, decode, rounded = [
    READER[k] for k in ("check", "identities", "tensors", "decode", "rounded")
]
VerificationError = READER["VerificationError"]
saved_change_witnesses = READER["saved_change_witnesses"]
FIELDS = ("saved_change_elements", "unchanged_saved_change_elements", "cast_erased_change_elements")
CASES = {f"merged-{p}-safe-{s}" for p in ("f32", "f16", "bf16") for s in ("true", "false")}


def ordered_float32(value: float) -> int:
    """Adjacent finite FP32 values differ by one; signed zeros share one position."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    magnitude = bits & 0x7FFFFFFF
    return -magnitude if bits >> 31 else magnitude


def verify(root: Path) -> dict[str, Any]:
    files = identities(root)
    results = []
    checked = 0
    for profile in ["bias", "head"]:
        source = root / profile
        observations = json.loads((source / "observations.json").read_text())
        check(observations["profile"] == profile, "Wrong precision-control profile")
        check(
            set(observations["merges"]) == CASES and len(observations["merges"]) == 6,
            "Incomplete precision-control declarations",
        )
        actual = {
            p.name
            for p in source.iterdir()
            if (p / "model.safetensors").exists() and p.name != "base"
        }
        check(actual == CASES, "Incomplete precision-control output matrix")
        original = tensors(source / "base")
        adapter = tensors(source / "adapter")
        check(
            all(t[0] == "F32" for t in [*original.values(), *adapter.values()]),
            "Precision controls must retain FP32 original and adapter source values",
        )
        for name in sorted(CASES):
            merged = tensors(source / name)
            dtype = "BF16" if "-bf16-" in name else "F16" if "-f16-" in name else "F32"
            check(
                all(t[0] == dtype for t in merged.values()),
                f"Mislabeled precision: {profile}/{name}",
            )
            report = audit(
                source / "base", source / "adapter", source / name, block_mib=1, max_adapter_mib=1
            )
            ordinary = {r["weight"]: r for r in report["trained_biases"]}
            full = {r["weight"]: r for r in report.get("saved_modules", [])}
            check(
                set(ordinary)
                == (
                    {"first.bias", "second.bias", "head.bias"}
                    if profile == "bias"
                    else {"first.bias", "second.bias"}
                ),
                "Incomplete bias coverage",
            )
            check(
                set(full) == (set() if profile == "bias" else {"head.weight", "head.bias"}),
                "Incomplete full-head coverage",
            )
            totals = {"bias": dict.fromkeys(FIELDS, 0), "head": dict.fromkeys(FIELDS, 0)}
            for weight, row in {**ordinary, **full}.items():
                key = "base_model.model." + weight
                if weight in ("first.bias", "second.bias"):
                    key = key.removesuffix(".bias") + ".base_layer.bias"
                saved_values = decode(adapter[key])
                base_values = decode(original[weight])
                for saved_value, base_value in zip(saved_values, base_values, strict=True):
                    check(
                        abs(ordered_float32(saved_value) - ordered_float32(base_value)) == 1,
                        "Control is not exactly one FP32 step from the base",
                    )
                    check(
                        rounded(base_value, "BF16") == base_value, "Control base is not BF16-exact"
                    )
                expected = [rounded(v, dtype) for v in saved_values]
                baseline = [rounded(v, dtype) for v in base_values]
                after = decode(merged[weight])
                for field, witness in saved_change_witnesses(
                    saved_values, base_values, expected, baseline, after
                ).items():
                    check(
                        row[field] == witness,
                        f"Saved change witness differs: {profile}/{name}/{weight}/{field}",
                    )
                check(
                    after == expected and row["target_mismatches"] == 0,
                    "Native control does not match its correct absolute target",
                )
                source_changed = [s != b for s, b in zip(saved_values, base_values, strict=True)]
                cast_changed = [s != b for s, b in zip(expected, baseline, strict=True)]
                after_same = [s == b for s, b in zip(after, baseline, strict=True)]
                counts = {
                    "saved_change_elements": sum(source_changed),
                    "unchanged_saved_change_elements": sum(
                        s and same for s, same in zip(source_changed, after_same, strict=True)
                    ),
                    "cast_erased_change_elements": sum(
                        s and not cast for s, cast in zip(source_changed, cast_changed, strict=True)
                    ),
                }
                for field, count in counts.items():
                    check(row[field] == count, f"{field} differs: {profile}/{name}/{weight}")
                    totals["bias" if weight in ordinary else "head"][field] += count
                check(
                    row["expected_change_elements"] == sum(cast_changed),
                    "Post-cast change count differs",
                )
                check(
                    row["unchanged_expected_change_elements"] == 0,
                    "Post-cast targets are incorrect",
                )
                check(
                    row["changed_from_base_elements"] == len(after) - sum(after_same),
                    "Observed change count differs",
                )
                checked += len(after)
            for field in FIELDS:
                check(
                    report["summary"]["trained_bias_" + field] == totals["bias"][field],
                    "Bias summary differs",
                )
            for category, expected_changes in [
                ("bias", observations["ordinary_bias_elements"]),
                ("head", observations["full_saved_elements"]),
            ]:
                counts = totals[category]
                check(
                    counts["saved_change_elements"] == expected_changes,
                    "Not every source change was retained in FP32",
                )
                lost = 0 if dtype == "F32" else expected_changes
                check(
                    counts["unchanged_saved_change_elements"] == lost
                    and counts["cast_erased_change_elements"] == lost,
                    "One-ulp control did not exercise complete reference-cast erasure",
                )
            results.append(
                {
                    "profile": profile,
                    "merge": name,
                    "summary": report["summary"],
                    "independent_counts": totals,
                    "trained_biases": report["trained_biases"],
                    "saved_modules": report.get("saved_modules", []),
                }
            )
    check(len(results) == 12, "Incomplete precision matrix")
    return {
        "state": "verified",
        "manifest_files_checked": files,
        "measured_cases": len(results),
        "physical_entries_checked": checked,
        "results": results,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: verify_precision.py PRODUCER_DIRECTORY OUTPUT.json.gz", file=sys.stderr)
        return 2
    root, output = map(Path, sys.argv[1:])
    try:
        check(
            not output.resolve().is_relative_to(root.resolve()),
            "Write output outside producer directory",
        )
        receipt = verify(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(gzip.compress((json.dumps(receipt, indent=2) + "\n").encode(), mtime=0))
    except (VerificationError, AuditError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"verification failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in receipt.items() if k != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
