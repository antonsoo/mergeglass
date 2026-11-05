"""Independent byte/scalar oracle for trained biases combined with saved heads.

Reuses the earlier study's stdlib-only reader and mandatory check function.
No audit implementation helpers construct targets, aliases or integer values.
"""

from __future__ import annotations

import gzip
import json
import math
import runpy
import struct
import sys
from pathlib import Path
from typing import Any

from mergeglass import AuditError, audit

READER = runpy.run_path(str(Path(__file__).parents[1] / "trained-bias/verify.py"))
check, identities, tensors, decode, rounded = [
    READER[name] for name in ("check", "identities", "tensors", "decode", "rounded")
]
VerificationError = READER["VerificationError"]
saved_change_counts = READER["saved_change_counts"]
saved_change_witnesses = READER["saved_change_witnesses"]
PROFILES = ("all-head", "lora-only-head", "dora-all-head", "nested-all-head")
NATIVE = {f"reloaded-{p}-safe-{s}" for p in ("f32", "f16", "bf16") for s in ("true", "false")}
OUTPUTS = NATIVE | {"in-memory-f32", "omitted-bias-f32", "scaled-bias-f32", "omitted-head-f32"}


def values(tensor: tuple[str, list[int], bytes]) -> list[Any]:
    if tensor[0] == "I64":
        result = [v for (v,) in struct.iter_unpack("<q", tensor[2])]
        check(len(result) == math.prod(tensor[1]), "Invalid integer tensor shape")
        return result
    return decode(tensor)


def target(tensor: tuple[str, list[int], bytes], dtype: str) -> list[Any]:
    if dtype == "I64":
        check(tensor[0] == dtype, "Integer target dtype differs")
        return values(tensor)
    return [rounded(v, dtype) for v in decode(tensor)]


def verify(root: Path) -> dict[str, Any]:
    file_count = identities(root)
    results = []
    physical_entries = alias_entries = 0
    for profile in PROFILES:
        source = root / profile
        observations = json.loads((source / "observations.json").read_text())
        check(observations["profile"] == profile, f"Wrong profile: {profile}")
        names = [r["name"] for r in observations["native_merges"]]
        check(
            set(names) == NATIVE and len(names) == 6, f"Incomplete native declarations: {profile}"
        )
        actual = {
            p.name
            for p in source.iterdir()
            if (p / "model.safetensors").exists() and p.name != "base"
        }
        check(actual == OUTPUTS, f"Incomplete output matrix: {profile}")
        original = tensors(source / "base")
        adapter = tensors(source / "adapter")
        native = tensors(source / "in-memory-f32")
        heads = set(observations["full_saved_tensors"])
        check(
            heads == {n for n in original if n.startswith("head.")}, "Full head inventory differs"
        )
        bias_mode = "lora_only" if profile == "lora-only-head" else "all"
        for name in sorted(OUTPUTS):
            saved = tensors(source / name)
            label = f"{profile}/{name}"
            check(set(saved) == set(original), f"Tensor inventory differs: {label}")
            dtype = "BF16" if "-bf16-" in name else "F16" if "-f16-" in name else "F32"
            check(
                all(
                    t[0] == ("I64" if original[n][0] == "I64" else dtype) for n, t in saved.items()
                ),
                f"Stored precision differs: {label}",
            )
            report = audit(
                source / "base", source / "adapter", source / name, block_mib=1, max_adapter_mib=1
            )
            biases = {r["weight"]: r for r in report["trained_biases"]}
            full = {r["weight"]: r for r in report["saved_modules"]}
            check(
                set(biases) == {"first.bias", "second.bias"}, f"Standalone bias inventory: {label}"
            )
            check(set(full) == heads, f"Head report inventory: {label}")
            check(
                set(observations["trained_biases"])
                == set(biases) | {n for n in heads if n.endswith(".bias")},
                f"Trainable bias coverage differs: {label}",
            )
            counts = {"trained": 0, "full": 0}
            retention = dict.fromkeys(
                (
                    "saved_change_elements",
                    "unchanged_saved_change_elements",
                    "cast_erased_change_elements",
                ),
                0,
            )
            for weight, row in {**biases, **full}.items():
                key = "base_model.model." + weight
                if weight in biases:
                    key = key.removesuffix(".bias") + ".base_layer.bias"
                expected = target(adapter[key], saved[weight][0])
                actual_values = values(saved[weight])
                baseline = target(original[weight], saved[weight][0])
                for field, count in saved_change_counts(
                    values(adapter[key]),
                    values(original[weight]),
                    expected,
                    baseline,
                    actual_values,
                ).items():
                    check(
                        row[field] == count, f"Saved change count differs: {label}/{weight}/{field}"
                    )
                    if weight in biases:
                        retention[field] += count
                for field, witness in saved_change_witnesses(
                    values(adapter[key]),
                    values(original[weight]),
                    expected,
                    baseline,
                    actual_values,
                ).items():
                    check(
                        row[field] == witness,
                        f"Saved change witness differs: {label}/{weight}/{field}",
                    )
                bad = [
                    i
                    for i, (a, b) in enumerate(zip(expected, actual_values, strict=True))
                    if a != b
                ]
                left = sum(
                    e != b and a == b
                    for e, a, b in zip(expected, actual_values, baseline, strict=True)
                )
                check(
                    row["target_mismatches"] == len(bad),
                    f"Mismatch count differs: {label}/{weight}",
                )
                check(
                    row["unchanged_expected_change_elements"] == left,
                    f"Base equality differs: {label}/{weight}",
                )
                first = (
                    None
                    if not bad
                    else {
                        "flat_index": bad[0],
                        "cast_target": expected[bad[0]],
                        "merged": actual_values[bad[0]],
                    }
                )
                check(
                    row["first_mismatch"] == first, f"Mismatch coordinate differs: {label}/{weight}"
                )
                if saved[weight][0] == "I64":
                    check(row["norms"] is None, f"Integer norm was invented: {label}/{weight}")
                    if name == "omitted-head-f32":
                        check(
                            first == {"flat_index": 0, "cast_target": 2**60 + 12, "merged": 2**60},
                            "Integer counter lost identity above 2**53",
                        )
                else:
                    norm = math.hypot(
                        *(a - b for a, b in zip(expected, actual_values, strict=True))
                    )
                    check(
                        math.isclose(row["norms"]["error"], norm, rel_tol=1e-13, abs_tol=1e-15),
                        f"Error norm differs: {label}/{weight}",
                    )
                counts["trained" if weight in biases else "full"] += len(bad)
                physical_entries += len(expected)
            summary = report["summary"]
            for field, count in retention.items():
                check(
                    summary["trained_bias_" + field] == count,
                    f"Saved change summary differs: {label}",
                )
            check(
                summary["trained_bias_target_mismatches"] == counts["trained"],
                "Trained bias summary differs",
            )
            check(
                summary["saved_module_target_mismatches"] == counts["full"],
                "Saved module summary differs",
            )
            check(summary["unadapted_changed_tensors"] == 0, f"Unadapted tensors changed: {label}")
            # Independently map the two native wrapper copies to their known references.
            expected_aliases = {}
            if bias_mode == "all":
                for weight in heads:
                    if not weight.endswith(".bias"):
                        continue
                    suffix = weight.removeprefix("head.")
                    slot = ".default" if profile == "nested-all-head" else ""
                    expected_aliases[f"base_model.model.head.modules_to_save{slot}.{suffix}"] = (
                        "saved-full-module",
                        "base_model.model." + weight,
                        adapter["base_model.model." + weight],
                    )
                    expected_aliases[f"base_model.model.head.original_module.{suffix}"] = (
                        "original-base",
                        weight,
                        original[weight],
                    )
            aliases = {r["tensor"]: r for r in report.get("saved_bias_aliases", [])}
            check(set(aliases) == set(expected_aliases), f"Alias coverage differs: {label}")
            for key, (kind, reference_name, reference) in expected_aliases.items():
                copy = adapter[key]
                check(values(copy) == target(reference, copy[0]), f"Alias bytes differ: {key}")
                check(
                    aliases[key]["reference_kind"] == kind
                    and aliases[key]["reference_tensor"] == reference_name,
                    f"Alias reference differs: {key}",
                )
                alias_entries += len(values(copy))
            if name in NATIVE or name == "in-memory-f32":
                check(counts == {"trained": 0, "full": 0}, f"Incorrect native merge: {label}")
            else:
                check(
                    all(
                        v == native[k]
                        for k, v in saved.items()
                        if k not in heads and not k.endswith(".bias")
                    ),
                    f"Fault changed a matrix: {label}",
                )
                if name == "omitted-head-f32":
                    check(
                        counts["trained"] == 0 and counts["full"] > 0, "Lost head was not detected"
                    )
                    check(
                        all(saved[k] == original[k] for k in heads),
                        "Head omission is not original bytes",
                    )
                else:
                    check(
                        counts["trained"] == 10 and counts["full"] > 0,
                        "Bias fault was not detected",
                    )
                    check(
                        all(v == native[k] for k, v in saved.items() if not k.endswith(".bias")),
                        "Bias fault changed a non-bias tensor",
                    )
            results.append(
                {
                    "profile": profile,
                    "merge": name,
                    "summary": summary,
                    "trained_biases": report["trained_biases"],
                    "saved_modules": report["saved_modules"],
                    "saved_bias_aliases": report.get("saved_bias_aliases", []),
                }
            )
        conflicts = observations["conflicting_adapters"]
        expected_conflicts = (
            set()
            if bias_mode == "lora_only"
            else {"conflicting-original_module-adapter", "conflicting-modules_to_save-adapter"}
        )
        check(
            {r["name"] for r in conflicts} == expected_conflicts
            and len(conflicts) == len(expected_conflicts),
            f"Incomplete conflict matrix: {profile}",
        )
        for conflict in conflicts:
            try:
                audit(
                    source / "base", source / conflict["name"], source / "reloaded-f32-safe-false"
                )
            except AuditError as error:
                check(
                    str(error).startswith("Redundant bias copy differs"),
                    f"Wrong conflict refusal: {error}",
                )
                results.append(
                    {"profile": profile, "adapter": conflict["name"], "refusal": str(error)}
                )
            else:
                check(False, "Conflicting wrapper copy was ignored")
    measured = sum("summary" in r for r in results)
    refused = sum("refusal" in r for r in results)
    check(measured == 40 and refused == 6, "Incomplete measured/refused matrix")
    return {
        "state": "verified",
        "manifest_files_checked": file_count,
        "physical_entries_checked": physical_entries,
        "alias_entries_checked": alias_entries,
        "measured_cases": measured,
        "refused_cases": refused,
        "results": results,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: verify.py PRODUCER_DIRECTORY OUTPUT.json.gz", file=sys.stderr)
        return 2
    root, destination = map(Path, sys.argv[1:])
    try:
        check(
            not destination.resolve().is_relative_to(root.resolve()),
            "Write output outside producer directory",
        )
        output = verify(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            gzip.compress((json.dumps(output, indent=2) + "\n").encode(), mtime=0)
        )
    except (
        VerificationError,
        AuditError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        struct.error,
    ) as error:
        print(f"verification failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in output.items() if k != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
