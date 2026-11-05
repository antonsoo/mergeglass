"""Independent saved-bias oracle with mandatory case and precision coverage.

No NumPy, Torch, PEFT, or Mergeglass storage helpers are used by the oracle.
The auditor is imported only to obtain the report under comparison. Explicit
checks remain active under python -O and PYTHONOPTIMIZE.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

from mergeglass import AuditError, audit

PROFILES = ("all", "lora_only", "dora-all", "conv1d-all")
NATIVE_MERGES = tuple(
    f"reloaded-{precision}-safe-{safe}"
    for precision in ("f32", "f16", "bf16")
    for safe in ("false", "true")
)
CONTROLS = ("in-memory-f32", "omitted-bias-f32", "scaled-bias-f32")


class VerificationError(ValueError):
    """The complete declared evidence contract was not verified."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def identities(root: Path) -> int:
    manifest = json.loads((root / "manifest.json").read_text())
    names = [entry["file"] for entry in manifest]
    check(len(names) == len(set(names)), "Duplicate manifest file identity")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    actual.discard("manifest.json")
    check(set(names) == actual, "Manifest must cover exactly every producer file")
    for entry in manifest:
        data = (root / entry["file"]).read_bytes()
        check(len(data) == entry["bytes"], f"Byte count differs: {entry['file']}")
        check(
            hashlib.sha256(data).hexdigest() == entry["sha256"], f"SHA-256 differs: {entry['file']}"
        )
    return len(manifest)


def tensors(directory: Path) -> dict[str, tuple[str, list[int], bytes]]:
    result = {}
    for path in directory.glob("*.safetensors"):
        raw = path.read_bytes()
        size = int.from_bytes(raw[:8], "little")
        header = json.loads(raw[8 : 8 + size])
        for name, info in header.items():
            if name == "__metadata__":
                continue
            check(name not in result, f"Duplicate stored tensor: {name}")
            start, end = info["data_offsets"]
            result[name] = (info["dtype"], info["shape"], raw[8 + size + start : 8 + size + end])
    check(bool(result), f"No stored tensors: {directory.name}")
    return result


def decode(tensor: tuple[str, list[int], bytes]) -> list[float]:
    dtype, shape, raw = tensor
    if dtype == "BF16":
        values = [
            struct.unpack("<f", (v << 16).to_bytes(4, "little"))[0]
            for (v,) in struct.iter_unpack("<H", raw)
        ]
    else:
        values = [float(v) for (v,) in struct.iter_unpack({"F32": "<f", "F16": "<e"}[dtype], raw)]
    check(len(values) == math.prod(shape), "Decoded tensor size differs from its shape")
    check(all(math.isfinite(v) for v in values), "Nonfinite oracle input")
    return values


def rounded(value: float, dtype: str) -> float:
    if dtype == "F32":
        return struct.unpack("<f", struct.pack("<f", value))[0]
    if dtype == "F16":
        return struct.unpack("<e", struct.pack("<e", value))[0]
    check(dtype == "BF16", f"Unsupported oracle dtype: {dtype}")
    bits = int.from_bytes(struct.pack("<f", value), "little")
    high, low = bits >> 16, bits & 65535
    high += int(low > 32768 or low == 32768 and high & 1)
    return struct.unpack("<f", (high << 16).to_bytes(4, "little"))[0]


def saved_change_counts(
    target: list[Any],
    original: list[Any],
    expected: list[Any],
    baseline: list[Any],
    actual: list[Any],
) -> dict[str, int]:
    rows = list(zip(target, original, expected, baseline, actual, strict=True))
    return {
        "saved_change_elements": sum(t != o for t, o, _e, _b, _a in rows),
        "unchanged_saved_change_elements": sum(t != o and a == b for t, o, _e, b, a in rows),
        "cast_erased_change_elements": sum(t != o and e == b for t, o, e, b, _a in rows),
    }


def saved_change_witnesses(
    target: list[Any],
    original: list[Any],
    expected: list[Any],
    baseline: list[Any],
    actual: list[Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "first_unchanged_saved_change": None,
        "first_cast_erased_change": None,
    }
    for index, (t, o, e, b, a) in enumerate(
        zip(target, original, expected, baseline, actual, strict=True)
    ):
        if t == o:
            continue
        witness = {
            "flat_index": index,
            "base": o,
            "saved_target": t,
            "cast_base": b,
            "cast_target": e,
            "merged": a,
            "reference_cast_erased": e == b,
        }
        if a == b and result["first_unchanged_saved_change"] is None:
            result["first_unchanged_saved_change"] = witness
        if e == b and result["first_cast_erased_change"] is None:
            result["first_cast_erased_change"] = witness
    return result


def verify(root: Path) -> dict[str, Any]:
    file_count = identities(root)
    results: list[dict[str, Any]] = []
    entries_checked = expected_measured = expected_refused = 0
    for profile in PROFILES:
        source = root / profile
        observations = json.loads((source / "observations.json").read_text())
        check(observations["profile"] == profile, f"Profile identity differs: {profile}")
        names = [row["name"] for row in observations["merges"]]
        check(
            len(names) == len(set(names)) and set(names) == set(NATIVE_MERGES),
            f"Incomplete native precision/safe-merge declarations: {profile}",
        )
        missing = bool(observations["missing_trained_bias_keys"])
        check(not missing or profile == "lora_only", "Unexpected missing-bias profile")
        expected_names = set(NATIVE_MERGES + CONTROLS)
        if missing:
            expected_names.add("captured-reloaded-f32")
        outputs = {
            p.name
            for p in source.iterdir()
            if p.is_dir() and (p / "model.safetensors").exists() and p.name != "base"
        }
        check(
            outputs == expected_names,
            f"Incomplete or unexpected merge cases: {profile}; "
            f"missing={sorted(expected_names - outputs)}, extra={sorted(outputs - expected_names)}",
        )
        original = tensors(source / "base")
        adapter_names = ["adapter"]
        captured = source / "captured-complete-adapter"
        check(captured.is_dir() == missing, f"Captured adapter coverage differs: {profile}")
        if missing:
            adapter_names.append(captured.name)
            expected_refused += len(outputs)
        expected_measured += len(outputs)
        for adapter_name in adapter_names:
            adapter = tensors(source / adapter_name)
            for name in sorted(outputs):
                merge = source / name
                identity = {"profile": profile, "adapter": adapter_name, "merge": name}
                label = f"{profile}/{adapter_name}/{name}"
                saved = tensors(merge)
                dtype = "BF16" if "-bf16-" in name else "F16" if "-f16-" in name else "F32"
                check(
                    all(t[0] == dtype for t in saved.values()),
                    f"Stored precision differs from declared {dtype}: {label}",
                )
                try:
                    report = audit(
                        source / "base",
                        source / adapter_name,
                        merge,
                        block_mib=1,
                        max_adapter_mib=1,
                    )
                except AuditError as error:
                    check(
                        missing and adapter_name == "adapter",
                        f"Unexpected refusal: {label}: {error}",
                    )
                    check(
                        str(error).startswith("Missing trained base bias:"),
                        f"Wrong refusal reason: {label}: {error}",
                    )
                    results.append({**identity, "refusal": str(error)})
                    continue
                check(
                    not (missing and adapter_name == "adapter"),
                    f"Incomplete adapter was accepted: {label}",
                )
                by_name = {row["weight"]: row for row in report["trained_biases"]}
                check(
                    sorted(by_name) == observations["trained_biases"],
                    f"Trained bias inventory differs: {label}",
                )
                mismatches = left_at_base = 0
                retention = dict.fromkeys(
                    (
                        "saved_change_elements",
                        "unchanged_saved_change_elements",
                        "cast_erased_change_elements",
                    ),
                    0,
                )
                for weight in observations["trained_biases"]:
                    key = "base_model.model." + weight
                    wrapped = key.removesuffix(".bias") + ".base_layer.bias"
                    check(
                        (key in adapter) != (wrapped in adapter), f"Bias alias ambiguity: {weight}"
                    )
                    target = decode(adapter[key if key in adapter else wrapped])
                    expected = [rounded(v, dtype) for v in target]
                    actual = decode(saved[weight])
                    row = by_name[weight]
                    base_values = decode(original[weight])
                    baseline = [rounded(v, dtype) for v in base_values]
                    for field, count in saved_change_counts(
                        target, base_values, expected, baseline, actual
                    ).items():
                        check(
                            row[field] == count,
                            f"Saved change count differs: {label}/{weight}/{field}",
                        )
                        retention[field] += count
                    for field, witness in saved_change_witnesses(
                        target, base_values, expected, baseline, actual
                    ).items():
                        check(
                            row[field] == witness,
                            f"Saved change witness differs: {label}/{weight}/{field}",
                        )
                    bad = [
                        i for i, (a, b) in enumerate(zip(expected, actual, strict=True)) if a != b
                    ]
                    left = sum(
                        e != b and a == b
                        for e, a, b in zip(expected, actual, baseline, strict=True)
                    )
                    check(
                        row["target_mismatches"] == len(bad),
                        f"Mismatch count differs: {label}/{weight}",
                    )
                    check(
                        row["unchanged_expected_change_elements"] == left,
                        f"Base equality count differs: {label}/{weight}",
                    )
                    error = math.hypot(*(a - b for a, b in zip(expected, actual, strict=True)))
                    check(
                        math.isclose(row["norms"]["error"], error, rel_tol=1e-13, abs_tol=1e-15),
                        f"Error norm differs: {label}/{weight}",
                    )
                    first = (
                        None
                        if not bad
                        else {
                            "flat_index": bad[0],
                            "cast_target": expected[bad[0]],
                            "merged": actual[bad[0]],
                        }
                    )
                    check(
                        row["first_mismatch"] == first,
                        f"Mismatch coordinate differs: {label}/{weight}",
                    )
                    mismatches += len(bad)
                    left_at_base += left
                    entries_checked += len(expected)
                summary = report["summary"]
                for field, count in retention.items():
                    check(
                        summary["trained_bias_" + field] == count,
                        f"Saved change summary differs: {label}",
                    )
                check(
                    summary["trained_bias_target_mismatches"] == mismatches,
                    f"Summary mismatches differ: {label}",
                )
                check(
                    summary["trained_bias_unchanged_expected_change_elements"] == left_at_base,
                    f"Summary base equality differs: {label}",
                )
                check(summary["unadapted_changed_tensors"] == 0, f"Unadapted change: {label}")
                if name in ("omitted-bias-f32", "scaled-bias-f32"):
                    native = tensors(source / "in-memory-f32")
                    check(set(saved) == set(native), f"Fault tensor inventory differs: {label}")
                    check(
                        all(
                            value == native[key]
                            for key, value in saved.items()
                            if key not in by_name
                        ),
                        f"Fault changes a non-bias tensor: {label}",
                    )
                    check(
                        mismatches == sum(row["elements"] for row in by_name.values()),
                        f"Fault does not affect every declared bias: {label}",
                    )
                elif not missing or name in ("in-memory-f32", "captured-reloaded-f32"):
                    check(mismatches == 0, f"Native merge has incorrect bias targets: {label}")
                else:
                    check(mismatches == 10, f"Historical omission count differs: {label}")
                results.append(
                    {**identity, "summary": summary, "trained_biases": report["trained_biases"]}
                )
    measured = sum("summary" in r for r in results)
    refused = sum("refusal" in r for r in results)
    check(
        measured == expected_measured and refused == expected_refused,
        "Complete case matrix was not verified",
    )
    return {
        "state": "verified",
        "manifest_files_checked": file_count,
        "bias_entries_checked": entries_checked,
        "measured_cases": measured,
        "refused_cases": refused,
        "expected_measured_cases": expected_measured,
        "expected_refused_cases": expected_refused,
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
            "Write verification output outside the producer directory",
        )
        output = verify(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            gzip.compress((json.dumps(output, indent=2) + "\n").encode(), mtime=0)
        )
    except (VerificationError, OSError, ValueError, KeyError, TypeError, struct.error) as error:
        print(f"verification failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({key: value for key, value in output.items() if key != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
