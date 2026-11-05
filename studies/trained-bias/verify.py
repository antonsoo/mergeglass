"""Independent saved-bias oracle: stdlib safetensors bytes and scalar rounding.

No NumPy, Torch, PEFT, or Mergeglass storage helpers are used by the oracle.
The auditor is imported only to obtain the report under comparison.
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


def tensors(directory: Path) -> dict[str, tuple[str, list[int], bytes]]:
    result = {}
    for path in directory.glob("*.safetensors"):
        raw = path.read_bytes()
        size = int.from_bytes(raw[:8], "little")
        header = json.loads(raw[8 : 8 + size])
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            result[name] = (info["dtype"], info["shape"], raw[8 + size + start : 8 + size + end])
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
    assert len(values) == math.prod(shape)
    return values


def rounded(value: float, dtype: str) -> float:
    if dtype == "F32":
        return struct.unpack("<f", struct.pack("<f", value))[0]
    if dtype == "F16":
        return struct.unpack("<e", struct.pack("<e", value))[0]
    assert dtype == "BF16"
    bits = int.from_bytes(struct.pack("<f", value), "little")
    high, low = bits >> 16, bits & 65535
    high += int(low > 32768 or low == 32768 and high & 1)
    return struct.unpack("<f", (high << 16).to_bytes(4, "little"))[0]


def main() -> None:
    root, destination = map(Path, sys.argv[1:])
    manifest = json.loads((root / "manifest.json").read_text())
    for entry in manifest:
        data = (root / entry["file"]).read_bytes()
        assert len(data) == entry["bytes"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    results: list[dict[str, Any]] = []
    entries_checked = 0
    for profile in ["all", "lora_only", "dora-all", "conv1d-all"]:
        source = root / profile
        observations = json.loads((source / "observations.json").read_text())
        original = tensors(source / "base")
        outputs = [
            p
            for p in sorted(source.iterdir())
            if (p / "model.safetensors").exists() and p.name != "base"
        ]
        adapter_names = ["adapter"]
        if (source / "captured-complete-adapter").is_dir():
            adapter_names.append("captured-complete-adapter")
        for adapter_name in adapter_names:
            adapter = tensors(source / adapter_name)
            for merge in outputs:
                identity = {"profile": profile, "adapter": adapter_name, "merge": merge.name}
                try:
                    report = audit(
                        source / "base",
                        source / adapter_name,
                        merge,
                        block_mib=1,
                        max_adapter_mib=1,
                    )
                except AuditError as error:
                    assert observations["missing_trained_bias_keys"] and adapter_name == "adapter"
                    assert str(error).startswith("Missing trained base bias:")
                    results.append({**identity, "refusal": str(error)})
                    continue
                assert not (observations["missing_trained_bias_keys"] and adapter_name == "adapter")
                saved = tensors(merge)
                by_name = {row["weight"]: row for row in report["trained_biases"]}
                assert sorted(by_name) == observations["trained_biases"]
                mismatches = left_at_base = 0
                for name in observations["trained_biases"]:
                    key = "base_model.model." + name
                    wrapped = key.removesuffix(".bias") + ".base_layer.bias"
                    assert (key in adapter) != (wrapped in adapter)
                    target = decode(adapter[key if key in adapter else wrapped])
                    dtype = saved[name][0]
                    expected = [rounded(v, dtype) for v in target]
                    actual = decode(saved[name])
                    baseline = [rounded(v, dtype) for v in decode(original[name])]
                    bad = [
                        i for i, (a, b) in enumerate(zip(expected, actual, strict=True)) if a != b
                    ]
                    left = sum(
                        e != b and a == b
                        for e, a, b in zip(expected, actual, baseline, strict=True)
                    )
                    row = by_name[name]
                    assert row["target_mismatches"] == len(bad)
                    assert row["unchanged_expected_change_elements"] == left
                    error = math.hypot(*(a - b for a, b in zip(expected, actual, strict=True)))
                    assert math.isclose(row["norms"]["error"], error, rel_tol=1e-13, abs_tol=1e-15)
                    first = (
                        None
                        if not bad
                        else {
                            "flat_index": bad[0],
                            "cast_target": expected[bad[0]],
                            "merged": actual[bad[0]],
                        }
                    )
                    assert row["first_mismatch"] == first
                    mismatches += len(bad)
                    left_at_base += left
                    entries_checked += len(expected)
                assert report["summary"]["trained_bias_target_mismatches"] == mismatches
                assert (
                    report["summary"]["trained_bias_unchanged_expected_change_elements"]
                    == left_at_base
                )
                assert report["summary"]["unadapted_changed_tensors"] == 0
                # Independently verify each fault preserves every saved matrix byte.
                if merge.name in ["omitted-bias-f32", "scaled-bias-f32"]:
                    native = tensors(source / "in-memory-f32")
                    assert all(
                        value == native[name]
                        for name, value in saved.items()
                        if name not in by_name
                    )
                    assert mismatches == sum(row["elements"] for row in by_name.values())
                elif not observations["missing_trained_bias_keys"] or merge.name in [
                    "in-memory-f32",
                    "captured-reloaded-f32",
                ]:
                    assert mismatches == 0
                else:
                    assert mismatches == 10
                results.append(
                    {
                        **identity,
                        "summary": report["summary"],
                        "trained_biases": report["trained_biases"],
                    }
                )
    output = {
        "manifest_files_checked": len(manifest),
        "bias_entries_checked": entries_checked,
        "measured_cases": sum("summary" in r for r in results),
        "refused_cases": sum("refusal" in r for r in results),
        "results": results,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(gzip.compress((json.dumps(output, indent=2) + "\n").encode(), mtime=0))
    print(json.dumps({key: value for key, value in output.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
