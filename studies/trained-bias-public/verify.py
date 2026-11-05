"""Stream identities and independently check every public saved-bias coordinate.

The scalar oracle uses only stdlib readers and round-to-nearest-even arithmetic.
Mergeglass is invoked separately to obtain the reports under comparison.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path

from mergeglass import audit

READER = runpy.run_path(str(Path(__file__).parents[1] / "trained-bias/verify.py"))
check, decode, rounded, saved_change_counts, saved_change_witnesses = [
    READER[k]
    for k in ("check", "decode", "rounded", "saved_change_counts", "saved_change_witnesses")
]
CASES = {"native-f32", "native-f16", "native-bf16", "omitted-bias-f32", "remapped-f32"}


def digest(path: Path, start: int = 0, length: int | None = None) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(start)
        remaining = path.stat().st_size - start if length is None else length
        while remaining:
            block = stream.read(min(remaining, 1024 * 1024))
            check(bool(block), f"Truncated payload: {path.name}")
            result.update(block)
            remaining -= len(block)
    return result.hexdigest()


class Index:
    """Retain headers, decode only bias vectors, stream large matrix hashes."""

    def __init__(self, directory: Path):
        self.entries = {}
        for path in sorted(directory.glob("*.safetensors")):
            with path.open("rb") as stream:
                size = int.from_bytes(stream.read(8), "little")
                check(0 < size < 16 * 1024 * 1024, "Invalid study header size")
                header = json.loads(stream.read(size))
            for key, info in header.items():
                if key == "__metadata__":
                    continue
                check(key not in self.entries, f"Duplicate tensor: {key}")
                start, end = info["data_offsets"]
                width = {"F32": 4, "F16": 2, "BF16": 2}[info["dtype"]]
                check(
                    0 <= start <= end and end + 8 + size <= path.stat().st_size,
                    "Invalid tensor offsets",
                )
                check(end - start == math.prod(info["shape"]) * width, "Invalid payload length")
                self.entries[key] = (
                    path,
                    8 + size + start,
                    end - start,
                    info["dtype"],
                    info["shape"],
                )
        check(bool(self.entries), f"No tensors: {directory.name}")

    def vector(self, key: str) -> list[float]:
        path, offset, length, dtype, shape = self.entries[key]
        check(len(shape) == 1 and length <= 1024 * 1024, "Expected bounded bias vector")
        with path.open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(length)
        return decode((dtype, shape, raw))

    def payload_hash(self, key: str) -> tuple:
        path, offset, length, dtype, shape = self.entries[key]
        return dtype, shape, digest(path, offset, length)


def verify(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    names = [r["file"] for r in manifest]
    check(len(names) == len(set(names)), "Duplicate producer file")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    check(set(names) == actual - {"manifest.json"}, "Incomplete producer file inventory")
    for row in manifest:
        path = root / row["file"]
        check(
            path.stat().st_size == row["bytes"] and digest(path) == row["sha256"],
            f"Producer identity differs: {row['file']}",
        )
    producer = json.loads((root / "producer.json").read_text())
    observed_names = [r["name"] for r in producer["outputs"]]
    check(
        set(observed_names) == CASES - {"omitted-bias-f32"} and len(observed_names) == 4,
        "Incomplete native producer output matrix",
    )
    folders = {p.name for p in root.iterdir() if p.is_dir()}
    check(folders == CASES | {"base", "adapter", "remapped-adapter"}, "Incomplete native folders")
    base, adapter = Index(root / "base"), Index(root / "adapter")
    mapped = Index(root / "remapped-adapter")
    mapping = {r["original"]: r["mapped"] for r in producer["key_only_mapping"]}
    check(len(mapping) == len(producer["key_only_mapping"]) == 41, "Incomplete key control")
    check(len(set(mapping.values())) == 41, "Colliding mapped keys")
    check(
        set(mapped.entries) == {mapping.get(k, k) for k in adapter.entries},
        "Key control has extra or missing tensors",
    )
    for key in adapter.entries:
        check(
            adapter.payload_hash(key) == mapped.payload_hash(mapping.get(key, key)),
            f"Key-only control changed tensor payload: {key}",
        )
    check(
        (root / "adapter/adapter_config.json").read_bytes()
        == (root / "remapped-adapter/adapter_config.json").read_bytes(),
        "Control config differs",
    )
    bias_keys = {k for k in adapter.entries if k.endswith(".bias")}
    check(len(adapter.entries) == 224 and len(bias_keys) == 62, "Incomplete adapter inventory")
    check(len(base.entries) == 165, "Incomplete base inventory")
    ordinary = Index(root / "native-f32")
    nonbias = set(ordinary.entries) - {k.removeprefix("base_model.model.") for k in bias_keys}
    check(len(nonbias) == 103, "Unexpected non-bias inventory")
    # Establish that the explicit controls changed only bias payloads: every
    # matrix and other persistent tensor is byte-identical to the native F32 merge.
    original_hashes = {k: ordinary.payload_hash(k) for k in nonbias}
    for case in ("remapped-f32", "omitted-bias-f32"):
        control = Index(root / case)
        check(set(control.entries) == set(ordinary.entries), "Control coverage differs")
        for key, expected in original_hashes.items():
            check(control.payload_hash(key) == expected, f"Control changed non-bias tensor: {key}")
    reports, results = {}, []
    for name in sorted(CASES):
        merged = Index(root / name)
        dtype = "BF16" if name == "native-bf16" else "F16" if name == "native-f16" else "F32"
        check(set(merged.entries) == set(base.entries), "Incomplete merged tensor coverage")
        check(all(v[3] == dtype for v in merged.entries.values()), "Mislabeled output precision")
        report = audit(
            root / "base", root / "adapter", root / name, block_mib=16, max_adapter_mib=16
        )
        rows = {r["weight"]: r for r in report["trained_biases"]}
        check(
            len(rows) == 62
            and set(rows) == {k.removeprefix("base_model.model.") for k in bias_keys},
            "Bias coverage differs",
        )
        totals = dict.fromkeys(
            (
                "target_mismatches",
                "saved_change_elements",
                "unchanged_saved_change_elements",
                "cast_erased_change_elements",
            ),
            0,
        )
        loaded = next((r for r in producer["outputs"] if r["name"] == name), None)
        for key in sorted(bias_keys):
            weight = key.removeprefix("base_model.model.")
            target, original, after = (
                adapter.vector(key),
                base.vector(weight),
                merged.vector(weight),
            )
            expected, baseline = (
                [rounded(v, dtype) for v in values] for values in (target, original)
            )
            counts = saved_change_counts(target, original, expected, baseline, after)
            counts["target_mismatches"] = sum(a != e for a, e in zip(after, expected, strict=True))
            counts["expected_change_elements"] = sum(
                e != b for e, b in zip(expected, baseline, strict=True)
            )
            counts["unchanged_expected_change_elements"] = sum(
                e != b and a == b for e, b, a in zip(expected, baseline, after, strict=True)
            )
            for field, value in counts.items():
                check(rows[weight][field] == value, f"{name}/{weight}: {field}")
            for field, value in saved_change_witnesses(
                target, original, expected, baseline, after
            ).items():
                check(rows[weight][field] == value, f"{name}/{weight}: {field}")
            if loaded:
                candidates = [r for r in loaded["native_loaded_biases"] if r["weight"] == weight]
                check(len(candidates) == 1, "Missing/duplicate native loaded-bias witness")
                check(
                    candidates[0]["loaded_target_mismatches"] == counts["target_mismatches"],
                    "Saved bias differs from native pre-merge observation",
                )
            for field in totals:
                totals[field] += counts[field]
        summary = report["summary"]
        for field, value in totals.items():
            check(summary["trained_bias_" + field] == value, f"{name}: summary {field}")
        check(
            summary["adapted_tensors"] == 81 and summary["checked_tensors"] == 165,
            "Incomplete matrix/tensor audit coverage",
        )
        check(summary["unadapted_changed_tensors"] == 0, "Unexpected outside-target changes")
        if name == "remapped-f32":
            check(totals["target_mismatches"] == 0, "Key-only control failed")
        if name == "omitted-bias-f32":
            check(
                totals["target_mismatches"] == totals["saved_change_elements"] == 175104,
                "Omission control did not omit every saved bias change",
            )
        results.append({"name": name, **totals})
        reports[name] = report
        print(name, totals, flush=True)
    return {
        "state": "verified",
        "runtime_code": {
            p.name: digest(p) for p in sorted(Path(audit.__code__.co_filename).parent.glob("*.py"))
        },
        "oracle_code": {
            "verify.py": digest(Path(__file__)),
            "trained-bias/verify.py": digest(Path(__file__).parents[1] / "trained-bias/verify.py"),
        },
        "producer_manifest_sha256": digest(root / "manifest.json"),
        "manifest": manifest,
        "producer": producer,
        "results": results,
        "reports": reports,
        "independent_bias_coordinates": 5 * 175104,
        "byte_identical_nonbias_control_tensors": 2 * len(nonbias),
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: verify.py PRODUCER_OUTPUT REPORT.json.gz")
    result = verify(Path(sys.argv[1]))
    with gzip.open(sys.argv[2], "wt", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
