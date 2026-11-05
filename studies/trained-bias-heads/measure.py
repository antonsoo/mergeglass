"""Measure installed/source CLI processes against the optional native wide head.

The reference is a streaming hash of the exact FP32 saved head payload; no
NumPy/Torch arithmetic constructs it. RSS includes each fresh interpreter.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import mergeglass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            result.update(data)
    return result.hexdigest()


def head_payload(path: Path, name: str) -> dict[str, Any]:
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
        info = header[name]
        start, end = info["data_offsets"]
        stream.seek(8 + length + start)
        result = hashlib.sha256()
        remaining = end - start
        while remaining:
            data = stream.read(min(1024 * 1024, remaining))
            require(bool(data), "Truncated head payload")
            remaining -= len(data)
            result.update(data)
    return {
        "dtype": info["dtype"],
        "shape": info["shape"],
        "bytes": end - start,
        "sha256": result.hexdigest(),
    }


def measure(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text())
    names = [e["file"] for e in manifest]
    require(len(names) == len(set(names)), "Duplicate producer identity")
    require(
        {e["file"] for e in manifest}
        == {
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        },
        "Producer manifest inventory differs",
    )
    for entry in manifest:
        path = root / entry["file"]
        require(
            path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"],
            f"Producer file identity differs: {entry['file']}",
        )
    observations = json.loads((root / "observations.json").read_text())
    require(observations["head_shape"] == [4096, 4096], "Wrong wide-head dimensions")
    target = head_payload(
        root / "adapter/adapter_model.safetensors", "base_model.model.head.weight"
    )
    actual = head_payload(root / "merged/model.safetensors", "head.weight")
    require(
        target == actual and target["dtype"] == "F32" and target["bytes"] == 4096**2 * 4,
        "Native merged head differs from the exact absolute FP32 target",
    )
    original = head_payload(root / "base/model.safetensors", "head.weight")
    require(original["sha256"] != target["sha256"], "Training did not change the saved head")
    runs = []
    baseline = None
    with tempfile.TemporaryDirectory(prefix="mergeglass-wide-measure-") as directory:
        scratch = Path(directory)
        for block in [1, 16]:
            timing, report_file = scratch / f"{block}.time", scratch / f"{block}.json"
            command = [
                "/usr/bin/time",
                "-v",
                "-o",
                str(timing),
                sys.executable,
                "-m",
                "mergeglass",
                str(root / "base"),
                str(root / "adapter"),
                str(root / "merged"),
                "--block-mib",
                str(block),
                "--max-adapter-mib",
                "1",
                "--summary",
                "--json",
                str(report_file),
            ]
            process = subprocess.run(command, capture_output=True, text=True, check=False)
            require(process.returncode == 0, f"Audit failed: {process.stderr}")
            report = json.loads(report_file.read_text())
            summary = report["summary"]
            require(
                summary["saved_module_target_mismatches"] == 0
                and summary["trained_bias_target_mismatches"] == 0
                and summary["saved_bias_alias_tensors"] == 2
                and summary["unadapted_changed_tensors"] == 0,
                "Wide head/bias targets or copies disagree",
            )
            head = next(r for r in report["saved_modules"] if r["weight"] == "head.weight")
            require(head["elements"] == 4096**2, "Wrong audited head size")
            if baseline is not None:
                require(
                    report["inputs"] == baseline["inputs"], "Block setting changed input identities"
                )
                require(summary.keys() == baseline["summary"].keys(), "Summary fields differ")
                for key, value in summary.items():
                    previous_value = baseline["summary"][key]
                    equal = (
                        math.isclose(value, previous_value, rel_tol=3e-13, abs_tol=0)
                        if isinstance(value, float)
                        else value == previous_value
                    )
                    require(equal, f"Block setting changed summary: {key}")
                previous = next(
                    r for r in baseline["saved_modules"] if r["weight"] == "head.weight"
                )
                for key, value in head["norms"].items():
                    require(
                        math.isclose(value, previous["norms"][key], rel_tol=3e-13, abs_tol=1e-15),
                        f"Block setting changed head norm: {key}",
                    )
            baseline = report
            time_text = timing.read_text()
            rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", time_text)
            wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(.+)", time_text)
            require(rss is not None and wall is not None, "Missing GNU time measurements")
            runs.append(
                {
                    "block_mib": block,
                    "max_adapter_mib": 1,
                    "peak_rss_kib": int(rss[1]),
                    "elapsed": wall[1].strip(),
                    "summary": summary,
                    "head": head,
                }
            )
    package = Path(mergeglass.__file__).resolve().parent
    return {
        "state": "verified",
        "python": platform.python_version(),
        "machine": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        },
        "runtime_versions": {
            n: importlib.metadata.version(n) for n in ["mergeglass", "numpy", "safetensors"]
        },
        "runtime_code": [
            {"file": p.name, "sha256": digest(p)} for p in sorted(package.glob("*.py"))
        ],
        "producer_observations": observations,
        "producer_files": manifest,
        "head_payload": target,
        "dense_float64_head_bytes": 4096**2 * 8,
        "runs": runs,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: measure.py WIDE_PRODUCER_DIRECTORY OUTPUT.json", file=sys.stderr)
        return 2
    root, output = map(Path, sys.argv[1:])
    try:
        require(
            not output.resolve().is_relative_to(root.resolve()),
            "Write output outside producer directory",
        )
        receipt = measure(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"measurement failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "state": receipt["state"],
                "runs": [
                    {k: v for k, v in r.items() if k in ("block_mib", "peak_rss_kib", "elapsed")}
                    for r in receipt["runs"]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
