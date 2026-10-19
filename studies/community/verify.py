"""Check community artifacts against the separate dense PyTorch study oracle."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    root, out = (Path(p) for p in sys.argv[1:])
    torch.set_num_threads(1)
    oracle_path = Path(__file__).parents[1] / "native-peft/verify.py"
    spec = importlib.util.spec_from_file_location("native_oracle", oracle_path)
    assert spec is not None and spec.loader is not None
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    receipt = json.loads((root / "merges.json").read_text())
    for source in receipt["sources"]["sources"]:
        for item in source["files"]:
            path = root / source["role"] / item["file"]
            assert path.stat().st_size == item["bytes"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    for merge in receipt["merges"]:
        for item in merge["files"]:
            path = root / merge["name"] / item["file"]
            assert path.stat().st_size == item["bytes"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    cases = [oracle.verify(root, "merged-f32"), oracle.verify(root, "merged-bf16")]
    for case in cases:
        report = case["report"]
        case["aggregate"] = {
            "adapted_elements": sum(layer["elements"] for layer in report["layers"]),
            "nonzero_reference_elements": sum(
                layer["intended_nonzero_elements"] for layer in report["layers"]
            ),
            "unchanged_nonzero_updates": sum(
                layer["unchanged_nonzero_update_elements"] for layer in report["layers"]
            ),
            "changed_elements": sum(layer["changed_elements"] for layer in report["layers"]),
            "relative_residual_min": min(layer["relative_residual"] for layer in report["layers"]),
            "relative_residual_max": max(layer["relative_residual"] for layer in report["layers"]),
        }
    payload = {
        "producer": receipt,
        "oracle_source_sha256": hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
        "cases": cases,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                case["name"]: {"summary": case["summary"], "aggregate": case["aggregate"]}
                for case in cases
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
