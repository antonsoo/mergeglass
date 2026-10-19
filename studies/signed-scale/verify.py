"""Before refusals and independently checked saved merges for signed coefficients."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

import mergeglass
from mergeglass import AuditError, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--before", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    producer = json.loads((args.directory / "producer.json").read_text())
    for item in producer["files"]:
        data = (args.directory / item["file"]).read_bytes()
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
    oracle_path = Path(__file__).parents[1] / "native-peft/verify.py"
    spec = importlib.util.spec_from_file_location("dense_oracle", oracle_path)
    assert spec is not None and spec.loader is not None
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    results = []
    for case in producer["cases"]:
        root = args.directory / case["name"]
        for name in ["merged-f32", "merged-f16", "merged-bf16"]:
            if args.before:
                try:
                    audit(root / "base", root / "adapter", root / name)
                except AuditError as exc:
                    message = str(exc)
                    assert "nonnegative" in message
                    results.append({"case": case["name"], "name": name, "refusal": message})
                else:
                    raise AssertionError("Baseline unexpectedly accepted the signed coefficient")
            else:
                result = oracle.verify(root, name)
                results.append({"case": case["name"], **result})
    payload = {
        "producer": producer,
        "before": args.before,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "oracle_sha256": hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
        "auditor_modules": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(mergeglass.__file__).parent.glob("*.py"))
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Verified {len(results)} signed-scale native cases; before={args.before}")


if __name__ == "__main__":
    main()
