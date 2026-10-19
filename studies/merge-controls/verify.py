"""Check all saved controls with the independent dense PyTorch oracle."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    root, out = map(Path, sys.argv[1:])
    torch.set_num_threads(1)
    oracle_path = Path(__file__).parents[1] / "native-peft/verify.py"
    spec = importlib.util.spec_from_file_location("dense_oracle", oracle_path)
    assert spec is not None and spec.loader is not None
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    producer = json.loads((root / "producer.json").read_text())
    for item in producer["files"]:
        data = (root / item["file"]).read_bytes()
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
    cases = [oracle.verify(root, name) for name in producer["cases"]]
    # These are independent native artifacts, not values edited into a report.
    # Keep concrete cross-artifact counts for the safe_merge operation-order study.
    unsafe = oracle.load(root / "bf16-safe-false")
    safe = oracle.load(root / "bf16-safe-true")
    differences = {
        key: int(torch.count_nonzero(unsafe[key] != safe[key]))
        for key in sorted(safe)
        if not torch.equal(unsafe[key], safe[key])
    }
    payload = {
        "producer": producer,
        "oracle_source_sha256": hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
        "cases": cases,
        "safe_merge_artifact_differences": differences,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    for case in cases:
        print(case["name"])
        for layer in case["report"]["layers"]:
            print(
                f"  {layer['module']}: changed={layer['changed_elements']}/{layer['elements']} "
                f"gain={layer['projection_gain']:.6g}, "
                f"cosine={layer['cosine_similarity']}, residual={layer['relative_residual']:.6g}"
            )
    print(
        f"safe_merge changes {sum(differences.values())} saved entries "
        f"in {len(differences)} tensors"
    )


if __name__ == "__main__":
    main()
