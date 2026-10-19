"""Large native DoRA control, closed-form measurements, fresh audit RSS workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

ROWS, COLUMNS, RANK = 8192, 4096, 8


def prepare(root: Path) -> None:
    import importlib.metadata

    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(COLUMNS, ROWS, bias=False)
            self.config = {"model_type": "mergeglass_synthetic_dora_streaming"}

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    root.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(71121)
    base = Model()
    with torch.no_grad():
        base.linear.weight.fill_(1)
    (root / "base").mkdir()
    save_file(base.state_dict(), root / "base/model.safetensors")
    model = get_peft_model(
        base, LoraConfig(r=RANK, lora_alpha=RANK, target_modules=["linear"], use_dora=True)
    )
    layer = model.base_model.model.linear
    with torch.no_grad():
        layer.lora_A["default"].weight.fill_(1 / 32)
        layer.lora_B["default"].weight.fill_(1 / 32)
        layer.lora_magnitude_vector["default"].weight.copy_(
            torch.tensor([0.0, 32, 65, -65]).repeat(ROWS // 4)
        )
    model.save_pretrained(root / "adapter")
    (root / "adapter/README.md").unlink(missing_ok=True)
    merged = model.merge_and_unload(safe_merge=False)
    (root / "merged").mkdir()
    save_file(merged.state_dict(), root / "merged/model.safetensors")
    receipt = {
        "synthetic": True,
        "trained": False,
        "producer": "PEFT save_pretrained and merge_and_unload(safe_merge=False)",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shape": [ROWS, COLUMNS],
        "rank": RANK,
        "base": "all ones",
        "factors": "A and B entries all 1/32, scale=1; BA entries all 1/128",
        "row_magnitudes_repeating": [0, 32, 65, -65],
        "closed_form_merged_rows": [0, 0.5, 65 / 64, -65 / 64],
        "versions": {
            name: importlib.metadata.version(name) for name in ["torch", "peft", "safetensors"]
        },
        "files": [
            {
                "file": str(p.relative_to(root)),
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in sorted(root.rglob("*"))
            if p.is_file()
        ],
    }
    (root / "producer.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"prepared_native_elements": ROWS * COLUMNS}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dora-norms", action="store_true")
    args = parser.parse_args()
    root = args.directory
    if args.prepare:
        prepare(root)
        return
    if args.worker is not None:
        from mergeglass import audit, audit_dora_norms

        started = time.perf_counter()
        report = (
            audit_dora_norms(root / "adapter", root / "merged", block_mib=args.worker)
            if args.dora_norms
            else audit(root / "base", root / "adapter", root / "merged", block_mib=args.worker)
        )
        print(
            json.dumps(
                {
                    "block_mib": args.worker,
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "layer": report["layers"][0],
                    "inputs": report["inputs"],
                }
            )
        )
        return
    receipt = json.loads((root / "producer.json").read_text())
    # Exact dyadic arithmetic establishes the full update independently of
    # either native merging or the auditor's row-normalization implementation.
    deltas = [Fraction(-1), Fraction(-1, 2), Fraction(1, 64), Fraction(-129, 64)]
    expected_norm = math.sqrt(float(sum(d * d for d in deltas) * (ROWS // 4) * COLUMNS))
    results = []
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for block in [1, 16, 64]:
        process = subprocess.run(
            [sys.executable, __file__, str(root), "--worker", str(block)]
            + (["--dora-norms"] if args.dora_norms else []),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        result = json.loads(process.stdout)
        layer = result["layer"]
        if args.dora_norms:
            rows = layer["dora_rows"]
            assert rows["rows"] == ROWS
            assert rows["row_norm_factor_range"] == [1.0, 1.0]
            assert rows["zero_magnitude_rows"] == rows["zero_saved_rows"] == ROWS // 4
            assert rows["negative_magnitude_rows"] == ROWS // 4
            assert rows["norm_comparable_rows"] == 3 * ROWS // 4
            assert rows["nonzero_magnitude_zero_saved_rows"] == 0
            assert rows["zero_magnitude_nonzero_saved_rows"] == 0
            assert "base" not in result["inputs"] and "direction_cosine_range" not in rows
        else:
            assert layer["changed_elements"] == ROWS * COLUMNS
            assert layer["intended_nonzero_elements"] == ROWS * COLUMNS
            assert layer["reference_merge_mismatches"] == 0
            assert layer["norms"]["residual"] == 0
            assert layer["relative_residual"] == 0
            assert math.isclose(layer["norms"]["delta"], expected_norm, rel_tol=1e-13)
        for role in result["inputs"]:
            for item in result["inputs"][role]:
                source = next(x for x in receipt["files"] if x["file"] == role + "/" + item["file"])
                assert source["sha256"] == item["sha256"]
                assert source["bytes"] == item["bytes"]
        results.append(result)
    output = {
        "mode": "dora-norms" if args.dora_norms else "complete-audit",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "producer": receipt,
        "machine": {"platform": platform.platform(), "logical_cpus": os.cpu_count(), "threads": 1},
        "full_float64_delta_bytes": ROWS * COLUMNS * 8,
        "closed_form_update_norm": expected_norm,
        "results": results,
    }
    destination = args.out or root / "benchmark.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            [
                {key: value for key, value in row.items() if key not in {"layer", "inputs"}}
                for row in results
            ]
        )
    )


if __name__ == "__main__":
    main()
