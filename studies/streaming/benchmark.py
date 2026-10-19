"""Measure installed auditor RSS on synthetic, explicitly labelled large tensors."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from mergeglass import audit


def prepare(root: Path, rows: int, columns: int, rank: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in ["base", "adapter", "merged"]:
        (root / name).mkdir()
    original = np.ones((rows, columns), dtype=np.float32)
    save_file({"linear.weight": original}, root / "base/model.safetensors")
    original += np.float32(1 / 128)
    save_file({"linear.weight": original}, root / "merged/model.safetensors")
    del original
    save_file(
        {
            "base_model.model.linear.lora_A.weight": np.full(
                (rank, columns), 1 / 32, dtype=np.float32
            ),
            "base_model.model.linear.lora_B.weight": np.full(
                (rows, rank), 1 / 32, dtype=np.float32
            ),
        },
        root / "adapter/adapter_model.safetensors",
    )
    (root / "adapter/adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": rank, "lora_alpha": rank, "bias": "none"})
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    root = args.directory
    rows, columns, rank = 4096, 8192, 8
    if args.prepare:
        prepare(root, rows, columns, rank)
        return
    if args.worker is not None:
        started = time.perf_counter()
        report = audit(root / "base", root / "adapter", root / "merged", block_mib=args.worker)
        print(
            json.dumps(
                {
                    "block_mib": args.worker,
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "layer": report["layers"][0],
                }
            )
        )
        return
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    # Preparation must have its own process: allocating the dense input here
    # contaminates the high-water RSS subsequently inherited by children on Linux.
    subprocess.run([sys.executable, __file__, str(root), "--prepare"], env=env, check=True)
    results = []
    for block in [1, 16, 64]:
        result = subprocess.run(
            [sys.executable, __file__, str(root), "--worker", str(block)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        row = json.loads(result.stdout)
        assert row["layer"]["relative_residual"] == 0
        assert row["layer"]["changed_elements"] == rows * columns
        results.append(row)
    output = {
        "synthetic": True,
        "shape": [rows, columns],
        "rank": rank,
        "machine": {
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "blas_threads": 1,
        },
        "checkpoint_tensor_bytes_each": rows * columns * 4,
        "full_float64_delta_bytes": rows * columns * 8,
        "results": results,
    }
    (root / "benchmark.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
