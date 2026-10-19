"""Merge an unchanged public adapter using only local built-in Llama model code."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


def main() -> None:
    root = Path(sys.argv[1])
    manifest = json.loads(Path(__file__).with_name("sources.json").read_text())
    for source in manifest["sources"]:
        for item in source["files"]:
            path = root / source["role"] / item["file"]
            assert path.stat().st_size == item["bytes"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    torch.set_num_threads(1)
    merges = []
    for name, dtype in [
        ("merged-f32", torch.float32),
        ("merged-f16", torch.float16),
        ("merged-bf16", torch.bfloat16),
    ]:
        base = AutoModelForCausalLM.from_pretrained(
            root / "base", dtype=dtype, local_files_only=True, trust_remote_code=False
        )
        model = PeftModel.from_pretrained(
            base, root / "adapter", local_files_only=True, autocast_adapter_dtype=True
        )
        adapter_dtypes = sorted(
            {
                str(p.dtype)
                for key, p in model.named_parameters()
                if ".lora_A." in key or ".lora_B." in key
            }
        )
        merged = model.merge_and_unload(safe_merge=False)
        merged.save_pretrained(root / name, safe_serialization=True, max_shard_size="32MB")
        merges.append(
            {
                "name": name,
                "base_dtype": str(dtype),
                "adapter_dtypes": adapter_dtypes,
                "files": [
                    {
                        "file": p.name,
                        "bytes": p.stat().st_size,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    }
                    for p in sorted((root / name).iterdir())
                    if p.is_file()
                ],
            }
        )
        del merged, model, base
    receipt = {
        "sources": manifest,
        "versions": {
            key: importlib.metadata.version(key)
            for key in ["torch", "peft", "transformers", "numpy", "safetensors"]
        },
        "merge": "CPU, 1 thread, FP32 adapters, merge_and_unload(safe_merge=False)",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "merges": merges,
    }
    (root / "merges.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
