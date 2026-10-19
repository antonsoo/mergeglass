"""Native PEFT signed alpha coefficients, with unchanged saved factor bytes."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM


def main() -> None:
    source, root = map(Path, sys.argv[1:])
    root.mkdir(parents=True)
    torch.set_num_threads(1)
    original = json.loads((source / "producer.json").read_text())
    for item in original["files"]:
        data = (source / item["file"]).read_bytes()
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
    factor_hash = hashlib.sha256(
        (source / "adapter/adapter_model.safetensors").read_bytes()
    ).hexdigest()
    cases = []
    for name, alpha, override in [("negative-global", -8, 4), ("negative-pattern", 8, -4)]:
        out = root / name
        shutil.copytree(source / "base", out / "base")
        base = AutoModelForCausalLM.from_pretrained(
            out / "base", local_files_only=True, trust_remote_code=False
        )
        base.name_or_path = ""
        config = LoraConfig.from_pretrained(source / "adapter")
        config.lora_alpha = alpha
        config.alpha_pattern = {"v_proj": override}
        model = PeftModel.from_pretrained(
            base, source / "adapter", config=config, local_files_only=True
        )
        model.save_pretrained(out / "adapter", save_embedding_layers=False)
        (out / "adapter/README.md").unlink(missing_ok=True)
        assert (
            hashlib.sha256((out / "adapter/adapter_model.safetensors").read_bytes()).hexdigest()
            == factor_hash
        )
        for suffix, dtype in [
            ("f32", torch.float32),
            ("f16", torch.float16),
            ("bf16", torch.bfloat16),
        ]:
            candidate = copy.deepcopy(model)
            for key, parameter in candidate.named_parameters():
                if ".lora_A." not in key and ".lora_B." not in key:
                    parameter.data = parameter.data.to(dtype)
            candidate.merge_and_unload(safe_merge=False).save_pretrained(
                out / f"merged-{suffix}", safe_serialization=True, max_shard_size="8KB"
            )
        cases.append(
            {
                "name": name,
                "global_alpha": alpha,
                "alpha_pattern": {"v_proj": override},
                "rsLoRA": config.use_rslora,
                "rank_pattern": config.rank_pattern,
                "factor_bytes_unchanged": True,
            }
        )
    payload = {
        "synthetic": True,
        "control": "signed alpha overrides on the unchanged native synthetic Llama factors",
        "input_producer": original,
        "factor_sha256": factor_hash,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {
            key: importlib.metadata.version(key)
            for key in ["torch", "peft", "transformers", "numpy", "safetensors"]
        },
        "cases": cases,
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
    (root / "producer.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
