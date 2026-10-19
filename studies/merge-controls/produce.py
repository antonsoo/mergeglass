"""Native PEFT controls: complete artifacts can still contain the wrong update.

All models and training data in this study are synthetic. No pretrained weights
or remote code are fetched. Input comes from native-peft/transformer.py.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM


def main() -> None:
    source, out = map(Path, sys.argv[1:])
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(71021)
    original = json.loads((source / "producer.json").read_text())
    for item in original["files"]:
        data = (source / item["file"]).read_bytes()
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
    for name in ("base", "adapter"):
        shutil.copytree(source / name, out / name)
    base = AutoModelForCausalLM.from_pretrained(
        out / "base", local_files_only=True, trust_remote_code=False
    )
    # This is a from-config synthetic model, not a Hub identity. Avoid saving
    # the throwaway output directory as the second adapter's base identifier.
    base.name_or_path = ""

    def intended(base_model: torch.nn.Module | None = None) -> PeftModel:
        return PeftModel.from_pretrained(
            copy.deepcopy(base) if base_model is None else base_model,
            out / "adapter",
            local_files_only=True,
        )

    def save(model: torch.nn.Module, name: str) -> None:
        model.save_pretrained(out / name, safe_serialization=True, max_shard_size="8KB")

    save(intended().merge_and_unload(), "correct-f32")
    save(copy.deepcopy(base), "omitted")
    once = intended().merge_and_unload()
    save(intended(once).merge_and_unload(), "repeated-f32")
    reversed_model = intended()
    for module in reversed_model.modules():
        if isinstance(module, LoraLayer):
            module.scale_layer(-1)
    save(reversed_model.merge_and_unload(), "reversed-f32")
    standard_config = LoraConfig.from_pretrained(out / "adapter")
    standard_config.use_rslora = False
    standard = PeftModel.from_pretrained(
        copy.deepcopy(base), out / "adapter", config=standard_config, local_files_only=True
    )
    save(standard.merge_and_unload(), "rs-as-standard-f32")

    # A different trained adapter on the same base has identical target shapes.
    other_config = LoraConfig.from_pretrained(out / "adapter")
    other_config.inference_mode = False
    other = get_peft_model(copy.deepcopy(base), other_config)
    other.train()
    optimizer = torch.optim.AdamW([p for p in other.parameters() if p.requires_grad], lr=0.002)
    tokens = torch.randint(3, 64, (4, 16))
    losses = []
    for _ in range(8):
        optimizer.zero_grad()
        loss = other(input_ids=tokens, labels=tokens).loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    other.save_pretrained(out / "other-adapter", safe_serialization=True)
    (out / "other-adapter/README.md").unlink(missing_ok=True)
    # Exercise the actual saved adapter loading path, not just an in-memory copy.
    wrong = PeftModel.from_pretrained(
        copy.deepcopy(base), out / "other-adapter", local_files_only=True
    )
    save(wrong.merge_and_unload(), "wrong-adapter-f32")
    for safe in (False, True):
        candidate = intended()
        for name, parameter in candidate.named_parameters():
            if ".lora_A." not in name and ".lora_B." not in name:
                parameter.data = parameter.data.to(torch.bfloat16)
        save(candidate.merge_and_unload(safe_merge=safe), f"bf16-safe-{str(safe).lower()}")

    receipt = {
        "synthetic": True,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_producer": original,
        "python": platform.python_version(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "seed": 71021,
        "other_adapter_training": {
            "steps": 8,
            "optimizer": "AdamW",
            "lr": 0.002,
            "data": "different seeded random token IDs (4x16), next-token objective",
            "losses": losses,
        },
        "cases": {
            "correct-f32": "saved original adapter, one ordinary native merge",
            "omitted": "base saved without loading an adapter",
            "repeated-f32": "load original adapter into an already merged base and merge again",
            "reversed-f32": "original adapter with scale_layer(-1) before merging",
            "rs-as-standard-f32": "original factors loaded with use_rslora=False",
            "wrong-adapter-f32": "different trained adapter; audit still receives original adapter",
            "bf16-safe-false": "original adapter; BF16 base, FP32 factors, safe_merge=False",
            "bf16-safe-true": "original adapter; BF16 base, FP32 factors, safe_merge=True",
        },
        "versions": {
            key: importlib.metadata.version(key)
            for key in ["torch", "peft", "transformers", "numpy", "safetensors"]
        },
        "files": [
            {
                "file": str(p.relative_to(out)),
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in sorted(out.rglob("*"))
            if p.is_file()
        ],
    }
    (out / "producer.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
