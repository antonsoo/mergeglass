"""Offline native Llama/PEFT training, sharding, rsLoRA and pattern controls."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(71020)
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            tie_word_embeddings=False,
            attention_dropout=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    base.save_pretrained(out / "base", safe_serialization=True, max_shard_size="8KB")
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        use_rslora=True,
        rank_pattern={"model.layers.1.self_attn.q_proj": 2},
        alpha_pattern={"v_proj": 4},
    )
    model = get_peft_model(copy.deepcopy(base), config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.002)
    tokens = torch.randint(3, 64, (4, 16))
    losses = []
    for _ in range(8):
        optimizer.zero_grad()
        output = model(input_ids=tokens, labels=tokens)
        output.loss.backward()
        optimizer.step()
        losses.append(float(output.loss.detach()))
    model.save_pretrained(out / "adapter", safe_serialization=True)
    (out / "adapter/README.md").unlink(missing_ok=True)
    for suffix, dtype in [("f32", torch.float32), ("bf16", torch.bfloat16), ("f16", torch.float16)]:
        candidate = copy.deepcopy(model)
        # Cast all unadapted weights and base layers, while retaining FP32
        # adapter factors to match the ordinary promoted-adapter merge path.
        for name, parameter in candidate.named_parameters():
            if ".lora_A." not in name and ".lora_B." not in name:
                parameter.data = parameter.data.to(dtype)
        merged = candidate.merge_and_unload(safe_merge=False)
        merged.save_pretrained(
            out / f"merged-{suffix}", safe_serialization=True, max_shard_size="8KB"
        )
    receipt = {
        "synthetic": True,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "pretrained_weights": False,
        "seed": 71020,
        "training": {
            "steps": 8,
            "optimizer": "AdamW",
            "lr": 0.002,
            "data": "seeded random token IDs (4x16), next-token objective",
            "losses": losses,
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
