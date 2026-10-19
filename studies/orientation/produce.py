"""Native mixed Conv1D/Linear exports with one misleading global orientation flag."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    ImageGPTConfig,
    ImageGPTForCausalImageModeling,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
)


def main() -> None:
    root = Path(sys.argv[1])
    root.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(7821)
    cases = []
    for name, config_class, model_class in [
        ("gpt2", GPT2Config, GPT2LMHeadModel),
        ("openai-gpt", OpenAIGPTConfig, OpenAIGPTLMHeadModel),
        ("imagegpt", ImageGPTConfig, ImageGPTForCausalImageModeling),
    ]:
        config = config_class(
            n_embd=16,
            n_head=2,
            n_layer=1,
            n_positions=32,
            vocab_size=32,
            tie_word_embeddings=False,
            bos_token_id=1,
            eos_token_id=2,
        )
        original = model_class(config)
        for kind, targets in [
            ("mixed", ["transformer.h.0.attn.c_proj", "lm_head"]),
            ("linear-head", ["lm_head"]),
        ]:
            out = root / f"{name}-{kind}"
            original.save_pretrained(out / "base")
            model = get_peft_model(
                copy.deepcopy(original), LoraConfig(r=2, lora_alpha=4, target_modules=targets)
            )
            with torch.no_grad():
                for parameter_name, parameter in model.named_parameters():
                    if ".lora_B." in parameter_name:
                        parameter.normal_(0, 0.01)
            # Explicitly omit redundant embedding payloads: this is a LoRA-only
            # adapter. It does not add a new vocabulary or train the base head.
            model.save_pretrained(out / "adapter", save_embedding_layers=False)
            (out / "adapter/README.md").unlink(missing_ok=True)
            orientations = {
                path.removeprefix("base_model.model."): module.fan_in_fan_out
                for path, module in model.named_modules()
                if isinstance(module, LoraLayer)
            }
            saved_config = json.loads((out / "adapter/adapter_config.json").read_text())
            assert saved_config["fan_in_fan_out"] is False
            merged = model.merge_and_unload()
            merged.save_pretrained(out / "merged")
            cases.append(
                {
                    "name": out.name,
                    "kind": kind,
                    "model_type": config.model_type,
                    "saved_fan_in_fan_out": saved_config["fan_in_fan_out"],
                    "native_module_orientations": orientations,
                }
            )
    receipt = {
        "synthetic": True,
        "training": False,
        "control": "seeded random nonzero B factors, native PEFT exports and merges",
        "seed": 7821,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": cases,
        "versions": {
            key: importlib.metadata.version(key)
            for key in ["torch", "peft", "transformers", "numpy", "safetensors"]
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


if __name__ == "__main__":
    main()
