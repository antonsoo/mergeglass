"""Native DoRA saves and deliberately wrong merges; all training is synthetic."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import warnings
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM


class TeachingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.direction = torch.nn.Linear(8, 12, bias=False)
        self.magnitude_only = torch.nn.Linear(8, 5, bias=False)
        self.zero_magnitude = torch.nn.Linear(8, 4, bias=False)
        self.control = torch.nn.Linear(8, 3, bias=False)
        self.config = {"model_type": "mergeglass_synthetic_dora"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.direction(x), self.magnitude_only(x), self.zero_magnitude(x), self.control(x)],
            dim=-1,
        )


def fingerprints(root: Path) -> list[dict[str, object]]:
    return [
        {
            "file": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(71119)
    base = TeachingModel()
    # Exact later base conversions keep merge arithmetic separate from base drift.
    with torch.no_grad():
        for parameter in base.parameters():
            parameter.copy_(parameter.bfloat16().float())
    (out / "base").mkdir()
    save_file(base.state_dict(), out / "base/model.safetensors")
    model = get_peft_model(
        copy.deepcopy(base),
        LoraConfig(
            r=4,
            lora_alpha=8,
            use_rslora=True,
            use_dora=True,
            target_modules=["direction", "magnitude_only", "zero_magnitude"],
            rank_pattern={"magnitude_only": 2},
            alpha_pattern={"direction": 6},
        ),
    )
    x = torch.randn(64, 8)
    target = torch.randn(64, 24)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    losses = []
    for _ in range(16):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        # Authored controls: the magnitudes alone can change a zero BA product.
        model.base_model.model.magnitude_only.lora_B["default"].weight.zero_()
        model.base_model.model.zero_magnitude.lora_magnitude_vector["default"].weight.zero_()
    model.eval()
    model.save_pretrained(out / "adapter", safe_serialization=True)
    (out / "adapter/README.md").unlink(missing_ok=True)

    def load(*, ordinary_lora: bool = False) -> PeftModel:
        config = LoraConfig.from_pretrained(out / "adapter")
        config.use_dora = not ordinary_lora
        return PeftModel.from_pretrained(
            copy.deepcopy(base), out / "adapter", config=config, local_files_only=True
        )

    def save(model_to_save: torch.nn.Module, name: str) -> None:
        (out / name).mkdir()
        save_file(model_to_save.state_dict(), out / name / "model.safetensors")

    cases = {}
    forward_checks = []
    for label, dtype in [("f32", torch.float32), ("f16", torch.float16), ("bf16", torch.bfloat16)]:
        for safe in (False, True):
            candidate = load()
            for module in candidate.modules():
                if isinstance(module, LoraLayer):
                    module.base_layer.to(dtype)
            candidate.base_model.model.control.to(dtype)
            name = f"correct-{label}-safe-{str(safe).lower()}"
            with torch.no_grad():
                unmerged_output = candidate(x.to(dtype)).float()
            merged = candidate.merge_and_unload(safe_merge=safe)
            save(merged, name)
            with torch.no_grad():
                merged_output = merged(x.to(dtype)).float()
            forward_checks.append(
                {
                    "case": name,
                    "max_absolute_output_difference": float(
                        (merged_output - unmerged_output).abs().max()
                    ),
                    "positions": merged_output.numel(),
                }
            )
            cases[name] = "native DoRA merge; FP32 adapter, specified base dtype"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wrong = load(ordinary_lora=True)
        for module in wrong.modules():
            if isinstance(module, LoraLayer):
                module.base_layer.to(dtype)
        wrong.base_model.model.control.to(dtype)
        name = f"as-lora-{label}"
        save(wrong.merge_and_unload(), name)
        cases[name] = {
            "control": "native load with use_dora=False; magnitudes omitted from merge",
            "warnings": [str(w.message) for w in caught],
        }
    half_magnitudes = load()
    with torch.no_grad():
        for module in half_magnitudes.modules():
            if isinstance(module, LoraLayer):
                module.lora_magnitude_vector["default"].weight.mul_(0.5)
    save(half_magnitudes.merge_and_unload(), "half-magnitudes-f32")
    cases["half-magnitudes-f32"] = "native DoRA merge after halving every learned magnitude"
    save(base, "omitted")
    cases["omitted"] = "base copied without any adapter"

    # Separate native Transformers checkpoint: actual sharding, module paths and
    # model save/reload conventions rather than a second hand-authored inventory.
    transformer = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            tie_word_embeddings=False,
        )
    )
    transformer.save_pretrained(out / "transformer-base", max_shard_size="8KB")
    adapted = get_peft_model(
        transformer,
        LoraConfig(r=2, lora_alpha=4, use_dora=True, target_modules=["q_proj", "v_proj"]),
    )
    tokens = torch.randint(3, 64, (4, 16))
    optimizer = torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad], lr=0.002)
    transformer_losses = []
    for _ in range(8):
        optimizer.zero_grad()
        loss = adapted(input_ids=tokens, labels=tokens).loss
        loss.backward()
        optimizer.step()
        transformer_losses.append(float(loss.detach()))
    adapted.save_pretrained(out / "transformer-adapter")
    (out / "transformer-adapter/README.md").unlink(missing_ok=True)
    adapted.merge_and_unload().save_pretrained(out / "transformer-merged", max_shard_size="8KB")

    receipt = {
        "synthetic": True,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "seed": 71119,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "training": {"steps": 16, "optimizer": "AdamW", "lr": 0.01, "losses": losses},
        "data": "seeded random 64x8 inputs and 64x24 regression targets",
        "post_training_controls": {
            "magnitude_only": "zero B, retain trained magnitudes",
            "zero_magnitude": "zero magnitude, retain trained A/B",
        },
        "transformer_training": {
            "steps": 8,
            "optimizer": "AdamW",
            "lr": 0.002,
            "data": "seeded random 4x16 token IDs; next-token objective",
            "losses": transformer_losses,
        },
        "cases": cases,
        "forward_checks": forward_checks,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "peft", "transformers", "safetensors", "numpy"]
        },
        "files": fingerprints(out),
    }
    (out / "producer.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"cases": len(cases) + 1, "files": len(receipt["files"])}))


if __name__ == "__main__":
    main()
