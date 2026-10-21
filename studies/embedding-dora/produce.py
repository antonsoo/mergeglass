"""Native mixed Embedding/Linear DoRA saves, with explicit wrong-merge controls.

Synthetic training and deliberate post-training changes; no task-quality claim.
Run in the separately pinned native PEFT environment.
"""

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
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM


class TokenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokens = torch.nn.Embedding(37, 11)
        self.square = torch.nn.Embedding(11, 11)
        self.magnitude_only = torch.nn.Embedding(23, 5)
        self.projection = torch.nn.Linear(11, 7, bias=False)
        self.control = torch.nn.Linear(11, 3, bias=False)
        self.config = {"model_type": "mergeglass_synthetic_embedding_dora"}

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        values = self.tokens(ids % 37) + self.square(ids % 11)
        return torch.cat(
            [self.projection(values), self.control(values), self.magnitude_only(ids % 23)],
            dim=-1,
        )


def save(model: torch.nn.Module, directory: Path) -> None:
    directory.mkdir(parents=True)
    save_file(model.state_dict(), directory / "model.safetensors")


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(397143)
    base = TokenModel()
    with torch.no_grad():
        # Exact later base casts isolate merge arithmetic from conversion drift.
        for parameter in base.parameters():
            parameter.copy_(parameter.bfloat16().float())
    save(base, out / "base")
    model = get_peft_model(
        copy.deepcopy(base),
        LoraConfig(
            r=4,
            lora_alpha=8,
            use_rslora=True,
            use_dora=True,
            target_modules=["tokens", "square", "magnitude_only", "projection"],
            rank_pattern={"square": 3, "magnitude_only": 2},
            alpha_pattern={"tokens": 6, "square": 5, "projection": -3},
        ),
    )
    ids = torch.arange(111)
    target = torch.randn(len(ids), 15)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.008)
    losses = []
    for _ in range(24):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(ids), target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        square = model.base_model.model.square
        square.lora_magnitude_vector["default"].weight[0] = 0
        square.lora_magnitude_vector["default"].weight[1] *= -1
        only = model.base_model.model.magnitude_only
        only.lora_embedding_A["default"].zero_()
        norm = torch.linalg.vector_norm(only.base_layer.weight, dim=0)
        only.lora_magnitude_vector["default"].weight.copy_(
            norm * torch.tensor([0.5, 0, -1.25, 1.5, 2])
        )
    model.save_pretrained(out / "adapter", save_embedding_layers=False)
    (out / "adapter/README.md").unlink(missing_ok=True)
    layers = {}
    for name, module in model.base_model.model.named_modules():
        if isinstance(module, LoraLayer):
            layers[name] = {
                "rank": module.r["default"],
                "alpha": module.lora_alpha["default"],
                "scale": module.scaling["default"],
                "embedding": bool(module.lora_embedding_A),
                "magnitude_shape": list(module.lora_magnitude_vector["default"].weight.shape),
            }
    cases = []
    for label, dtype in [("f32", torch.float32), ("f16", torch.float16), ("bf16", torch.bfloat16)]:
        for safe in (False, True):
            candidate = PeftModel.from_pretrained(
                copy.deepcopy(base), out / "adapter", local_files_only=True
            )
            for module in candidate.modules():
                if isinstance(module, LoraLayer):
                    module.base_layer.to(dtype)
            candidate.base_model.model.control.to(dtype)
            name = f"correct-{label}-safe-{str(safe).lower()}"
            merged = candidate.merge_and_unload(safe_merge=safe)
            save(merged, out / name)
            cases.append({"merged": name, "base": "base", "adapter": "adapter", "kind": "native"})
        candidate = PeftModel.from_pretrained(
            copy.deepcopy(base), out / "adapter", local_files_only=True
        )
        with torch.no_grad():
            for module in candidate.modules():
                if isinstance(module, LoraLayer):
                    module.base_layer.to(dtype)
                    if module.lora_embedding_A:
                        delta = module.get_delta_weight("default")
                        magnitude = module.lora_magnitude_vector["default"]
                        magnitude.weight.copy_(
                            magnitude.get_weight_norm(module.base_layer.weight, delta.T, scaling=1)
                        )
            candidate.base_model.model.control.to(dtype)
        name = f"reset-embedding-magnitudes-{label}"
        save(candidate.merge_and_unload(safe_merge=False), out / name)
        cases.append(
            {
                "merged": name,
                "base": "base",
                "adapter": "adapter",
                "kind": "embedding magnitudes reset to direction norms; Linear stays DoRA",
            }
        )

    correct = PeftModel.from_pretrained(
        copy.deepcopy(base), out / "adapter", local_files_only=True
    ).merge_and_unload(safe_merge=False)
    wrong = copy.deepcopy(correct)
    with torch.no_grad():
        module = model.base_model.model.square
        direction = (
            base.square.weight.double()
            + (
                module.lora_embedding_B["default"].double()
                @ module.lora_embedding_A["default"].double()
            ).T
            * module.scaling["default"]
        )
        # Same square shape, different normalization axis. All other weights
        # remain the native correct merge, so Linear cannot reveal this error.
        wrong.square.weight.copy_(
            direction
            / direction.norm(dim=1, keepdim=True)
            * module.lora_magnitude_vector["default"].weight.double()[:, None]
        )
    save(wrong, out / "wrong-row-axis-f32")
    cases.append(
        {
            "merged": "wrong-row-axis-f32",
            "base": "base",
            "adapter": "adapter",
            "kind": "authored square-only row-axis error",
        }
    )
    rotated = copy.deepcopy(correct)
    with torch.no_grad():
        for module in rotated.modules():
            if isinstance(module, torch.nn.Embedding):
                module.weight.copy_(module.weight.roll(1, dims=0))
    save(rotated, out / "rotated-token-rows-f32")
    cases.append(
        {
            "merged": "rotated-token-rows-f32",
            "base": "base",
            "adapter": "adapter",
            "kind": "authored token-row rotation; saved column multisets unchanged",
        }
    )
    edited = copy.deepcopy(correct)
    with torch.no_grad():
        edited.control.weight[0, 0] += 0.125
    save(edited, out / "unrelated-edit-f32")
    cases.append(
        {
            "merged": "unrelated-edit-f32",
            "base": "base",
            "adapter": "adapter",
            "kind": "native correct adapted weights plus authored unrelated edit",
        }
    )
    save(base, out / "omitted-all-f32")
    cases.append(
        {
            "merged": "omitted-all-f32",
            "base": "base",
            "adapter": "adapter",
            "kind": "unchanged original base",
        }
    )

    # Actual Transformers state names, default PEFT embedding snapshots and a
    # sharded saved model complement the small custom-module controls.
    torch.manual_seed(397144)
    llama = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=31,
            hidden_size=12,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=3,
            num_key_value_heads=1,
            tie_word_embeddings=False,
            max_position_embeddings=64,
        )
    )
    with torch.no_grad():
        for parameter in llama.parameters():
            parameter.copy_(parameter.bfloat16().float())
    llama.save_pretrained(out / "llama-base", max_shard_size="2KB")
    adapted = get_peft_model(
        copy.deepcopy(llama),
        LoraConfig(
            r=2,
            lora_alpha=3,
            use_dora=True,
            use_rslora=True,
            target_modules=["embed_tokens", "q_proj", "v_proj"],
        ),
    )
    optimizer = torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad], lr=0.004)
    for _ in range(4):
        batch = torch.randint(0, 31, (2, 12))
        optimizer.zero_grad()
        adapted(input_ids=batch, labels=batch).loss.backward()
        optimizer.step()
    adapted.eval()
    adapted.save_pretrained(out / "llama-adapter")
    adapted.save_pretrained(out / "llama-factors", save_embedding_layers=False)
    for name in ["llama-adapter", "llama-factors"]:
        (out / name / "README.md").unlink(missing_ok=True)
    llama_merged = adapted.merge_and_unload(safe_merge=False)
    llama_merged.save_pretrained(out / "llama-merged", max_shard_size="2KB")
    for name in ["llama-adapter", "llama-factors"]:
        cases.append(
            {
                "merged": "llama-merged",
                "base": "llama-base",
                "adapter": name,
                "kind": "native sharded Llama",
            }
        )
    receipt = {
        "producer_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": (
            "synthetic training, explicit post-training magnitudes and controlled wrong merges"
        ),
        "python": platform.python_version(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "peft", "transformers", "safetensors", "numpy"]
        },
        "training_losses": losses,
        "layers": layers,
        "post_training_controls": [
            "square magnitude 0 set to zero, 1 negated",
            "magnitude_only A zeroed; original column norms multiplied by [0.5, 0, -1.25, 1.5, 2]",
        ],
        "cases": cases,
        "files": [
            {
                "path": str(path.relative_to(out)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(out.rglob("*"))
            if path.is_file()
        ],
    }
    (out / "producer.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {"cases": len(cases), "native_files": len(receipt["files"]), "layers": layers}, indent=2
        )
    )


if __name__ == "__main__":
    main()
