"""Native PEFT embedding saves; synthetic training and explicit wrong-merge controls."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import platform
import sys
import warnings
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
        self.projection = torch.nn.Linear(11, 7, bias=False)
        self.control = torch.nn.Linear(11, 3, bias=False)
        self.config = {"model_type": "mergeglass_synthetic_embeddings"}

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        values = self.tokens(ids) + self.square(ids % 11)
        return torch.cat([self.projection(values), self.control(values)], dim=-1)


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
    torch.manual_seed(91871)
    base = TokenModel()
    with torch.no_grad():
        # Dtype-exact bases separate merge rounding from conversion drift.
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
            target_modules=["tokens", "square", "projection"],
            rank_pattern={"square": 2},
            alpha_pattern={"projection": -3, "square": 5},
        ),
    )
    ids = torch.arange(37).repeat(3)
    target = torch.randn(len(ids), 10)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.008)
    losses = []
    for _ in range(24):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(ids), target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    model.save_pretrained(out / "adapter", save_embedding_layers=False)
    (out / "adapter/README.md").unlink(missing_ok=True)

    def load() -> PeftModel:
        return PeftModel.from_pretrained(
            copy.deepcopy(base), out / "adapter", local_files_only=True
        )

    def save(candidate: torch.nn.Module, name: str) -> None:
        (out / name).mkdir()
        save_file(candidate.state_dict(), out / name / "model.safetensors")

    cases = {}
    forwards = []
    for label, dtype in [("f32", torch.float32), ("f16", torch.float16), ("bf16", torch.bfloat16)]:
        for safe in (False, True):
            candidate = load()
            for module in candidate.modules():
                if isinstance(module, LoraLayer):
                    module.base_layer.to(dtype)
            candidate.base_model.model.control.to(dtype)
            name = f"correct-{label}-safe-{str(safe).lower()}"
            with torch.no_grad():
                before = candidate(ids).float()
                merged = candidate.merge_and_unload(safe_merge=safe)
                after = merged(ids).float()
            save(merged, name)
            cases[name] = "native merge; FP32 factors, selected base dtype"
            forwards.append(
                {"case": name, "max_absolute_difference": float((after - before).abs().max())}
            )
        for control in ("omit-embeddings", "reverse-embeddings"):
            candidate = load()
            with torch.no_grad():
                for module in candidate.modules():
                    if isinstance(module, LoraLayer):
                        module.base_layer.to(dtype)
                        if "default" in module.lora_embedding_A:
                            module.lora_embedding_A["default"].mul_(
                                0 if control == "omit-embeddings" else -1
                            )
                candidate.base_model.model.control.to(dtype)
            name = f"{control}-{label}"
            save(candidate.merge_and_unload(), name)
            cases[name] = (
                f"native merge after {'zeroing' if control == 'omit-embeddings' else 'negating'} "
                "embedding A; Linear unchanged"
            )

    # A square embedding makes the wrong untransposed product shape-compatible.
    wrong = load().merge_and_unload()
    with torch.no_grad():
        layer = model.base_model.model.square
        untransposed = (
            layer.lora_embedding_B["default"] @ layer.lora_embedding_A["default"]
        ) * layer.scaling["default"]
        wrong.square.weight.copy_(base.square.weight + untransposed)
    save(wrong, "wrong-orientation-f32")
    cases["wrong-orientation-f32"] = (
        "native merge with square embedding replaced by base + untransposed B @ A"
    )

    # A real model class provides its own saved module names and shard indexes.
    llama = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            tie_word_embeddings=False,
        )
    )
    llama.save_pretrained(out / "transformer-base", max_shard_size="4KB")
    adapted = get_peft_model(
        llama, LoraConfig(r=2, lora_alpha=4, target_modules=["embed_tokens", "q_proj"])
    )
    batch = torch.randint(3, 64, (4, 16))
    optimizer = torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad], lr=0.004)
    for _ in range(8):
        optimizer.zero_grad()
        loss = adapted(input_ids=batch, labels=batch).loss
        loss.backward()
        optimizer.step()
    adapted.save_pretrained(out / "transformer-adapter", save_embedding_layers=False)
    (out / "transformer-adapter/README.md").unlink(missing_ok=True)
    with warnings.catch_warnings(record=True):
        adapted.save_pretrained(out / "transformer-default-save")
    (out / "transformer-default-save/README.md").unlink(missing_ok=True)
    adapted.merge_and_unload().save_pretrained(out / "transformer-merged", max_shard_size="4KB")
    receipt = {
        "scope": (
            "synthetic training; wrong merges are explicit authored controls, "
            "not observed incidents"
        ),
        "seed": 91871,
        "python": platform.python_version(),
        "libraries": {
            key: importlib.metadata.version(key)
            for key in ["torch", "peft", "transformers", "safetensors", "numpy"]
        },
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_losses": losses,
        "cases": cases,
        "native_forward_differences": forwards,
        "files": fingerprints(out),
    }
    (out / "producer.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(f"Produced {len(cases)} mixed embedding/Linear controls and sharded Llama saves")


if __name__ == "__main__":
    main()
