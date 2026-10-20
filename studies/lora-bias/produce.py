"""Native PEFT LoRA B-bias merges and explicitly authored omission/scale faults.

The tiny models, optimization targets and faults are synthetic. No pretrained
weights, network calls or inference-quality claims are involved.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import sys
import warnings
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from safetensors.torch import load_file, save_file
from transformers import GPT2Config, GPT2LMHeadModel


class Affine(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = {"model_type": "synthetic-affine"}
        self.first = torch.nn.Linear(5, 7)
        self.second = torch.nn.Linear(7, 3)
        self.head = torch.nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.second(self.first(x).tanh()).tanh())


def save_model(model: torch.nn.Module, destination: Path) -> None:
    destination.mkdir(parents=True)
    if isinstance(model, GPT2LMHeadModel):
        model.save_pretrained(destination, max_shard_size="8KB")
    else:
        save_file(model.state_dict(), destination / "model.safetensors")
        (destination / "config.json").write_text(json.dumps(model.config) + "\n")


def main() -> None:
    root = Path(sys.argv[1])
    root.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(74192)
    cases = []
    for profile in ["affine", "bias-only", "affine-saved-head", "gpt2-conv1d"]:
        out = root / profile
        if profile == "gpt2-conv1d":
            original = GPT2LMHeadModel(
                GPT2Config(
                    n_embd=12,
                    n_head=2,
                    n_layer=1,
                    n_inner=28,
                    n_positions=24,
                    vocab_size=31,
                    tie_word_embeddings=False,
                    resid_pdrop=0,
                    embd_pdrop=0,
                    attn_pdrop=0,
                    bos_token_id=1,
                    eos_token_id=2,
                )
            ).eval()
            targets = ["c_attn", "c_proj", "c_fc"]
            x = torch.randint(1, 31, (3, 10))
            target = None
            rank_pattern = {"transformer.h.0.mlp.c_fc": 3}
            alpha_pattern = {"transformer.h.0.mlp.c_proj": -3}
        else:
            original = Affine().eval()
            targets = ["first", "second"]
            x = torch.randn(6, 5)
            target = torch.randn(6, 2)
            rank_pattern = {"second": 3}
            alpha_pattern = {"second": -3}
        # Exact BF16 base values make the separate low-precision controls easier
        # to interpret without changing the producer's normal merge arithmetic.
        with torch.no_grad():
            for parameter in original.parameters():
                parameter.copy_(parameter.bfloat16().float())
        save_model(original, out / "base")
        adapted = get_peft_model(
            copy.deepcopy(original),
            LoraConfig(
                target_modules=targets,
                r=2,
                lora_alpha=6,
                lora_bias=True,
                use_rslora=profile == "affine-saved-head",
                modules_to_save=["head"] if profile == "affine-saved-head" else None,
                rank_pattern=rank_pattern,
                alpha_pattern=alpha_pattern,
            ),
        )
        if profile == "bias-only":
            for name, parameter in adapted.named_parameters():
                parameter.requires_grad_(name.endswith("lora_B.default.bias"))
        optimizer = torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad], lr=0.01)
        losses = []
        for _ in range(6):
            optimizer.zero_grad()
            result = adapted(x)
            loss = (
                torch.nn.functional.cross_entropy(
                    result.logits[:, :-1].reshape(-1, 31), x[:, 1:].reshape(-1)
                )
                if target is None
                else (result - target).square().mean()
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        with warnings.catch_warnings(record=True) as save_warnings:
            adapted.eval().save_pretrained(out / "adapter", save_embedding_layers=False)
        (out / "adapter/README.md").unlink(missing_ok=True)
        factors = load_file(out / "adapter/adapter_model.safetensors")
        layers = {}
        for name, layer in adapted.named_modules():
            if isinstance(layer, LoraLayer):
                path = name.removeprefix("base_model.model.")
                layers[path] = {
                    "scale": layer.scaling["default"],
                    "rank": layer.r["default"],
                    "kind": type(layer.get_base_layer()).__name__,
                }
        merges = []
        for label, dtype in [
            ("f32", torch.float32),
            ("f16", torch.float16),
            ("bf16", torch.bfloat16),
        ]:
            for safe in [False, True]:
                candidate = PeftModel.from_pretrained(
                    copy.deepcopy(original).to(dtype), out / "adapter", local_files_only=True
                ).eval()
                loaded_biases = 0
                for path, layer in candidate.named_modules():
                    if isinstance(layer, LoraLayer):
                        key = path + ".lora_B.bias"
                        assert torch.equal(layer.lora_B["default"].bias.cpu(), factors[key])
                        loaded_biases += 1
                assert loaded_biases == len(layers)
                args = x.to(dtype) if x.is_floating_point() else x
                with torch.inference_mode():
                    raw = candidate(args)
                    before = raw.logits if target is None else raw
                    merged = candidate.merge_and_unload(safe_merge=safe).eval()
                    raw = merged(args)
                    after = raw.logits if target is None else raw
                name = f"correct-{label}-safe-{str(safe).lower()}"
                save_model(merged, out / name)
                merges.append(
                    {
                        "name": name,
                        "purpose": "native merge",
                        "loaded_biases_equal_saved_adapter": loaded_biases,
                        "max_forward_difference": float(
                            (before.float() - after.float()).abs().max()
                        ),
                    }
                )
        # Start from the real F32 merger's output; change only its bias vectors.
        correct = (
            PeftModel.from_pretrained(
                copy.deepcopy(original), out / "adapter", local_files_only=True
            )
            .merge_and_unload()
            .eval()
        )
        for name in ["omitted-bias", "unscaled-bias"]:
            wrong = copy.deepcopy(correct)
            with torch.no_grad():
                for path in layers:
                    key = path + ".bias"
                    value = original.state_dict()[key]
                    if name == "unscaled-bias":
                        value = value + factors[f"base_model.model.{path}.lora_B.bias"]
                    wrong.state_dict()[key].copy_(value)
            save_model(wrong, out / name)
            merges.append(
                {"name": name, "purpose": "authored bias fault; matrix updates unchanged"}
            )
        cases.append(
            {
                "profile": profile,
                "training_loss": losses,
                "layers": layers,
                "merges": merges,
                "native_save_warnings": [str(w.message) for w in save_warnings],
            }
        )
    artifacts = [
        {
            "file": str(p.relative_to(root)),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]
    manifest = {
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "peft", "transformers", "safetensors", "numpy"]
        },
        "cases": cases,
        "artifacts": artifacts,
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Produced {len(cases)} native profiles and "
        f"{sum(len(c['merges']) for c in cases)} saved merges."
    )


if __name__ == "__main__":
    main()
