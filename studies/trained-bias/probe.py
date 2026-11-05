"""Native PEFT trained-base-bias artifacts and explicitly authored merge faults.

All training data and models are synthetic. No pretrained weights or network
calls are needed. A separately named complete adapter captures trained values
from memory; it is not represented as an unchanged save_pretrained output.
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
from safetensors.torch import load_file, save_file
from transformers import GPT2Config, GPT2LMHeadModel


class Affine(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(5, 7)
        self.second = torch.nn.Linear(7, 3)
        self.head = torch.nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.second(self.first(x).tanh()).tanh())


def save(model: torch.nn.Module, path: Path) -> None:
    path.mkdir(parents=True)
    if isinstance(model, GPT2LMHeadModel):
        model.save_pretrained(path)
    else:
        save_file(model.state_dict(), path / "model.safetensors")


def output(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    result = model(x)
    return result if isinstance(result, torch.Tensor) else result.logits


def main() -> None:
    root = Path(sys.argv[1])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(37192)
    for profile in ["all", "lora_only", "dora-all", "conv1d-all"]:
        out = root / profile
        mode = "lora_only" if profile == "lora_only" else "all"
        if profile == "conv1d-all":
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
            target = torch.randn(3, 10, 31)
        else:
            original = Affine().eval()
            targets = ["first", "second"]
            x, target = torch.randn(8, 5), torch.randn(8, 2)
        # Representable base values keep casting distinct from lost training changes.
        with torch.no_grad():
            for p in original.parameters():
                p.copy_(p.bfloat16().float())
        save(original, out / "base")
        model = get_peft_model(
            copy.deepcopy(original),
            LoraConfig(
                target_modules=targets,
                r=2,
                lora_alpha=6,
                bias=mode,
                use_dora=profile == "dora-all",
            ),
        )
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.02)
        for _ in range(12):
            optimizer.zero_grad()
            loss = (output(model, x) - target).square().mean()
            loss.backward()
            optimizer.step()
        with warnings.catch_warnings(record=True) as observed_warnings:
            model.eval().save_pretrained(out / "adapter", save_embedding_layers=False)
        (out / "adapter/README.md").unlink(missing_ok=True)
        in_memory = copy.deepcopy(model).merge_and_unload().eval()
        save(in_memory, out / "in-memory-f32")
        payload = load_file(out / "adapter/adapter_model.safetensors")
        # Capture every actually trainable base bias before unload. Plain bias=all
        # also selects LayerNorm biases in the GPT case, beyond LoRA targets.
        trained = {}
        for name, p in model.named_parameters():
            if p.requires_grad and name.endswith(".bias"):
                trained[name.removeprefix("base_model.model.").replace(".base_layer.", ".")] = (
                    p.detach()
                )
        missing = []
        complete = dict(payload)
        for name, p in trained.items():
            wrapped = name.removesuffix(".bias") + ".base_layer.bias"
            key = "base_model.model." + (
                wrapped if "base_model.model." + wrapped in model.state_dict() else name
            )
            if key not in payload:
                missing.append(key)
                complete[key] = p
        if missing:
            # Deliberately authored recovery using values still available in memory.
            destination = out / "captured-complete-adapter"
            destination.mkdir()
            (destination / "adapter_config.json").write_bytes(
                (out / "adapter/adapter_config.json").read_bytes()
            )
            save_file(complete, destination / "adapter_model.safetensors")
            recovered = PeftModel.from_pretrained(
                copy.deepcopy(original), destination, local_files_only=True
            ).eval()
            with torch.inference_mode():
                if not torch.equal(output(recovered, x), output(model, x)):
                    raise ValueError("Captured trained biases did not recover the original forward")
            save(recovered.merge_and_unload(), out / "captured-reloaded-f32")
        measurements = []
        for label, dtype in [
            ("f32", torch.float32),
            ("f16", torch.float16),
            ("bf16", torch.bfloat16),
        ]:
            for safe in [False, True]:
                reloaded = PeftModel.from_pretrained(
                    copy.deepcopy(original).to(dtype), out / "adapter", local_files_only=True
                ).eval()
                args = x if profile == "conv1d-all" else x.to(dtype)
                with torch.inference_mode():
                    expected = output(model, x)
                    loaded = output(reloaded, args).float()
                    merged = reloaded.merge_and_unload(safe_merge=safe).eval()
                    actual = output(merged, args).float()
                name = f"reloaded-{label}-safe-{str(safe).lower()}"
                save(merged, out / name)
                measurements.append(
                    {
                        "name": name,
                        "max_reload_forward_difference": float((expected - loaded).abs().max()),
                        "max_merge_forward_difference": float((loaded - actual).abs().max()),
                    }
                )
        # Authored faults start from the true native FP32 in-memory merger.
        # They change only absolute base biases; every adapted weight survives.
        for label in ["omitted-bias-f32", "scaled-bias-f32"]:
            wrong = copy.deepcopy(in_memory)
            with torch.no_grad():
                for name in trained:
                    dest = wrong.get_parameter(name)
                    dest.copy_(
                        original.get_parameter(name) if label.startswith("omitted") else dest * 3
                    )
            save(wrong, out / label)
        observations = {
            "synthetic": True,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ["torch", "peft", "transformers", "safetensors"]
            },
            "profile": profile,
            "bias_mode": mode,
            "saved_keys": sorted(payload),
            "trained_biases": sorted(trained),
            "missing_trained_bias_keys": missing,
            "save_warnings": [str(w.message) for w in observed_warnings],
            "merges": measurements,
        }
        (out / "observations.json").write_text(json.dumps(observations, indent=2) + "\n")
        print(
            profile,
            "missing:",
            missing,
            "reload error:",
            measurements[0]["max_reload_forward_difference"],
        )
    # Exact installed producer dependency set, including transitive dependencies.
    (root / "environment.txt").write_text(
        "\n".join(
            sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
        )
        + "\n"
    )
    manifest = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            manifest.append(
                {
                    "file": path.relative_to(root).as_posix(),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
