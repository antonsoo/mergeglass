"""Native trained base biases plus full saved heads; all data/models are synthetic."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file

PROFILES = ("all-head", "lora-only-head", "dora-all-head", "nested-all-head")


class Head(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = torch.nn.Linear(3, 4)
        self.out = torch.nn.Linear(4, 2)
        self.register_buffer("updates", torch.tensor(2**60, dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.updates += 1
        return self.out(self.dense(x).tanh())


class Network(torch.nn.Module):
    def __init__(self, nested: bool) -> None:
        super().__init__()
        self.first = torch.nn.Linear(5, 7)
        self.second = torch.nn.Linear(7, 3)
        self.head = Head() if nested else torch.nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.second(self.first(x).tanh()).tanh())


def save(model: torch.nn.Module, path: Path) -> None:
    path.mkdir(parents=True)
    save_file(model.state_dict(), path / "model.safetensors")


def main() -> None:
    root = Path(sys.argv[1])
    root.mkdir()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(46891)
    for profile in PROFILES:
        out = root / profile
        original = Network(profile == "nested-all-head").eval()
        with torch.no_grad():
            for parameter in original.parameters():
                parameter.copy_(parameter.bfloat16().float())
        save(original, out / "base")
        config = LoraConfig(
            target_modules=["first", "second"],
            modules_to_save=["head"],
            r=2,
            lora_alpha=6,
            rank_pattern={"second": 3},
            alpha_pattern={"second": -3},
            use_rslora=profile == "nested-all-head",
            use_dora=profile == "dora-all-head",
            bias="lora_only" if profile == "lora-only-head" else "all",
        )
        model = get_peft_model(copy.deepcopy(original), config).train()
        x, target = torch.randn(8, 5), torch.randn(8, 2)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.02)
        for _ in range(12):
            optimizer.zero_grad()
            loss = (model(x) - target).square().mean()
            loss.backward()
            optimizer.step()
        model.eval().save_pretrained(out / "adapter", save_embedding_layers=False)
        (out / "adapter/README.md").unlink(missing_ok=True)
        payload = load_file(out / "adapter/adapter_model.safetensors")
        native = copy.deepcopy(model).merge_and_unload().eval()
        save(native, out / "in-memory-f32")
        biases = [name for name in native.state_dict() if name.endswith(".bias")]
        observations = []
        for label, dtype in [
            ("f32", torch.float32),
            ("f16", torch.float16),
            ("bf16", torch.bfloat16),
        ]:
            for safe in [False, True]:
                reloaded = PeftModel.from_pretrained(
                    copy.deepcopy(original).to(dtype), out / "adapter", local_files_only=True
                ).eval()
                with torch.inference_mode():
                    expected = model(x)
                    before = reloaded(x.to(dtype)).float()
                    merged = reloaded.merge_and_unload(safe_merge=safe).eval()
                    after = merged(x.to(dtype)).float()
                name = f"reloaded-{label}-safe-{str(safe).lower()}"
                save(merged, out / name)
                observations.append(
                    {
                        "name": name,
                        "max_reload_forward_difference": float((expected - before).abs().max()),
                        "max_merge_forward_difference": float((before - after).abs().max()),
                    }
                )
        for label in ["omitted-bias-f32", "scaled-bias-f32", "omitted-head-f32"]:
            wrong = copy.deepcopy(native)
            if label == "omitted-head-f32":
                wrong.head.load_state_dict(original.head.state_dict())
            else:
                with torch.no_grad():
                    for name in biases:
                        tensor = wrong.get_parameter(name)
                        tensor.copy_(
                            original.get_parameter(name)
                            if label.startswith("omitted")
                            else tensor * 3
                        )
            save(wrong, out / label)
        bad_adapters = []
        if config.bias == "all":
            for wrapper in ["original_module", "modules_to_save"]:
                fault = {name: tensor.clone() for name, tensor in payload.items()}
                key = next(k for k in fault if f".head.{wrapper}." in k and k.endswith(".bias"))
                fault[key].reshape(-1)[0] += 1
                name = f"conflicting-{wrapper}-adapter"
                path = out / name
                path.mkdir()
                (path / "adapter_config.json").write_bytes(
                    (out / "adapter/adapter_config.json").read_bytes()
                )
                save_file(fault, path / "adapter_model.safetensors")
                bad_adapters.append({"name": name, "changed_tensor": key})
        (out / "observations.json").write_text(
            json.dumps(
                {
                    "profile": profile,
                    "synthetic": True,
                    "versions": {
                        name: importlib.metadata.version(name)
                        for name in ["torch", "peft", "transformers", "safetensors"]
                    },
                    "bias_mode": config.bias,
                    "trained_biases": sorted(biases),
                    "full_saved_tensors": ["head." + name for name in native.head.state_dict()],
                    "saved_keys": sorted(payload),
                    "native_merges": observations,
                    "conflicting_adapters": bad_adapters,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            profile, len(payload), "adapter tensors;", len(bad_adapters), "authored alias conflicts"
        )
    (root / "environment.txt").write_text(
        "\n".join(
            sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
        )
        + "\n"
    )
    manifest = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            raw = path.read_bytes()
            manifest.append(
                {
                    "file": path.relative_to(root).as_posix(),
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
