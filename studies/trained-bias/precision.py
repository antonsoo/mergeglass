"""Native saved-bias/head rounding controls with explicit one-FP32-ulp updates.

Train small original models, then retain the sign of each trained bias/head
change while setting its size to one FP32 step from the original BF16-exact
base. This post-training control is authored, not a naturally trained adapter.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import save_file


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
    save_file(model.state_dict(), path / "model.safetensors")


def one_step(original: torch.Tensor, trained: torch.Tensor) -> torch.Tensor:
    if not torch.all(original != trained):
        raise ValueError("Every controlled value needs an actual trained direction")
    return torch.nextafter(original, trained)


def main() -> None:
    root = Path(sys.argv[1])
    root.mkdir()
    torch.manual_seed(52731)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for profile in ["bias", "head"]:
        out = root / profile
        original = Affine().eval()
        with torch.no_grad():
            for p in original.parameters():
                p.copy_(p.bfloat16().float())
        save(original, out / "base")
        model = get_peft_model(
            copy.deepcopy(original),
            LoraConfig(
                target_modules=["first", "second"],
                r=2,
                lora_alpha=6,
                bias="all",
                modules_to_save=["head"] if profile == "head" else None,
            ),
        ).train()
        x, target = torch.randn(8, 5), torch.randn(8, 2)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.02)
        for _ in range(8):
            optimizer.zero_grad()
            loss = (model(x) - target).square().mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            for name in ["first", "second"]:
                actual = model.base_model.model.get_submodule(name).base_layer.bias
                actual.copy_(one_step(original.get_submodule(name).bias, actual))
            head = model.base_model.model.head
            if profile == "head":
                active = head.modules_to_save["default"]
                for name, actual in active.named_parameters():
                    actual.copy_(one_step(original.head.get_parameter(name), actual))
            else:
                head.bias.copy_(one_step(original.head.bias, head.bias))
        model.eval().save_pretrained(out / "adapter", save_embedding_layers=False)
        (out / "adapter/README.md").unlink(missing_ok=True)
        cases = []
        for label, dtype in [
            ("f32", torch.float32),
            ("f16", torch.float16),
            ("bf16", torch.bfloat16),
        ]:
            for safe in [False, True]:
                loaded = PeftModel.from_pretrained(
                    copy.deepcopy(original).to(dtype), out / "adapter", local_files_only=True
                ).eval()
                merged = loaded.merge_and_unload(safe_merge=safe)
                name = f"merged-{label}-safe-{str(safe).lower()}"
                save(merged, out / name)
                cases.append(name)
        (out / "observations.json").write_text(
            json.dumps(
                {
                    "profile": profile,
                    "synthetic": True,
                    "control": "one FP32 step toward trained values",
                    "ordinary_bias_elements": 12 if profile == "bias" else 10,
                    "full_saved_elements": 0 if profile == "bias" else 8,
                    "merges": cases,
                },
                indent=2,
            )
            + "\n"
        )
        print(profile, len(cases), "native precision/safe-merge outputs")
    (root / "environment.json").write_text(
        json.dumps(
            {
                n: importlib.metadata.version(n)
                for n in ["torch", "peft", "transformers", "safetensors"]
            },
            indent=2,
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
