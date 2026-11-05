"""Optional native wide saved-head producer for bounded-memory verification.

Creates approximately 193 MiB of synthetic weights in a fresh directory.
No pretrained model, tokenizer, network access or quality evaluation is used.
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


class Wide(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(16, 4096)
        self.head = torch.nn.Linear(4096, 4096)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.first(x).tanh())


def save(model: torch.nn.Module, path: Path) -> None:
    path.mkdir(parents=True)
    save_file(model.state_dict(), path / "model.safetensors")


def main() -> None:
    root = Path(sys.argv[1])
    root.mkdir()
    torch.manual_seed(612983)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    original = Wide().eval()
    save(original, root / "base")
    model = get_peft_model(
        copy.deepcopy(original),
        LoraConfig(
            r=2,
            lora_alpha=6,
            target_modules=["first"],
            bias="all",
            modules_to_save=["head"],
        ),
    ).train()
    x, target = torch.randn(4, 16), torch.randn(4, 4096)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05)
    for _ in range(3):
        optimizer.zero_grad()
        loss = (model(x) - target).square().mean()
        loss.backward()
        optimizer.step()
    model.eval().save_pretrained(root / "adapter", save_embedding_layers=False)
    (root / "adapter/README.md").unlink(missing_ok=True)
    loaded = PeftModel.from_pretrained(
        copy.deepcopy(original), root / "adapter", local_files_only=True
    ).eval()
    with torch.inference_mode():
        discrepancy = float((model(x) - loaded(x)).abs().max())
    if discrepancy != 0:
        raise ValueError(f"Reload changed the FP32 synthetic forward: {discrepancy}")
    save(loaded.merge_and_unload(), root / "merged")
    (root / "observations.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "head_shape": [4096, 4096],
                "head_elements": 4096 * 4096,
                "steps": 3,
                "max_reload_forward_difference": discrepancy,
                "versions": {
                    n: importlib.metadata.version(n)
                    for n in ["torch", "peft", "transformers", "safetensors"]
                },
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
    print(
        json.dumps(
            {
                "head_elements": 4096**2,
                "files": len(manifest),
                "bytes": sum(e["bytes"] for e in manifest),
                "reload_difference": discrepancy,
            }
        )
    )


if __name__ == "__main__":
    main()
