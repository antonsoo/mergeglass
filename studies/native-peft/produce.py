"""Create native PEFT artifacts from labelled synthetic training, without downloads.

Run with the pinned producer environment documented in this study. Audit runtime
does not depend on torch, transformers or PEFT. No pretrained weights are used.
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
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file


class TeachingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.small = torch.nn.Linear(8, 8, bias=False)
        self.large = torch.nn.Linear(8, 8, bias=False)
        self.zero = torch.nn.Linear(8, 8, bias=False)
        self.control = torch.nn.Linear(8, 8, bias=False)
        self.config = {"model_type": "mergeglass_synthetic"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.large(self.small(x)) + self.zero(x) + self.control(x)


def fingerprint(root: Path) -> list[dict[str, object]]:
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
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(71019)
    base = TeachingModel()
    # Exactly representable, nonzero BF16 base. This isolates merge rounding
    # from an earlier conversion of the base, and makes the counterexample legible.
    with torch.no_grad():
        for parameter in base.parameters():
            parameter.copy_(parameter.bfloat16().float())
    (out / "base").mkdir()
    save_file(base.state_dict(), out / "base/model.safetensors")
    config = LoraConfig(r=2, lora_alpha=2, target_modules=["small", "large", "zero"])
    model = get_peft_model(copy.deepcopy(base), config)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    x = torch.randn(32, 8)
    target = x * 0.2
    losses = []
    for _ in range(12):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        # Controlled post-training scales: do not claim these are discovered
        # rates in a production model or unmodified optimizer outputs.
        model.base_model.model.small.lora_B["default"].weight.mul_(1e-4)
        model.base_model.model.large.lora_B["default"].weight.mul_(100)
        model.base_model.model.zero.lora_B["default"].weight.zero_()
    model.save_pretrained(out / "adapter", safe_serialization=True)
    (out / "adapter/README.md").unlink(missing_ok=True)
    # The generic auto_mapping names this script's class. It is inert metadata:
    # mergeglass never imports it or calls any model-loading API.
    for name, dtype in [
        ("merged-f32", torch.float32),
        ("merged-bf16", torch.bfloat16),
        ("merged-f16", torch.float16),
    ]:
        candidate = copy.deepcopy(model)
        # Keep adapters in FP32, matching the common promoted-adapter path.
        for module in candidate.modules():
            if hasattr(module, "base_layer"):
                module.base_layer.to(dtype)
        candidate.base_model.model.control.to(dtype)
        merged = candidate.merge_and_unload(safe_merge=False)
        folder = out / name
        folder.mkdir()
        save_file(merged.state_dict(), folder / "model.safetensors")
    # A copied base and an unrelated edit demonstrate that numeric evidence
    # cannot infer why an update is absent, and catches changes outside LoRA.
    (out / "omitted").mkdir()
    save_file(base.state_dict(), out / "omitted/model.safetensors")
    bad = copy.deepcopy(base.state_dict())
    bad["control.weight"][0, 0] += 0.125
    (out / "unrelated-edit").mkdir()
    save_file(bad, out / "unrelated-edit/model.safetensors")
    receipt = {
        "synthetic": True,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "seed": 71019,
        "training": {
            "steps": 12,
            "optimizer": "SGD",
            "lr": 1e-3,
            "data": "seeded random 32x8 inputs; targets = 0.2 * input",
            "losses": losses,
        },
        "post_training_controls": {
            "small_B_multiplier": 1e-4,
            "large_B_multiplier": 100,
            "zero_B": "set to zero",
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "peft", "transformers", "safetensors", "numpy"]
        },
        "merge": "PEFT merge_and_unload(safe_merge=False), FP32 adapters, CPU, one thread",
        "files": fingerprint(out),
    }
    (out / "producer.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
