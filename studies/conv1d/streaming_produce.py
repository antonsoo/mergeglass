"""A native wide-input Conv1D whose decoded A exceeds a one-MiB factor limit."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import GPT2Config, GPT2LMHeadModel


def main() -> None:
    root = Path(sys.argv[1])
    root.mkdir(parents=True)
    torch.manual_seed(819512)
    torch.set_num_threads(1)
    config = GPT2Config(
        n_embd=512,
        n_head=8,
        n_layer=1,
        n_inner=32768,
        n_positions=32,
        vocab_size=43,
        tie_word_embeddings=False,
    )
    original = GPT2LMHeadModel(config)
    original.save_pretrained(root / "base", max_shard_size="64MB")
    adapted = get_peft_model(
        original, LoraConfig(r=8, lora_alpha=16, target_modules=["transformer.h.0.mlp.c_proj"])
    )
    with torch.no_grad():
        layer = adapted.base_model.model.transformer.h[0].mlp.c_proj
        assert isinstance(layer, LoraLayer)
        layer.lora_B["default"].weight.normal_(0, 0.001)
        a = layer.lora_A["default"].weight.double()
        b = layer.lora_B["default"].weight.double()
        before = layer.base_layer.weight.detach().clone().double()
        delta = (b @ a).T * 2
    adapted.save_pretrained(root / "adapter", save_embedding_layers=False)
    (root / "adapter/README.md").unlink(missing_ok=True)
    merged = adapted.merge_and_unload()
    merged.save_pretrained(root / "merged", max_shard_size="64MB")
    after = merged.transformer.h[0].mlp.c_proj.weight.detach().double()
    observed = after - before
    residual = observed - delta
    index = int(residual.abs().reshape(-1).argmax())
    row, column = divmod(index, 512)
    reference = {
        "synthetic": True,
        "training": False,
        "control": "native seeded nonzero B; native PEFT merge; no trained-task claim",
        "seed": 819512,
        "shape": list(delta.shape),
        "decoded_A_bytes": a.numel() * 8,
        "decoded_B_bytes": b.numel() * 8,
        "full_float64_delta_bytes": delta.numel() * 8,
        "norms": {
            "delta": float(delta.norm()),
            "observed": float(observed.norm()),
            "residual": float(residual.norm()),
        },
        "relative_residual": float(residual.norm() / delta.norm()),
        "changed_elements": int(observed.count_nonzero()),
        "intended_nonzero_elements": int(delta.count_nonzero()),
        "unchanged_nonzero_update_elements": int(((delta != 0) & (observed == 0)).sum()),
        "largest_residual": {
            "row": row,
            "column": column,
            "residual": float(residual[row, column]),
        },
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (root / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(json.dumps(reference, indent=2))


if __name__ == "__main__":
    main()
