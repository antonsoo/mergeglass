"""Pinned public GPT-2 adapter and native PEFT merges; no weight redistribution."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from peft.tuners.lora import LoraLayer
from safetensors import safe_open
from transformers import GPT2LMHeadModel

BASE = ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e")
ADAPTER = ("palsp/gpt2-lora", "4235d59235a7269b34119195e63e1424d2a530e1")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(sys.argv[1])
    root.mkdir(parents=True)
    torch.set_num_threads(2)
    sources = []
    for label, (repo, revision), names in [
        ("base-source", BASE, ["config.json", "model.safetensors", "README.md"]),
        ("adapter", ADAPTER, ["adapter_config.json", "adapter_model.safetensors", "README.md"]),
    ]:
        for name in names:
            path = Path(hf_hub_download(repo, name, revision=revision, local_dir=root / label))
            sources.append(
                {
                    "repo": repo,
                    "revision": revision,
                    "file": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": file_hash(path),
                }
            )
    base = GPT2LMHeadModel.from_pretrained(root / "base-source", local_files_only=True).eval()
    # The historical public file uses bare GPT2Model names and includes twelve
    # nonpersistent causal-mask buffers. Native LM-head saves use transformer.*
    # names and omit those buffers. Establish equality of every retained value
    # before using the native original-base save as the audit input.
    base.save_pretrained(root / "base", max_shard_size="64MB")
    normalized = {}
    source_names = set()
    with safe_open(root / "base-source/model.safetensors", framework="pt") as source:
        source_names.update(source.keys())
        for path in sorted((root / "base").glob("*.safetensors")):
            with safe_open(path, framework="pt") as saved:
                for key in saved.keys():
                    original_key = key.removeprefix("transformer.")
                    assert original_key in source_names
                    assert torch.equal(saved.get_tensor(key), source.get_tensor(original_key))
                    normalized[key] = original_key
        omitted = sorted(source_names - set(normalized.values()))
        assert len(omitted) == 12 and all(name.endswith(".attn.bias") for name in omitted)
        omitted_buffers = [
            {"tensor": key, "shape": list(source.get_slice(key).get_shape())} for key in omitted
        ]
    tokens = torch.tensor(
        [[15496, 11, 616, 1438, 318, 257, 1332, 13], [40, 716, 1016, 284, 466, 340, 0, 198]]
    )
    cases = []
    layouts = None
    for label, dtype in [("f32", torch.float32), ("f16", torch.float16), ("bf16", torch.bfloat16)]:
        candidate = PeftModel.from_pretrained(
            copy.deepcopy(base).to(dtype), root / "adapter", local_files_only=True
        ).eval()
        if layouts is None:
            layouts = {
                path.removeprefix("base_model.model."): {
                    "kind": type(layer.get_base_layer()).__name__,
                    "transpose": layer.fan_in_fan_out,
                    "scale": layer.scaling["default"],
                    "rank": layer.r["default"],
                }
                for path, layer in candidate.named_modules()
                if isinstance(layer, LoraLayer)
            }
        with torch.inference_mode():
            unmerged = candidate(tokens).logits.float()
            merged = candidate.merge_and_unload(safe_merge=False)
            saved = merged(tokens).logits.float()
        name = f"native-{label}"
        merged.save_pretrained(root / name, max_shard_size="64MB")
        cases.append(
            {
                "name": name,
                "native_merge": True,
                "forward": {
                    "input_token_ids": tokens.tolist(),
                    "prediction_positions": tokens.numel(),
                    "max_logit_difference": float((saved - unmerged).abs().max()),
                    "argmax_changes": int((saved.argmax(-1) != unmerged.argmax(-1)).sum()),
                },
            }
        )
        if label == "f32":
            # A shape-compatible authored wrong conversion for all square
            # attention output projections. Other adapted modules stay native.
            wrong = copy.deepcopy(merged)
            with torch.no_grad():
                for path, layer in wrong.named_modules():
                    if path.endswith(".attn.c_proj"):
                        old = base.get_submodule(path).weight
                        delta = layer.weight - old
                        layer.weight.copy_(old + delta.T)
            wrong.save_pretrained(root / "wrong-square-orientation", max_shard_size="64MB")
            cases.append({"name": "wrong-square-orientation", "native_merge": False})
    result = {
        "sources": sources,
        "native_base_serialization": {
            "all_retained_values_equal": True,
            "saved_to_original_names": normalized,
            "omitted_nonpersistent_causal_mask_buffers": omitted_buffers,
            "audited_base": "base",
            "unchanged_downloaded_base": "base-source",
        },
        "training_provenance": (
            "unknown; the public card does not identify the training dataset or base revision"
        ),
        "merging": (
            "locally produced with native PEFT; no claim about the adapter author's merged output"
        ),
        "native_layouts": layouts,
        "cases": cases,
        "script_sha256": file_hash(Path(__file__)),
        "versions": {
            key: importlib.metadata.version(key)
            for key in ["torch", "peft", "transformers", "numpy", "safetensors"]
        },
        "files": [
            {
                "file": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
            for path in sorted(root.rglob("*"))
            if path.is_file() and ".cache" not in path.parts
        ],
    }
    (root / "producer.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        f"Produced {len(cases)} public artifact merges across {len(layouts or {})} native layouts."
    )


if __name__ == "__main__":
    main()
