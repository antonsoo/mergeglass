"""Reproduce a pinned public legacy-bias load and an explicit key-only control.

Requires the study's Torch/Transformers/PEFT environment, not runtime dependencies.
Downloads are pinned and verified before weights_only=True pickle loading.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import importlib.metadata
import json
import shutil
import sys
import urllib.request
import warnings
from pathlib import Path

import torch
from peft import PeftModel
from peft.tuners.lora import LoraLayer
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def identity(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def produce(cache: Path, out: Path) -> None:
    require(not out.exists(), "Use a new output directory")
    sources = json.loads(Path(__file__).with_name("sources.json").read_text())
    for source in sources:
        for entry in source["files"]:
            path = cache / entry["file"]
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".partial")
                with urllib.request.urlopen(entry["url"], timeout=120) as response:
                    with temporary.open("wb") as target:
                        shutil.copyfileobj(response, target, length=1024 * 1024)
                temporary.rename(path)
            require(
                identity(path) == {k: entry[k] for k in ("bytes", "sha256")},
                f"Pinned source identity differs: {entry['file']}",
            )
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    raw = torch.load(cache / "adapter/adapter_model.bin", map_location="cpu", weights_only=True)
    raw_base = torch.load(cache / "base/pytorch_model.bin", map_location="cpu", weights_only=True)
    with warnings.catch_warnings(record=True) as base_warnings:
        original = AutoModelForCausalLM.from_pretrained(
            cache / "base", dtype=torch.float32, local_files_only=True, trust_remote_code=False
        ).eval()
    state = original.state_dict()
    require(not state.keys() - raw_base.keys(), "Native base has unverified new tensors")
    dropped = sorted(raw_base.keys() - state.keys())
    require(
        dropped == sorted(f"transformer.h.{i}.attn.causal_mask" for i in range(20)),
        "Unexpected raw base exclusions",
    )
    for key, value in state.items():
        require(
            value.shape == raw_base[key].shape
            and torch.equal(value, raw_base[key].to(value.dtype)),
            f"Base changed: {key}",
        )
    base_check = {
        "tensors": len(state),
        "elements": sum(v.numel() for v in state.values()),
        "dropped_raw_tensors": [
            {"name": k, "shape": list(raw_base[k].shape), "dtype": str(raw_base[k].dtype)}
            for k in dropped
        ],
        "new_native_tensors": [],
    }
    original.save_pretrained(out / "base", max_shard_size="1GB")
    normalized = out / "adapter"
    normalized.mkdir()
    shutil.copyfile(cache / "adapter/adapter_config.json", normalized / "adapter_config.json")
    save_file({k: v.contiguous() for k, v in raw.items()}, normalized / "adapter_model.safetensors")
    reloaded = load_file(normalized / "adapter_model.safetensors")
    require(
        set(reloaded) == set(raw)
        and all(
            torch.equal(reloaded[k], v) and reloaded[k].dtype == v.dtype for k, v in raw.items()
        ),
        "Adapter conversion changed names, values or dtypes",
    )
    factors = {k: v for k, v in raw.items() if ".lora_" in k}
    biases = sorted(k for k in state if k.endswith(".bias"))
    require(len(factors) == 162 and len(biases) == 62, "Unexpected adapter inventory")
    modules = {
        k.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
        for k in factors
        if k.endswith(".lora_A.weight")
    }
    fixed, mapping = {}, []
    for key, value in raw.items():
        path = key.removeprefix("base_model.model.").removesuffix(".bias")
        dest = key
        if key.endswith(".bias") and path in modules:
            dest = key.removesuffix(".bias") + ".base_layer.bias"
            mapping.append({"original": key, "mapped": dest})
        require(dest not in fixed, "Mapping collision")
        fixed[dest] = value.contiguous()
    require(len(mapping) == 41, "Unexpected key-only mapping scope")
    remapped = out / "remapped-adapter"
    remapped.mkdir()
    shutil.copyfile(normalized / "adapter_config.json", remapped / "adapter_config.json")
    save_file(fixed, remapped / "adapter_model.safetensors")
    del raw_base, reloaded, state
    gc.collect()
    results = []
    cases = [
        ("native-f32", torch.float32),
        ("native-f16", torch.float16),
        ("native-bf16", torch.bfloat16),
        ("remapped-f32", torch.float32),
    ]
    for name, dtype in cases:
        source = remapped if name == "remapped-f32" else cache / "adapter"
        with warnings.catch_warnings(record=True) as caught:
            model = PeftModel.from_pretrained(
                copy.deepcopy(original).to(dtype), source, local_files_only=True
            ).eval()
        # PEFT may retain FP32 adapter factors over a lower-precision base.
        for key, value in factors.items():
            live = key.removesuffix(".weight") + ".default.weight"
            actual = model.get_parameter(live).detach()
            require(torch.equal(actual, value.to(actual.dtype)), f"Factor load differs: {key}")
        before, bias_rows = {}, []
        for key in biases:
            module = model.base_model.model.get_submodule(key.rsplit(".", 1)[0])
            if isinstance(module, LoraLayer):
                module = module.get_base_layer()
            actual = module.bias.detach().cpu().clone()
            before[key] = actual
            expected = raw["base_model.model." + key].to(actual.dtype)
            mismatches = int((actual != expected).sum())
            bias_rows.append(
                {"weight": key, "elements": actual.numel(), "loaded_target_mismatches": mismatches}
            )
            if name == "remapped-f32":
                require(mismatches == 0, f"Mapped bias load differs: {key}")
        merged = model.merge_and_unload(safe_merge=False).eval()
        for key, value in before.items():
            require(torch.equal(merged.get_parameter(key), value), f"Merge changed bias: {key}")
        merged.save_pretrained(out / name, max_shard_size="1GB")
        results.append(
            {
                "name": name,
                "native_loaded_biases": bias_rows,
                "factors_loaded_exactly": len(factors),
                "merge_preserved_loaded_biases": True,
                "load_warnings": [str(w.message) for w in caught],
            }
        )
        print(name, sum(r["loaded_target_mismatches"] for r in bias_rows), flush=True)
        if name == "native-f32":
            with torch.no_grad():
                for key in biases:
                    merged.get_parameter(key).copy_(original.get_parameter(key))
            merged.save_pretrained(out / "omitted-bias-f32", max_shard_size="1GB")
        del model, merged, before
        gc.collect()
    receipt = {
        "versions": {
            n: importlib.metadata.version(n)
            for n in ["torch", "transformers", "peft", "safetensors"]
        },
        "base_value_check": base_check,
        "sources": sources,
        "adapter_conversion": {"tensors": len(raw), "values_names_dtypes_exact": True},
        "key_only_mapping": mapping,
        "base_load_warnings": [str(w.message) for w in base_warnings],
        "outputs": results,
    }
    (out / "producer.json").write_text(json.dumps(receipt, indent=2) + "\n")
    manifest = [
        {"file": p.relative_to(out).as_posix(), **identity(p)}
        for p in sorted(out.rglob("*"))
        if p.is_file()
    ]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: produce.py CACHE NEW_OUTPUT_DIRECTORY")
    produce(Path(sys.argv[1]), Path(sys.argv[2]))
