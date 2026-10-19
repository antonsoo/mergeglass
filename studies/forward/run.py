"""Actual CPU forward comparisons on the pinned public adapter.

This is a numerical equivalence probe, not an accuracy benchmark. No generation
or model-supplied executable code is used. Full logits remain outside the repo.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def compare(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = torch.from_numpy(np.load(reference_path, allow_pickle=False)).double()
    candidate = torch.from_numpy(np.load(candidate_path, allow_pickle=False)).double()
    assert reference.shape == candidate.shape
    assert torch.isfinite(reference).all() and torch.isfinite(candidate).all()
    difference = candidate - reference
    ref_ids, candidate_ids = reference.argmax(-1), candidate.argmax(-1)
    changed = torch.nonzero(ref_ids != candidate_ids).flatten()
    total_variation = (reference.softmax(-1) - candidate.softmax(-1)).abs().sum(-1) / 2
    assert torch.all((total_variation >= 0) & (total_variation <= 1))
    return {
        "positions": reference.shape[0],
        "vocabulary": reference.shape[1],
        "max_abs_logit_difference": float(difference.abs().max()),
        "logit_rmse": float((difference.square().mean()).sqrt()),
        "changed_top1_positions": len(changed),
        "mean_probability_total_variation": float(total_variation.mean()),
        "max_probability_total_variation": float(total_variation.max()),
        "first_top1_changes": [
            {
                "position": int(position),
                "reference_token_id": int(ref_ids[position]),
                "candidate_token_id": int(candidate_ids[position]),
            }
            for position in changed[:5]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("community", type=Path, help="existing pinned community merge artifacts")
    parser.add_argument("directory", type=Path, help="throwaway logits/output directory")
    parser.add_argument("out", type=Path, help="small JSON receipt, not full logits")
    args = parser.parse_args()
    root, cache = args.community, args.directory
    cache.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((root / "merges.json").read_text())
    recorded = Path(__file__).parents[1] / "community/results/verification.json"
    assert receipt == json.loads(recorded.read_text())["producer"], "Not the recorded public merges"
    for source in receipt["sources"]["sources"]:
        for item in source["files"]:
            path = root / source["role"] / item["file"]
            assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
    for merge in receipt["merges"]:
        for item in merge["files"]:
            path = root / merge["name"] / item["file"]
            assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
    tokenizer_source = json.loads(Path(__file__).with_name("tokenizer-source.json").read_text())
    tokenizer_file = cache / "tokenizer.json"
    if not tokenizer_file.exists() or sha(tokenizer_file) != tokenizer_source["sha256"]:
        with urllib.request.urlopen(tokenizer_source["url"], timeout=60) as response:
            data = response.read()
        assert len(data) == tokenizer_source["bytes"]
        assert hashlib.sha256(data).hexdigest() == tokenizer_source["sha256"]
        tokenizer_file.write_bytes(data)
    assert tokenizer_file.stat().st_size == tokenizer_source["bytes"]
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    probes = json.loads(Path(__file__).with_name("probes.json").read_text())
    for probe in probes:
        probe["token_ids"] = tokenizer.encode(probe["text"], add_special_tokens=False).ids
        assert 2 <= len(probe["token_ids"]) <= 128
    torch.set_num_threads(1)
    torch.manual_seed(71022)
    profiles = []
    for precision, dtype in [("f32", torch.float32), ("bf16", torch.bfloat16)]:
        for kind in ("base", "wrapped", "merged"):
            folder = root / (f"merged-{precision}" if kind == "merged" else "base")
            model = AutoModelForCausalLM.from_pretrained(
                folder,
                dtype=dtype,
                local_files_only=True,
                trust_remote_code=False,
                attn_implementation="eager",
            )
            if kind == "wrapped":
                model = PeftModel.from_pretrained(
                    model, root / "adapter", local_files_only=True, autocast_adapter_dtype=True
                )
            model.eval()
            profile = f"{kind}-{precision}"
            adapter_dtypes = sorted(
                {
                    str(p.dtype)
                    for name, p in model.named_parameters()
                    if ".lora_A." in name or ".lora_B." in name
                }
            )
            outputs = []
            with torch.inference_mode():
                for probe in probes:
                    inputs = torch.tensor([probe["token_ids"]], dtype=torch.long)
                    logits = model(input_ids=inputs, use_cache=False).logits[0]
                    assert torch.isfinite(logits).all()
                    # A repeated actual forward is the determinism control.
                    repeated = model(input_ids=inputs, use_cache=False).logits[0]
                    assert torch.equal(logits, repeated), (profile, probe["name"])
                    path = cache / f"{profile}-{probe['name']}.npy"
                    np.save(path, logits.float().cpu().numpy(), allow_pickle=False)
                    outputs.append(
                        {
                            "probe": probe["name"],
                            "file": path.name,
                            "bytes": path.stat().st_size,
                            "sha256": sha(path),
                            "native_logit_dtype": str(logits.dtype),
                            "shape": list(logits.shape),
                            "repeated_forward_identical": True,
                        }
                    )
            profiles.append({"name": profile, "adapter_dtypes": adapter_dtypes, "outputs": outputs})
            del model
            print(f"Captured and repeated {profile}: {len(probes)} probes", flush=True)
    comparisons = []
    for reference, candidate, meaning in [
        ("wrapped-f32", "merged-f32", "FP32 saved merge versus unmerged FP32 adapter"),
        ("wrapped-bf16", "merged-bf16", "BF16 saved merge versus unmerged BF16 base/FP32 factors"),
        ("base-f32", "wrapped-f32", "adapter's observed effect in the FP32 runtime"),
        ("base-bf16", "wrapped-bf16", "adapter's observed effect in the BF16 runtime"),
        ("wrapped-f32", "wrapped-bf16", "runtime dtype difference; not solely merge rounding"),
    ]:
        measurements = []
        for probe in probes:
            point = compare(
                cache / f"{reference}-{probe['name']}.npy",
                cache / f"{candidate}-{probe['name']}.npy",
            )
            measurements.append({"probe": probe["name"], **point})
        positions = sum(m["positions"] for m in measurements)
        comparisons.append(
            {
                "reference": reference,
                "candidate": candidate,
                "meaning": meaning,
                "positions": positions,
                "changed_top1_positions": sum(m["changed_top1_positions"] for m in measurements),
                "max_abs_logit_difference": max(
                    m["max_abs_logit_difference"] for m in measurements
                ),
                "mean_probability_total_variation": sum(
                    m["mean_probability_total_variation"] * m["positions"] for m in measurements
                )
                / positions,
                "measurements": measurements,
            }
        )
    payload = {
        "scope": "numerical forward probes on authored text; not a task-quality evaluation",
        "community_merge_receipt_sha256": sha(root / "merges.json"),
        "script_sha256": sha(Path(__file__)),
        "probes_source_sha256": sha(Path(__file__).with_name("probes.json")),
        "tokenizer": tokenizer_source,
        "probes": probes,
        "settings": {
            "device": "cpu",
            "threads": 1,
            "attention": "eager",
            "use_cache": False,
            "eval": True,
            "add_special_tokens": False,
            "seed": 71022,
            "python": platform.python_version(),
        },
        "versions": {
            key: importlib.metadata.version(key)
            for key in ["torch", "peft", "transformers", "tokenizers", "numpy", "safetensors"]
        },
        "profiles": profiles,
        "comparisons": comparisons,
        "limits": [
            "Six authored probes are not a sampled benchmark or accuracy evaluation.",
            "Identical observed prefixes at every position; no free-running generation test.",
            "No chat template is inferred from the adapter or used to claim instruction quality.",
            "Greedy next-token changes at near ties do not directly measure semantic impact.",
            "Results depend on the pinned runtime, CPU and explicit attention implementation.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            [{k: v for k, v in c.items() if k != "measurements"} for c in comparisons], indent=2
        )
    )


if __name__ == "__main__":
    main()
