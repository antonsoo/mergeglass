"""Matched greedy tool-call generations on pinned public BFCL records.

No generated function is executed. The auditor runtime does not import this
optional producer/evaluation environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import peft.tuners.lora.dora as dora_runtime
import torch
from bfcl import BFCL, canonical, digest
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def file_sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def state_sha(model: torch.nn.Module) -> str:
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value.update(canonical([name, str(tensor.dtype), list(tensor.shape)]) + b"\n")
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1).numpy()
        for start in range(0, len(raw), 1024 * 1024):
            value.update(raw[start : start + 1024 * 1024].tobytes())
    return value.hexdigest()


def validate_inputs(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(
        Path(__file__).parents[1].joinpath("dora/public-sources.json").read_text()
    )
    public = json.loads(Path(__file__).parents[1].joinpath("dora/results/public.json").read_text())
    for source in manifest["sources"]:
        for item in source["files"]:
            path = root / source["role"] / item["file"]
            assert path.stat().st_size == item["bytes"] and file_sha(path) == item["sha256"]
    for merge in public["merges"]:
        for item in merge["files"]:
            path = root / merge["case"] / item["file"]
            assert path.stat().st_size == item["bytes"] and file_sha(path) == item["sha256"]
    for item in protocol["tokenizer_files"]:
        path = root / "adapter" / item["file"]
        assert path.stat().st_size == item["bytes"] and file_sha(path) == item["sha256"]
    return {"public_sources": manifest, "native_merges": public["merges"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", type=Path)
    parser.add_argument("bfcl", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache-control", action="store_true")
    args = parser.parse_args()
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    script_hash = file_sha(Path(__file__))
    helper_path = Path(__file__).with_name("bfcl.py")
    helper_hash = file_sha(helper_path)
    sources = validate_inputs(args.models, protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    benchmark = BFCL(args.bfcl, args.output / "bfcl-work", protocol)
    tokenizer = AutoTokenizer.from_pretrained(
        args.models / "adapter", local_files_only=True, trust_remote_code=False
    )
    torch.set_num_threads(protocol["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(71123)
    dora_runtime.ENABLE_DORA_CACHING = protocol["dora_weight_caching"]
    records = []
    for row in benchmark.rows[: args.limit]:
        messages = benchmark.messages(row)
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        assert isinstance(ids, list) and all(type(value) is int for value in ids)
        records.append(
            {
                "source": row,
                "messages": messages,
                "input_ids": ids,
                "question_hash": digest(canonical([row["question"], row["function"]])),
                "prompt_hash": digest(canonical(ids)),
            }
        )
    prompt_receipt = [
        {
            "id": r["source"]["id"],
            "question_hash": r["question_hash"],
            "prompt_hash": r["prompt_hash"],
            "input_ids": r["input_ids"],
        }
        for r in records
    ]
    (args.output / "prompts.json").write_text(json.dumps(prompt_receipt) + "\n")
    print(
        json.dumps(
            {
                "prepared_records": len(records),
                "max_prompt_tokens": max(len(r["input_ids"]) for r in records),
                "incomplete_references": benchmark.incomplete_references,
            }
        ),
        flush=True,
    )
    profiles = args.profile or protocol["profiles"]
    assert set(profiles) <= set(protocol["profiles"])
    summaries = []
    generation = GenerationConfig(
        **protocol["generation"],
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    for profile in profiles:
        assert not (args.output / f"{profile}.jsonl").exists(), "Use a fresh output directory"
        dtype = torch.bfloat16 if profile.endswith("bf16") else torch.float32
        folder = {
            "merged-f32": "local-f32",
            "merged-bf16": "local-bf16",
            "as-lora-f32": "local-as-lora-f32",
        }.get(profile, "base")
        started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            args.models / folder,
            dtype=dtype,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation=protocol["attention"],
        )
        if profile.startswith("unmerged"):
            model = PeftModel.from_pretrained(
                model, args.models / "adapter", local_files_only=True, autocast_adapter_dtype=True
            )
        model.eval()
        assert model.config._attn_implementation == protocol["attention"]
        assert (
            max(len(r["input_ids"]) for r in records) + generation.max_new_tokens
            <= model.config.max_position_embeddings
        )
        identity = state_sha(model)
        adapter_dtypes = sorted(
            {str(p.dtype) for name, p in model.named_parameters() if ".lora_" in name}
        )
        counts: dict[str, int] = {}
        correct, generated, decode_seconds = 0, 0, 0.0
        control = None
        if args.cache_control and profile.startswith("unmerged"):
            inputs = torch.tensor([records[0]["input_ids"]], dtype=torch.long)
            with torch.inference_mode():
                dora_runtime.ENABLE_DORA_CACHING = False
                uncached = model.generate(
                    inputs,
                    attention_mask=torch.ones_like(inputs),
                    generation_config=generation,
                    max_new_tokens=16,
                )
                dora_runtime.ENABLE_DORA_CACHING = True
                cached = model.generate(
                    inputs,
                    attention_mask=torch.ones_like(inputs),
                    generation_config=generation,
                    max_new_tokens=16,
                )
            assert torch.equal(uncached, cached), "Caching changed the control generation"
            control = {
                "max_new_tokens": 16,
                "generated_tokens": uncached.shape[1] - inputs.shape[1],
                "cached_and_uncached_ids_identical": True,
            }
        with (args.output / f"{profile}.jsonl").open("w") as stream, torch.inference_mode():
            for index, record in enumerate(records):
                inputs = torch.tensor([record["input_ids"]], dtype=torch.long)
                began = time.perf_counter()
                output = model.generate(
                    inputs, attention_mask=torch.ones_like(inputs), generation_config=generation
                )
                seconds = time.perf_counter() - began
                new_ids = output[0, inputs.shape[1] :].tolist()
                text = tokenizer.decode(new_ids, skip_special_tokens=True)
                grade = benchmark.grade(record["source"], text)
                counts[grade["status"]] = counts.get(grade["status"], 0) + 1
                correct += grade["score"] == 1
                generated += len(new_ids)
                decode_seconds += seconds
                row = {
                    "id": record["source"]["id"],
                    "category": record["source"]["category"],
                    "profile": profile,
                    "question_hash": record["question_hash"],
                    "prompt_hash": record["prompt_hash"],
                    "input_tokens": inputs.shape[1],
                    "generated_ids": new_ids,
                    "output": text,
                    "grade": grade,
                    "stop": "eos"
                    if new_ids and new_ids[-1] == tokenizer.eos_token_id
                    else "token_limit",
                    "generation_seconds": seconds,
                    "function_name": record["source"]["function"][0]["name"],
                }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if (index + 1) % 10 == 0 or index + 1 == len(records):
                    print(
                        json.dumps(
                            {
                                "profile": profile,
                                "finished": index + 1,
                                "total": len(records),
                                "correct": correct,
                                "seconds": round(time.perf_counter() - started, 2),
                            }
                        ),
                        flush=True,
                    )
        after_identity = state_sha(model)
        assert identity == after_identity, "Generation modified named model weights"
        summary = {
            "profile": profile,
            "model_state_sha256": identity,
            "post_generation_state_sha256": after_identity,
            "adapter_dtypes": adapter_dtypes,
            "records": len(records),
            "correct": correct,
            "status_counts": counts,
            "generated_tokens": generated,
            "generation_seconds": decode_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "cache_control": control,
            "output_sha256": file_sha(args.output / f"{profile}.jsonl"),
        }
        summaries.append(summary)
        (args.output / f"{profile}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        del model
    assert script_hash == file_sha(Path(__file__)), "Study code changed during generation"
    assert helper_hash == file_sha(helper_path), "Grader wrapper changed during generation"
    assert digest(protocol_bytes) == file_sha(protocol_path), "Protocol changed during generation"
    receipt = {
        "protocol": protocol,
        "protocol_sha256": digest(protocol_bytes),
        "script_sha256": script_hash,
        "grader_wrapper_sha256": helper_hash,
        "incomplete_references": benchmark.incomplete_references,
        "registry_entry": {
            "model_name": "mergeglass-smollm2",
            "underscore_to_dot": False,
            "provider_clients_imported": False,
        },
        "sources": sources,
        "benchmark_files": benchmark.data_files,
        "benchmark_modules": benchmark.loaded_sources(),
        "generation_config": generation.to_dict(),
        "profiles": summaries,
        "machine": {"platform": platform.platform(), "logical_cpus": os.cpu_count()},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "peft", "transformers", "tokenizers", "numpy", "safetensors"]
        },
        "prompt_tokens": {
            "min": min(len(r["input_ids"]) for r in records),
            "max": max(len(r["input_ids"]) for r in records),
        },
        "prompts_sha256": file_sha(args.output / "prompts.json"),
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({"profiles_completed": len(summaries), "records_per_profile": len(records)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
