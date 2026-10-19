"""Measure native probability drift at exactly matched function-call prefixes.

This is a conditional numerical experiment, not a generation accuracy score.
The selected target is checked by BFCL; no function call is executed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import peft.tuners.lora.dora as dora_runtime
import torch
from bfcl import BFCL, canonical, digest
from peft import PeftModel
from replay import file_sha, select_records, state_sha, validate_inputs
from transformers import AutoModelForCausalLM, AutoTokenizer
from verify_grader import literal

HERE = Path(__file__).resolve().parent


def selected_call(benchmark: BFCL, row: dict[str, Any]) -> str:
    answer = benchmark.answers[row["id"]][0]
    assert len(answer) == 1
    name, options = next(iter(answer.items()))
    schema = row["function"][0]["parameters"]
    values = {}
    for param, alternatives in options.items():
        if "" in alternatives and param not in schema["required"]:
            continue
        value = next(x for x in alternatives if x != "")
        values[param] = literal(value, schema["properties"][param])
    text = "[" + name + "(" + ", ".join(f"{k}={v!r}" for k, v in values.items()) + ")]"
    assert benchmark.grade(row, text)["score"] == 1, row["id"]
    return text


def drift(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    p, q = reference.exp(), candidate.exp()
    kl = (p * (reference - candidate)).sum(dim=-1)
    # Summation can produce a tiny negative zero-level KL; retain the raw
    # minimum as well as the nonnegative numerical diagnostic.
    assert float(kl.min()) >= -1e-12
    total_variation = 0.5 * (p - q).abs().sum(dim=-1)
    return {
        "mean_kl_nats": float(kl.clamp_min(0).mean()),
        "max_kl_nats": float(kl.max().clamp_min(0)),
        "raw_min_kl_nats": float(kl.min()),
        "mean_total_variation": float(total_variation.mean()),
        "max_total_variation": float(total_variation.max()),
        "argmax_disagreements": int((reference.argmax(-1) != candidate.argmax(-1)).sum()),
        "positions": reference.shape[0],
    }


def check_probability_math() -> dict[str, Any]:
    p = torch.tensor([[0.0, 0.0]], dtype=torch.float64).log_softmax(-1)
    q = torch.tensor([[0.0, math.log(3)]], dtype=torch.float64).log_softmax(-1)
    observed = drift(p, q)
    expected = 0.5 * math.log(4 / 3)
    assert abs(observed["mean_kl_nats"] - expected) < 1e-15
    assert abs(observed["mean_total_variation"] - 0.25) < 1e-15
    assert abs(float(-q[0, 1]) - math.log(4 / 3)) < 1e-15
    assert drift(p, p)["mean_kl_nats"] == 0
    return {
        "two_class_kl_nats": observed["mean_kl_nats"],
        "closed_form_kl_nats": expected,
        "total_variation": 0.25,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path)
    parser.add_argument("bfcl", type=Path)
    parser.add_argument("prompt_receipt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--limit", type=int, help="diagnostic pilot only, not the complete protocol"
    )
    parser.add_argument("--chunk-start", type=int, default=0)
    parser.add_argument("--chunk-stop", type=int)
    args = parser.parse_args()
    frozen = {
        name: file_sha(HERE / name)
        for name in (
            "teacher-forced-protocol.json",
            "teacher_forced.py",
            "protocol.json",
            "replay.py",
            "bfcl.py",
            "verify_grader.py",
        )
    }
    protocol = json.loads((HERE / "protocol.json").read_text())
    experiment = json.loads((HERE / "teacher-forced-protocol.json").read_text())
    probability_control = check_probability_math()
    sources = validate_inputs(args.models, protocol)
    prompt_hash = file_sha(args.prompt_receipt)
    expected_prompts = {row["id"]: row for row in json.loads(args.prompt_receipt.read_text())}
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / "questions.jsonl").exists(), "Use a fresh output directory"
    benchmark = BFCL(args.bfcl, args.output / "bfcl-work", protocol)
    tokenizer = AutoTokenizer.from_pretrained(
        args.models / "adapter", local_files_only=True, trust_remote_code=False
    )
    torch.set_num_threads(protocol["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(71123)
    dora_runtime.ENABLE_DORA_CACHING = protocol["dora_weight_caching"]
    records, excluded = [], []
    for row in benchmark.rows:
        if row["id"] in benchmark.incomplete_references:
            excluded.append(
                {
                    "id": row["id"],
                    "status": "incomplete_reference",
                    "parameters": benchmark.incomplete_references[row["id"]],
                }
            )
            continue
        call = selected_call(benchmark, row)
        target = tokenizer.encode(call, add_special_tokens=False) + [tokenizer.eos_token_id]
        prefix = tokenizer.apply_chat_template(
            benchmark.messages(row), tokenize=True, add_generation_prompt=True, return_dict=False
        )
        identity = {
            "id": row["id"],
            "question_hash": digest(canonical([row["question"], row["function"]])),
            "prompt_hash": digest(canonical(prefix)),
            "input_ids": prefix,
        }
        assert identity == expected_prompts[row["id"]]
        records.append(
            {
                **identity,
                "target_ids": target,
                "selected_call": call,
                "category": row["category"],
                "function_name": row["function"][0]["name"],
                "target_hash": digest(canonical(target)),
            }
        )
    assert len(records) + len(excluded) == len(benchmark.rows)
    assert len(benchmark.rows) == len(expected_prompts)
    complete_records = len(records)
    try:
        records, selection = select_records(records, args.chunk_start, args.chunk_stop, args.limit)
    except ValueError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "prepared": len(records),
                "excluded": excluded,
                "complete_protocol_records": complete_records,
                "selection": selection,
            }
        ),
        flush=True,
    )
    profiles = experiment["profiles"]
    models, identities = {}, {}
    for profile in profiles:
        dtype = torch.bfloat16 if profile.endswith("bf16") else torch.float32
        folder = {
            "merged-f32": "local-f32",
            "merged-bf16": "local-bf16",
            "as-lora-f32": "local-as-lora-f32",
        }.get(profile, "base")
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
            max(len(r["input_ids"]) + len(r["target_ids"]) for r in records)
            <= model.config.max_position_embeddings
        )
        identities[profile] = {
            "before_state_sha256": state_sha(model),
            "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
            "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
        }
        models[profile] = model
    summaries = {
        profile: {"question_nll": [], "token_nll_sum": 0.0, "tokens": 0} for profile in profiles
    }
    began = time.perf_counter()
    with (args.output / "questions.jsonl").open("w") as stream, torch.inference_mode():
        for index, record in enumerate(records):
            prefix, target = record["input_ids"], record["target_ids"]
            inputs = torch.tensor([prefix + target], dtype=torch.long)
            positions = torch.arange(len(prefix) - 1, inputs.shape[1] - 1)
            target_ids = torch.tensor(target, dtype=torch.long)
            probabilities, measurements = {}, {}
            for profile, model in models.items():
                start = time.perf_counter()
                logits = model(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    logits_to_keep=positions,
                    use_cache=False,
                ).logits[0]
                assert logits.shape[0] == len(target)
                assert torch.isfinite(logits).all(), (profile, record["id"])
                logp = logits.to(torch.float64).log_softmax(-1)
                nll = -logp.gather(1, target_ids[:, None])[:, 0]
                assert torch.isfinite(nll).all()
                probabilities[profile] = logp
                predicted = logp.argmax(-1)
                measurements[profile] = {
                    "mean_target_nll_nats": float(nll.mean()),
                    "target_nll_sum_nats": float(nll.sum()),
                    "target_tokens": len(target),
                    "teacher_forced_target_argmax_matches": int((predicted == target_ids).sum()),
                    "first_token_id": int(predicted[0]),
                    "first_token_text": tokenizer.decode([int(predicted[0])]),
                    "forward_seconds": time.perf_counter() - start,
                }
                summaries[profile]["question_nll"].append(float(nll.mean()))
                summaries[profile]["token_nll_sum"] += float(nll.sum())
                summaries[profile]["tokens"] += len(target)
                del logits
            comparisons = {
                reference: {
                    profile: drift(probabilities[reference], probabilities[profile])
                    for profile in profiles
                    if profile != reference
                }
                for reference in ("unmerged-f32", "unmerged-bf16")
            }
            result = {key: value for key, value in record.items() if key != "input_ids"}
            result.update(
                {"input_tokens": len(prefix), "profiles": measurements, "drift": comparisons}
            )
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            del probabilities
            if (index + 1) % 10 == 0 or index + 1 == len(records):
                print(
                    json.dumps(
                        {
                            "finished": index + 1,
                            "total": len(records),
                            "seconds": round(time.perf_counter() - began, 2),
                        }
                    ),
                    flush=True,
                )
    for profile, model in models.items():
        identities[profile]["after_state_sha256"] = state_sha(model)
        assert (
            identities[profile]["after_state_sha256"] == identities[profile]["before_state_sha256"]
        )
    for name, checksum in frozen.items():
        assert file_sha(HERE / name) == checksum, f"Study code changed: {name}"
    assert file_sha(args.prompt_receipt) == prompt_hash
    result = {
        "protocol": experiment,
        "source_protocol": protocol,
        "source_hashes": frozen,
        "prompt_receipt_sha256": prompt_hash,
        "input_sources": sources,
        "questions": len(records),
        "complete_protocol_records": complete_records,
        "is_limited_pilot": args.limit is not None,
        "selection": selection,
        "excluded": excluded,
        "weight_identities": identities,
        "probability_control": probability_control,
        "question_output_sha256": file_sha(args.output / "questions.jsonl"),
        "elapsed_seconds": time.perf_counter() - began,
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "tokenizers", "numpy", "safetensors")
        },
        "upstream_sources": benchmark.loaded_sources(),
        "data_files": benchmark.data_files,
        "summaries": {
            profile: {
                "mean_question_nll_nats": math.fsum(data["question_nll"]) / len(records),
                "token_weighted_nll_nats": data["token_nll_sum"] / data["tokens"],
                "tokens": data["tokens"],
            }
            for profile, data in summaries.items()
        },
    }
    (args.output / "teacher-forced.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"profiles_completed": len(profiles), "questions": len(records)}), flush=True)


if __name__ == "__main__":
    main()
