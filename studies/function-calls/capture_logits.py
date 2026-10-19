"""Export fixed native vocabulary logits for an independent probability oracle.

First and last questions in simple_python, first/middle/EOS prediction positions.
These are numerical controls, not a sampled estimate of inference quality.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
from pathlib import Path

import numpy as np
import peft.tuners.lora.dora as dora_runtime
import torch
from bfcl import BFCL, canonical, digest
from peft import PeftModel
from replay import state_sha, validate_inputs
from run_chunks import HERE, read_rows, sha, validate_chunk
from teacher_forced import drift, selected_call
from transformers import AutoModelForCausalLM, AutoTokenizer

IDS = ("simple_python_0", "simple_python_399")


def equivalent(observed: dict, expected: dict) -> None:
    for key, value in observed.items():
        if isinstance(value, float):
            assert math.isclose(value, expected[key], rel_tol=1e-10, abs_tol=2e-13), key
        else:
            assert value == expected[key], key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "bfcl", "teacher", "prompts", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    prompts_list = json.loads(args.prompts.read_text())
    protocol = json.loads((HERE / "protocol.json").read_text())
    contract = json.loads((args.teacher / "contract.json").read_text())
    assert contract["mode"] == "teacher" and contract["profiles"] == protocol["profiles"]
    assert contract["available_records"] == contract["target_records"] == 656
    assert contract["prompts_sha256"] == sha(args.prompts)
    assert sha(args.teacher / "grader.json") == contract["grader_sha256"]
    assert all(sha(HERE / name) == value for name, value in contract["source_hashes"].items())
    assert contract["versions"] == {
        name: importlib.metadata.version(name) for name in contract["versions"]
    }
    grader = json.loads((args.teacher / "grader.json").read_text())
    eligible = {row["id"] for row in grader["results"] if row["score"] == 1}
    ordered_ids = [row["id"] for row in prompts_list if row["id"] in eligible]
    prompt_map = {row["id"]: row for row in prompts_list}
    progress = json.loads((args.teacher / "progress.json").read_text())
    recorded, reference_chunks, weight_states = {}, [], None
    for chunk in progress["chunks"]:
        start, stop = chunk["start"], chunk["stop"]
        if not set(IDS).intersection(ordered_ids[start:stop]):
            continue
        folder = args.teacher / chunk["directory"]
        item = validate_chunk(
            folder,
            "teacher",
            "all-profiles",
            start,
            stop,
            ordered_ids[start:stop],
            prompt_map,
            contract,
        )
        assert {"directory": chunk["directory"], **item} == chunk
        assert item == json.loads((folder / "completed.json").read_text())
        assert weight_states is None or weight_states == item["weight_states"]
        weight_states = item["weight_states"]
        reference_chunks.append(chunk)
        recorded.update({row["id"]: row for row in read_rows(folder / item["records_file"])})
    assert set(IDS) <= recorded.keys() and weight_states is not None
    validate_inputs(args.models, protocol)
    args.output.mkdir(parents=True)
    benchmark = BFCL(args.bfcl, args.output / "bfcl-work", protocol)
    questions = {row["id"]: row for row in benchmark.rows}
    tokenizer = AutoTokenizer.from_pretrained(
        args.models / "adapter", local_files_only=True, trust_remote_code=False
    )
    torch.set_num_threads(protocol["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(71123)
    dora_runtime.ENABLE_DORA_CACHING = protocol["dora_weight_caching"]
    models = {}
    for profile in protocol["profiles"]:
        folder = {
            "merged-f32": "local-f32",
            "merged-bf16": "local-bf16",
            "as-lora-f32": "local-as-lora-f32",
        }.get(profile, "base")
        model = AutoModelForCausalLM.from_pretrained(
            args.models / folder,
            dtype=torch.bfloat16 if profile.endswith("bf16") else torch.float32,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation=protocol["attention"],
        )
        if profile.startswith("unmerged"):
            model = PeftModel.from_pretrained(
                model, args.models / "adapter", local_files_only=True, autocast_adapter_dtype=True
            )
        model.eval()
        assert {str(p.device) for p in model.parameters()} == {"cpu"}
        assert state_sha(model) == weight_states[profile]
        models[profile] = model
    arrays, samples = {}, []
    with torch.inference_mode():
        for number, question_id in enumerate(IDS):
            row, original = questions[question_id], recorded[question_id]
            prefix = tokenizer.apply_chat_template(
                benchmark.messages(row),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            target = tokenizer.encode(selected_call(benchmark, row), add_special_tokens=False)
            target.append(tokenizer.eos_token_id)
            assert (
                digest(canonical([row["question"], row["function"]])) == original["question_hash"]
            )
            assert digest(canonical(prefix)) == original["prompt_hash"]
            assert digest(canonical(target)) == original["target_hash"]
            assert prefix == prompt_map[question_id]["input_ids"]
            assert target == original["target_ids"]
            indices = sorted({0, len(target) // 2, len(target) - 1})
            inputs = torch.tensor([prefix + target], dtype=torch.long)
            positions = torch.arange(len(prefix) - 1, inputs.shape[1] - 1)
            targets = torch.tensor(target, dtype=torch.long)
            probabilities, selected = {}, {}
            sample = {
                "id": question_id,
                "question_hash": original["question_hash"],
                "prompt_hash": original["prompt_hash"],
                "target_hash": original["target_hash"],
                "prediction_indices": indices,
                "target_ids": [target[i] for i in indices],
                "profiles": {},
            }
            for profile, model in models.items():
                logits = model(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    logits_to_keep=positions,
                    use_cache=False,
                ).logits[0]
                assert logits.shape[0] == len(target) and torch.isfinite(logits).all()
                logp = logits.to(torch.float64).log_softmax(-1)
                nll = -logp.gather(1, targets[:, None])[:, 0]
                equivalent(
                    {
                        "mean_target_nll_nats": float(nll.mean()),
                        "target_nll_sum_nats": float(nll.sum()),
                        "target_tokens": len(target),
                        "teacher_forced_target_argmax_matches": int(
                            (logp.argmax(-1) == targets).sum()
                        ),
                        "first_token_id": int(logp[0].argmax()),
                    },
                    original["profiles"][profile],
                )
                probabilities[profile] = logp
                selected[profile] = logp[indices]
                key = f"q{number}_{profile}"
                arrays[key] = logits[indices].to(torch.float32).cpu().numpy().copy()
                sample["profiles"][profile] = {
                    "array": key,
                    "native_logits_dtype": str(logits.dtype),
                    "target_nll_nats": nll[indices].tolist(),
                    "argmax_ids": logp[indices].argmax(-1).tolist(),
                }
            sample["drift"] = {}
            for reference in ("unmerged-f32", "unmerged-bf16"):
                sample["drift"][reference] = {}
                for profile in models:
                    if profile == reference:
                        continue
                    equivalent(
                        drift(probabilities[reference], probabilities[profile]),
                        original["drift"][reference][profile],
                    )
                    sample["drift"][reference][profile] = drift(
                        selected[reference], selected[profile]
                    )
            samples.append(sample)
            print(json.dumps({"captured_question": question_id, "positions": indices}), flush=True)
    for profile, model in models.items():
        assert state_sha(model) == weight_states[profile]
    assert all(sha(HERE / name) == value for name, value in contract["source_hashes"].items())
    destination = args.output / "logits.npz"
    np.savez_compressed(destination, **arrays)
    receipt = {
        "scope": "Fixed numerical controls, not inference-quality estimates. Native BF16 values "
        "are exactly representable in the exported float32 arrays.",
        "script_sha256": sha(Path(__file__)),
        "python": platform.python_version(),
        "versions": contract["versions"],
        "source_hashes": contract["source_hashes"],
        "reference_chunks": reference_chunks,
        "teacher_contract_sha256": sha(args.teacher / "contract.json"),
        "prompt_receipt_sha256": sha(args.prompts),
        "weight_states": weight_states,
        "all_full_question_measurements_match_canonical_rows": True,
        "matching_tolerance": {"relative": 1e-10, "absolute": 2e-13},
        "logits_sha256": sha(destination),
        "samples": samples,
    }
    (args.output / "capture.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
