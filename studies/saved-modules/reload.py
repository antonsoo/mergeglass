"""Compare an actual saved public checkpoint with a fresh native PEFT merge."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "case", choices=("correct-f32", "correct-f16", "correct-bf16", "omit-full-modules-f32")
    )
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    root = args.root
    producer_path = root / "public-producer.json"
    producer = json.loads(producer_path.read_text())
    case = next(item for item in producer["cases"] if item["name"] == args.case)
    inventories = [
        (root / source["role"], source["files"]) for source in producer["sources"]["sources"]
    ]
    inventories.append((root / args.case, case["files"]))
    for directory, items in inventories:
        for item in items:
            path = directory / item["file"]
            assert path.stat().st_size == item["bytes"]
            with path.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
    torch.set_num_threads(1)
    dtype = (
        torch.bfloat16
        if args.case == "correct-bf16"
        else torch.float16
        if args.case == "correct-f16"
        else torch.float32
    )
    base = AutoModelForCausalLM.from_pretrained(
        root / "base", dtype=dtype, local_files_only=True, trust_remote_code=False
    )
    old_embedding = base.get_input_embeddings().weight.detach().clone()
    old_head = base.get_output_embeddings().weight.detach().clone()
    candidate = PeftModel.from_pretrained(
        base, root / "adapter", local_files_only=True, autocast_adapter_dtype=True
    ).eval()
    native = candidate.merge_and_unload(safe_merge=False).eval()
    if args.case == "omit-full-modules-f32":
        with torch.no_grad():
            native.get_input_embeddings().weight.copy_(old_embedding)
            native.get_output_embeddings().weight.copy_(old_head)
    saved = AutoModelForCausalLM.from_pretrained(
        root / args.case, dtype=dtype, local_files_only=True, trust_remote_code=False
    ).eval()
    native_state, saved_state = native.state_dict(), saved.state_dict()
    assert native_state.keys() == saved_state.keys()
    for key in native_state:
        assert native_state[key].dtype == saved_state[key].dtype
        assert torch.equal(native_state[key], saved_state[key]), key
    assert native.config._attn_implementation == saved.config._attn_implementation
    assert native.config.tie_word_embeddings is False and saved.config.tie_word_embeddings is False
    assert (
        native.get_input_embeddings().weight.data_ptr()
        != native.get_output_embeddings().weight.data_ptr()
    )
    assert (
        saved.get_input_embeddings().weight.data_ptr()
        != saved.get_output_embeddings().weight.data_ptr()
    )
    ids = torch.tensor(producer["token_ids"])
    with torch.inference_mode():
        expected = native(input_ids=ids).logits.float()
        actual = saved(input_ids=ids).logits.float()
    assert torch.equal(actual, expected)
    receipt = {
        "case": args.case,
        "method": (
            "Fresh native PEFT merge compared with AutoModelForCausalLM.from_pretrained "
            "of the actual saved checkpoint, local files and built-in model code only"
        ),
        "producer_manifest_sha256": hashlib.sha256(producer_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "peft", "transformers", "numpy", "safetensors")
        },
        "matching_state_tensors": len(native_state),
        "matching_state_elements": sum(value.numel() for value in native_state.values()),
        "dtype": str(dtype),
        "attention_implementation": native.config._attn_implementation,
        "input_and_output_weights_tied": False,
        "native_vs_saved_max_logit_difference": float((expected - actual).abs().max()),
        "native_vs_saved_argmax_changes": int(
            (expected.argmax(-1) != actual.argmax(-1)).count_nonzero()
        ),
        "logit_elements": actual.numel(),
        "matching_float32_logit_sha256": hashlib.sha256(
            actual.contiguous().numpy().tobytes()
        ).hexdigest(),
    }
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        args.case,
        receipt["matching_state_tensors"],
        "matching tensors; identical saved/native logits",
    )


if __name__ == "__main__":
    main()
