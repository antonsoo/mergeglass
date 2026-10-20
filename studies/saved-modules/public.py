"""Native public PEFT merges and explicit lost-full-module controls, CPU only.

Download the pinned inputs in sources.json first. No remote Python is loaded.
The original adapter and base remain unchanged; outputs are not redistributed.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import sys
import warnings
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


def files(root: Path) -> list[dict[str, object]]:
    return [
        {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
        }
        for path in sorted(root.iterdir())
        if path.is_file()
    ]


def main() -> None:
    root = Path(sys.argv[1])
    sources = json.loads(Path(__file__).with_name("sources.json").read_text())
    for source in sources["sources"]:
        for item in source["files"]:
            path = root / source["role"] / item["file"]
            assert path.stat().st_size == item["bytes"]
            with path.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
    torch.set_num_threads(1)
    # Deterministic token IDs, not decoded language examples or a task evaluation.
    ids = torch.tensor([[1, 28, 409, 1217, 37, 901], [55, 72, 91, 110, 129, 148]])
    cases = []
    for label, dtype in [("f32", torch.float32), ("f16", torch.float16), ("bf16", torch.bfloat16)]:
        base = AutoModelForCausalLM.from_pretrained(
            root / "base", dtype=dtype, local_files_only=True, trust_remote_code=False
        )
        original_embedding = base.get_input_embeddings().weight.detach().clone()
        original_head = base.get_output_embeddings().weight.detach().clone()
        with warnings.catch_warnings(record=True) as caught:
            model = PeftModel.from_pretrained(
                base, root / "adapter", local_files_only=True, autocast_adapter_dtype=True
            ).eval()
        adapter_dtypes = sorted(
            {str(p.dtype) for name, p in model.named_parameters() if ".lora_A." in name}
        )
        with torch.inference_mode():
            unmerged = model(input_ids=ids).logits.float()
            merged = model.merge_and_unload(safe_merge=False).eval()
            correct = merged(input_ids=ids).logits.float()
        name = f"correct-{label}"
        merged.save_pretrained(root / name, max_shard_size="32MB")
        forward = {
            "unmerged_vs_merged_max_logit_difference": float((unmerged - correct).abs().max()),
            "unmerged_vs_merged_argmax_changes": int(
                (unmerged.argmax(-1) != correct.argmax(-1)).count_nonzero()
            ),
        }
        cases.append(
            {
                "name": name,
                "purpose": "unchanged public adapter; native merge_and_unload(safe_merge=False)",
                "requested_base_dtype": str(dtype),
                "loaded_factor_dtypes": adapter_dtypes,
                "full_weight_dtypes": {
                    "embedding": str(merged.get_input_embeddings().weight.dtype),
                    "head": str(merged.get_output_embeddings().weight.dtype),
                },
                "warnings": [str(item.message) for item in caught],
                "forward": forward,
                "files": files(root / name),
            }
        )
        if label == "f32":
            # Keep every already-merged LoRA weight. Only restore both original
            # full modules, simulating a conversion that discarded their saves.
            control = copy.deepcopy(merged)
            with torch.no_grad():
                control.get_input_embeddings().weight.copy_(original_embedding)
                control.get_output_embeddings().weight.copy_(original_head)
            with torch.inference_mode():
                omitted = control(input_ids=ids).logits.float()
            control_name = "omit-full-modules-f32"
            control.save_pretrained(root / control_name, max_shard_size="32MB")
            cases.append(
                {
                    "name": control_name,
                    "purpose": (
                        "authored wrong-conversion control, not an observed incident; "
                        "all native LoRA weights retained, embedding and head restored to base"
                    ),
                    "forward": {
                        "correct_vs_control_max_logit_difference": float(
                            (correct - omitted).abs().max()
                        ),
                        "correct_vs_control_argmax_changes": int(
                            (correct.argmax(-1) != omitted.argmax(-1)).count_nonzero()
                        ),
                        "token_positions": ids.numel(),
                    },
                    "files": files(root / control_name),
                }
            )
            del control, omitted
        del merged, model, base, unmerged, correct, original_embedding, original_head
    receipt = {
        "scope": "saved-weight and deterministic token-ID observations; no quality assessment",
        "sources": sources,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "peft", "transformers", "safetensors", "numpy")
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "token_ids": ids.tolist(),
        "cases": cases,
    }
    (root / "public-producer.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
