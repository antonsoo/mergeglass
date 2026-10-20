"""Native standard GPT layouts, mixed heads, precision and faulty conversions.

All models and training tokens here are synthetic. No pretrained weights are
needed. PEFT's per-module flags are captured *before* merge_and_unload.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from safetensors.torch import load_file, save_file
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Model,
    ImageGPTConfig,
    ImageGPTForCausalImageModeling,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
)


def fingerprints(root: Path) -> list[dict[str, object]]:
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
    root = Path(sys.argv[1])
    root.mkdir(parents=True)
    torch.manual_seed(20319)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    profiles = [
        ("gpt2-mixed", GPT2Config, GPT2LMHeadModel, False, False),
        ("openai-gpt-mixed", OpenAIGPTConfig, OpenAIGPTLMHeadModel, False, False),
        ("imagegpt-mixed", ImageGPTConfig, ImageGPTForCausalImageModeling, False, False),
        ("gpt2-cross", GPT2Config, GPT2LMHeadModel, True, False),
        ("gpt2-bare", GPT2Config, GPT2Model, False, True),
        ("gpt2-saved-head", GPT2Config, GPT2LMHeadModel, False, False),
    ]
    cases = []
    for name, config_type, model_type, cross, bare in profiles:
        saved_head = name == "gpt2-saved-head"
        out = root / name
        config = config_type(
            n_embd=16,
            n_head=2,
            n_layer=1,
            n_inner=40,
            n_positions=32,
            vocab_size=43,
            tie_word_embeddings=False,
            add_cross_attention=cross,
            bos_token_id=1,
            eos_token_id=2,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
        )
        original = model_type(config).eval()
        with torch.no_grad():
            for parameter in original.parameters():
                parameter.copy_(parameter.bfloat16().float())
        original.save_pretrained(out / "base", max_shard_size="12KB")
        prefix = "" if bare else "transformer."
        targets = [
            prefix + "h.0.attn.c_attn",
            prefix + "h.0.attn.c_proj",
            prefix + "h.0.mlp.c_fc",
            prefix + "h.0.mlp.c_proj",
        ]
        if cross:
            targets += [prefix + "h.0.crossattention." + s for s in ("c_attn", "q_attn", "c_proj")]
        if not bare:
            targets += (["lm_head"] if not saved_head else []) + [prefix + "wte"]
        adapted = get_peft_model(
            copy.deepcopy(original),
            LoraConfig(
                r=4,
                lora_alpha=8,
                use_rslora=True,
                target_modules=targets,
                modules_to_save=["lm_head"] if saved_head else None,
                rank_pattern={prefix + "h.0.mlp.c_fc": 2},
                alpha_pattern={prefix + "h.0.mlp.c_proj": -2},
            ),
        )
        # ImageGPT reserves its final vocabulary entry for the mask token and
        # predicts only the other entries. Use each native output dimension.
        classes = 42 if config.model_type == "imagegpt" else 43
        tokens = torch.randint(1, classes, (4, 12))
        forward_args = {"input_ids": tokens}
        if cross:
            forward_args["encoder_hidden_states"] = torch.randn(4, 5, 16)
        optimizer = torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad], lr=0.01)
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            output = adapted(**forward_args)
            result = output.last_hidden_state if bare else output.logits
            loss = (
                result.square().mean()
                if bare
                else torch.nn.functional.cross_entropy(
                    result[:, :-1].reshape(-1, classes), tokens[:, 1:].reshape(-1)
                )
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        adapted.eval().save_pretrained(out / "adapter", save_embedding_layers=False)
        (out / "adapter/README.md").unlink(missing_ok=True)
        orientations = {}
        for path, layer in adapted.named_modules():
            if isinstance(layer, LoraLayer):
                orientations[path.removeprefix("base_model.model.")] = {
                    "kind": type(layer.get_base_layer()).__name__,
                    "transpose": getattr(layer, "fan_in_fan_out", False),
                    "scale": layer.scaling["default"],
                    "rank": layer.r["default"],
                }
        merges = []
        for label, dtype in [
            ("f32", torch.float32),
            ("f16", torch.float16),
            ("bf16", torch.bfloat16),
        ]:
            for safe in (False, True):
                candidate = PeftModel.from_pretrained(
                    copy.deepcopy(original).to(dtype), out / "adapter", local_files_only=True
                ).eval()
                args = {
                    key: value.to(dtype) if value.is_floating_point() else value
                    for key, value in forward_args.items()
                }
                with torch.inference_mode():
                    output = candidate(**args)
                    unmerged = (output.last_hidden_state if bare else output.logits).float()
                    merged = candidate.merge_and_unload(safe_merge=safe).eval()
                    output = merged(**args)
                    after = (output.last_hidden_state if bare else output.logits).float()
                filename = f"correct-{label}-safe-{str(safe).lower()}"
                merged.save_pretrained(out / filename, max_shard_size="12KB")
                merges.append(
                    {
                        "name": filename,
                        "purpose": "native merge",
                        "max_forward_difference": float((after - unmerged).abs().max()),
                        "argmax_changes": int((after.argmax(-1) != unmerged.argmax(-1)).sum()),
                        "prediction_positions": after.shape[0] * after.shape[1],
                    }
                )
        # These two intentionally wrong conversions retain correct rectangular
        # and Linear updates, isolating a square Conv1D orientation/omission.
        correct = model_type.from_pretrained(out / "correct-f32-safe-false", local_files_only=True)
        square = prefix + "h.0.attn.c_proj.weight"
        baseline = original.state_dict()[square]
        delta = correct.state_dict()[square] - baseline
        factors = load_file(out / "adapter/adapter_model.safetensors")
        module = square.removesuffix(".weight")
        scale = orientations[module]["scale"]
        product = (
            factors[f"base_model.model.{module}.lora_B.weight"]
            @ factors[f"base_model.model.{module}.lora_A.weight"]
        )
        with torch.no_grad():
            wrong = copy.deepcopy(correct)
            wrong.state_dict()[square].copy_(baseline + product * scale)
            wrong.save_pretrained(out / "wrong-square-orientation", max_shard_size="12KB")
            omitted = copy.deepcopy(correct)
            omitted.state_dict()[square].copy_(baseline)
            omitted.save_pretrained(out / "omitted-square", max_shard_size="12KB")
        for control in ["wrong-square-orientation", "omitted-square"]:
            merges.append({"name": control, "purpose": "authored faulty conversion"})
        cases.append(
            {
                "name": name,
                "losses": losses,
                "native_layouts": orientations,
                "saved_fan_in_fan_out": json.loads(
                    (out / "adapter/adapter_config.json").read_text()
                )["fan_in_fan_out"],
                "square_control_weight": square,
                "native_square_changed": int(delta.count_nonzero()),
                "merges": merges,
            }
        )
    # A minimal runnable example uses real native output, with the complete
    # base/adapter state and all saved tensor names retained.
    small = root / "gpt2-mixed"
    sample = root / "example"
    for stem in [
        "base",
        "adapter",
        "correct-f32-safe-false",
        "wrong-square-orientation",
        "omitted-square",
    ]:
        source = small / stem
        destination = sample / stem
        destination.mkdir(parents=True)
        if stem == "adapter":
            for file in source.iterdir():
                if file.name != "README.md":
                    (destination / file.name).write_bytes(file.read_bytes())
        else:
            model = GPT2LMHeadModel.from_pretrained(source, local_files_only=True)
            save_file(model.state_dict(), destination / "model.safetensors")
            model.config.to_json_file(destination / "config.json")
    receipt = {
        "synthetic": True,
        "training": (
            "5 AdamW steps on seeded random token IDs; no pretrained model or task benchmark"
        ),
        "seed": 20319,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {
            key: importlib.metadata.version(key)
            for key in ["torch", "peft", "transformers", "numpy", "safetensors"]
        },
        "cases": cases,
        "files": fingerprints(root),
    }
    (root / "producer.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Produced {sum(len(c['merges']) for c in cases)} native/control merges.")


if __name__ == "__main__":
    main()
