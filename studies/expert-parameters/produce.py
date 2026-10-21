"""Native Qwen3 MoE ParamWrapper saves, normal merges and authored expert faults.

Models, tokens and training are synthetic. Parameter identity is captured from
the actual pre-injection nn.Parameter objects, never inferred from tensor fit.
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
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import ParamWrapper
from safetensors.torch import load_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM


def save(model: Qwen3MoeForCausalLM, path: Path) -> None:
    model.save_pretrained(path, max_shard_size="8KB")


def main(native_notices: list[warnings.WarningMessage]) -> None:
    root = Path(sys.argv[1])
    root.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    profiles = ["both", "reversed-rs", "mixed-zero", "singleton-down", "two-layer-gate"]
    cases = []
    for index, profile in enumerate(profiles):
        torch.manual_seed(80700 + index)
        out = root / profile
        experts = 1 if profile == "singleton-down" else 4 if profile == "two-layer-gate" else 3
        layers = 2 if profile == "two-layer-gate" else 1
        config = Qwen3MoeConfig(
            vocab_size=31,
            hidden_size=12,
            moe_intermediate_size=8,
            num_hidden_layers=layers,
            num_attention_heads=3,
            num_key_value_heads=1,
            head_dim=4,
            num_experts=experts,
            num_experts_per_tok=min(2, experts),
            tie_word_embeddings=False,
            attention_dropout=0,
        )
        original = Qwen3MoeForCausalLM(config).eval()
        # Match one explicit CPU forward implementation at every precision.
        # Automatic grouped_mm rejects some tiny FP16 stride shapes.
        original.set_experts_implementation("eager")
        with torch.no_grad():
            for parameter in original.parameters():
                parameter.copy_(parameter.bfloat16().float())
        save(original, out / "base")
        parameters = (
            ["down_proj"]
            if profile == "singleton-down"
            else ["mlp.experts.gate_up_proj"]
            if profile == "two-layer-gate"
            else ["down_proj", "gate_up_proj"]
            if profile == "reversed-rs"
            else ["gate_up_proj", "down_proj"]
        )
        working = copy.deepcopy(original)
        source_names = {id(parameter): name for name, parameter in working.named_parameters()}
        adapted = get_peft_model(
            working,
            LoraConfig(
                r=2,
                lora_alpha=6,
                target_parameters=parameters,
                target_modules=["q_proj", "embed_tokens"] if profile == "mixed-zero" else [],
                modules_to_save=["lm_head"] if profile == "mixed-zero" else None,
                use_rslora=profile == "reversed-rs",
                rank_pattern=(
                    {"mlp.experts.gate_up_proj": 3}
                    if profile == "reversed-rs"
                    else {"model.layers.1.mlp.experts.gate_up_proj": 3}
                    if profile == "two-layer-gate"
                    else {}
                ),
                alpha_pattern=(
                    {"mlp.experts.down_proj": -2}
                    if profile == "reversed-rs"
                    else {"model.layers.0.mlp.experts.gate_up_proj": -4}
                    if profile == "two-layer-gate"
                    else {}
                ),
            ),
        )
        x = torch.randint(0, 31, (4, 12))
        optimizer = torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad], lr=0.01)
        losses = []
        for _ in range(6):
            optimizer.zero_grad()
            logits = adapted(x).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 31), x[:, 1:].reshape(-1)
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if profile == "mixed-zero":
            # Authored post-training zero-reference control. No claim that the
            # optimizer or router naturally left this expert untouched.
            with torch.no_grad():
                for module in adapted.modules():
                    if isinstance(module, ParamWrapper):
                        module.lora_B["default"].weight[:, experts - 1 :: experts].zero_()
        identities = {}
        for name, module in adapted.named_modules():
            if isinstance(module, ParamWrapper):
                parameter = source_names[id(module.get_param())]
                prefix = name.removeprefix("base_model.model.")
                identities[parameter] = {
                    "factor_module": prefix,
                    "A": name + ".lora_A.weight",
                    "B": name + ".lora_B.weight",
                    "rank": module.r["default"],
                    "scale": module.scaling["default"],
                    "experts": module.num_experts,
                    "runtime_shape": list(module.get_param().shape),
                    "native_swapped_in_out": module._did_swap_in_out_features,
                }
        with warnings.catch_warnings(record=True) as notices:
            adapted.eval().save_pretrained(out / "adapter", save_embedding_layers=False)
        (out / "adapter/README.md").unlink(missing_ok=True)
        factors = load_file(out / "adapter/adapter_model.safetensors")
        for identity in identities.values():
            assert identity["A"] in factors and identity["B"] in factors
        merges = []
        for label, dtype in [
            ("f32", torch.float32),
            ("f16", torch.float16),
            ("bf16", torch.bfloat16),
        ]:
            for safe in [False, True]:
                candidate = PeftModel.from_pretrained(
                    copy.deepcopy(original).to(dtype), out / "adapter", local_files_only=True
                ).eval()
                for name, module in candidate.named_modules():
                    if isinstance(module, ParamWrapper):
                        assert torch.equal(
                            module.lora_A["default"].weight.cpu(), factors[name + ".lora_A.weight"]
                        )
                        assert torch.equal(
                            module.lora_B["default"].weight.cpu(), factors[name + ".lora_B.weight"]
                        )
                with torch.inference_mode():
                    before = candidate(x).logits.float()
                    merged = candidate.merge_and_unload(safe_merge=safe).eval()
                    after = merged(x).logits.float()
                name = f"correct-{label}-safe-{str(safe).lower()}"
                save(merged, out / name)
                merges.append(
                    {
                        "name": name,
                        "purpose": "native merge",
                        "loaded_factors_equal_saved": True,
                        "max_forward_difference": float((before - after).abs().max()),
                        "next_token_argmax_changes": int(
                            (before.argmax(-1) != after.argmax(-1)).sum()
                        ),
                        "positions": before.shape[0] * before.shape[1],
                    }
                )
        correct = (
            PeftModel.from_pretrained(
                copy.deepcopy(original), out / "adapter", local_files_only=True
            )
            .merge_and_unload()
            .eval()
        )
        target = next(
            (name for name in identities if name.endswith("gate_up_proj")), next(iter(identities))
        )
        identity = identities[target]
        rank, scale = identity["rank"], identity["scale"]
        output, input_ = identity["runtime_shape"][1:]
        a, b = factors[identity["A"]], factors[identity["B"]]
        omitted_expert = min(1, experts - 1)
        controls = ["omitted-expert", "unscaled-parameter"] + (
            ["contiguous-B"] if experts > 1 else []
        )
        for name in controls:
            wrong = copy.deepcopy(correct)
            with torch.no_grad():
                value = wrong.state_dict()[target]
                baseline = original.state_dict()[target]
                if name == "omitted-expert":
                    value[omitted_expert].copy_(baseline[omitted_expert])
                elif name == "unscaled-parameter":
                    delta = torch.einsum(
                        "ore,eri->eoi",
                        b.reshape(output, rank, experts),
                        a.reshape(experts, rank, input_),
                    )
                    value.copy_(baseline + delta)
                else:
                    delta = torch.einsum(
                        "eor,eri->eoi",
                        b.reshape(experts, output, rank),
                        a.reshape(experts, rank, input_),
                    )
                    value.copy_(baseline + delta * scale)
            save(wrong, out / name)
            merges.append(
                {
                    "name": name,
                    "purpose": "authored parameter fault; other native weights unchanged",
                    "parameter": target,
                    "expert": omitted_expert if name == "omitted-expert" else None,
                }
            )
        cases.append(
            {
                "profile": profile,
                "training_loss": losses,
                "identity_source": "original pre-injection nn.Parameter object identities",
                "parameters": identities,
                "merges": merges,
                "zero_expert_control": experts - 1 if profile == "mixed-zero" else None,
                "native_save_warnings": [str(w.message) for w in notices],
                "expert_implementation": original.config._experts_implementation,
            }
        )
    artifacts = [
        {
            "file": str(p.relative_to(root)),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]
    manifest = {
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "peft", "transformers", "safetensors", "numpy"]
        },
        "cases": cases,
        "artifacts": artifacts,
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_warnings": sorted({str(w.message) for w in native_notices}),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Produced {len(cases)} expert profiles / "
        f"{sum(len(c['merges']) for c in cases)} saved merges."
    )


if __name__ == "__main__":
    with warnings.catch_warnings(record=True) as native_notices:
        warnings.simplefilter("always")
        main(native_notices)
