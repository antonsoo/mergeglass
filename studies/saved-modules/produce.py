"""Native trained heads, full-module buffers and resized/tied token tables.

Synthetic training and explicitly authored faulty conversions; no pretrained
weights, training-data claims, task benchmark or measured incident frequency.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    LlamaConfig,
    LlamaForCausalLM,
)


class Head(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = torch.nn.Linear(8, 5)
        self.out = torch.nn.Linear(5, 3)
        self.register_buffer("seen", torch.tensor(2**60, dtype=torch.int64))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.out(self.dense(value).tanh())


class Classifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(8, 8, bias=False)
        self.classifier = Head()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.projection(value))


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


def save(model: torch.nn.Module, path: Path) -> None:
    path.mkdir()
    save_file(model.state_dict(), path / "model.safetensors")


def main() -> None:
    root = Path(sys.argv[1])
    root.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(21706)
    torch.use_deterministic_algorithms(True)
    base = Classifier()
    save(base, root / "base")
    model = get_peft_model(
        copy.deepcopy(base),
        LoraConfig(
            r=2, lora_alpha=4, target_modules=["projection"], modules_to_save=["classifier"]
        ),
    )
    values = torch.randn(64, 8)
    labels = (values @ torch.randn(8, 3)).argmax(-1)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.025)
    losses = []
    for _ in range(64):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(values), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.base_model.model.classifier.modules_to_save.default.seen.add_(3)
    model.eval().save_pretrained(root / "adapter")
    (root / "adapter/README.md").unlink(missing_ok=True)
    probes = torch.randn(96, 8)
    cases = []
    forwards = []
    for label, dtype in [("f32", torch.float32), ("f16", torch.float16), ("bf16", torch.bfloat16)]:
        candidate = PeftModel.from_pretrained(
            copy.deepcopy(base).to(dtype), root / "adapter", local_files_only=True
        ).eval()
        with torch.inference_mode():
            unmerged = candidate(probes.to(dtype)).float()
            correct = candidate.merge_and_unload(safe_merge=False).eval()
            merged = correct(probes.to(dtype)).float()
        name = f"correct-{label}"
        save(correct, root / name)
        cases.append(
            {"base": "base", "adapter": "adapter", "merged": name, "purpose": "native merge"}
        )
        forwards.append(
            {
                "case": name,
                "max_logit_difference": float((merged - unmerged).abs().max()),
                "argmax_changes": int((merged.argmax(-1) != unmerged.argmax(-1)).count_nonzero()),
                "probes": len(probes),
            }
        )
        if label == "f32":
            omitted = copy.deepcopy(correct)
            omitted.classifier.load_state_dict(base.classifier.state_dict())
            save(omitted, root / "omit-head-f32")
            cases.append(
                {
                    "base": "base",
                    "adapter": "adapter",
                    "merged": "omit-head-f32",
                    "purpose": "native LoRA retained; complete original classifier restored",
                }
            )
            with torch.inference_mode():
                control = omitted(probes)
            forwards.append(
                {
                    "case": "omit-head-f32",
                    "correct_vs_control_max_logit_difference": float(
                        (merged - control).abs().max()
                    ),
                    "correct_vs_control_argmax_changes": int(
                        (merged.argmax(-1) != control.argmax(-1)).count_nonzero()
                    ),
                    "probes": len(probes),
                }
            )
            counter = copy.deepcopy(correct)
            counter.classifier.seen.sub_(1)
            save(counter, root / "wrong-counter-f32")
            cases.append(
                {
                    "base": "base",
                    "adapter": "adapter",
                    "merged": "wrong-counter-f32",
                    "purpose": (
                        "all float weights correct; integer buffer differs by one above 2**60"
                    ),
                }
            )
    # Sequence classification adds classifier/score selectors automatically;
    # only classifier exists in this actual built-in model architecture.
    bert = BertForSequenceClassification(
        BertConfig(
            vocab_size=37,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=32,
            num_labels=3,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
    )
    bert.save_pretrained(root / "bert-base", max_shard_size="4KB")
    adapted = get_peft_model(
        bert,
        LoraConfig(
            r=2, lora_alpha=4, target_modules=["query", "value"], task_type=TaskType.SEQ_CLS
        ),
    )
    batch = torch.randint(1, 37, (6, 12))
    optimizer = torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad], lr=0.01)
    for _ in range(8):
        optimizer.zero_grad()
        loss = adapted(input_ids=batch, labels=torch.arange(6) % 3).loss
        loss.backward()
        optimizer.step()
    adapted.save_pretrained(root / "bert-adapter")
    (root / "bert-adapter/README.md").unlink(missing_ok=True)
    adapted.merge_and_unload().save_pretrained(root / "bert-merged", max_shard_size="4KB")
    cases.append(
        {
            "base": "bert-base",
            "adapter": "bert-adapter",
            "merged": "bert-merged",
            "purpose": "native sequence classifier; automatically added classifier/score selectors",
        }
    )
    for vocabulary in (37, 41):
        prefix = f"llama-{vocabulary}"
        llama = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=37,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=32,
                tie_word_embeddings=True,
            )
        )
        llama.save_pretrained(root / f"{prefix}-base", max_shard_size="4KB")
        if vocabulary != 37:
            llama.resize_token_embeddings(vocabulary, mean_resizing=False)
        adapted = get_peft_model(
            llama,
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["q_proj"],
                modules_to_save=["embed_tokens", "lm_head"],
            ),
        )
        batch = torch.randint(1, vocabulary, (4, 12))
        optimizer = torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad], lr=0.01)
        for _ in range(6):
            optimizer.zero_grad()
            loss = adapted(input_ids=batch, labels=batch).loss
            loss.backward()
            optimizer.step()
        adapted.save_pretrained(root / f"{prefix}-adapter", save_embedding_layers=False)
        (root / f"{prefix}-adapter/README.md").unlink(missing_ok=True)
        adapted.merge_and_unload().save_pretrained(root / f"{prefix}-merged", max_shard_size="4KB")
        cases.append(
            {
                "base": f"{prefix}-base",
                "adapter": f"{prefix}-adapter",
                "merged": f"{prefix}-merged",
                "purpose": "native tied base omits head; "
                + ("token table resized 37 -> 41" if vocabulary == 41 else "original 37 tokens"),
            }
        )
    # Full-module replacement also composes with Linear DoRA's different
    # directional reference; neither contract may swallow the other's state.
    adapted = get_peft_model(
        copy.deepcopy(base),
        LoraConfig(
            r=2,
            lora_alpha=4,
            use_dora=True,
            target_modules=["projection"],
            modules_to_save=["classifier"],
        ),
    )
    optimizer = torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad], lr=0.02)
    for _ in range(24):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(adapted(values), labels)
        loss.backward()
        optimizer.step()
    adapted.base_model.model.classifier.modules_to_save.default.seen.add_(5)
    adapted.eval().save_pretrained(root / "dora-adapter")
    (root / "dora-adapter/README.md").unlink(missing_ok=True)
    for label, dtype in [("f32", torch.float32), ("f16", torch.float16), ("bf16", torch.bfloat16)]:
        candidate = PeftModel.from_pretrained(
            copy.deepcopy(base).to(dtype), root / "dora-adapter", local_files_only=True
        )
        name = f"dora-correct-{label}"
        save(candidate.merge_and_unload(), root / name)
        cases.append(
            {
                "base": "base",
                "adapter": "dora-adapter",
                "merged": name,
                "purpose": "native Linear DoRA with a complete trained classifier",
            }
        )
    # Malformed evidence controls are derived and clearly separate from native saves.
    bad = root / "incomplete-adapter"
    bad.mkdir()
    (bad / "adapter_config.json").write_bytes((root / "adapter/adapter_config.json").read_bytes())
    data = load_file(root / "adapter/adapter_model.safetensors")
    del data["base_model.model.classifier.dense.bias"]
    save_file(data, bad / "adapter_model.safetensors")
    cases.append(
        {
            "base": "base",
            "adapter": "incomplete-adapter",
            "merged": "correct-f32",
            "purpose": "derived malformed evidence: one trained full-module bias deleted",
            "expect_refusal": True,
        }
    )
    receipt = {
        "scope": (
            "synthetic training, native saves and explicitly authored faulty conversions; "
            "no quality benchmark"
        ),
        "seed": 21706,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "peft", "transformers", "numpy", "safetensors")
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "classifier_training_losses": [losses[0], losses[-1]],
        "forward_observations": forwards,
        "cases": cases,
        "files": fingerprints(root),
    }
    (root / "producer.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
