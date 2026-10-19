"""Inspect pinned public DoRA bytes and compare separate local native merges."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
import warnings
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM
from verify import load, verify

from mergeglass import AuditError


def main() -> None:
    root, destination = map(Path, sys.argv[1:])
    manifest = json.loads(Path(__file__).with_name("public-sources.json").read_text())
    for source in manifest["sources"]:
        for item in source["files"]:
            data = (root / source["role"] / item["file"]).read_bytes()
            assert len(data) == item["bytes"]
            assert hashlib.sha256(data).hexdigest() == item["sha256"]
    torch.set_num_threads(1)
    cases = []
    try:
        cases.append(verify(root / "base", root / "adapter", root / "published"))
    except AuditError as exc:
        cases.append({"case": "published", "audit_refused": str(exc)})
    merges = []
    for name, dtype, dora in [
        ("local-f32", torch.float32, True),
        ("local-bf16", torch.bfloat16, True),
        ("local-as-lora-f32", torch.float32, False),
    ]:
        base = AutoModelForCausalLM.from_pretrained(
            root / "base", dtype=dtype, local_files_only=True, trust_remote_code=False
        )
        config = LoraConfig.from_pretrained(root / "adapter")
        config.use_dora = dora
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = PeftModel.from_pretrained(
                base,
                root / "adapter",
                config=config,
                local_files_only=True,
                autocast_adapter_dtype=True,
            )
        adapter_dtypes = sorted(
            {str(p.dtype) for name, p in model.named_parameters() if ".lora_" in name}
        )
        merged = model.merge_and_unload(safe_merge=False)
        merged.save_pretrained(root / name, safe_serialization=True, max_shard_size="64MB")
        merges.append(
            {
                "case": name,
                "dtype": str(dtype),
                "adapter_dtypes": adapter_dtypes,
                "use_dora": dora,
                "warnings": [str(w.message) for w in caught],
                "files": [
                    {
                        "file": p.name,
                        "bytes": p.stat().st_size,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    }
                    for p in sorted((root / name).iterdir())
                    if p.is_file()
                ],
            }
        )
        del merged, model, base
        cases.append(verify(root / "base", root / "adapter", root / name))
        print(f"Verified {name}", flush=True)
    published = load(root / "published")
    local = load(root / "local-f32")
    assert published.keys() == local.keys()
    comparison = {
        "checked_tensors": len(published),
        "checked_elements": sum(t.numel() for t in published.values()),
        "identical_dtypes": all(published[k].dtype == local[k].dtype for k in published),
        "numeric_differences": sum(
            int(torch.count_nonzero(published[k] != local[k])) for k in published
        ),
    }
    payload = {
        "sources": manifest,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "peft", "transformers", "safetensors", "numpy"]
        },
        "merge": "local built-in Llama code, CPU, one thread, safe_merge=False",
        "merges": merges,
        "cases": cases,
        "published_vs_local_f32": comparison,
        "provenance_limit": (
            "These are numerical comparisons against a selected pinned base. The adapter's "
            "training revision and its relationship to the separately published model are unknown."
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "layers": sum(len(c.get("report", {}).get("layers", [])) for c in cases),
            }
        )
    )


if __name__ == "__main__":
    main()
