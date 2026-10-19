"""Independent dense PyTorch measurements of native artifact bytes.

The auditor uses NumPy blocks. This oracle uses complete tensors, PEFT's own
configuration loader/pattern resolver and torch float64 matrix multiplication.
Producer merge arithmetic remains separate from both references.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig
from peft.utils.other import get_pattern_key
from safetensors.torch import load_file

from mergeglass import audit


def load(root: Path, adapter: bool = False) -> dict[str, torch.Tensor]:
    stem = "adapter_model" if adapter else "model"
    index = root / f"{stem}.safetensors.index.json"
    names = (
        set(json.loads(index.read_text())["weight_map"].values())
        if index.exists()
        else {f"{stem}.safetensors"}
    )
    result: dict[str, torch.Tensor] = {}
    for name in sorted(names):
        weights = load_file(root / name)
        assert not result.keys() & weights.keys()
        result.update(weights)
    return result


def close(actual: float | None, expected: float | None) -> None:
    if expected is None:
        assert actual is None
    else:
        assert actual is not None
        assert math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-15), (actual, expected)


def equivalent(actual: Any, expected: Any) -> None:
    if isinstance(expected, float):
        close(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            equivalent(actual[key], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            equivalent(a, b)
    else:
        assert actual == expected, (actual, expected)


def verify(root: Path, name: str) -> dict[str, Any]:
    report = audit(root / "base", root / "adapter", root / name, block_mib=1)
    original = load(root / "base")
    adapter = load(root / "adapter", adapter=True)
    merged = load(root / name)
    config = LoraConfig.from_pretrained(root / "adapter")
    results = []
    adapted = set()
    for layer in report["layers"]:
        module = layer["module"]
        weight = layer["weight"]
        adapted.add(weight)
        rank = config.rank_pattern.get(get_pattern_key(list(config.rank_pattern), module), config.r)
        alpha = config.alpha_pattern.get(
            get_pattern_key(list(config.alpha_pattern), module), config.lora_alpha
        )
        scale = alpha / (math.sqrt(rank) if config.use_rslora else rank)
        a = adapter[f"base_model.model.{module}.lora_A.weight"].double()
        b = adapter[f"base_model.model.{module}.lora_B.weight"].double()
        delta = (b @ a) * scale
        baseline = original[weight].to(merged[weight].dtype).double()
        observed = merged[weight].double() - baseline
        residual = observed - delta
        denominator = float(torch.linalg.vector_norm(delta))
        norm_values = {
            "delta": denominator,
            "observed": float(torch.linalg.vector_norm(observed)),
            "residual": float(torch.linalg.vector_norm(residual)),
            "base_cast_drift": float(
                torch.linalg.vector_norm(baseline - original[weight].double())
            ),
            "unchanged_reference": float(torch.linalg.vector_norm(delta[observed == 0])),
        }
        for key, expected in norm_values.items():
            close(layer["norms"][key], expected)
        dot = float(torch.sum(observed * delta))
        close(layer["projection_gain"], dot / denominator**2 if denominator else None)
        observed_norm = norm_values["observed"]
        close(
            layer["cosine_similarity"],
            dot / denominator / observed_norm if denominator and observed_norm else None,
        )
        close(
            layer["relative_residual"],
            norm_values["residual"] / denominator if denominator else None,
        )
        close(
            layer["unchanged_reference_energy_fraction"],
            (norm_values["unchanged_reference"] / denominator) ** 2 if denominator else None,
        )
        assert layer["rank"] == rank
        close(layer["scale"], scale)
        nonzero = int(torch.count_nonzero(delta))
        erased = int(torch.count_nonzero((delta != 0) & (observed == 0)))
        changed = int(torch.count_nonzero(observed))
        assert layer["changed_elements"] == changed
        assert layer["intended_nonzero_elements"] == nonzero
        assert layer["unchanged_nonzero_update_elements"] == erased
        # Compare concrete largest-residual evidence, not just aggregates.
        point = layer["largest_residual"]
        row, col = point["row"], point["column"]
        close(point["intended_delta"], float(delta[row, col]))
        close(point["observed_delta"], float(observed[row, col]))
        close(abs(point["residual"]), float(torch.max(torch.abs(residual))))
        results.append(
            {
                "module": module,
                "changed_elements": changed,
                "nonzero_update_elements": nonzero,
                "unchanged_nonzero_updates": erased,
                "norms": norm_values,
                "scale": scale,
                "rank": rank,
                "projection_gain": dot / denominator**2 if denominator else None,
                "cosine_similarity": (
                    dot / denominator / observed_norm if denominator and observed_norm else None
                ),
            }
        )
    unadapted = []
    for key in original.keys() - adapted:
        changed = int(torch.count_nonzero(merged[key] != original[key].to(merged[key].dtype)))
        if changed:
            unadapted.append({"weight": key, "changed": changed})
    assert len(unadapted) == report["summary"]["unadapted_changed_tensors"]
    assert (
        sum(row["changed"] for row in unadapted) == report["summary"]["unadapted_changed_elements"]
    )
    reference_energy = sum(layer["norms"]["delta"] ** 2 for layer in results)
    unchanged_energy = sum(layer["norms"]["unchanged_reference"] ** 2 for layer in results)
    close(
        report["summary"]["unchanged_reference_energy_fraction"],
        unchanged_energy / reference_energy if reference_energy else None,
    )
    # Blocks are a resource choice, not an arithmetic or evidence choice.
    large_blocks = audit(root / "base", root / "adapter", root / name, block_mib=16)
    for key in ("layers", "unadapted_changes", "summary", "inputs"):
        equivalent(report[key], large_blocks[key])
    return {
        "name": name,
        "summary": report["summary"],
        "layers": results,
        "unadapted": sorted(unadapted, key=lambda x: x["weight"]),
        "report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("teaching", type=Path)
    parser.add_argument("transformer", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = []
    for root, names in [
        (args.teaching, ["merged-f32", "merged-f16", "merged-bf16", "omitted", "unrelated-edit"]),
        (args.transformer, ["merged-f32", "merged-f16", "merged-bf16"]),
    ]:
        # Establish that the files under test are the producer's actual bytes.
        producer = json.loads((root / "producer.json").read_text())
        for row in producer["files"]:
            data = (root / row["file"]).read_bytes()
            assert len(data) == row["bytes"]
            assert hashlib.sha256(data).hexdigest() == row["sha256"]
        cases = [verify(root, name) for name in names]
        results.append(
            {
                "kind": "teaching" if root == args.teaching else "transformer",
                "producer": producer,
                "cases": cases,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "versions": {
            key: importlib.metadata.version(key)
            for key in ["torch", "peft", "transformers", "numpy", "safetensors"]
        },
        "oracle": "dense torch.float64; native PEFT scale resolution; saved tensor comparison",
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Verified {sum(len(row['cases']) for row in results)} native artifact/control cases")


if __name__ == "__main__":
    main()
