"""Independent dense torch.float64 replay of the native DoRA artifact bytes."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig
from peft.utils.other import get_pattern_key
from safetensors.torch import load_file

from mergeglass import audit


@lru_cache
def rounding_grid(dtype: torch.dtype) -> tuple[torch.Tensor, float]:
    if dtype == torch.bfloat16:
        entries = [struct.unpack("<f", struct.pack("<I", bits << 16))[0] for bits in range(0x7F80)]
        overflow = float.fromhex("0x1.ffp+127")
    else:
        assert dtype == torch.float16
        entries = [struct.unpack("<e", struct.pack("<H", bits))[0] for bits in range(0x7C00)]
        overflow = 65520.0
    return torch.tensor(entries, dtype=torch.float64), overflow


def reference_cast(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Independent direct-F64 rounding by adjacent representable bit patterns.

    In the pinned CPU runtime, torch F64-to-BF16 conversion rounds through F32.
    It cannot serve as the oracle for the auditor's declared direct cast.
    """
    if dtype not in [torch.bfloat16, torch.float16]:
        return values.to(dtype).double()
    grid, overflow = rounding_grid(dtype)
    magnitude = values.abs().flatten().contiguous()
    position = torch.searchsorted(grid, magnitude)
    low = grid[(position - 1).clamp(min=0)]
    high = grid[position.clamp(max=len(grid) - 1)]
    midpoint = (low + high) / 2
    choose_high = (magnitude > midpoint) | ((magnitude == midpoint) & (position % 2 == 0))
    rounded = torch.where(choose_high, high, low)
    rounded[magnitude >= overflow] = float("inf")
    return torch.copysign(rounded.reshape(values.shape), values)


def load(root: Path, *, adapter: bool = False) -> dict[str, torch.Tensor]:
    stem = "adapter_model" if adapter else "model"
    index = root / f"{stem}.safetensors.index.json"
    files = (
        set(json.loads(index.read_text())["weight_map"].values())
        if index.exists()
        else {f"{stem}.safetensors"}
    )
    result: dict[str, torch.Tensor] = {}
    for name in sorted(files):
        shard = load_file(root / name)
        assert not result.keys() & shard.keys()
        result.update(shard)
    return result


def close(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            close(actual[key], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            close(a, b)
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-15), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def verify(base: Path, adapter: Path, merged: Path) -> dict[str, Any]:
    before, factors, after = load(base), load(adapter, adapter=True), load(merged)
    config = LoraConfig.from_pretrained(adapter)
    assert config.use_dora
    report = audit(base, adapter, merged, block_mib=1)
    rows = []
    adapted = set()
    for layer in report["layers"]:
        name = layer["module"]
        key = layer["weight"]
        adapted.add(key)
        prefix = f"base_model.model.{name}."
        a, b = (factors[prefix + f"lora_{side}.weight"].double() for side in ["A", "B"])
        magnitude = factors[prefix + "lora_magnitude_vector"].double()
        rank = config.rank_pattern.get(get_pattern_key(list(config.rank_pattern), name), config.r)
        alpha = config.alpha_pattern.get(
            get_pattern_key(list(config.alpha_pattern), name), config.lora_alpha
        )
        scale = alpha / (math.sqrt(rank) if config.use_rslora else rank)
        # Different implementation and operation order from the block auditor:
        # dense PyTorch product, direct row norm, then magnitude/norm scaling.
        direction = before[key].double() + (b @ a) * scale
        row_norm = torch.linalg.vector_norm(direction, dim=1)
        assert bool((row_norm > 0).all())
        reference = (magnitude / row_norm)[:, None] * direction
        delta = reference - before[key].double()
        baseline = before[key].to(after[key].dtype).double()
        observed = after[key].double() - baseline
        residual = observed - delta
        unchanged = observed == 0
        norms = {
            "delta": float(torch.linalg.vector_norm(delta)),
            "observed": float(torch.linalg.vector_norm(observed)),
            "residual": float(torch.linalg.vector_norm(residual)),
            "base_cast_drift": float(torch.linalg.vector_norm(baseline - before[key].double())),
            "unchanged_reference": float(torch.linalg.vector_norm(delta[unchanged])),
        }
        close(layer["norms"], norms)
        denominator = norms["delta"]
        observed_norm = norms["observed"]
        dot = float(torch.sum(observed * delta))
        close(layer["projection_gain"], dot / denominator**2 if denominator else None)
        close(
            layer["cosine_similarity"],
            dot / denominator / observed_norm if denominator and observed_norm else None,
        )
        close(layer["relative_residual"], norms["residual"] / denominator if denominator else None)
        close(
            layer["observed_to_intended_norm"],
            observed_norm / denominator if denominator else None,
        )
        close(
            layer["unchanged_reference_energy_fraction"],
            (norms["unchanged_reference"] / denominator) ** 2 if denominator else None,
        )
        changed = int(torch.count_nonzero(observed))
        nonzero = int(torch.count_nonzero(delta))
        erased = int(torch.count_nonzero((delta != 0) & unchanged))
        assert layer["changed_elements"] == changed
        assert layer["intended_nonzero_elements"] == nonzero
        assert layer["unchanged_nonzero_update_elements"] == erased
        assert layer["adapter_variant"] == "dora"
        assert layer["rank"] == rank
        close(layer["scale"], scale)
        rounded_reference = reference_cast(reference, after[key].dtype)
        assert layer["reference_merge_mismatches"] == int(
            torch.count_nonzero(after[key].double() != rounded_reference)
        )
        point = layer["largest_residual"]
        row, column = point["row"], point["column"]
        close(point["intended_delta"], float(delta[row, column]))
        close(point["observed_delta"], float(observed[row, column]))
        close(abs(point["residual"]), float(residual.abs().max()))
        rows.append(
            {
                "module": name,
                "norms": norms,
                "changed": changed,
                "nonzero": nonzero,
                "framework_reference_cast_disagreements": int(
                    torch.count_nonzero(
                        rounded_reference != reference.to(after[key].dtype).double()
                    )
                ),
            }
        )
    unrelated = {}
    for key in before.keys() - adapted:
        changed = int(torch.count_nonzero(after[key] != before[key].to(after[key].dtype)))
        if changed:
            unrelated[key] = changed
    assert {
        row["weight"]: row["changed_elements"] for row in report["unadapted_changes"]
    } == unrelated
    assert report["summary"]["unadapted_changed_tensors"] == len(unrelated)
    assert report["summary"]["unadapted_changed_elements"] == sum(unrelated.values())
    full_energy = sum(row["norms"]["delta"] ** 2 for row in rows)
    unchanged_energy = sum(row["norms"]["unchanged_reference"] ** 2 for row in rows)
    close(
        report["summary"]["unchanged_reference_energy_fraction"],
        unchanged_energy / full_energy if full_energy else None,
    )
    larger = audit(base, adapter, merged, block_mib=16)
    for key in ["inputs", "layers", "summary", "unadapted_changes"]:
        close(report[key], larger[key])
    return {"case": merged.name, "dense_oracle": rows, "report": report}


def main() -> None:
    root, destination = map(Path, sys.argv[1:])
    torch.set_num_threads(1)
    receipt = json.loads((root / "producer.json").read_text())
    for item in receipt["files"]:
        data = (root / item["file"]).read_bytes()
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
    cases = [verify(root / "base", root / "adapter", root / name) for name in receipt["cases"]]
    cases.append(
        verify(root / "transformer-base", root / "transformer-adapter", root / "transformer-merged")
    )
    payload = {
        "oracle": "dense torch.float64; direct row norms; native PEFT rank/scale resolution",
        "reference_cast": "independent adjacent-bit-pattern grid; direct F64 ties to even",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tolerance": {"relative": 1e-8, "absolute": 1e-15},
        "counts_and_reference_casts": "exact",
        "producer": receipt,
        "cases": cases,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Verified {len(cases)} native DoRA artifacts against the independent dense oracle")


if __name__ == "__main__":
    main()
