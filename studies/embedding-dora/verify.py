"""Independent dense Torch and 80-digit scalar reference for native DoRA files."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import struct
import subprocess
from decimal import Decimal, localcontext
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig
from peft.utils.other import get_pattern_key
from safetensors.torch import load_file


def load(root: Path, *, adapter: bool = False) -> dict[str, torch.Tensor]:
    stem = "adapter_model" if adapter else "model"
    index = root / f"{stem}.safetensors.index.json"
    files = (
        sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if index.exists()
        else [f"{stem}.safetensors"]
    )
    result = {}
    for name in files:
        tensors = load_file(root / name)
        assert not result.keys() & tensors.keys()
        result.update(tensors)
    return result


@lru_cache
def grid(dtype: torch.dtype) -> tuple[torch.Tensor, float]:
    # Independent enumeration of adjacent representable values. The pinned
    # native runtime's F64 -> BF16 cast can double-round through F32.
    if dtype == torch.bfloat16:
        values = [struct.unpack("<f", struct.pack("<I", n << 16))[0] for n in range(0x7F80)]
        overflow = float.fromhex("0x1.ffp+127")
    else:
        assert dtype == torch.float16
        values = [struct.unpack("<e", struct.pack("<H", n))[0] for n in range(0x7C00)]
        overflow = 65520.0
    return torch.tensor(values, dtype=torch.float64), overflow


def rounded(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype not in (torch.bfloat16, torch.float16):
        return values.to(dtype).double()
    entries, overflow = grid(dtype)
    magnitude = values.abs().flatten().contiguous()
    position = torch.searchsorted(entries, magnitude)
    low = entries[(position - 1).clamp(min=0)]
    high = entries[position.clamp(max=len(entries) - 1)]
    midpoint = (low + high) / 2
    upper = (magnitude > midpoint) | ((magnitude == midpoint) & (position % 2 == 0))
    result = torch.where(upper, high, low)
    result[magnitude >= overflow] = float("inf")
    return torch.copysign(result.reshape(values.shape), values)


def close(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), (actual.keys(), expected.keys())
        for key, value in expected.items():
            close(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            close(a, b)
    elif expected is None:
        assert actual is None
    elif isinstance(expected, float):
        assert actual is not None and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-14), (
            actual,
            expected,
        )
    else:
        assert actual == expected, (actual, expected)


def scalar_reference(
    original: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    magnitude: torch.Tensor,
    scale: float,
    embedding: bool,
) -> torch.Tensor:
    """Every dot product, norm and final coordinate without a matrix operation."""
    rows, columns = original.shape
    with localcontext() as context:
        context.prec = 80
        values = []
        factor = Decimal.from_float(scale)
        for row in range(rows):
            values.append([])
            for column in range(columns):
                left, right = (column, row) if embedding else (row, column)
                product = sum(
                    (
                        Decimal.from_float(float(b[left, k]))
                        * Decimal.from_float(float(a[k, right]))
                        for k in range(a.shape[0])
                    ),
                    Decimal(0),
                )
                values[-1].append(
                    Decimal.from_float(float(original[row, column])) + product * factor
                )
        lengths = (
            [
                sum((values[row][column] ** 2 for row in range(rows)), Decimal(0)).sqrt()
                for column in range(columns)
            ]
            if embedding
            else [sum((value**2 for value in line), Decimal(0)).sqrt() for line in values]
        )
        result = [
            [
                float(
                    values[row][column]
                    * Decimal.from_float(float(magnitude[column if embedding else row]))
                    / lengths[column if embedding else row]
                )
                for column in range(columns)
            ]
            for row in range(rows)
        ]
    return torch.tensor(result, dtype=torch.float64)


def measure(
    args: argparse.Namespace, base: Path, adapter: Path, merged: Path, *, partial: bool = False
) -> tuple[dict[str, Any] | None, str | None]:
    if args.auditor_python:
        command = [
            args.auditor_python,
            "-I",
            "-c",
            "from mergeglass.cli import main; raise SystemExit(main())",
        ]
        command += ["--dora-norms"] if partial else [str(base)]
        command += [str(adapter), str(merged), "--block-mib", "1", "--json", "-"]
        if not partial:
            command += ["--max-adapter-mib", "1"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if result.returncode == 2:
            return None, result.stderr.strip()
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout), None
    from mergeglass import AuditError, audit, audit_dora_norms

    try:
        return (
            audit_dora_norms(adapter, merged, block_mib=1)
            if partial
            else audit(base, adapter, merged, block_mib=1, max_adapter_mib=1)
        ), None
    except AuditError as exc:
        return None, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--auditor-python")
    parser.add_argument("--before", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    producer = json.loads((args.directory / "producer.json").read_text())
    for item in producer["files"]:
        raw = (args.directory / item["path"]).read_bytes()
        assert len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"]
    results = []
    scalar_checked: dict[tuple[str, str, str], int] = {}
    for case in producer["cases"]:
        base, adapter, merged = (
            args.directory / case[role] for role in ["base", "adapter", "merged"]
        )
        report, error = measure(args, base, adapter, merged)
        if args.before:
            assert report is None and error and "Embedding DoRA" in error, (case, error)
            results.append({"case": case, "refusal": error})
            continue
        assert report is not None, (case, error)
        before, factors, saved = load(base), load(adapter, adapter=True), load(merged)
        config = LoraConfig.from_pretrained(adapter)
        verified = []
        for layer in report["layers"]:
            name, weight = layer["module"], layer["weight"]
            prefix = "base_model.model." + name
            embedding = prefix + ".lora_embedding_A" in factors
            a, b = (
                factors[prefix + key].double()
                for key in (
                    (".lora_embedding_A", ".lora_embedding_B")
                    if embedding
                    else (".lora_A.weight", ".lora_B.weight")
                )
            )
            m = factors[prefix + ".lora_magnitude_vector"].double()
            rank = config.rank_pattern.get(
                get_pattern_key(list(config.rank_pattern), name), config.r
            )
            alpha = config.alpha_pattern.get(
                get_pattern_key(list(config.alpha_pattern), name), config.lora_alpha
            )
            scale = alpha / (math.sqrt(rank) if config.use_rslora else rank)
            original, after = before[weight].double(), saved[weight].double()
            direction = original + ((b @ a).T if embedding else b @ a) * scale
            axis = 0 if embedding else 1
            norms = torch.linalg.vector_norm(direction, dim=axis, keepdim=True)
            reference = direction / norms * (m[None, :] if embedding else m[:, None])
            key = (case["base"], case["adapter"], name)
            if key not in scalar_checked:
                decimal = scalar_reference(original, a, b, m, scale, embedding)
                torch.testing.assert_close(reference, decimal, rtol=2e-13, atol=2e-14)
                scalar_checked[key] = reference.numel()
            delta = reference - original
            baseline = rounded(original, saved[weight].dtype)
            observed = after - baseline
            residual = observed - delta
            unchanged, nonzero = observed == 0, delta != 0
            observed_norm = float(torch.linalg.vector_norm(observed))
            intended_norm = float(torch.linalg.vector_norm(delta))
            expected_norms = {
                "delta": intended_norm,
                "observed": observed_norm,
                "residual": float(torch.linalg.vector_norm(residual)),
                "base_cast_drift": float(torch.linalg.vector_norm(baseline - original)),
                "unchanged_reference": float(torch.linalg.vector_norm(delta[unchanged])),
            }
            close(layer["norms"], expected_norms)
            dot = float(torch.sum(delta * observed))
            close(layer["projection_gain"], dot / intended_norm**2 if intended_norm else None)
            close(
                layer["cosine_similarity"],
                dot / intended_norm / observed_norm if intended_norm and observed_norm else None,
            )
            close(
                layer["relative_residual"],
                expected_norms["residual"] / intended_norm if intended_norm else None,
            )
            close(
                layer["observed_to_intended_norm"],
                observed_norm / intended_norm if intended_norm else None,
            )
            close(
                layer["unchanged_reference_energy_fraction"],
                (expected_norms["unchanged_reference"] / intended_norm) ** 2
                if intended_norm
                else None,
            )
            assert layer["rank"] == rank and layer["alpha"] == alpha and layer["scale"] == scale
            assert layer["changed_elements"] == int(torch.count_nonzero(observed))
            assert layer["intended_nonzero_elements"] == int(torch.count_nonzero(nonzero))
            assert layer["unchanged_nonzero_update_elements"] == int(
                torch.count_nonzero(nonzero & unchanged)
            )
            assert layer["reference_merge_mismatches"] == int(
                torch.count_nonzero(after != rounded(reference, saved[weight].dtype))
            )
            point = layer["largest_residual"]
            row, column = point["row"], point["column"]
            assert point["base"] == float(original[row, column]) and point["merged"] == float(
                after[row, column]
            )
            close(point["intended_delta"], float(delta[row, column]))
            close(point["observed_delta"], float(observed[row, column]))
            close(abs(point["residual"]), float(residual.abs().max()))
            vector = layer["dora_columns" if embedding else "dora_rows"]
            label = "column" if embedding else "row"
            saved_norm = torch.linalg.vector_norm(after, dim=axis)
            reference_norm = torch.linalg.vector_norm(reference, dim=axis)
            comparable = (m != 0) & (saved_norm != 0)
            factors_norm = saved_norm[comparable] / m[comparable].abs()
            logs = torch.log2(factors_norm)
            close(
                vector[f"{label}_norm_factor_range"],
                [float(factors_norm.min()), float(factors_norm.max())]
                if len(factors_norm)
                else None,
            )
            close(
                vector[f"log2_{label}_norm_factor_range"],
                [float(logs.min()), float(logs.max())] if len(logs) else None,
            )
            close(
                vector[f"mean_absolute_log2_{label}_norm_factor"],
                float(logs.abs().mean()) if len(logs) else None,
            )
            for field, expected in [
                (f"{label}s", m.numel()),
                (f"zero_magnitude_{label}s", int((m == 0).sum())),
                (f"negative_magnitude_{label}s", int((m < 0).sum())),
                (f"zero_saved_{label}s", int((saved_norm == 0).sum())),
                (
                    f"nonzero_magnitude_zero_saved_{label}s",
                    int(((m != 0) & (saved_norm == 0)).sum()),
                ),
                (
                    f"zero_magnitude_nonzero_saved_{label}s",
                    int(((m == 0) & (saved_norm != 0)).sum()),
                ),
                (f"norm_comparable_{label}s", int(comparable.sum())),
            ]:
                assert vector[field] == expected, (field, vector[field], expected)
            directions = (saved_norm != 0) & (reference_norm != 0)
            cosines = (
                (after * reference).sum(dim=axis)[directions]
                / saved_norm[directions]
                / reference_norm[directions]
            )
            close(
                vector["direction_cosine_range"],
                [float(cosines.min()), float(cosines.max())] if len(cosines) else None,
            )
            close(vector["mean_direction_cosine"], float(cosines.mean()) if len(cosines) else None)
            assert vector[f"direction_comparable_{label}s"] == len(cosines)
            if embedding:
                assert layer["dora_normalization_axis"] == 0
                assert layer["coordinate_axes"] == {
                    "row": "token_index",
                    "column": "embedding_dimension",
                }
            verified.append(
                {
                    "module": name,
                    "embedding": embedding,
                    "changed": layer["changed_elements"],
                    "intended_nonzero": layer["intended_nonzero_elements"],
                    "unchanged_nonzero": layer["unchanged_nonzero_update_elements"],
                    "gain": layer["projection_gain"],
                    "relative_residual": layer["relative_residual"],
                    "vectors": vector,
                    "witness": point,
                }
            )
        mapped = {layer["weight"] for layer in report["layers"]}
        unrelated = {
            name: int(
                (saved[name].double() != rounded(before[name].double(), saved[name].dtype)).sum()
            )
            for name in before.keys() - mapped
        }
        assert report["summary"]["unadapted_changed_elements"] == sum(unrelated.values())
        partial, partial_error = measure(args, base, adapter, merged, partial=True)
        if partial is None:
            assert partial_error and "snapshots require" in partial_error
        else:
            for layer in report["layers"]:
                p = next(item for item in partial["layers"] if item["module"] == layer["module"])
                key = "dora_columns" if "dora_columns" in layer else "dora_rows"
                expected = {
                    name: value
                    for name, value in layer[key].items()
                    if "direction" not in name and "zero_numerical_reference" not in name
                }
                assert p[key] == expected
        results.append(
            {
                "case": case,
                "summary": report["summary"],
                "layers": verified,
                "partial_refusal": partial_error,
            }
        )
    if args.auditor_python:
        command = [
            args.auditor_python,
            "-I",
            "-c",
            "import mergeglass, pathlib, hashlib, json; "
            "root=pathlib.Path(mergeglass.__file__).parent; "
            "print(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() "
            "for p in sorted(root.glob('*.py'))}))",
        ]
        runtime = json.loads(subprocess.check_output(command, text=True))
    else:
        import mergeglass

        runtime = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(mergeglass.__file__).parent.glob("*.py"))
        }
    result = {
        "scope": "native saved weights only; no model or task quality inference",
        "oracle": "dense Torch float64, 80-digit scalar products and vector roots; "
        "adjacent-bit-pattern direct rounding",
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "auditor_runtime_sha256": runtime,
        "tolerance": {"relative": 1e-8, "absolute": 1e-14, "counts": "exact"},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "peft", "numpy", "safetensors"]
        },
        "auditor_mode": "installed isolated interpreter"
        if args.auditor_python
        else "in-process source",
        "producer_sha256": hashlib.sha256(
            (args.directory / "producer.json").read_bytes()
        ).hexdigest(),
        "scalar_checked_coordinates": sum(scalar_checked.values()),
        "cases": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "cases": len(results),
                "scalar_checked_coordinates": result["scalar_checked_coordinates"],
                "before": args.before,
            }
        )
    )


if __name__ == "__main__":
    main()
