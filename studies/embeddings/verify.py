"""Replay original files against a dense Torch reference and scalar Decimal sums."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

import mergeglass
from mergeglass import AuditError, audit
from mergeglass.cli import render


def load(root: Path, stem: str = "model") -> dict[str, torch.Tensor]:
    index = root / f"{stem}.safetensors.index.json"
    paths = (
        sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if index.exists()
        else [f"{stem}.safetensors"]
    )
    return {name: tensor for path in paths for name, tensor in load_file(root / path).items()}


def close(actual: float | None, expected: float | None) -> None:
    if expected is None:
        assert actual is None
    else:
        assert actual is not None and math.isclose(
            actual, expected, rel_tol=2e-12, abs_tol=2e-14
        ), (actual, expected)


def scalar_delta(a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
    """Independent full product, without a matmul or auditor layout helper."""
    result = torch.empty((a.shape[1], b.shape[0]), dtype=torch.float64)
    with localcontext() as context:
        context.prec = 80
        for token in range(a.shape[1]):
            for dimension in range(b.shape[0]):
                value = sum(
                    (
                        Decimal.from_float(float(a[k, token]))
                        * Decimal.from_float(float(b[dimension, k]))
                        for k in range(a.shape[0])
                    ),
                    Decimal(0),
                ) * Decimal.from_float(scale)
                result[token, dimension] = float(value)
    return result


def verify_case(base: Path, adapter: Path, merged: Path, *, decimal: bool) -> dict[str, Any]:
    original, factors, saved = load(base), load(adapter, "adapter_model"), load(merged)
    config = json.loads((adapter / "adapter_config.json").read_text())
    report = audit(base, adapter, merged, block_mib=1)
    results = []
    for layer in report["layers"]:
        path = layer["module"]
        prefix = f"base_model.model.{path}"
        embedding = prefix + ".lora_embedding_A" in factors
        a_key = ".lora_embedding_A" if embedding else ".lora_A.weight"
        b_key = ".lora_embedding_B" if embedding else ".lora_B.weight"
        a, b = factors[prefix + a_key].double(), factors[prefix + b_key].double()
        rank = next(
            (
                value
                for key, value in config["rank_pattern"].items()
                if path == key or path.endswith("." + key)
            ),
            config["r"],
        )
        alpha = next(
            (
                value
                for key, value in config["alpha_pattern"].items()
                if path == key or path.endswith("." + key)
            ),
            config["lora_alpha"],
        )
        scale = alpha / (math.sqrt(rank) if config["use_rslora"] else rank)
        delta = (b @ a) * scale
        if embedding:
            delta = delta.T
            assert layer["weight_layout"] == "embedding"
            if decimal:
                torch.testing.assert_close(delta, scalar_delta(a, b, scale), rtol=2e-14, atol=1e-16)
        before, after = original[path + ".weight"].double(), saved[path + ".weight"]
        baseline = before.to(after.dtype).double()
        observed = after.double() - baseline
        residual = observed - delta
        nonzero = delta != 0
        unchanged = after.double() == baseline
        count = int(nonzero.count_nonzero())
        erased = int((nonzero & unchanged).count_nonzero())
        assert layer["intended_nonzero_elements"] == count
        assert layer["changed_elements"] == int((~unchanged).count_nonzero())
        assert layer["unchanged_nonzero_update_elements"] == erased
        close(layer["unchanged_nonzero_update_fraction"], erased / count if count else None)
        norms = {
            "delta": float(delta.norm()),
            "observed": float(observed.norm()),
            "residual": float(residual.norm()),
            "base_cast_drift": float((baseline - before).norm()),
            "unchanged_reference": float(delta[unchanged].norm()),
        }
        for key, value in norms.items():
            close(layer["norms"][key], value)
        length = norms["delta"]
        close(layer["relative_residual"], norms["residual"] / length if length else None)
        close(
            layer["unchanged_reference_energy_fraction"],
            norms["unchanged_reference"] ** 2 / length**2 if length else None,
        )
        inner = float((observed * delta).sum())
        close(layer["projection_gain"], inner / length**2 if length else None)
        close(
            layer["cosine_similarity"],
            inner / length / norms["observed"] if length and norms["observed"] else None,
        )
        witness = layer["largest_residual"]
        row, column = witness["row"], witness["column"]
        close(abs(witness["residual"]), float(residual.abs().max()))
        close(witness["intended_delta"], float(delta[row, column]))
        close(witness["merged"], float(after[row, column]))
        close(witness["residual"], float(residual[row, column]))
        results.append(
            {
                "module": path,
                "layout": "embedding" if embedding else "linear",
                "shape": layer["shape"],
                "changed": layer["changed_elements"],
                "nonzero_reference": count,
                "unchanged_nonzero": erased,
                "gain": layer["projection_gain"],
                "relative_residual": layer["relative_residual"],
                "witness": witness,
            }
        )
    adapted = {layer["weight"] for layer in report["layers"]}
    unrelated = {
        key: int((saved[key] != original[key].to(saved[key].dtype)).count_nonzero())
        for key in original.keys() - adapted
    }
    assert report["summary"]["unadapted_changed_elements"] == sum(unrelated.values()) == 0
    return {"case": merged.name, "summary": report["summary"], "layers": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--before", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    producer = json.loads((args.directory / "producer.json").read_text())
    for item in producer["files"]:
        data = (args.directory / item["file"]).read_bytes()
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
    root = args.directory
    cases = [(root / "base", root / "adapter", root / name) for name in producer["cases"]]
    cases.append(
        (root / "transformer-base", root / "transformer-adapter", root / "transformer-merged")
    )
    cases.append(
        (root / "transformer-base", root / "transformer-default-save", root / "transformer-merged")
    )
    results = []
    for base, adapter, merged in cases:
        label = (
            "transformer-default-save"
            if adapter.name == "transformer-default-save"
            else merged.name
        )
        if args.before:
            try:
                audit(base, adapter, merged)
            except AuditError as exc:
                assert "Unsupported adapter tensor" in str(exc)
                results.append({"case": label, "refusal": str(exc)})
            else:
                raise AssertionError("Baseline unexpectedly accepted embedding factors")
        else:
            result = verify_case(base, adapter, merged, decimal=base.name == "base")
            result["case"] = label
            results.append(result)
    if not args.before:
        report = audit(root / "base", root / "adapter", root / "omit-embeddings-bf16")
        args.out.with_suffix(".txt").write_text(render(report) + "\n")
    payload = {
        "before": args.before,
        "producer": producer,
        "verification": (
            "dense PyTorch complete count/norm/gain/coordinate replay; "
            "80-digit scalar sums at every teaching embedding coordinate"
        ),
        "decimal_embedding_coordinates": 13 * (37 * 11 + 11 * 11),
        "auditor_modules": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(mergeglass.__file__).parent.glob("*.py"))
        },
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Verified {len(results)} native mixed embedding/Linear saves; before={args.before}")


if __name__ == "__main__":
    main()
