"""Full Torch and scalar Decimal checks, independent of auditor layout decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from decimal import Decimal, localcontext
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

import mergeglass
from mergeglass import AuditError, audit


def load(root: Path, stem: str = "model") -> dict[str, torch.Tensor]:
    index = root / f"{stem}.safetensors.index.json"
    paths = (
        sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if index.exists()
        else [f"{stem}.safetensors"]
    )
    return {name: value for path in paths for name, value in load_file(root / path).items()}


def close(actual: float | None, expected: float | None) -> None:
    if expected is None:
        assert actual is None
    else:
        assert actual is not None and math.isclose(
            actual, expected, rel_tol=2e-10, abs_tol=1e-13
        ), (actual, expected)


def scalar_product(a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
    """80-digit scalar sums, deliberately without a matmul or layout helper."""
    output = torch.empty((a.shape[1], b.shape[0]), dtype=torch.float64)
    with localcontext() as context:
        context.prec = 80
        for ins in range(a.shape[1]):
            for outs in range(b.shape[0]):
                value = sum(
                    (
                        Decimal.from_float(float(a[k, ins])) * Decimal.from_float(float(b[outs, k]))
                        for k in range(a.shape[0])
                    ),
                    Decimal(0),
                ) * Decimal.from_float(scale)
                output[ins, outs] = float(value)
    return output


@lru_cache(maxsize=2)
def representable(dtype: torch.dtype) -> np.ndarray[Any, Any]:
    if dtype == torch.bfloat16:
        return np.array(
            [struct.unpack("<f", struct.pack("<I", bits << 16))[0] for bits in range(0x7F80)],
            dtype=np.float64,
        )
    return np.arange(0x7C00, dtype=np.uint16).view(np.float16).astype(np.float64)


def rounded(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Choose the nearest adjacent representable number, with even-bit ties.

    No frexp, spacing calculation or integer-significand rounding from the
    auditor is reused. Torch's double -> half/BF16 may round through float32.
    """
    if dtype not in (torch.float16, torch.bfloat16):
        return values.to(dtype).double()
    table = representable(dtype)
    x = np.abs(values.numpy())
    hi = np.minimum(np.searchsorted(table, x), len(table) - 1)
    lo = np.maximum(hi - 1, 0)
    distance_lo, distance_hi = x - table[lo], table[hi] - x
    upper = (distance_hi < distance_lo) | ((distance_hi == distance_lo) & (hi % 2 == 0))
    result = np.where(upper, table[hi], table[lo])
    overflow = 65520.0 if dtype == torch.float16 else float.fromhex("0x1.ffp+127")
    result = np.where(x >= overflow, np.inf, result)
    return torch.from_numpy(np.copysign(result, values.numpy()))


def check_layer(
    row: dict[str, Any], before: torch.Tensor, after: torch.Tensor, delta: torch.Tensor
) -> dict[str, Any]:
    original = before.double()
    baseline = before.to(after.dtype).double()
    observed = after.double() - baseline
    residual = observed - delta
    nonzero, unchanged = delta != 0, after.double() == baseline
    nonzero_count = int(nonzero.sum())
    erased = int((nonzero & unchanged).sum())
    assert row["changed_elements"] == int((~unchanged).sum())
    assert row["intended_nonzero_elements"] == nonzero_count
    assert row["unchanged_nonzero_update_elements"] == erased
    close(
        row["unchanged_nonzero_update_fraction"], erased / nonzero_count if nonzero_count else None
    )
    norms = {
        "delta": float(delta.norm()),
        "observed": float(observed.norm()),
        "residual": float(residual.norm()),
        "base_cast_drift": float((baseline - original).norm()),
        "unchanged_reference": float(delta[unchanged].norm()),
    }
    for key, value in norms.items():
        close(row["norms"][key], value)
    intended, saved = norms["delta"], norms["observed"]
    inner = float((delta * observed).sum())
    close(row["relative_residual"], norms["residual"] / intended if intended else None)
    close(row["projection_gain"], inner / intended**2 if intended else None)
    close(row["cosine_similarity"], inner / intended / saved if intended and saved else None)
    close(row["observed_to_intended_norm"], saved / intended if intended else None)
    close(
        row["unchanged_reference_energy_fraction"],
        norms["unchanged_reference"] ** 2 / intended**2 if intended else None,
    )
    reference = rounded(original + delta, after.dtype)
    assert row["reference_merge_mismatches"] == int((reference != after.double()).sum()), (
        row["module"],
        str(after.dtype),
        row["reference_merge_mismatches"],
        int((reference != after.double()).sum()),
    )
    witness = row["largest_residual"]
    coord = (witness["row"], witness["column"])
    close(abs(witness["residual"]), float(residual.abs().max()))
    close(witness["residual"], float(residual[coord]))
    close(witness["intended_delta"], float(delta[coord]))
    close(witness["merged"], float(after[coord]))
    return {
        "module": row["module"],
        "norms": norms,
        "largest_residual_coordinate": list(coord),
        "torch_cast_reference_disagreements": int(
            (reference != (original + delta).to(after.dtype).double()).sum()
        ),
    }


def verify_case(root: Path, case: dict[str, Any], *, before: bool) -> dict[str, Any]:
    original = load(root / "base")
    factors = load(root / "adapter", "adapter_model")
    deltas = {}
    decimal_entries = 0
    for module, layout in case["native_layouts"].items():
        prefix = f"base_model.model.{module}"
        embedding = layout["kind"] == "Embedding"
        a = factors[prefix + (".lora_embedding_A" if embedding else ".lora_A.weight")].double()
        b = factors[prefix + (".lora_embedding_B" if embedding else ".lora_B.weight")].double()
        product = (b @ a) * layout["scale"]
        transposed = embedding or layout["transpose"]
        delta = product.T if transposed else product
        if transposed and not before:
            torch.testing.assert_close(
                delta, scalar_product(a, b, layout["scale"]), rtol=2e-14, atol=1e-15
            )
            decimal_entries += delta.numel()
        deltas[module + ".weight"] = delta
    results = []
    for merge in case["merges"]:
        report = None
        refusal = None
        try:
            report = audit(root / "base", root / "adapter", root / merge["name"], block_mib=1)
        except AuditError as exc:
            refusal = str(exc)
        if before:
            assert refusal is not None
            results.append({"merge": merge, "refusal": refusal})
            continue
        assert report is not None, refusal
        saved = load(root / merge["name"])
        checked = []
        for row in report["layers"]:
            module, key = row["module"], row["weight"]
            layout = case["native_layouts"][module]
            if layout["transpose"]:
                assert row["weight_layout"] == "conv1d"
                assert row["layout_evidence"]
                assert row["coordinate_axes"] == {"row": "input", "column": "output"}
            close(row["scale"], layout["scale"])
            assert row["rank"] == layout["rank"]
            checked.append(check_layer(row, original[key], saved[key], deltas[key]))
        # Every tensor outside adapter factors must retain its cast base.
        assert not report["unadapted_changes"]
        replacements = {row["weight"] for row in report.get("saved_modules", [])}
        for row in report.get("saved_modules", []):
            key = row["weight"]
            target = factors[f"base_model.model.{key}"]
            assert torch.equal(rounded(target.double(), saved[key].dtype), saved[key].double())
            assert row["target_mismatches"] == 0
        for key in original.keys() - deltas.keys() - replacements:
            assert torch.equal(original[key].to(saved[key].dtype), saved[key])
        control = next(r for r in report["layers"] if r["weight"] == case["square_control_weight"])
        if merge["name"] == "omitted-square":
            close(control["relative_residual"], 1)
            assert control["status"] == "no-observed-update"
        elif merge["name"] == "wrong-square-orientation":
            assert control["changed_elements"] == control["elements"]
            assert control["relative_residual"] > 0.5
        elif "f32" in merge["name"]:
            assert max(r["relative_residual"] for r in report["layers"]) < 2e-5
        results.append({"merge": merge, "report": report, "independent": checked})
    return {"case": case["name"], "decimal_entries": decimal_entries, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--before", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    producer = json.loads((args.directory / "producer.json").read_text())
    for item in producer["files"]:
        payload = (args.directory / item["file"]).read_bytes()
        assert len(payload) == item["bytes"]
        assert hashlib.sha256(payload).hexdigest() == item["sha256"]
    cases = [
        verify_case(args.directory / case["name"], case, before=args.before)
        for case in producer["cases"]
    ]
    result = {
        "before": args.before,
        "producer": producer,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "auditor_modules": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(mergeglass.__file__).parent.glob("*.py"))
        },
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        f"Verified {sum(len(case['results']) for case in cases)} saved merges; before={args.before}"
    )


if __name__ == "__main__":
    main()
