"""Observed saved-weight changes, with the LoRA product as a float64 reference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .adapter import Pair, adapter_pairs
from .storage import FLOAT_DTYPES, AuditError, Checkpoint, FloatArray, Tensor, cast

CONTRACT = {
    "reference": "delta = (B_float64 @ A_float64) * scale; no dense full-layer delta retained",
    "observed": "merged_float64 - round_to_merged_dtype(base_float64)",
    "residual": "observed - delta; reported relative to the Frobenius norm of delta",
    "zero_change": "exact numeric equality to the base rounded to the merged dtype",
    "reference_merge": "round_to_merged_dtype(base_float64 + delta), ties to even",
    "unadapted": "must equal base rounded to merged dtype; integers require identical dtype/values",
    "scope": "saved plain Linear LoRA pairs only; no architecture execution or quality estimate",
}


@dataclass
class Squares:
    """Scaled sum of squares avoids overflow/underflow from squaring weights."""

    scale: float = 0.0
    total: float = 0.0

    def add(self, values: FloatArray) -> None:
        maximum = float(np.max(np.abs(values), initial=0.0))
        if not math.isfinite(maximum):
            raise AuditError("Nonfinite audit arithmetic; rescale the weights before comparing")
        if maximum == 0:
            return
        block = float(np.sum((values / maximum) ** 2))
        if maximum > self.scale:
            self.total = self.total * (self.scale / maximum) ** 2 + block
            self.scale = maximum
        else:
            self.total += block * (maximum / self.scale) ** 2

    def norm(self) -> float:
        result = self.scale * math.sqrt(self.total)
        if not math.isfinite(result):
            raise AuditError("Audit norm exceeds float64 range")
        return result


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    value = numerator / denominator
    if not math.isfinite(value):
        raise AuditError("Audit ratio exceeds float64 range")
    return value


def _compatible(base: Tensor, merged: Tensor) -> None:
    if base.shape != merged.shape:
        raise AuditError(f"Base/merged shape mismatch: {base.name}")
    if (base.dtype in FLOAT_DTYPES) != (merged.dtype in FLOAT_DTYPES):
        raise AuditError(f"Base/merged dtype category mismatch: {base.name}")
    if base.dtype not in FLOAT_DTYPES and base.dtype != merged.dtype:
        raise AuditError(f"Integer/bool dtype changes are unsupported: {base.name}")


def _layer(
    pair: Pair, base: Tensor, merged: Tensor, elements: int, max_adapter_bytes: int
) -> dict[str, Any]:
    if base.dtype not in FLOAT_DTYPES:
        raise AuditError(f"Adapted base tensor is not floating point: {base.name}")
    # Loading only A is enough: B rows are read alongside their base rows.
    if pair.a.elements * 8 > max_adapter_bytes:
        raise AuditError(f"A exceeds the configured adapter memory limit: {pair.module}")
    a = pair.a.floats().reshape(pair.a.shape)
    rows, columns = base.shape
    block_rows = max(1, elements // max(columns, pair.rank))
    norms = {key: Squares() for key in ("delta", "observed", "residual", "base_cast_drift")}
    changed = 0
    nonzero = 0
    erased = 0
    mismatches = 0
    reference_overflows = 0
    worst: dict[str, Any] | None = None
    worst_size = -1.0
    for row in range(0, rows, block_rows):
        count = min(rows - row, block_rows)
        b = pair.b.floats(row * pair.rank, count * pair.rank).reshape(count, pair.rank)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            delta = (b @ a) * pair.scale
        if not np.isfinite(delta).all():
            raise AuditError(f"Nonfinite LoRA product at {pair.module}")
        original = base.floats(row * columns, count * columns).reshape(count, columns)
        after = merged.floats(row * columns, count * columns).reshape(count, columns)
        baseline = cast(original, merged.dtype)
        if not np.isfinite(baseline).all():
            raise AuditError(f"Base conversion overflows the merged dtype at {pair.module}")
        observed = after - baseline
        residual = observed - delta
        drift = baseline - original
        for key, values in (
            ("delta", delta),
            ("observed", observed),
            ("residual", residual),
            ("base_cast_drift", drift),
        ):
            norms[key].add(values)
        nz = delta != 0
        changed += int(np.count_nonzero(after != baseline))
        nonzero += int(np.count_nonzero(nz))
        erased += int(np.count_nonzero(nz & (after == baseline)))
        with np.errstate(over="ignore", invalid="ignore"):
            reference = cast(original + delta, merged.dtype)
        reference_overflows += int(np.count_nonzero(~np.isfinite(reference)))
        mismatches += int(np.count_nonzero(after != reference))
        flat = int(np.argmax(np.abs(residual)))
        size = float(abs(residual.flat[flat]))
        if size > worst_size:
            worst_size = size
            local_row, column = np.unravel_index(flat, residual.shape)
            expected = float(reference.flat[flat])
            worst = {
                "row": row + int(local_row),
                "column": int(column),
                "base": float(original.flat[flat]),
                "cast_base": float(baseline.flat[flat]),
                "intended_delta": float(delta.flat[flat]),
                "merged": float(after.flat[flat]),
                "observed_delta": float(observed.flat[flat]),
                "residual": float(residual.flat[flat]),
                "reference_merged": expected if math.isfinite(expected) else None,
            }
    measured = {key: value.norm() for key, value in norms.items()}
    status = (
        "zero-update"
        if nonzero == 0 and changed == 0
        else "change-with-zero-reference"
        if nonzero == 0
        else "no-observed-update"
        if changed == 0
        else "observed-update"
    )
    return {
        "module": pair.module,
        "weight": pair.weight,
        "shape": list(base.shape),
        "rank": pair.rank,
        "alpha": pair.alpha,
        "scale": pair.scale,
        "adapter_dtypes": {"A": pair.a.dtype, "B": pair.b.dtype},
        "base_dtype": base.dtype,
        "merged_dtype": merged.dtype,
        "status": status,
        "elements": base.elements,
        "intended_nonzero_elements": nonzero,
        "changed_elements": changed,
        "unchanged_nonzero_update_elements": erased,
        "unchanged_nonzero_update_fraction": _ratio(erased, nonzero),
        "norms": measured,
        "relative_residual": _ratio(measured["residual"], measured["delta"]),
        "observed_to_intended_norm": _ratio(measured["observed"], measured["delta"]),
        "reference_merge_mismatches": mismatches,
        "reference_merge_overflows": reference_overflows,
        "largest_residual": worst,
    }


def _unadapted(base: Tensor, merged: Tensor, elements: int) -> dict[str, Any] | None:
    changed = 0
    example = None
    for start in range(0, base.elements, elements):
        count = min(elements, base.elements - start)
        if base.dtype in FLOAT_DTYPES:
            original = base.floats(start, count)
            baseline = cast(original, merged.dtype)
            if not np.isfinite(baseline).all():
                raise AuditError(f"Base conversion overflows the merged dtype: {base.name}")
            after = merged.floats(start, count)
        else:
            baseline = base.raw(start, count)
            after = merged.raw(start, count)
        different = after != baseline
        changed += int(np.count_nonzero(different))
        if example is None and different.any():
            index = int(np.flatnonzero(different)[0])
            example = {
                "flat_index": start + index,
                "cast_base": baseline[index].item(),
                "merged": after[index].item(),
            }
    if not changed:
        return None
    return {
        "weight": base.name,
        "elements": base.elements,
        "changed_elements": changed,
        "base_dtype": base.dtype,
        "merged_dtype": merged.dtype,
        "first_change": example,
    }


def audit(
    base: str | Path,
    adapter: str | Path,
    merged: str | Path,
    *,
    block_mib: int = 16,
    max_adapter_mib: int = 512,
) -> dict[str, Any]:
    """Read local artifacts only. No model imports, downloads, GPU or weight writes."""
    for key, value in (("block_mib", block_mib), ("max_adapter_mib", max_adapter_mib)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4096:
            raise AuditError(f"{key} must be an integer from 1 to 4096")
    inputs = {
        "base": Checkpoint(base),
        "adapter": Checkpoint(adapter, adapter=True),
        "merged": Checkpoint(merged),
    }
    before, update, after = inputs["base"], inputs["adapter"], inputs["merged"]
    if before.tensors.keys() != after.tensors.keys():
        missing = sorted(before.tensors.keys() - after.tensors.keys())
        extra = sorted(after.tensors.keys() - before.tensors.keys())
        raise AuditError(
            f"Base/merged tensor inventories differ; missing={missing[:5]}, extra={extra[:5]}"
        )
    pairs = adapter_pairs(update, before)
    by_weight = {pair.weight: pair for pair in pairs}
    # Workspace target for the several simultaneous binary64 blocks. A whole
    # row and the rank-wide B row are the minimum; A has a separate limit.
    elements = max(1, block_mib * 1024 * 1024 // (8 * 10))
    layers = []
    unrelated = []
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            for name, tensor in sorted(before.tensors.items()):
                counterpart = after.tensors[name]
                _compatible(tensor, counterpart)
                if name in by_weight:
                    layers.append(
                        _layer(
                            by_weight[name],
                            tensor,
                            counterpart,
                            elements,
                            max_adapter_mib * 1024 * 1024,
                        )
                    )
                else:
                    difference = _unadapted(tensor, counterpart, elements)
                    if difference is not None:
                        unrelated.append(difference)
        except FloatingPointError as exc:
            raise AuditError(f"Audit arithmetic exceeds float64 range: {exc}") from exc
    # All original files, including configs and indexes, are fingerprinted.
    fingerprints = {role: checkpoint.fingerprints() for role, checkpoint in inputs.items()}
    for checkpoint in inputs.values():
        checkpoint.verify_stable()
    return {
        "schema_version": 1,
        "arithmetic": CONTRACT,
        "inputs": fingerprints,
        "settings": {"block_mib": block_mib, "max_adapter_mib": max_adapter_mib},
        "summary": {
            "adapted_tensors": len(layers),
            "checked_tensors": len(before.tensors),
            "adapted_elements": sum(layer["elements"] for layer in layers),
            "intended_nonzero_elements": sum(
                layer["intended_nonzero_elements"] for layer in layers
            ),
            "changed_adapted_elements": sum(layer["changed_elements"] for layer in layers),
            "unchanged_nonzero_update_elements": sum(
                layer["unchanged_nonzero_update_elements"] for layer in layers
            ),
            "unchanged_nonzero_update_fraction": _ratio(
                sum(layer["unchanged_nonzero_update_elements"] for layer in layers),
                sum(layer["intended_nonzero_elements"] for layer in layers),
            ),
            "zero_update_tensors": sum(layer["status"] == "zero-update" for layer in layers),
            "no_observed_update_tensors": sum(
                layer["status"] == "no-observed-update" for layer in layers
            ),
            "unadapted_changed_tensors": len(unrelated),
            "unadapted_changed_elements": sum(layer["changed_elements"] for layer in unrelated),
        },
        "layers": layers,
        "unadapted_changes": unrelated,
        "limitations": [
            "Supplied base identity and training provenance are not independently established.",
            "Equal weights do not distinguish rounding from an omitted or wrong adapter.",
            "The float64 reference is numerical; it is not a PEFT merge emulator.",
            "Operation order and matmul precision can differ from the producer, notably at ties.",
            "Weight-space changes and residuals do not establish generation or task quality.",
            "Missing tensors, unsupported variants and nonfinite inputs refuse the whole audit.",
        ],
    }
