"""Partial DoRA observations when the original training base is unavailable."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .adapter import adapter_pairs
from .dora_rows import DoraRows
from .storage import FLOAT_DTYPES, AuditError, Checkpoint


def audit_dora_norms(
    adapter: str | Path, merged: str | Path, *, block_mib: int = 16
) -> dict[str, Any]:
    """Compare saved row lengths with |m|, without inferring reference directions.

    All adapter tensors and all floating checkpoint tensors are scanned for
    finite values. A/B are validated but no matrix product or base is needed.
    A matching norm is a necessary condition, not proof of a correct merge.
    """
    if isinstance(block_mib, bool) or not isinstance(block_mib, int) or not 1 <= block_mib <= 4096:
        raise AuditError("block_mib must be an integer from 1 to 4096")
    inputs = {"adapter": Checkpoint(adapter, adapter=True), "merged": Checkpoint(merged)}
    update, after = inputs["adapter"], inputs["merged"]
    pairs = adapter_pairs(update, after, weight_role="merged")
    if any(pair.base_snapshot is not None for pair in pairs):
        raise AuditError("Saved base snapshots require an original base for comparison")
    if any(pair.magnitude is None for pair in pairs):
        raise AuditError("Norm observations without a base require a saved DoRA magnitude vector")
    elements = max(1, block_mib * 1024 * 1024 // (8 * 10))
    # Do not produce a partial success over malformed or nonfinite inputs.
    # Read A/B in bounded slices; unlike a full audit, no decoded A is retained.
    for checkpoint in inputs.values():
        for tensor in checkpoint.tensors.values():
            if checkpoint is update or tensor.dtype in FLOAT_DTYPES:
                for start in range(0, tensor.elements, elements):
                    tensor.floats(start, min(elements, tensor.elements - start))
    layers: list[dict[str, Any]] = []
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            for pair in pairs:
                assert pair.magnitude is not None
                weight = after.tensors[pair.weight]
                rows, columns = weight.shape
                block_rows = max(1, elements // columns)
                observed = DoraRows()
                for row in range(0, rows, block_rows):
                    count = min(rows - row, block_rows)
                    values = weight.floats(row * columns, count * columns).reshape(count, columns)
                    observed.add(row, pair.magnitude.floats(row, count), None, values)
                layers.append(
                    {
                        "module": pair.module,
                        "weight": pair.weight,
                        "adapter_variant": "dora",
                        "shape": list(weight.shape),
                        "merged_dtype": weight.dtype,
                        "magnitude_dtype": pair.magnitude.dtype,
                        "dora_rows": observed.result(include_direction=False),
                    }
                )
        except FloatingPointError as exc:
            raise AuditError(f"Norm arithmetic exceeds float64 range: {exc}") from exc
    fingerprints = {role: checkpoint.fingerprints() for role, checkpoint in inputs.items()}
    for checkpoint in inputs.values():
        checkpoint.verify_stable()
    return {
        "schema": "mergeglass.dora-norms",
        "schema_version": 1,
        "scope": "saved DoRA row norms without an original base; partial observations only",
        "arithmetic": {
            "row_norm_factor": "saved full-row norm / abs(saved magnitude), retained in log2 form",
            "zero_rows": "zero targets and zero saved rows have explicit counts, no invented ratio",
        },
        "inputs": fingerprints,
        "settings": {"block_mib": block_mib},
        "summary": {
            "norm_observed_tensors": len(layers),
            "norm_observed_rows": sum(layer["dora_rows"]["rows"] for layer in layers),
            "checkpoint_tensors": len(after.tensors),
            "unmapped_tensors": len(after.tensors) - len(layers),
        },
        "layers": layers,
        "limitations": [
            "A matching row norm does not establish a correct merge or even a correct direction.",
            "Reference directions, complete updates, gain and residual require the original base.",
            "Unmapped tensors are not compared; their equality to a base is unknown.",
            "Saved magnitudes and adapter identity have no independent provenance verification.",
            "Tensor shapes and saved metadata do not establish every module's type.",
            "Rounding can change a correctly merged row's norm; no pass/fail threshold is applied.",
            "Weight-space observations do not establish generation or task quality.",
        ],
    }
