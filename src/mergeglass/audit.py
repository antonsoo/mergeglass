"""Observed saved-weight changes, with a declared float64 adapter reference."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .adapter import Pair, SavedTensor, adapter_pairs, saved_modules
from .dora_rows import DoraRows
from .storage import FLOAT_DTYPES, AuditError, Checkpoint, FloatArray, Tensor, cast

CONTRACT = {
    "reference": (
        "Linear LoRA: delta = (B_float64 @ A_float64) * scale; "
        "DoRA: direction = base_float64 + (B_float64 @ A_float64) * scale; "
        "reference_weight = row_normalize(direction) * magnitude_float64; "
        "delta = reference_weight - base_float64; no dense full-layer delta retained"
    ),
    "observed": "merged_float64 - round_to_merged_dtype(base_float64)",
    "residual": "observed - delta; reported relative to the Frobenius norm of delta",
    "zero_change": "exact numeric equality to the base rounded to the merged dtype",
    "reference_merge": (
        "LoRA: round_to_merged_dtype(base_float64 + delta); "
        "DoRA: round_to_merged_dtype(reference_weight); ties to even"
    ),
    "projection_gain": "dot(observed, delta) / dot(delta, delta); null for a zero reference",
    "cosine_similarity": (
        "dot(observed, delta) / (norm(observed) * norm(delta)); null if either is zero"
    ),
    "unchanged_reference_energy": (
        "squared delta norm at unchanged positions / squared delta norm; null if delta is zero"
    ),
    "unadapted": "must equal base rounded to merged dtype; integers require identical dtype/values",
    "scope": "saved Linear LoRA/DoRA adapters only; no architecture execution or quality estimate",
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


@dataclass
class Alignment:
    """Signed inner product with separately scaled operands, accumulated in blocks."""

    reference: Squares
    observed: Squares
    product: float = 0.0

    def add(self, reference: FloatArray, observed: FloatArray) -> None:
        old_reference, old_observed = self.reference.scale, self.observed.scale
        self.reference.add(reference)
        self.observed.add(observed)
        if self.reference.scale == 0 or self.observed.scale == 0:
            return
        retained = self.product * (old_reference / self.reference.scale)
        retained *= old_observed / self.observed.scale
        block = float(np.sum((reference / self.reference.scale) * (observed / self.observed.scale)))
        self.product = math.fsum((retained, block))

    def cosine(self) -> float | None:
        if self.reference.scale == 0 or self.observed.scale == 0:
            return None
        value = self.product / math.sqrt(self.reference.total) / math.sqrt(self.observed.total)
        # Summation can move a parallel pair a few ulps outside the mathematical range.
        return max(-1.0, min(1.0, value))


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    value = numerator / denominator
    if not math.isfinite(value):
        raise AuditError("Audit ratio exceeds float64 range")
    return value


def _energy_fraction(reference: Squares, unchanged: Squares) -> float | None:
    if reference.scale == 0:
        return None
    # Form a norm ratio before squaring. Raw squared norms can overflow or
    # underflow even when their dimensionless fraction is representable.
    ratio = (unchanged.scale / reference.scale) * math.sqrt(unchanged.total / reference.total)
    return min(1.0, ratio) ** 2


def _compatible(base: Tensor, merged: Tensor, *, role: str = "Base") -> None:
    if base.shape != merged.shape:
        raise AuditError(f"{role}/merged shape mismatch: {base.name}")
    if (base.dtype in FLOAT_DTYPES) != (merged.dtype in FLOAT_DTYPES):
        raise AuditError(f"{role}/merged dtype category mismatch: {base.name}")
    if base.dtype not in FLOAT_DTYPES and base.dtype != merged.dtype:
        raise AuditError(f"Integer/bool dtype changes are unsupported: {base.name}")


def _dora_reference(
    pair: Pair, original: FloatArray, lora_delta: FloatArray, row: int
) -> FloatArray:
    assert pair.magnitude is not None
    reference = original + lora_delta
    if not np.isfinite(reference).all():
        raise AuditError(f"Nonfinite DoRA direction at {pair.module}")
    maximum = np.max(np.abs(reference), axis=1)
    zero = np.flatnonzero(maximum == 0)
    if zero.size:
        raise AuditError(f"Undefined DoRA direction at {pair.module}, row {row + int(zero[0])}")
    # Normalize each complete output row. Scaling first avoids overflow from
    # squares or magnitude/norm and underflow for tiny but nonzero directions.
    reference /= maximum[:, None]
    length = np.sqrt(np.sum(reference**2, axis=1))
    reference /= length[:, None]
    magnitude = pair.magnitude.floats(row, original.shape[0])
    reference *= magnitude[:, None]
    return reference


def _layer(
    pair: Pair,
    base: Tensor,
    merged: Tensor,
    elements: int,
    max_adapter_bytes: int,
    *,
    bias: bool = False,
    product: Callable[[int, int], FloatArray] | None = None,
) -> dict[str, Any]:
    if base.dtype not in FLOAT_DTYPES:
        raise AuditError(f"Adapted base tensor is not floating point: {base.name}")
    # Linear: B rows @ A. Embedding/Conv1D: A columns, transposed @ B.T.
    # Keep the factor independent of saved-weight rows; a vocabulary-wide A
    # must not be decoded in full merely to audit a few token rows at a time.
    fixed = None
    if not bias and product is None:
        stationary = pair.b if pair.transposed else pair.a
        if stationary.elements * 8 > max_adapter_bytes:
            name = "B" if pair.transposed else "A"
            raise AuditError(f"{name} exceeds the configured adapter memory limit: {pair.module}")
        fixed = stationary.floats().reshape(stationary.shape)
        if pair.transposed:
            fixed = fixed.T
    # A bias vector uses the same observed/reference arithmetic in bounded
    # one-column blocks. Its saved shape and index remain one-dimensional.
    rows, columns = (base.elements, 1) if bias else base.shape
    block_rows = max(1, elements // max(columns, pair.rank))
    norms = {
        key: Squares()
        for key in ("delta", "observed", "residual", "base_cast_drift", "unchanged_reference")
    }
    alignment = Alignment(norms["delta"], norms["observed"])
    changed = 0
    nonzero = 0
    erased = 0
    mismatches = 0
    reference_overflows = 0
    dora_rows = DoraRows() if pair.magnitude is not None else None
    worst: dict[str, Any] | None = None
    worst_size = -1.0
    for row in range(0, rows, block_rows):
        count = min(rows - row, block_rows)
        if product is not None:
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                delta = product(row, count)
        elif bias:
            assert pair.bias is not None
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                delta = pair.bias.floats(row, count).reshape(count, 1) * pair.scale
        elif pair.transposed:
            factor = np.empty((count, pair.rank), dtype=np.float64)
            for component in range(pair.rank):
                factor[:, component] = pair.a.floats(component * rows + row, count)
        else:
            factor = pair.b.floats(row * pair.rank, count * pair.rank).reshape(count, pair.rank)
        if not bias and product is None:
            assert fixed is not None
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                delta = (factor @ fixed) * pair.scale
        if not np.isfinite(delta).all():
            raise AuditError(f"Nonfinite LoRA product at {pair.module}")
        original = base.floats(row * columns, count * columns).reshape(count, columns)
        if pair.base_snapshot is not None and not bias:
            snapshot = pair.base_snapshot.floats(row * columns, count * columns)
            expected_snapshot = cast(original.reshape(-1), pair.base_snapshot.dtype)
            if not np.isfinite(expected_snapshot).all() or not np.array_equal(
                snapshot, expected_snapshot
            ):
                raise AuditError(
                    f"Saved base snapshot differs from the supplied original base at {pair.module}"
                )
        after = merged.floats(row * columns, count * columns).reshape(count, columns)
        baseline = cast(original, merged.dtype)
        if not np.isfinite(baseline).all():
            raise AuditError(f"Base conversion overflows the merged dtype at {pair.module}")
        reference_weight = None
        if pair.magnitude is not None:
            reference_weight = _dora_reference(pair, original, delta, row)
            assert dora_rows is not None
            dora_rows.add(row, pair.magnitude.floats(row, count), reference_weight, after)
            delta = reference_weight - original
        observed = after - baseline
        residual = observed - delta
        drift = baseline - original
        alignment.add(delta, observed)
        for key, values in (
            ("residual", residual),
            ("base_cast_drift", drift),
        ):
            norms[key].add(values)
        nz = delta != 0
        unchanged = after == baseline
        norms["unchanged_reference"].add(np.where(unchanged, delta, 0.0))
        changed += int(np.count_nonzero(~unchanged))
        nonzero += int(np.count_nonzero(nz))
        erased += int(np.count_nonzero(nz & unchanged))
        with np.errstate(over="ignore", invalid="ignore"):
            reference = cast(
                original + delta if reference_weight is None else reference_weight, merged.dtype
            )
        reference_overflows += int(np.count_nonzero(~np.isfinite(reference)))
        mismatches += int(np.count_nonzero(after != reference))
        flat = int(np.argmax(np.abs(residual)))
        size = float(abs(residual.flat[flat]))
        if size > worst_size:
            worst_size = size
            local_row, column = np.unravel_index(flat, residual.shape)
            expected = float(reference.flat[flat])
            worst = {
                **(
                    {"index": row + int(local_row)}
                    if bias
                    else {"row": row + int(local_row), "column": int(column)}
                ),
                "base": float(original.flat[flat]),
                "cast_base": float(baseline.flat[flat]),
                "intended_delta": float(delta.flat[flat]),
                "merged": float(after.flat[flat]),
                "observed_delta": float(observed.flat[flat]),
                "residual": float(residual.flat[flat]),
                "reference_merged": expected if math.isfinite(expected) else None,
            }
    measured = {key: value.norm() for key, value in norms.items()}
    magnitude_ratio = _ratio(measured["observed"], measured["delta"])
    cosine = alignment.cosine()
    gain = None if magnitude_ratio is None else 0.0 if cosine is None else cosine * magnitude_ratio
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
        "weight": base.name,
        "adapter_variant": "lora" if pair.magnitude is None else "dora",
        **(
            {"weight_layout": "embedding", "reference_product": "(B @ A).T"}
            if pair.embedding and not bias
            else {}
        ),
        **(
            {
                "weight_layout": "conv1d",
                "reference_product": "(B @ A).T",
                "layout_evidence": list(pair.conv1d_evidence),
                "coordinate_axes": {"row": "input", "column": "output"},
            }
            if pair.conv1d_evidence and not bias
            else {}
        ),
        "shape": list(base.shape),
        "rank": pair.rank,
        "alpha": pair.alpha,
        "scale": pair.scale,
        "adapter_dtypes": {
            **(
                {"bias": pair.bias.dtype}
                if bias and pair.bias is not None
                else {"A": pair.a.dtype, "B": pair.b.dtype}
            ),
            **({"magnitude": pair.magnitude.dtype} if pair.magnitude is not None else {}),
        },
        **(
            {
                "base_snapshot": {
                    "tensor": pair.base_snapshot.name,
                    "dtype": pair.base_snapshot.dtype,
                    "comparison": "equal to supplied original base rounded to snapshot dtype",
                }
            }
            if pair.base_snapshot is not None and not bias
            else {}
        ),
        **(
            {"tensor_role": "lora_B_bias", "reference_product": "bias_float64 * scale"}
            if bias
            else {}
        ),
        "base_dtype": base.dtype,
        "merged_dtype": merged.dtype,
        "status": status,
        "elements": base.elements,
        "intended_nonzero_elements": nonzero,
        "changed_elements": changed,
        "unchanged_nonzero_update_elements": erased,
        "unchanged_nonzero_update_fraction": _ratio(erased, nonzero),
        "unchanged_reference_energy_fraction": _energy_fraction(
            norms["delta"], norms["unchanged_reference"]
        ),
        "norms": measured,
        "relative_residual": _ratio(measured["residual"], measured["delta"]),
        "observed_to_intended_norm": magnitude_ratio,
        "projection_gain": gain,
        "cosine_similarity": cosine,
        "reference_merge_mismatches": mismatches,
        "reference_merge_overflows": reference_overflows,
        "largest_residual": worst,
        **({"dora_rows": dora_rows.result()} if dora_rows is not None else {}),
    }


def _expert_parameter_layers(
    pair: Pair, base: Checkpoint, merged: Checkpoint, elements: int, max_adapter_bytes: int
) -> list[dict[str, Any]]:
    layout = pair.expert_layout
    assert layout is not None
    # B interleaves experts in its columns. Keep it once, then decode and
    # release one expert's contiguous A rows. Native checkpoints hold separate
    # Linear weights; no dense 3-D parameter or delta is reconstructed.
    if (pair.b.elements + pair.rank * layout.inputs) * 8 > max_adapter_bytes:
        raise AuditError(
            f"Expert B plus one A slice exceeds the adapter memory limit: {pair.module}"
        )
    b = pair.b.floats().reshape(layout.outputs, pair.rank, layout.experts)

    def one_expert(expert: int) -> list[dict[str, Any]]:
        a = pair.a.floats(expert * pair.rank * layout.inputs, pair.rank * layout.inputs)
        a = a.reshape(pair.rank, layout.inputs)
        results = []
        for matrix in layout.matrices:
            if matrix.expert != expert:
                continue
            original, after = base.tensors[matrix.weight], merged.tensors[matrix.weight]
            _compatible(original, after)

            def product(row: int, count: int, *, offset: int = matrix.output_start) -> FloatArray:
                return (b[offset + row : offset + row + count, :, expert] @ a) * pair.scale

            view = replace(
                pair,
                module=matrix.weight.removesuffix(".weight"),
                weight=matrix.weight,
                expert_layout=None,
            )
            result = _layer(view, original, after, elements, 0, product=product)
            result.update(
                {
                    "weight_layout": "expert_parameter_slice",
                    "expert_parameter": layout.parameter,
                    "expert_index": expert,
                    "parameter_output_rows": [
                        matrix.output_start,
                        matrix.output_start + matrix.rows,
                    ],
                    "coordinate_axes": {"row": "output", "column": "input"},
                    "factor_tensors": {"A": pair.a.name, "B": pair.b.name},
                    "layout_evidence": list(layout.evidence),
                    "reference_product": (
                        "sum_r B[o, r * experts + e] * A[e * rank + r, i] * scale; "
                        "o includes the declared gate/up output-row offset"
                    ),
                }
            )
            result["largest_residual"] = {
                "expert": expert,
                "parameter_output_row": matrix.output_start + result["largest_residual"]["row"],
                **result["largest_residual"],
            }
            results.append(result)
        return results

    return [row for expert in range(layout.experts) for row in one_expert(expert)]


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


def _saved_tensor(
    saved: SavedTensor, base: Tensor | None, merged: Tensor, elements: int
) -> dict[str, Any]:
    """Full trained state is an absolute target, never a low-rank update."""
    target = saved.tensor
    _compatible(target, merged, role="Saved-module")
    floating = target.dtype in FLOAT_DTYPES
    unavailable = (
        "absent"
        if base is None
        else "shape-changed"
        if base.shape != merged.shape
        else "dtype-category-changed"
        if (base.dtype in FLOAT_DTYPES) != floating
        else "integer-dtype-changed"
        if not floating and base.dtype != merged.dtype
        else None
    )
    comparable = unavailable is None
    # A replacement with a different shape still cannot hide nonfinite base input.
    if base is not None and not comparable and base.dtype in FLOAT_DTYPES:
        for start in range(0, base.elements, elements):
            base.floats(start, min(elements, base.elements - start))
    norms = {name: Squares() for name in ("target", "error", "target_cast_drift")}
    mismatches = changed = intended = unchanged = 0
    first = None
    for start in range(0, target.elements, elements):
        count = min(elements, target.elements - start)
        if floating:
            original = target.floats(start, count)
            expected = cast(original, merged.dtype)
            if not np.isfinite(expected).all():
                raise AuditError(
                    f"Saved module conversion overflows the merged dtype: {saved.weight}"
                )
            after = merged.floats(start, count)
            norms["target"].add(expected)
            norms["error"].add(after - expected)
            norms["target_cast_drift"].add(expected - original)
        else:
            # Preserve integer identity above 2**53, including scalar buffers.
            expected = target.raw(start, count)
            after = merged.raw(start, count)
        different = after != expected
        mismatches += int(np.count_nonzero(different))
        if first is None and different.any():
            index = int(np.flatnonzero(different)[0])
            first = {
                "flat_index": start + index,
                "cast_target": expected[index].item(),
                "merged": after[index].item(),
            }
        if comparable:
            assert base is not None
            if floating:
                baseline = cast(base.floats(start, count), merged.dtype)
                if not np.isfinite(baseline).all():
                    raise AuditError(f"Base conversion overflows the merged dtype: {saved.weight}")
            else:
                baseline = base.raw(start, count)
            intended_change = expected != baseline
            same_as_base = after == baseline
            changed += int(np.count_nonzero(~same_as_base))
            intended += int(np.count_nonzero(intended_change))
            unchanged += int(np.count_nonzero(intended_change & same_as_base))
    measured = {name: value.norm() for name, value in norms.items()} if floating else None
    return {
        "module": saved.module,
        "weight": saved.weight,
        "shape": list(merged.shape),
        "elements": merged.elements,
        "adapter_dtype": target.dtype,
        "merged_dtype": merged.dtype,
        "base_dtype": None if base is None else base.dtype,
        "base_shape": None if base is None else list(base.shape),
        "base_comparison_unavailable": unavailable,
        "status": "differs-from-saved-module" if mismatches else "matches-saved-module",
        "target_mismatches": mismatches,
        "norms": measured,
        "relative_error": (
            _ratio(measured["error"], measured["target"]) if measured is not None else None
        ),
        "changed_from_base_elements": changed if comparable else None,
        "expected_change_elements": intended if comparable else None,
        "unchanged_expected_change_elements": unchanged if comparable else None,
        "first_mismatch": first,
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
    full = saved_modules(update, before, after)
    replacements = {saved.weight: saved for saved in full.tensors}
    missing = sorted(before.tensors.keys() - after.tensors.keys())
    extra = sorted(after.tensors.keys() - before.tensors.keys() - replacements.keys())
    if missing or extra:
        raise AuditError(
            f"Base/merged tensor inventories differ; missing={missing[:5]}, extra={extra[:5]}"
        )
    pairs = adapter_pairs(
        update, before, full_weights=frozenset(saved.tensor.name for saved in full.tensors)
    )
    for pair in pairs:
        physical_weights = (
            {matrix.weight for matrix in pair.expert_layout.matrices}
            if pair.expert_layout is not None
            else {pair.weight}
        )
        if physical_weights & replacements.keys():
            raise AuditError(f"LoRA target overlaps a full saved module: {pair.module}")
        if any(
            pair.module == saved.module
            or pair.module.startswith(saved.module + ".")
            or saved.module.startswith(pair.module + ".")
            for saved in full.tensors
        ):
            raise AuditError(f"LoRA target overlaps a full saved module: {pair.module}")
    by_weight = {pair.weight: pair for pair in pairs if pair.expert_layout is None}
    parameter_pairs = [pair for pair in pairs if pair.expert_layout is not None]
    by_parameter_weight = {}
    for pair in parameter_pairs:
        assert pair.expert_layout is not None
        for matrix in pair.expert_layout.matrices:
            if matrix.weight in by_parameter_weight:
                raise AuditError(f"Overlapping expert parameter targets: {matrix.weight}")
            by_parameter_weight[matrix.weight] = pair
    measured_parameters: set[str] = set()
    by_bias = {pair.module + ".bias": pair for pair in pairs if pair.bias is not None}
    # Workspace target for the several simultaneous binary64 blocks. A whole
    # row and rank-wide factor row are the minimum; the fixed factor has a separate limit.
    elements = max(1, block_mib * 1024 * 1024 // (8 * 10))
    layers = []
    biases = []
    full_results = []
    unrelated = []
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            for name, counterpart in sorted(after.tensors.items()):
                if name in replacements:
                    full_results.append(
                        _saved_tensor(
                            replacements[name], before.tensors.get(name), counterpart, elements
                        )
                    )
                    continue
                tensor = before.tensors[name]
                _compatible(tensor, counterpart)
                if name in by_bias:
                    biases.append(
                        _layer(by_bias[name], tensor, counterpart, elements, 0, bias=True)
                    )
                elif name in by_parameter_weight:
                    pair = by_parameter_weight[name]
                    if pair.weight not in measured_parameters:
                        layers.extend(
                            _expert_parameter_layers(
                                pair, before, after, elements, max_adapter_mib * 1024 * 1024
                            )
                        )
                        measured_parameters.add(pair.weight)
                elif name in by_weight:
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
    if parameter_pairs:
        layers.sort(key=lambda row: row["weight"])
    # All original files, including configs and indexes, are fingerprinted.
    fingerprints = {role: checkpoint.fingerprints() for role, checkpoint in inputs.items()}
    for checkpoint in inputs.values():
        checkpoint.verify_stable()
    reference, unchanged_reference = Squares(), Squares()
    reference.add(np.array([layer["norms"]["delta"] for layer in layers]))
    unchanged_reference.add(np.array([layer["norms"]["unchanged_reference"] for layer in layers]))
    return {
        "schema_version": 1,
        "arithmetic": {
            **CONTRACT,
            **(
                {
                    "expert_parameter_reference": (
                        "one rank-r product per expert; A rows group experts, B columns "
                        "interleave experts. Native HF serialization splits fused gate/up "
                        "into separate per-expert Linear weights, preserving exact saved keys "
                        "and row/column coordinates. B and one expert A slice are retained "
                        "under the adapter memory cap"
                    ),
                }
                if any(pair.expert_layout is not None for pair in pairs)
                else {}
            ),
            **(
                {
                    "lora_bias_reference": (
                        "delta_bias = saved lora_B.bias_float64 * scale; "
                        "reference_bias = round_to_merged_dtype(base_bias_float64 + delta_bias); "
                        "reported separately from matrix updates and full saved-module biases"
                    ),
                }
                if biases
                else {}
            ),
            **(
                {
                    "conv1d_reference": (
                        "delta = (B_float64 @ A_float64).T * scale; saved weight rows "
                        "are inputs, columns are outputs; standard GPT module paths "
                        "resolved from declared model metadata, not a global orientation flag"
                    ),
                }
                if any(pair.conv1d_evidence for pair in pairs)
                else {}
            ),
            **(
                {
                    "embedding_reference": (
                        "delta = (B_float64 @ A_float64).T * scale; "
                        "weight rows are token indices, columns are embedding dimensions; "
                        "A columns are streamed, B is retained in full"
                    ),
                    "scope": (
                        "saved Linear LoRA/DoRA and Embedding LoRA adapters only; "
                        "no architecture execution or quality estimate"
                    ),
                }
                if any(pair.embedding for pair in pairs)
                else {}
            ),
            **(
                {
                    "saved_modules": (
                        "absolute saved state, rounded to merged dtype; numeric equality, "
                        "not a LoRA delta; integer/bool values require identical dtype and "
                        "exact values without float conversion; original base comparisons "
                        "are unavailable for absent or incompatible original tensors"
                    ),
                    "scope": (
                        "saved Linear LoRA/DoRA, Embedding LoRA and explicit full modules; "
                        "no architecture execution or quality estimate"
                    ),
                }
                if full_results
                else {}
            ),
            **(
                {
                    "adapter_base_snapshots": (
                        "saved <module>.base_layer.weight must equal the supplied original base "
                        "rounded to snapshot dtype; not a replacement weight or inferred base"
                    )
                }
                if any(pair.base_snapshot is not None for pair in pairs)
                else {}
            ),
            **(
                {
                    "dora_rows": (
                        "saved full-row norm / abs(saved magnitude), retained in log2 form; "
                        "direction cosine compares saved and float64 full-reference rows, "
                        "including the magnitude sign; zero rows have no norm ratio or direction"
                    )
                }
                if any("dora_rows" in layer for layer in layers)
                else {}
            ),
            **(
                {
                    "scope": (
                        "saved Linear LoRA/DoRA, standard GPT Conv1D LoRA, Embedding LoRA "
                        "and explicit full modules; no architecture execution or quality estimate"
                    )
                }
                if any(pair.conv1d_evidence for pair in pairs)
                else {}
            ),
            **(
                {
                    "scope": (
                        "saved Linear LoRA/DoRA, standard GPT Conv1D LoRA, Embedding LoRA, "
                        "scaled LoRA B bias and explicit full modules; "
                        "no architecture execution or quality estimate"
                    )
                }
                if biases
                else {}
            ),
            **(
                {
                    "scope": (
                        "saved Linear LoRA/DoRA, standard GPT Conv1D LoRA, Embedding LoRA, "
                        "explicit full modules and declared Qwen3 MoE parameter layouts; "
                        "no architecture execution or quality estimate"
                    )
                }
                if parameter_pairs
                else {}
            ),
        },
        "inputs": fingerprints,
        "settings": {"block_mib": block_mib, "max_adapter_mib": max_adapter_mib},
        "summary": {
            "adapted_tensors": len(layers),
            "checked_tensors": len(after.tensors),
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
            "unchanged_reference_energy_fraction": _energy_fraction(reference, unchanged_reference),
            "zero_update_tensors": sum(layer["status"] == "zero-update" for layer in layers),
            "no_observed_update_tensors": sum(
                layer["status"] == "no-observed-update" for layer in layers
            ),
            "unadapted_changed_tensors": len(unrelated),
            "unadapted_changed_elements": sum(layer["changed_elements"] for layer in unrelated),
            **(
                {
                    "logical_expert_parameters": len(parameter_pairs),
                    "expert_weight_tensors": sum("expert_parameter" in row for row in layers),
                    "expert_no_observed_update_tensors": sum(
                        row["status"] == "no-observed-update"
                        for row in layers
                        if "expert_parameter" in row
                    ),
                    "expert_zero_update_tensors": sum(
                        row["status"] == "zero-update"
                        for row in layers
                        if "expert_parameter" in row
                    ),
                }
                if parameter_pairs
                else {}
            ),
            **(
                {
                    "lora_bias_tensors": len(biases),
                    "lora_bias_elements": sum(row["elements"] for row in biases),
                    "lora_bias_intended_nonzero_elements": sum(
                        row["intended_nonzero_elements"] for row in biases
                    ),
                    "lora_bias_unchanged_nonzero_update_elements": sum(
                        row["unchanged_nonzero_update_elements"] for row in biases
                    ),
                    "lora_bias_reference_merge_mismatches": sum(
                        row["reference_merge_mismatches"] for row in biases
                    ),
                }
                if biases
                else {}
            ),
            **(
                {
                    "saved_module_tensors": len(full_results),
                    "saved_module_elements": sum(row["elements"] for row in full_results),
                    "saved_module_target_mismatches": sum(
                        row["target_mismatches"] for row in full_results
                    ),
                    "saved_module_base_comparable_tensors": sum(
                        row["base_comparison_unavailable"] is None for row in full_results
                    ),
                }
                if full_results
                else {}
            ),
        },
        "layers": layers,
        **({"lora_biases": biases} if biases else {}),
        **(
            {
                "saved_modules": full_results,
                "unmatched_saved_module_selectors": list(full.unmatched_selectors),
            }
            if full_results
            else {}
        ),
        "unadapted_changes": unrelated,
        "limitations": [
            "Supplied base identity and training provenance are not independently established.",
            "Equal weights do not distinguish rounding from an omitted or wrong adapter.",
            "The float64 reference is numerical; it is not a PEFT merge emulator.",
            "DoRA normalizes rows of the supplied original base plus its directional update.",
            "Tensor shapes and a global adapter flag do not establish every module's type.",
            "Operation order and matmul precision can differ from the producer, notably at ties.",
            "Weight-space changes and residuals do not establish generation or task quality.",
            "Missing tensors, unsupported variants and nonfinite inputs refuse the whole audit.",
            *(
                [
                    "Full modules are absolute adapter state, not inferred additive updates.",
                    "Absent original aliases and resized original values are not reconstructed.",
                    "Unmatched selectors have no observable tensor prefix; "
                    "module existence is unknown.",
                ]
                if full_results
                else []
            ),
        ],
    }
