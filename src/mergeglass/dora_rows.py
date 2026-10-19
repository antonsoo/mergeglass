"""Full-row DoRA norm and direction observations, without an epsilon floor.

Factored lengths and logarithmic ratios retain observations outside float64's
ordinary norm/factor range. Zero rows have explicit counts, never fake cosines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .storage import FloatArray


def _unit_rows(values: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Unit directions and factored lengths, including zero and extreme rows."""
    scale = np.max(np.abs(values), axis=1)
    units = np.zeros_like(values)
    np.divide(values, scale[:, None], out=units, where=scale[:, None] != 0)
    length = np.sqrt(np.einsum("ij,ij->i", units, units))
    np.divide(units, length[:, None], out=units, where=length[:, None] != 0)
    return units, scale, length


def _factor(log2: float) -> float | None:
    """Keep a finite logarithmic observation when its factor cannot be stored."""
    try:
        value = 2.0**log2
    except OverflowError:
        return None
    return value if math.isfinite(value) and value != 0 else None


@dataclass
class DoraRows:
    rows: int = 0
    zero_magnitude: int = 0
    negative_magnitude: int = 0
    zero_saved: int = 0
    nonzero_magnitude_zero_saved: int = 0
    zero_magnitude_nonzero_saved: int = 0
    norm_pairs: int = 0
    norm_log2_min: float = math.inf
    norm_log2_max: float = -math.inf
    norm_log2_abs_total: float = 0.0
    largest_norm_ratio_error: dict[str, Any] | None = None
    largest_log2_size: float = -1.0
    direction_pairs: int = 0
    direction_min: float = 1.0
    direction_max: float = -1.0
    direction_total: float = 0.0
    zero_reference: int = 0

    def add(
        self, row: int, magnitude: FloatArray, reference: FloatArray | None, after: FloatArray
    ) -> None:
        units, scale, length = _unit_rows(after)
        self.rows += len(magnitude)
        target = np.abs(magnitude)
        self.zero_magnitude += int(np.count_nonzero(target == 0))
        self.negative_magnitude += int(np.count_nonzero(magnitude < 0))
        self.zero_saved += int(np.count_nonzero(scale == 0))
        self.nonzero_magnitude_zero_saved += int(np.count_nonzero((target != 0) & (scale == 0)))
        self.zero_magnitude_nonzero_saved += int(np.count_nonzero((target == 0) & (scale != 0)))
        comparable = np.flatnonzero((target != 0) & (scale != 0))
        if comparable.size:
            a, a_exponent = np.frexp(scale[comparable])
            b, b_exponent = np.frexp(target[comparable])
            significand = (a / b) * length[comparable]
            exponent = a_exponent - b_exponent
            logs = np.log2(significand) + exponent
            # Close factors should not lose their tiny error by subtracting
            # logs of huge/tiny row norms. The factored ratio stays bounded.
            nearby = np.abs(exponent) < 500
            logs[nearby] = np.log2(np.ldexp(significand[nearby], exponent[nearby]))
            self.norm_pairs += len(comparable)
            self.norm_log2_min = min(self.norm_log2_min, float(np.min(logs)))
            self.norm_log2_max = max(self.norm_log2_max, float(np.max(logs)))
            self.norm_log2_abs_total = math.fsum(
                (self.norm_log2_abs_total, float(np.sum(np.abs(logs))))
            )
            index = int(np.argmax(np.abs(logs)))
            value = float(logs[index])
            if abs(value) > self.largest_log2_size:
                self.largest_log2_size = abs(value)
                local = int(comparable[index])
                observed_norm = float(scale[local]) * float(length[local])
                self.largest_norm_ratio_error = {
                    "row": row + local,
                    "saved_magnitude": float(magnitude[local]),
                    "target_row_norm": float(target[local]),
                    "saved_row_norm": observed_norm if math.isfinite(observed_norm) else None,
                    "saved_row_norm_scale": float(scale[local]),
                    "saved_row_norm_scaled_length": float(length[local]),
                    "row_norm_factor": _factor(value),
                    "log2_row_norm_factor": value,
                }
        if reference is None:
            return
        reference_units, reference_scale, _ = _unit_rows(reference)
        self.zero_reference += int(np.count_nonzero(reference_scale == 0))
        directions = (scale != 0) & (reference_scale != 0)
        if directions.any():
            cosine = np.clip(np.einsum("ij,ij->i", units, reference_units)[directions], -1, 1)
            self.direction_pairs += len(cosine)
            self.direction_min = min(self.direction_min, float(np.min(cosine)))
            self.direction_max = max(self.direction_max, float(np.max(cosine)))
            self.direction_total = math.fsum((self.direction_total, float(np.sum(cosine))))

    def result(self, *, include_direction: bool = True) -> dict[str, Any]:
        result = {
            "rows": self.rows,
            "zero_magnitude_rows": self.zero_magnitude,
            "negative_magnitude_rows": self.negative_magnitude,
            "zero_saved_rows": self.zero_saved,
            "nonzero_magnitude_zero_saved_rows": self.nonzero_magnitude_zero_saved,
            "zero_magnitude_nonzero_saved_rows": self.zero_magnitude_nonzero_saved,
            "norm_comparable_rows": self.norm_pairs,
            "row_norm_factor_range": (
                [_factor(self.norm_log2_min), _factor(self.norm_log2_max)]
                if self.norm_pairs
                else None
            ),
            "log2_row_norm_factor_range": (
                [self.norm_log2_min, self.norm_log2_max] if self.norm_pairs else None
            ),
            "mean_absolute_log2_row_norm_factor": (
                self.norm_log2_abs_total / self.norm_pairs if self.norm_pairs else None
            ),
            "largest_nonzero_log2_norm_error": self.largest_norm_ratio_error,
        }
        if not include_direction:
            return result
        return {
            **result,
            "zero_numerical_reference_rows": self.zero_reference,
            "direction_comparable_rows": self.direction_pairs,
            "direction_cosine_range": (
                [self.direction_min, self.direction_max] if self.direction_pairs else None
            ),
            "mean_direction_cosine": (
                self.direction_total / self.direction_pairs if self.direction_pairs else None
            ),
        }
