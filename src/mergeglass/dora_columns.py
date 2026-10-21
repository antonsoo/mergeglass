"""Embedding DoRA normalizes complete vocabulary columns, across row blocks.

Keep factored lengths and aligned inner products per embedding dimension.
No full vocabulary vector, product or transposed checkpoint is retained.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .dora_rows import DoraRows
from .storage import FloatArray


class ColumnNorms:
    def __init__(self, columns: int) -> None:
        self.scale = np.zeros(columns, dtype=np.float64)
        self.total = np.zeros(columns, dtype=np.float64)

    def add(self, values: FloatArray) -> tuple[FloatArray, FloatArray]:
        maximum = np.maximum(self.scale, np.max(np.abs(values), axis=0))
        ratio = np.zeros_like(maximum)
        np.divide(self.scale, maximum, out=ratio, where=maximum != 0)
        normalized = np.zeros_like(values)
        np.divide(values, maximum, out=normalized, where=maximum != 0)
        self.total = self.total * ratio**2 + np.einsum("ij,ij->j", normalized, normalized)
        self.scale = maximum
        return ratio, normalized

    def normalize(self, values: FloatArray, magnitude: FloatArray) -> FloatArray:
        # The caller rejects zero direction columns. Dividing before multiplying
        # by magnitude avoids overflow in either a raw norm or magnitude/norm.
        result = values / self.scale
        result /= np.sqrt(self.total)
        result *= magnitude
        return result


class DoraColumns:
    def __init__(self, columns: int) -> None:
        self.saved = ColumnNorms(columns)
        self.reference = ColumnNorms(columns)
        self.cross = np.zeros(columns, dtype=np.float64)

    def add(self, after: FloatArray, reference: FloatArray | None = None) -> None:
        saved_ratio, saved = self.saved.add(after)
        if reference is not None:
            reference_ratio, expected = self.reference.add(reference)
            self.cross = self.cross * saved_ratio * reference_ratio + np.einsum(
                "ij,ij->j", saved, expected
            )

    def result(self, magnitude: FloatArray, *, include_direction: bool = True) -> dict[str, Any]:
        vectors = DoraRows()
        vectors.add_norms(0, magnitude, self.saved.scale, np.sqrt(self.saved.total))
        if include_direction:
            vectors.zero_reference = int(np.count_nonzero(self.reference.scale == 0))
            comparable = (self.saved.scale != 0) & (self.reference.scale != 0)
            if comparable.any():
                cosine = np.clip(
                    self.cross[comparable]
                    / np.sqrt(self.saved.total[comparable])
                    / np.sqrt(self.reference.total[comparable]),
                    -1,
                    1,
                )
                vectors.direction_pairs = len(cosine)
                vectors.direction_min = float(np.min(cosine))
                vectors.direction_max = float(np.max(cosine))
                vectors.direction_total = float(np.sum(cosine))
        # Reuse vector diagnostics, retaining the actual axis in their schema.
        result = {
            key.replace("row", "column"): value
            for key, value in vectors.result(include_direction=include_direction).items()
        }
        witness = result["largest_nonzero_log2_norm_error"]
        if witness is not None:
            result["largest_nonzero_log2_norm_error"] = {
                key.replace("row", "column"): value for key, value in witness.items()
            }
        return result
