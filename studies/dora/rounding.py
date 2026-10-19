"""Check direct F64 rounding separately from the producer framework's casts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from verify import reference_cast, rounding_grid

from mergeglass.storage import cast


def main() -> None:
    torch.set_num_threads(1)
    results = []
    for name, dtype in [("BF16", torch.bfloat16), ("F16", torch.float16)]:
        grid, overflow = rounding_grid(dtype)
        positive = grid.numpy()
        low, high = positive[:-1], positive[1:]
        midpoint = (low + high) / 2
        tie = np.where(np.arange(len(low)) % 2 == 0, low, high)
        values = np.stack(
            [np.nextafter(midpoint, -np.inf), midpoint, np.nextafter(midpoint, np.inf)], axis=1
        ).ravel()
        expected = np.stack([low, tie, high], axis=1).ravel()
        values = np.concatenate(
            [values, [np.nextafter(overflow, 0), overflow, np.nextafter(overflow, np.inf)]]
        )
        expected = np.concatenate([expected, [positive[-1], np.inf, np.inf]])
        values, expected = np.concatenate([values, -values]), np.concatenate([expected, -expected])
        original = torch.from_numpy(values)
        independent = reference_cast(original, dtype).numpy()
        actual = cast(values, name)
        framework = original.to(dtype).double().numpy()
        intermediate = cast(values.astype(np.float32).astype(np.float64), name)
        assert np.array_equal(independent, expected)
        assert np.array_equal(actual, expected)
        assert np.array_equal(framework, intermediate)
        for result in [independent, actual]:
            assert not np.count_nonzero(
                (result == 0) & (np.signbit(result) != np.signbit(expected))
            )
        indices = np.flatnonzero(framework != expected)
        results.append(
            {
                "dtype": name,
                "cases": len(values),
                "auditor_disagreements": 0,
                "independent_grid_disagreements": 0,
                "torch_direct_cast_disagreements": len(indices),
                "torch_cast_matches_f32_intermediate_in_every_case": True,
                "examples": [
                    {
                        "input_hex": float(values[i]).hex(),
                        "expected": float(expected[i]),
                        "torch_cast": float(framework[i]),
                    }
                    for i in indices[:3]
                ],
            }
        )
    output = {
        "oracle": "every adjacent finite bit-pattern midpoint and both F64 neighbors, both signs",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "results": results,
    }
    destination = Path(sys.argv[1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
