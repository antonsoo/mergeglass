"""80-digit scalar DoRA reference; no NumPy/torch products or vector norms."""

from __future__ import annotations

import json
import math
import random
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from mergeglass import audit


def main() -> None:
    root, destination = map(Path, sys.argv[1:])
    root.mkdir(parents=True)
    rng = random.Random(71120)
    results = []
    with localcontext() as context:
        context.prec = 80
        for case in range(36):
            rows, columns, rank = 9, 13, 3

            def matrix(height: int, width: int, divisor: int) -> list[list[Decimal]]:
                return [
                    [Decimal(rng.randrange(-16, 17)) / divisor for _ in range(width)]
                    for _ in range(height)
                ]

            original = matrix(rows, columns, 8)
            a, b = matrix(rank, columns, 32), matrix(rows, rank, 32)
            magnitude = [Decimal(rng.randrange(-20, 21)) / 8 for _ in range(rows)]
            scale = [-2, 0, 1, 3][case % 4]
            reference = []
            for row in range(rows):
                direction = [
                    original[row][col] + scale * sum(b[row][k] * a[k][col] for k in range(rank))
                    for col in range(columns)
                ]
                norm = sum(value * value for value in direction).sqrt()
                assert norm != 0
                reference.append([value * magnitude[row] / norm for value in direction])
            # Three saved artifacts: rounded intended weights, magnitude halved,
            # or a copied base. These are constructed controls, not native saves.
            saved = [reference, [[v / 2 for v in row] for row in reference], original][case % 3]
            after = np.array(saved, dtype=np.float32)
            directory = root / str(case)
            for name in ["base", "adapter", "merged"]:
                (directory / name).mkdir(parents=True)
            save_file(
                {"linear.weight": np.array(original, dtype=np.float64)},
                directory / "base/model.safetensors",
            )
            save_file({"linear.weight": after}, directory / "merged/model.safetensors")
            save_file(
                {
                    "base_model.model.linear.lora_A.weight": np.array(a, dtype=np.float64),
                    "base_model.model.linear.lora_B.weight": np.array(b, dtype=np.float64),
                    "base_model.model.linear.lora_magnitude_vector": np.array(
                        magnitude, dtype=np.float64
                    ),
                },
                directory / "adapter/adapter_model.safetensors",
            )
            (directory / "adapter/adapter_config.json").write_text(
                json.dumps(
                    {"peft_type": "LORA", "r": rank, "lora_alpha": rank * scale, "use_dora": True}
                )
            )
            report = audit(*(directory / name for name in ["base", "adapter", "merged"]))
            layer = report["layers"][0]
            delta = []
            observed = []
            residual = []
            for row in range(rows):
                for col in range(columns):
                    d = reference[row][col] - original[row][col]
                    o = Decimal.from_float(float(after[row, col])) - original[row][col]
                    delta.append(d)
                    observed.append(o)
                    residual.append(o - d)

            def norm(values: list[Decimal]) -> float:
                return float(sum((v * v for v in values), Decimal(0)).sqrt())

            expected = {
                "delta": norm(delta),
                "observed": norm(observed),
                "residual": norm(residual),
                "base_cast_drift": 0.0,
                "unchanged_reference": norm(
                    [d for d, o in zip(delta, observed, strict=True) if o == 0]
                ),
            }
            for key, value in expected.items():
                assert math.isclose(layer["norms"][key], value, rel_tol=1e-8, abs_tol=2e-15), (
                    case,
                    key,
                    layer["norms"][key],
                    value,
                )
            dot = sum(d * o for d, o in zip(delta, observed, strict=True))
            gain = float(dot / sum(d * d for d in delta))
            assert math.isclose(layer["projection_gain"], gain, rel_tol=1e-12, abs_tol=1e-15)
            assert layer["intended_nonzero_elements"] == sum(d != 0 for d in delta)
            assert layer["changed_elements"] == sum(o != 0 for o in observed)
            assert layer["unchanged_nonzero_update_elements"] == sum(
                d != 0 and o == 0 for d, o in zip(delta, observed, strict=True)
            )
            point = layer["largest_residual"]
            index = point["row"] * columns + point["column"]
            assert abs(point["intended_delta"] - float(delta[index])) < 2e-15
            assert abs(abs(point["residual"]) - float(max(abs(v) for v in residual))) < 2e-15
            results.append(
                {
                    "case": case,
                    "scale": scale,
                    "saved_control": ["rounded intended", "halved magnitude", "base copy"][
                        case % 3
                    ],
                    "maximum_norm_absolute_difference": max(
                        abs(layer["norms"][key] - value) for key, value in expected.items()
                    ),
                    "projection_gain": layer["projection_gain"],
                    "reference_gain": gain,
                    "changed": layer["changed_elements"],
                    "nonzero": layer["intended_nonzero_elements"],
                }
            )
    payload = {
        "synthetic": True,
        "seed": 71120,
        "oracle": "80-digit Decimal scalar dot products, row norms, complete updates and metrics",
        "shape": [9, 13],
        "rank": 3,
        "cases": len(results),
        "compared_coordinates": len(results) * 9 * 13,
        "results": results,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}))


if __name__ == "__main__":
    main()
