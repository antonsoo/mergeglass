"""Compare two executed attention paths with identical named model values."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


def measure(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = np.load(reference_path, allow_pickle=False).astype(np.float64)
    candidate = np.load(candidate_path, allow_pickle=False).astype(np.float64)
    assert reference.shape == candidate.shape and reference.ndim == 2
    assert np.isfinite(reference).all() and np.isfinite(candidate).all()
    difference = candidate - reference
    before, after = reference.argmax(-1), candidate.argmax(-1)
    changed = np.flatnonzero(before != after)
    reference = np.exp(reference - reference.max(-1, keepdims=True))
    reference /= reference.sum(-1, keepdims=True)
    candidate = np.exp(candidate - candidate.max(-1, keepdims=True))
    candidate /= candidate.sum(-1, keepdims=True)
    variation = np.abs(reference - candidate).sum(-1) / 2
    assert np.all((variation >= 0) & (variation <= 1 + 1e-15))
    return {
        "positions": reference.shape[0],
        "vocabulary": reference.shape[1],
        "max_abs_logit_difference": float(np.max(np.abs(difference))),
        "logit_rmse": float(np.sqrt(np.mean(difference**2))),
        "changed_top1_positions": len(changed),
        "mean_probability_total_variation": float(variation.mean()),
        "max_probability_total_variation": float(variation.max()),
        "first_top1_changes": [
            {
                "position": int(position),
                "reference_token_id": int(before[position]),
                "candidate_token_id": int(after[position]),
            }
            for position in changed[:5]
        ],
    }


def same(actual: Any, expected: Any) -> None:
    if isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-15), (actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            same(actual[key], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            same(a, b)
    else:
        assert actual == expected, (actual, expected)


def aggregate(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    positions = sum(m["positions"] for m in measurements)
    return {
        "positions": positions,
        "changed_top1_positions": sum(m["changed_top1_positions"] for m in measurements),
        "mean_probability_total_variation": sum(
            m["mean_probability_total_variation"] * m["positions"] for m in measurements
        )
        / positions,
        "max_abs_logit_difference": max(m["max_abs_logit_difference"] for m in measurements),
    }


def verify_within_path(receipt: dict[str, Any], cache: Path) -> int:
    count = 0
    for case in receipt["comparisons"]:
        measurements = []
        for row in case["measurements"]:
            result = measure(
                cache / f"{case['reference']}-{row['probe']}.npy",
                cache / f"{case['candidate']}-{row['probe']}.npy",
            )
            result["probe"] = row["probe"]
            same(result, row)
            measurements.append(result)
            count += 1
        for key, value in aggregate(measurements).items():
            same(value, case[key])
    return count


def main() -> None:
    eager_path, sdpa_path, eager_cache, sdpa_cache, out = map(Path, sys.argv[1:])
    eager, sdpa = json.loads(eager_path.read_text()), json.loads(sdpa_path.read_text())
    for key in ("community_merge_receipt_sha256", "probes", "versions", "script_sha256"):
        assert eager[key] == sdpa[key], key
    assert eager["settings"]["attention"] == "eager"
    assert sdpa["settings"]["attention"] == "sdpa"
    assert {k: v for k, v in eager["settings"].items() if k != "attention"} == {
        k: v for k, v in sdpa["settings"].items() if k != "attention"
    }
    module_path = Path(__file__).with_name("run.py")
    assert hashlib.sha256(module_path.read_bytes()).hexdigest() == eager["script_sha256"]
    assert [p["name"] for p in eager["profiles"]] == [p["name"] for p in sdpa["profiles"]]
    checked_within = verify_within_path(eager, eager_cache) + verify_within_path(sdpa, sdpa_cache)
    cases = []
    for first, second in zip(eager["profiles"], sdpa["profiles"], strict=True):
        assert first["model_state_sha256"] == second["model_state_sha256"]
        assert first["adapter_dtypes"] == second["adapter_dtypes"]
        measurements = []
        for a, b in zip(first["outputs"], second["outputs"], strict=True):
            assert a["probe"] == b["probe"]
            assert a["shape"] == b["shape"]
            assert a["native_logit_dtype"] == b["native_logit_dtype"]
            for cache, row in ((eager_cache, a), (sdpa_cache, b)):
                path = cache / row["file"]
                assert path.stat().st_size == row["bytes"]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
            measurements.append(
                {
                    "probe": a["probe"],
                    **measure(eager_cache / a["file"], sdpa_cache / b["file"]),
                }
            )
        cases.append(
            {
                "profile": first["name"],
                "identical_model_state_sha256": first["model_state_sha256"],
                **aggregate(measurements),
                "measurements": measurements,
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "eager_receipt_sha256": hashlib.sha256(eager_path.read_bytes()).hexdigest(),
        "sdpa_receipt_sha256": hashlib.sha256(sdpa_path.read_bytes()).hexdigest(),
        "comparison_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "same IDs, versions, dtypes and named model values; eager versus SDPA on CPU",
        "independent_backend": "NumPy float64; no Torch/PEFT imports",
        "numpy": np.__version__,
        "checked_logit_arrays": sum(len(p["outputs"]) for p in eager["profiles"]) * 2,
        "checked_within_attention_measurements": checked_within,
        "cases": cases,
    }
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    for case in cases:
        print(
            case["profile"],
            "changed top1:",
            case["changed_top1_positions"],
            "mean TV:",
            case["mean_probability_total_variation"],
        )


if __name__ == "__main__":
    main()
