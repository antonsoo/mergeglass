"""Recompute fixed native vocabulary probabilities with NumPy extended precision.

No Torch, PEFT, model weights or inference is needed. The arrays contain actual
native logits, not saved probabilities; the independent normalization is new.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    receipt = args.capture / "capture.json"
    logits = args.capture / "logits.npz"
    assert args.output.resolve() not in {receipt.resolve(), logits.resolve()}
    observed = json.loads(receipt.read_text())
    assert sha(logits) == observed["logits_sha256"]
    assert np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant
    errors, comparisons = [], []

    def close(native: float, independent: float) -> None:
        error = abs(native - independent)
        errors.append(error)
        assert math.isclose(native, independent, rel_tol=5e-12, abs_tol=2e-14), (
            native,
            independent,
            error,
        )

    with np.load(logits, allow_pickle=False) as archive:
        expected = {p["array"] for s in observed["samples"] for p in s["profiles"].values()}
        assert set(archive.files) == expected
        for sample in observed["samples"]:
            probabilities, log_probabilities = {}, {}
            targets = np.asarray(sample["target_ids"], dtype=np.int64)
            for profile, native in sample["profiles"].items():
                values = archive[native["array"]]
                assert values.dtype == np.float32 and values.ndim == 2
                assert values.shape[0] == len(targets) and np.isfinite(values).all()
                values = values.astype(np.longdouble)
                centered = values - values.max(axis=1, keepdims=True)
                logp = centered - np.log(np.exp(centered).sum(axis=1, keepdims=True))
                log_probabilities[profile] = logp
                probabilities[profile] = np.exp(logp)
                nll = -logp[np.arange(len(targets)), targets]
                for a, b in zip(native["target_nll_nats"], nll, strict=True):
                    close(a, float(b))
                assert np.argmax(logp, axis=1).tolist() == native["argmax_ids"]
            for reference, candidates in sample["drift"].items():
                p, logp = probabilities[reference], log_probabilities[reference]
                for candidate, native in candidates.items():
                    q, logq = probabilities[candidate], log_probabilities[candidate]
                    kl = np.sum(p * (logp - logq), axis=1)
                    tv = np.sum(np.abs(p - q), axis=1) / 2
                    extended = {
                        "mean_kl_nats": float(np.maximum(kl, 0).mean()),
                        "max_kl_nats": float(max(kl.max(), 0)),
                        "raw_min_kl_nats": float(kl.min()),
                        "mean_total_variation": float(tv.mean()),
                        "max_total_variation": float(tv.max()),
                    }
                    for field, value in extended.items():
                        close(native[field], value)
                    assert native["argmax_disagreements"] == int(
                        np.sum(np.argmax(logp, axis=1) != np.argmax(logq, axis=1))
                    )
                    assert native["positions"] == len(targets)
                    comparisons.append(
                        {
                            "id": sample["id"],
                            "reference": reference,
                            "candidate": candidate,
                            "extended": extended,
                            "native": native,
                        }
                    )
    assert sha(logits) == observed["logits_sha256"]
    args.output.write_text(
        json.dumps(
            {
                "scope": "Two fixed questions, first/middle/EOS positions. Numerical probability "
                "verification, not model quality or full-cohort distribution coverage.",
                "python": platform.python_version(),
                "numpy": np.__version__,
                "extended_mantissa_bits_excluding_implicit_bit": int(np.finfo(np.longdouble).nmant),
                "script_sha256": sha(Path(__file__)),
                "capture_receipt_sha256": sha(receipt),
                "logits_sha256": sha(logits),
                "comparisons": comparisons,
                "all_selected_argmax_ids_match": True,
                "max_absolute_scalar_difference": max(errors),
                "tolerance": {"relative": 5e-12, "absolute": 2e-14},
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "probability_comparisons": len(comparisons),
                "max_absolute_scalar_difference": max(errors),
            }
        )
    )


if __name__ == "__main__":
    main()
