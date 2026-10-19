"""Independently check full-row DoRA observations using dense torch norms.

These retained producer inputs are moderate finite values. Separate closed-form
tests exercise zero/sign cases and norms/factors outside float64 range.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import torch
from peft import LoraConfig
from peft.utils.other import get_pattern_key
from verify import load

import mergeglass
from mergeglass import audit


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-8, abs_tol=2e-14), (actual, expected)


def verify(base: Path, adapter: Path, merged: Path) -> dict:
    report = audit(base, adapter, merged, block_mib=1)
    before, factors, after = load(base), load(adapter, adapter=True), load(merged)
    config = LoraConfig.from_pretrained(adapter)
    observations = []
    for layer in report["layers"]:
        name = layer["module"]
        key = layer["weight"]
        prefix = f"base_model.model.{name}."
        magnitude = factors[prefix + "lora_magnitude_vector"].double()
        a, b = (factors[prefix + f"lora_{side}.weight"].double() for side in ("A", "B"))
        rank = config.rank_pattern.get(get_pattern_key(list(config.rank_pattern), name), config.r)
        alpha = config.alpha_pattern.get(
            get_pattern_key(list(config.alpha_pattern), name), config.lora_alpha
        )
        scale = alpha / (math.sqrt(rank) if config.use_rslora else rank)
        direction = before[key].double() + (b @ a) * scale
        reference = direction * (magnitude / torch.linalg.vector_norm(direction, dim=1))[:, None]
        saved = after[key].double()
        saved_norm = torch.linalg.vector_norm(saved, dim=1)
        target = magnitude.abs()
        comparable = (target != 0) & (saved_norm != 0)
        ratios = saved_norm[comparable] / target[comparable]
        logs = ratios.log2()
        rows = layer["dora_rows"]
        assert rows["rows"] == len(magnitude)
        assert rows["zero_magnitude_rows"] == int((target == 0).sum())
        assert rows["negative_magnitude_rows"] == int((magnitude < 0).sum())
        assert rows["zero_saved_rows"] == int((saved_norm == 0).sum())
        assert rows["nonzero_magnitude_zero_saved_rows"] == int(
            ((target != 0) & (saved_norm == 0)).sum()
        )
        assert rows["zero_magnitude_nonzero_saved_rows"] == int(
            ((target == 0) & (saved_norm != 0)).sum()
        )
        assert rows["norm_comparable_rows"] == int(comparable.sum())
        if comparable.any():
            for actual, expected in zip(
                rows["row_norm_factor_range"],
                [float(ratios.min()), float(ratios.max())],
                strict=True,
            ):
                close(actual, expected)
            for actual, expected in zip(
                rows["log2_row_norm_factor_range"],
                [float(logs.min()), float(logs.max())],
                strict=True,
            ):
                close(actual, expected)
            close(rows["mean_absolute_log2_row_norm_factor"], float(logs.abs().mean()))
            witness = rows["largest_nonzero_log2_norm_error"]
            index = witness["row"]
            assert bool(comparable[index])
            ratio = float(saved_norm[index] / target[index])
            close(witness["row_norm_factor"], ratio)
            close(witness["saved_row_norm"], float(saved_norm[index]))
            close(abs(witness["log2_row_norm_factor"]), float(logs.abs().max()))
            assert witness["saved_magnitude"] == float(magnitude[index])
        else:
            assert rows["row_norm_factor_range"] is None
            assert rows["log2_row_norm_factor_range"] is None
            assert rows["largest_nonzero_log2_norm_error"] is None
        reference_norm = torch.linalg.vector_norm(reference, dim=1)
        comparable_directions = (reference_norm != 0) & (saved_norm != 0)
        cosine = torch.nn.functional.cosine_similarity(
            saved[comparable_directions], reference[comparable_directions], dim=1, eps=0
        ).clamp(-1, 1)
        assert rows["direction_comparable_rows"] == len(cosine)
        assert rows["zero_numerical_reference_rows"] == int((reference_norm == 0).sum())
        if len(cosine):
            close(rows["direction_cosine_range"][0], float(cosine.min()))
            close(rows["direction_cosine_range"][1], float(cosine.max()))
            close(rows["mean_direction_cosine"], float(cosine.mean()))
        else:
            assert rows["direction_cosine_range"] is None
            assert rows["mean_direction_cosine"] is None
        observations.append({"weight": key, "dora_rows": rows})
    return {"case": merged.name, "inputs": report["inputs"], "layers": observations}


def check_files(root: Path, files: list[dict]) -> None:
    for item in files:
        path = root / item["file"]
        data = path.read_bytes()
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("public", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    implementation = Path(mergeglass.__file__).parent
    frozen_paths = [
        Path(__file__),
        Path(__file__).with_name("verify.py"),
        *sorted(implementation.glob("*.py")),
    ]
    frozen = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in frozen_paths}
    torch.set_num_threads(1)
    producer = json.loads((args.native / "producer.json").read_text())
    check_files(args.native, producer["files"])
    sources = json.loads(Path(__file__).with_name("public-sources.json").read_text())
    for source in sources["sources"]:
        check_files(args.public / source["role"], source["files"])
    public_merges = json.loads(
        Path(__file__).with_name("results").joinpath("public.json").read_text()
    )
    for merge in public_merges["merges"]:
        check_files(args.public / merge["case"], merge["files"])
    native = [
        verify(args.native / "base", args.native / "adapter", args.native / case)
        for case in producer["cases"]
    ]
    native.append(
        verify(
            args.native / "transformer-base",
            args.native / "transformer-adapter",
            args.native / "transformer-merged",
        )
    )
    public = [
        verify(args.public / "base", args.public / "adapter", args.public / case)
        for case in ["published", *[m["case"] for m in public_merges["merges"]]]
    ]
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == checksum
        for path, checksum in frozen.items()
    ), "Study or auditor source changed during replay"
    result = {
        "oracle": "dense torch.float64 vector_norm and cosine_similarity(eps=0)",
        "tolerance": {"relative": 1e-8, "absolute": 2e-14},
        "script_sha256": frozen[Path(__file__)],
        "helper_sha256": frozen[Path(__file__).with_name("verify.py")],
        "auditor_sources": {path.name: frozen[path] for path in implementation.glob("*.py")},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "peft", "numpy", "safetensors")
        },
        "native": native,
        "public": public,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "native_cases": len(native),
                "public_cases": len(public),
                "matrices": sum(len(c["layers"]) for c in native + public),
                "rows": sum(
                    layer["dora_rows"]["rows"]
                    for case in native + public
                    for layer in case["layers"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
