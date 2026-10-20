"""Public native layouts checked with full dense Torch and sampled Decimal sums."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import torch
from public_produce import file_hash
from verify import check_layer, load

import mergeglass
from mergeglass import AuditError, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--before", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    producer = json.loads((args.directory / "producer.json").read_text())
    for item in producer["files"]:
        path = args.directory / item["file"]
        assert path.stat().st_size == item["bytes"]
        assert file_hash(path) == item["sha256"]
    original, factors = (
        load(args.directory / "base"),
        load(args.directory / "adapter", "adapter_model"),
    )
    results = []
    for case in producer["cases"]:
        refusal = None
        report = None
        try:
            report = audit(
                args.directory / "base",
                args.directory / "adapter",
                args.directory / case["name"],
                block_mib=1,
            )
        except AuditError as exc:
            refusal = str(exc)
        if args.before:
            assert refusal is not None
            results.append({"case": case, "refusal": refusal})
            continue
        assert report is not None, refusal
        saved = load(args.directory / case["name"])
        independent = []
        for row in report["layers"]:
            module, key = row["module"], row["weight"]
            layout = producer["native_layouts"][module]
            assert layout["kind"] == "Conv1D" and layout["transpose"]
            assert row["weight_layout"] == "conv1d" and row["layout_evidence"]
            a = factors[f"base_model.model.{module}.lora_A.weight"].double()
            b = factors[f"base_model.model.{module}.lora_B.weight"].double()
            delta = (b @ a).T * layout["scale"]
            point = check_layer(row, original[key], saved[key], delta)
            coords = [
                (0, 0),
                (delta.shape[0] - 1, delta.shape[1] - 1),
                (row["largest_residual"]["row"], row["largest_residual"]["column"]),
            ]
            scalar = []
            with localcontext() as context:
                context.prec = 80
                for ins, outs in coords:
                    value = sum(
                        (
                            Decimal.from_float(float(a[k, ins]))
                            * Decimal.from_float(float(b[outs, k]))
                            for k in range(a.shape[0])
                        ),
                        Decimal(0),
                    ) * Decimal.from_float(layout["scale"])
                    assert math.isclose(
                        float(value), float(delta[ins, outs]), rel_tol=2e-13, abs_tol=1e-17
                    )
                    scalar.append({"row": ins, "column": outs, "delta": float(value)})
            point["decimal_coordinates"] = scalar
            independent.append(point)
        assert not report["unadapted_changes"]
        adapted = {row["weight"] for row in report["layers"]}
        assert saved.keys() == original.keys()
        for key in original.keys() - adapted:
            assert torch.equal(original[key].to(saved[key].dtype), saved[key])
        if not case["native_merge"]:
            squares = [row for row in report["layers"] if row["module"].endswith(".attn.c_proj")]
            assert len(squares) == 12
            assert all(row["relative_residual"] > 0.5 for row in squares)
        results.append({"case": case, "report": report, "independent": independent})
        print(f"Checked {case['name']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "before": args.before,
                "producer": producer,
                "verifier_sha256": file_hash(Path(__file__)),
                "rounding_and_metric_verifier_sha256": file_hash(
                    Path(__file__).with_name("verify.py")
                ),
                "auditor_modules": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path(mergeglass.__file__).parent.glob("*.py"))
                },
                "results": results,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
