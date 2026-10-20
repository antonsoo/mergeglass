"""Retain the before/after audit and a separate orientation-aware dense oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

import mergeglass
from mergeglass import AuditError, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--before", action="store_true")
    parser.add_argument("--refused", action="store_true", help="check the historical refusal guard")
    args = parser.parse_args()
    torch.set_num_threads(1)
    receipt = json.loads((args.directory / "producer.json").read_text())
    for item in receipt["files"]:
        data = (args.directory / item["file"]).read_bytes()
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
    results = []
    for case in receipt["cases"]:
        root = args.directory / case["name"]
        base = load_file(root / "base/model.safetensors")
        adapter = load_file(root / "adapter/adapter_model.safetensors")
        merged = load_file(root / "merged/model.safetensors")
        measurements = []
        for module, transpose in case["native_module_orientations"].items():
            a = adapter[f"base_model.model.{module}.lora_A.weight"].double()
            b = adapter[f"base_model.model.{module}.lora_B.weight"].double()
            product = (b @ a) * 2
            delta = product.T if transpose else product
            observed = merged[module + ".weight"].double() - base[module + ".weight"].double()
            measurements.append(
                {
                    "module": module,
                    "native_transpose": transpose,
                    "relative_residual_with_native_orientation": float(
                        torch.linalg.vector_norm(observed - delta) / torch.linalg.vector_norm(delta)
                    ),
                    "relative_residual_assuming_linear": float(
                        torch.linalg.vector_norm(observed - product)
                        / torch.linalg.vector_norm(product)
                    ),
                }
            )
        refusal = None
        report = None
        try:
            report = audit(root / "base", root / "adapter", root / "merged")
        except AuditError as exc:
            refusal = str(exc)
        should_refuse = case["kind"] == "mixed" and args.refused
        assert (refusal is not None) == should_refuse, (case["name"], refusal)
        if should_refuse:
            assert "Conv1D" in refusal
        else:
            assert report is not None
            for point in measurements:
                layer = next(x for x in report["layers"] if x["module"] == point["module"])
                assert (
                    abs(
                        layer["relative_residual"]
                        - point[
                            "relative_residual_assuming_linear"
                            if args.before
                            else "relative_residual_with_native_orientation"
                        ]
                    )
                    < 1e-8
                )
        results.append(
            {"case": case, "dense_measurements": measurements, "refusal": refusal, "report": report}
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "producer": receipt,
                "before": args.before,
                "historical_refusal_guard": args.refused,
                "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "auditor_modules": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path(mergeglass.__file__).parent.glob("*.py"))
                },
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Verified {len(results)} native orientation cases; before={args.before}")


if __name__ == "__main__":
    main()
