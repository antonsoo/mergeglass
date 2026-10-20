"""Independent dense Torch comparison of full state and every LoRA update."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

import mergeglass
from mergeglass import AuditError, audit
from mergeglass.cli import render


class Weights:
    def __init__(self, root: Path, *, adapter: bool = False):
        self.root = root
        stem = "adapter_model" if adapter else "model"
        index = root / f"{stem}.safetensors.index.json"
        if index.exists():
            self.paths = json.loads(index.read_text())["weight_map"]
        else:
            name = f"{stem}.safetensors"
            with safe_open(root / name, framework="pt") as handle:
                self.paths = dict.fromkeys(handle.keys(), name)

    def __getitem__(self, key: str) -> torch.Tensor:
        with safe_open(self.root / self.paths[key], framework="pt") as handle:
            return handle.get_tensor(key)


def close(actual: float | None, expected: float | None) -> None:
    assert (actual is None) == (expected is None), (actual, expected)
    if expected is not None:
        assert actual is not None and math.isclose(
            actual, expected, rel_tol=2e-11, abs_tol=2e-14
        ), (actual, expected)


def rounded(values: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, int]:
    """Check direct rounding using adjacent representable values and exact ties.

    Torch's double -> half/BF16 can double-round through float32. Check its
    provisional result against the exact binary64 midpoint to each native
    nextafter neighbor. This differs from the auditor's exponent/spacing
    routine. Validate every repaired value using rational distances; half
    also has an independent CPython struct.pack('e') check.
    """
    if dtype not in (torch.float16, torch.bfloat16):
        return values.to(dtype).double(), 0
    provisional = values.to(dtype)
    center = provisional.double()
    lower = torch.nextafter(provisional, torch.full_like(provisional, -torch.inf)).double()
    upper = torch.nextafter(provisional, torch.full_like(provisional, torch.inf)).double()
    assert torch.isfinite(lower).all() and torch.isfinite(upper).all()
    low_midpoint, high_midpoint = (lower + center) / 2, (center + upper) / 2
    odd = (provisional.view(torch.int16) & 1) != 0
    choose_lower = (values < low_midpoint) | ((values == low_midpoint) & odd)
    choose_upper = (values > high_midpoint) | ((values == high_midpoint) & odd)
    repaired = choose_lower | choose_upper
    result = torch.where(choose_lower, lower, torch.where(choose_upper, upper, center))
    for index in repaired.reshape(-1).nonzero().flatten().tolist():
        value = float(values.reshape(-1)[index])
        candidates = [float(t.reshape(-1)[index]) for t in (lower, center, upper)]
        distances = [abs(Fraction.from_float(v) - Fraction.from_float(value)) for v in candidates]
        distance = abs(
            Fraction.from_float(float(result.reshape(-1)[index])) - Fraction.from_float(value)
        )
        assert distance == min(distances)
        if distances.count(min(distances)) > 1:
            assert int(result.reshape(-1)[index].to(dtype).view(torch.int16)) & 1 == 0
        if dtype == torch.float16:
            assert (
                float(result.reshape(-1)[index]) == struct.unpack("e", struct.pack("e", value))[0]
            )
    return result, int(repaired.count_nonzero())


def verify(base: Path, adapter: Path, merged: Path) -> dict[str, Any]:
    before, source, after = Weights(base), Weights(adapter, adapter=True), Weights(merged)
    report = audit(base, adapter, merged, block_mib=1)
    config = json.loads((adapter / "adapter_config.json").read_text())
    # Native cases use Linear LoRA/DoRA with one rank/alpha. Discover
    # their expected paths directly from the original adapter, not the report.
    assert not config.get("rank_pattern") and not config.get("alpha_pattern")
    assert not config.get("use_rslora")
    prefix = "base_model.model."
    suffix = ".lora_A.weight"
    expected_layers = {
        key[len(prefix) : -len(suffix)] for key in source.paths if key.endswith(suffix)
    }
    assert {layer["module"] for layer in report["layers"]} == expected_layers
    lora_results = []
    repaired_roundings = 0
    for layer in report["layers"]:
        path = layer["module"]
        a = source[prefix + path + ".lora_A.weight"].double()
        b = source[prefix + path + ".lora_B.weight"].double()
        delta = (b @ a) * (config["lora_alpha"] / config["r"])
        original, saved = before[path + ".weight"].double(), after[path + ".weight"]
        reference = original + delta
        if config.get("use_dora"):
            direction = reference / reference.norm(dim=1, keepdim=True)
            reference = (
                direction * source[prefix + path + ".lora_magnitude_vector"].double()[:, None]
            )
            delta = reference - original
        baseline = original.to(saved.dtype).double()
        observed = saved.double() - baseline
        residual = observed - delta
        unchanged = saved.double() == baseline
        length, observed_length = float(delta.norm()), float(observed.norm())
        inner = float((observed * delta).sum())
        assert layer["intended_nonzero_elements"] == int((delta != 0).count_nonzero())
        assert layer["changed_elements"] == int((~unchanged).count_nonzero())
        assert layer["unchanged_nonzero_update_elements"] == int(
            ((delta != 0) & unchanged).count_nonzero()
        )
        target, repairs = rounded(reference, saved.dtype)
        repaired_roundings += repairs
        assert layer["reference_merge_mismatches"] == int(
            (target != saved.double()).count_nonzero()
        ), path
        for field, value in {
            "delta": length,
            "observed": observed_length,
            "residual": float(residual.norm()),
            "base_cast_drift": float((baseline - original).norm()),
            "unchanged_reference": float(delta[unchanged].norm()),
        }.items():
            close(layer["norms"][field], value)
        close(layer["projection_gain"], inner / length**2 if length else None)
        close(layer["relative_residual"], float(residual.norm()) / length if length else None)
        close(
            layer["cosine_similarity"],
            inner / length / observed_length if length and observed_length else None,
        )
        witness = layer["largest_residual"]
        close(abs(witness["residual"]), float(residual.abs().max()))
        close(witness["residual"], float(residual[witness["row"], witness["column"]]))
        lora_results.append(
            {
                "module": path,
                "elements": layer["elements"],
                "gain": layer["projection_gain"],
                "relative_residual": layer["relative_residual"],
            }
        )
    full_keys = {key[len(prefix) :] for key in source.paths if ".lora_" not in key}
    assert full_keys == {row["weight"] for row in report["saved_modules"]}
    for row in report["saved_modules"]:
        key = row["weight"]
        original, saved = source[prefix + key], after[key]
        expected = original.to(saved.dtype)
        floating = expected.is_floating_point()
        mismatch = saved != expected
        count = int(mismatch.count_nonzero())
        assert row["target_mismatches"] == count
        if count:
            index = int(mismatch.reshape(-1).nonzero()[0].item())
            assert row["first_mismatch"] == {
                "flat_index": index,
                "cast_target": expected.reshape(-1)[index].item(),
                "merged": saved.reshape(-1)[index].item(),
            }
        else:
            assert row["first_mismatch"] is None
        if floating:
            for field, tensor in {
                "target": expected.double(),
                "error": saved.double() - expected.double(),
                "target_cast_drift": expected.double() - original.double(),
            }.items():
                close(row["norms"][field], float(tensor.norm()))
            denominator = float(expected.double().norm())
            close(
                row["relative_error"],
                float((saved.double() - expected.double()).norm()) / denominator
                if denominator
                else None,
            )
        else:
            assert original.dtype == saved.dtype
            assert row["norms"] is None and row["relative_error"] is None
        if key not in before.paths:
            assert row["base_comparison_unavailable"] == "absent"
        elif before[key].shape != saved.shape:
            assert row["base_comparison_unavailable"] == "shape-changed"
        else:
            baseline = before[key].to(saved.dtype)
            assert row["base_comparison_unavailable"] is None
            assert row["changed_from_base_elements"] == int((saved != baseline).count_nonzero())
            assert row["expected_change_elements"] == int((expected != baseline).count_nonzero())
            assert row["unchanged_expected_change_elements"] == int(
                ((expected != baseline) & (saved == baseline)).count_nonzero()
            )
    adapted = {path + ".weight" for path in expected_layers}
    unrelated = {
        key: int((after[key] != before[key].to(after[key].dtype)).count_nonzero())
        for key in before.paths.keys() - adapted - full_keys
    }
    assert report["summary"]["checked_tensors"] == len(after.paths)
    assert report["summary"]["unadapted_changed_elements"] == sum(unrelated.values())
    assert report["summary"]["unadapted_changed_tensors"] == sum(
        bool(count) for count in unrelated.values()
    )
    return {
        "case": merged.name,
        "summary": report["summary"],
        "saved_modules": report["saved_modules"],
        "unmatched_saved_module_selectors": report["unmatched_saved_module_selectors"],
        "all_lora_layers_checked": len(lora_results),
        "lora_elements_checked": sum(row["elements"] for row in lora_results),
        "factor_measurements_sha256": hashlib.sha256(
            json.dumps(report["layers"], sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "torch_provisional_low_precision_casts_repaired": repaired_roundings,
        "lora_gain_range": [
            min(row["gain"] for row in lora_results),
            max(row["gain"] for row in lora_results),
        ],
        "lora_relative_residual_range": [
            min(row["relative_residual"] for row in lora_results),
            max(row["relative_residual"] for row in lora_results),
        ],
        "complete_report_sha256": hashlib.sha256(
            json.dumps(report, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--before", action="store_true")
    parser.add_argument("--case", help="verify one merged case, for resumable public runs")
    args = parser.parse_args()
    torch.set_num_threads(1)
    producer = json.loads(
        (args.root / ("public-producer.json" if args.public else "producer.json")).read_text()
    )
    if args.public:
        cases = [
            {"base": "base", "adapter": "adapter", "merged": item["name"]}
            for item in producer["cases"]
        ]
        inventories = [(args.root / item["name"], item["files"]) for item in producer["cases"]]
        inventories += [
            (args.root / source["role"], source["files"])
            for source in producer["sources"]["sources"]
        ]
    else:
        cases = producer["cases"]
        inventories = [(args.root, producer["files"])]
    if args.case:
        cases = [item for item in cases if item["merged"] == args.case]
        if not cases:
            parser.error("--case does not name a producer case")
    for directory, items in inventories:
        for item in items:
            path = directory / item["file"]
            assert path.stat().st_size == item["bytes"]
            with path.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
    results = []
    for item in cases:
        paths = [args.root / item[key] for key in ("base", "adapter", "merged")]
        if args.before or item.get("expect_refusal"):
            try:
                audit(*paths)
            except AuditError as exc:
                assert (
                    ("modules_to_save" in str(exc) or "inventories differ" in str(exc))
                    if args.before
                    else "Incomplete saved module" in str(exc)
                )
                result = {
                    "case": item["adapter"] if item.get("expect_refusal") else item["merged"],
                    "refusal": str(exc),
                }
            else:
                raise AssertionError("Expected a refusal")
        else:
            result = verify(*paths)
        results.append(result)
        print(
            result["case"],
            "refused"
            if "refusal" in result
            else result["summary"]["saved_module_target_mismatches"],
            flush=True,
        )
    if not args.before and args.case in (None, "omit-full-modules-f32", "omit-head-f32"):
        selected = "omit-full-modules-f32" if args.public else "omit-head-f32"
        args.out.with_suffix(".txt").write_text(
            render(
                audit(args.root / "base", args.root / "adapter", args.root / selected),
                summary_only=args.public,
            )
            + "\n"
        )
    payload = {
        "scope": (
            "independent dense Torch replay of complete full-module state and all LoRA matrices"
        ),
        "before": args.before,
        "public": args.public,
        "producer": producer,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "auditor_modules": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(mergeglass.__file__).parent.glob("*.py"))
        },
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
