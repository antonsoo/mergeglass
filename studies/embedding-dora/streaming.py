"""Native vocabulary-scale DoRA, independently counted from 33 integer groups.

Heavy native production/checking and each installed audit use separate processes.
The measured auditor imports neither Torch nor PEFT.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from decimal import Decimal, localcontext
from pathlib import Path

VOCABULARY = 250_003
DIMENSIONS = 128
RANK = 8


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def produce(root: Path) -> None:
    import importlib.metadata

    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tokens = torch.nn.Embedding(VOCABULARY, DIMENSIONS)
            self.config = {"model_type": "mergeglass_synthetic_embedding_dora"}

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return self.tokens(ids)

    root.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(397145)
    base = Model()
    with torch.no_grad():
        base.tokens.weight.fill_(1)
    (root / "base").mkdir()
    save_file(base.state_dict(), root / "base/model.safetensors")
    model = get_peft_model(
        base,
        LoraConfig(
            r=RANK,
            lora_alpha=RANK,
            use_dora=True,
            target_modules=["tokens"],
        ),
    )
    module = model.base_model.model.tokens
    with torch.no_grad():
        module.lora_embedding_A["default"].zero_()
        module.lora_embedding_A["default"][0].copy_(
            (torch.arange(VOCABULARY) % 33 - 16).float() / 4096
        )
        module.lora_embedding_B["default"].zero_()
        module.lora_embedding_B["default"][:, 0].copy_((torch.arange(DIMENSIONS) % 4 != 0).float())
        module.lora_magnitude_vector["default"].weight.fill_(500)
    model.save_pretrained(root / "adapter", save_embedding_layers=False)
    (root / "adapter/README.md").unlink(missing_ok=True)
    module.base_layer.to(torch.bfloat16)
    merged = model.merge_and_unload(safe_merge=False)
    (root / "merged").mkdir()
    save_file(merged.state_dict(), root / "merged/model.safetensors")
    receipt = {
        "scope": "native PEFT BF16 merge of explicit synthetic factors; no training",
        "producer_sha256": digest(Path(__file__)),
        "shape": [VOCABULARY, DIMENSIONS],
        "rank": RANK,
        "scale": 1,
        "formula": (
            "base=1; directional delta=(token%33-16)/4096 when dimension%4!=0, "
            "else 0; every magnitude=500"
        ),
        "versions": {
            name: importlib.metadata.version(name) for name in ["torch", "peft", "safetensors"]
        },
        "files": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ],
    }
    (root / "producer.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"weights": VOCABULARY * DIMENSIONS, "native_files": len(receipt["files"])}))


def reference() -> tuple[dict[str, object], list[list[float]]]:
    # Complete positive finite BF16 grid, decoded independently from its bits.
    grid = [struct.unpack("<f", struct.pack("<I", value << 16))[0] for value in range(0x7F80)]

    def nearest(value: float) -> float:
        position = bisect.bisect_left(grid, value)
        low, high = grid[position - 1], grid[position]
        midpoint = (low + high) / 2
        return high if value > midpoint or (value == midpoint and position % 2 == 0) else low

    counts = [VOCABULARY // 33 + (index < VOCABULARY % 33) for index in range(33)]
    expected_saved: list[list[float]] = [[], []]
    with localcontext() as context:
        context.prec = 80
        expected = {
            name: Decimal(0)
            for name in ["delta", "observed", "residual", "unchanged_reference", "dot"]
        }
        changed = 0
        erased = 0
        norm_factors = []
        cosines = []
        largest_residual = Decimal(0)
        for family, dimensions in [(0, DIMENSIONS // 4), (1, DIMENSIONS - DIMENSIONS // 4)]:
            directions = [
                Decimal(1) + (Decimal(index - 16) / 4096 if family else 0) for index in range(33)
            ]
            length = sum(
                (count * direction**2 for count, direction in zip(counts, directions, strict=True)),
                Decimal(0),
            ).sqrt()
            weights = [direction * 500 / length for direction in directions]
            expected_saved[family] = [nearest(float(value)) for value in weights]
            saved_length = sum(
                (
                    count * Decimal.from_float(value) ** 2
                    for count, value in zip(counts, expected_saved[family], strict=True)
                ),
                Decimal(0),
            ).sqrt()
            norm_factors.append(float(saved_length / 500))
            vector_dot = sum(
                (
                    count * weight * Decimal.from_float(saved)
                    for count, weight, saved in zip(
                        counts, weights, expected_saved[family], strict=True
                    )
                ),
                Decimal(0),
            )
            cosines.append(float(vector_dot / saved_length / 500))
            for count, weight, saved in zip(counts, weights, expected_saved[family], strict=True):
                delta = weight - 1
                observed = Decimal.from_float(saved) - 1
                residual = observed - delta
                largest_residual = max(largest_residual, abs(residual))
                assert delta != 0
                changed += count * dimensions * (observed != 0)
                erased += count * dimensions * (observed == 0)
                for name, value in [
                    ("delta", delta**2),
                    ("observed", observed**2),
                    ("residual", residual**2),
                    ("dot", observed * delta),
                ]:
                    expected[name] += count * dimensions * value
                if observed == 0:
                    expected["unchanged_reference"] += count * dimensions * delta**2
        measured = {
            "intended_nonzero_elements": VOCABULARY * DIMENSIONS,
            "changed_elements": changed,
            "unchanged_nonzero_update_elements": erased,
            "norms": {
                name: float(expected[name].sqrt())
                for name in ["delta", "observed", "residual", "unchanged_reference"]
            },
            "projection_gain": float(expected["dot"] / expected["delta"]),
            "relative_residual": float((expected["residual"] / expected["delta"]).sqrt()),
            "unchanged_reference_energy_fraction": float(
                expected["unchanged_reference"] / expected["delta"]
            ),
            "column_norm_factor_range": [min(norm_factors), max(norm_factors)],
            "direction_cosine_range": [min(cosines), max(cosines)],
            "largest_absolute_residual": float(largest_residual),
            "group_counts": counts,
        }
    return measured, expected_saved


def verify_native(root: Path) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    _reference, table = reference()
    factors = load_file(root / "adapter/adapter_model.safetensors")
    a = factors["base_model.model.tokens.lora_embedding_A"]
    b = factors["base_model.model.tokens.lora_embedding_B"]
    m = factors["base_model.model.tokens.lora_magnitude_vector"]
    assert torch.equal(a[0], (torch.arange(VOCABULARY) % 33 - 16).float() / 4096)
    assert not bool(a[1:].count_nonzero()) and not bool(b[:, 1:].count_nonzero())
    assert torch.equal(b[:, 0], (torch.arange(DIMENSIONS) % 4 != 0).float())
    assert bool((m == 500).all())
    group = torch.tensor(table, dtype=torch.bfloat16).T
    family = (torch.arange(DIMENSIONS) % 4 != 0).long()
    for role in ["base", "merged"]:
        with safe_open(root / role / "model.safetensors", framework="pt") as handle:
            tensor = handle.get_slice("tokens.weight")
            assert tensor.get_shape() == [VOCABULARY, DIMENSIONS]
            for row in range(0, VOCABULARY, 4096):
                count = min(4096, VOCABULARY - row)
                values = tensor[row : row + count]
                expected = (
                    torch.ones_like(values)
                    if role == "base"
                    else group[(torch.arange(row, row + count) % 33)[:, None], family[None, :]]
                )
                assert torch.equal(values, expected), (role, row)
    print(
        json.dumps(
            {
                "all_base_entries_checked": VOCABULARY * DIMENSIONS,
                "all_merged_entries_checked": VOCABULARY * DIMENSIONS,
                "all_factors_checked": True,
            }
        )
    )


PROGRAM = r"""
import contextlib, hashlib, importlib.metadata, io, json, pathlib, resource, sys, time
started = time.perf_counter()
from mergeglass.cli import main
import mergeglass
root = pathlib.Path(sys.argv[1])
block = sys.argv[2]
output = io.StringIO()
with contextlib.redirect_stdout(output):
    status = main([str(root/'base'), str(root/'adapter'), str(root/'merged'),
                   '--block-mib', block, '--max-adapter-mib', '1', '--json', '-'])
assert status == 0
assert not any(name.split('.')[0] in {'torch', 'peft', 'transformers'} for name in sys.modules)
print(json.dumps({'report': json.loads(output.getvalue()),
  'elapsed_seconds': time.perf_counter()-started,
  'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  'python': sys.version.split()[0],
  'versions': {name: importlib.metadata.version(name) for name in ['numpy','safetensors']},
  'runtime_sha256': {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(pathlib.Path(mergeglass.__file__).parent.glob('*.py'))},
  'model_runtime_imported':False}))
"""


def measure(args: argparse.Namespace) -> None:
    producer = json.loads((args.directory / "producer.json").read_text())
    for item in producer["files"]:
        path = args.directory / item["path"]
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
    native = json.loads(
        subprocess.check_output(
            [sys.executable, str(Path(__file__).resolve()), "verify-native", str(args.directory)],
            text=True,
        )
    )
    expected, _table = reference()
    results = []
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for block in [1, 16]:
        result = json.loads(
            subprocess.check_output(
                [args.python, "-I", "-c", PROGRAM, str(args.directory.resolve()), str(block)],
                env=environment,
                text=True,
            )
        )
        layer = result["report"]["layers"][0]
        for name in [
            "intended_nonzero_elements",
            "changed_elements",
            "unchanged_nonzero_update_elements",
        ]:
            assert layer[name] == expected[name], (name, layer[name], expected[name])
        for name in ["projection_gain", "relative_residual", "unchanged_reference_energy_fraction"]:
            assert math.isclose(layer[name], expected[name], rel_tol=1e-8, abs_tol=1e-12), (
                name,
                layer[name],
                expected[name],
            )
        for name, value in expected["norms"].items():
            assert math.isclose(layer["norms"][name], value, rel_tol=1e-8, abs_tol=1e-10), (
                name,
                layer["norms"][name],
                value,
            )
        for name in ["column_norm_factor_range", "direction_cosine_range"]:
            assert all(
                math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-12)
                for a, b in zip(layer["dora_columns"][name], expected[name], strict=True)
            )
        assert layer["reference_merge_mismatches"] == layer["reference_merge_overflows"] == 0
        assert math.isclose(
            abs(layer["largest_residual"]["residual"]),
            expected["largest_absolute_residual"],
            rel_tol=1e-8,
            abs_tol=1e-12,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "block_mib": block,
                    "peak_rss_kib": result["peak_rss_kib"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "changed": layer["changed_elements"],
                }
            ),
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "scope": (
                    "one isolated installed CLI process per block setting; "
                    "not a statistical timing estimate"
                ),
                "oracle": (
                    "80-digit roots of integer-residue moments; independently enumerated "
                    "BF16 values; all native entries checked"
                ),
                "producer_sha256": digest(args.directory / "producer.json"),
                "driver_sha256": digest(Path(__file__)),
                "native_check": native,
                "expected": expected,
                "measurements": results,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["produce", "verify-native", "measure"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.mode == "produce":
        produce(args.directory)
    elif args.mode == "verify-native":
        verify_native(args.directory)
    else:
        if args.out is None:
            parser.error("measure requires --out")
        measure(args)


if __name__ == "__main__":
    main()
