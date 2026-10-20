"""Native vocabulary-scale embedding merge and isolated installed-auditor RSS replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path


def produce(root: Path) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file

    class Vocabulary(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tokens = torch.nn.Embedding(250_003, 128, dtype=torch.bfloat16)
            self.config = {"model_type": "mergeglass_synthetic_vocabulary"}

    root.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(716)
    base = Vocabulary()
    with torch.no_grad():
        base.tokens.weight.fill_(1)
    (root / "base").mkdir()
    save_file(base.state_dict(), root / "base/model.safetensors")
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=8, target_modules=["tokens"]))
    layer = model.base_model.model.tokens
    with torch.no_grad():
        # These are authored binary-fraction controls, not trained parameters.
        a = layer.lora_embedding_A["default"]
        a.copy_(((torch.arange(250_003) % 33 - 16).float() / 4096).expand_as(a))
        layer.lora_embedding_B["default"].fill_(1 / 8)
    model.save_pretrained(root / "adapter", save_embedding_layers=False)
    (root / "adapter/README.md").unlink(missing_ok=True)
    (root / "merged").mkdir()
    merged = model.merge_and_unload()
    save_file(merged.state_dict(), root / "merged/model.safetensors")
    (root / "producer.json").write_text(
        json.dumps(
            {
                "scope": "native PEFT merge of explicitly controlled synthetic embedding weights",
                "shape": [250_003, 128],
                "rank": 8,
                "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "torch": torch.__version__,
            },
            indent=2,
        )
        + "\n"
    )


def child(root: Path, block: int) -> None:
    import mergeglass
    from mergeglass import audit

    start = time.perf_counter()
    report = audit(
        root / "base", root / "adapter", root / "merged", block_mib=block, max_adapter_mib=1
    )
    elapsed = time.perf_counter() - start
    layer = report["layers"][0]
    # Independent integer control: delta[token, dimension] = (token % 33 - 16) / 4096.
    # BF16 spacing below 1 is 1/256, above 1 is 1/128. Exactly half rounds to 1.
    nonzero = changed = erased = squares = residual_squares = erased_squares = 0
    observed_squares = dot = 0
    for residue in range(33):
        count = (250_003 + 32 - residue) // 33 * 128
        numerator = residue - 16
        observed = -16 if numerator < -8 else 0
        nonzero += count * (numerator != 0)
        changed += count * (observed != 0)
        erased += count * (numerator != 0 and observed == 0)
        squares += count * numerator**2
        residual_squares += count * (observed - numerator) ** 2
        erased_squares += count * numerator**2 * (observed == 0)
        observed_squares += count * observed**2
        dot += count * observed * numerator
    assert layer["intended_nonzero_elements"] == nonzero
    assert layer["changed_elements"] == changed
    assert layer["unchanged_nonzero_update_elements"] == erased
    expected = {
        "delta": math.sqrt(squares) / 4096,
        "observed": math.sqrt(observed_squares) / 4096,
        "residual": math.sqrt(residual_squares) / 4096,
        "unchanged_reference": math.sqrt(erased_squares) / 4096,
        "base_cast_drift": 0,
    }
    for key, value in expected.items():
        assert math.isclose(layer["norms"][key], value, rel_tol=2e-13, abs_tol=1e-14)
    assert math.isclose(layer["projection_gain"], dot / squares, rel_tol=2e-13)
    assert math.isclose(
        layer["unchanged_reference_energy_fraction"], erased_squares / squares, rel_tol=2e-13
    )
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "block_mib": block,
                "max_adapter_mib": 1,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "elapsed_seconds": elapsed,
                "dense_float64_delta_bytes": 250_003 * 128 * 8,
                "decoded_full_a_bytes": 8 * 250_003 * 8,
                "decoded_full_b_bytes": 128 * 8 * 8,
                "torch_imported": "torch" in sys.modules,
                "summary": report["summary"],
                "inputs": report["inputs"],
                "auditor_modules": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(Path(mergeglass.__file__).parent.glob("*.py"))
                },
                "oracle": (
                    "integer residue counts and exact BF16 spacing; all counts and squared norms"
                ),
            },
            allow_nan=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["produce", "measure", "child"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.mode == "produce":
        produce(args.directory)
    elif args.mode == "child":
        child(args.directory, args.block)
    else:
        records = []
        for block in (1, 16):
            result = subprocess.run(
                [
                    args.python,
                    "-I",
                    str(Path(__file__).resolve()),
                    "child",
                    str(args.directory),
                    "--block",
                    str(block),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            records.append(json.loads(result.stdout))
        assert records[0]["summary"] == records[1]["summary"]
        assert args.out is not None
        args.out.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        print(f"Verified {250_003 * 128:,} embedding weights at two block sizes")


if __name__ == "__main__":
    main()
