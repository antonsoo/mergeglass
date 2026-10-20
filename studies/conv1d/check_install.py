"""Run a wheel's CLI outside the checkout and measure fresh-process peak RSS."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("python", type=Path)
    parser.add_argument("streaming", type=Path)
    parser.add_argument("public", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    python, streaming, public = (
        # Resolving a virtualenv interpreter's symlink discards its environment.
        args.python.absolute(),
        args.streaming.resolve(),
        args.public.resolve(),
    )
    repository = Path(__file__).resolve().parents[2]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="conv1d-installed-") as tmp:
        root = Path(tmp)
        identity_code = """
import hashlib, importlib.metadata, json, pathlib, sys
import mergeglass
root = pathlib.Path(mergeglass.__file__).parent
print(json.dumps({
    "version": importlib.metadata.version("mergeglass"),
    "python": sys.version,
    "auditor_modules": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))
    },
    "torch_imported": "torch" in sys.modules,
    "dependencies": importlib.metadata.requires("mergeglass"),
}))
"""
        identity = json.loads(
            subprocess.check_output(
                [str(python), "-I", "-c", identity_code], cwd=root, env=env, text=True
            )
        )
        source = repository / "src/mergeglass"
        assert identity["auditor_modules"] == {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("*.py"))
        }
        assert not identity["torch_imported"]

        def run(base: Path, adapter: Path, merged: Path, *options: str) -> dict[str, Any]:
            output, timing = root / "report.json", root / "time.txt"
            command = [
                "/usr/bin/time",
                "-f",
                "%M %e",
                "-o",
                str(timing),
                str(python),
                "-I",
                "-m",
                "mergeglass",
                str(base),
                str(adapter),
                str(merged),
                *options,
                "--json",
                str(output),
            ]
            result = subprocess.run(
                command, cwd=root, env=env, capture_output=True, text=True, check=True
            )
            peak, elapsed = timing.read_text().split()
            return {
                "peak_rss_kib": int(peak),
                "elapsed_seconds": float(elapsed),
                "report": json.loads(output.read_text()),
                "stdout": result.stdout,
                "report_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }

        examples = []
        fixture = repository / "examples/conv1d"
        for name in ["correct-f32-safe-false", "wrong-square-orientation", "omitted-square"]:
            checked = run(fixture / "base", fixture / "adapter", fixture / name)
            row = next(
                row
                for row in checked["report"]["layers"]
                if row["module"] == "transformer.h.0.attn.c_proj"
            )
            if name.startswith("correct"):
                assert row["relative_residual"] < 1e-6
            elif name.startswith("wrong"):
                assert row["changed_elements"] == 256 and row["relative_residual"] > 1
            else:
                assert row["changed_elements"] == 0 and row["relative_residual"] == 1
            examples.append({"case": name, **checked})
        memory = []
        reference = json.loads((streaming / "reference.json").read_text())
        for block in (1, 16):
            checked = run(
                streaming / "base",
                streaming / "adapter",
                streaming / "merged",
                "--block-mib",
                str(block),
                "--max-adapter-mib",
                "1",
            )
            row = checked["report"]["layers"][0]
            for field in [
                "changed_elements",
                "intended_nonzero_elements",
                "unchanged_nonzero_update_elements",
            ]:
                assert row[field] == reference[field]
            for field, value in reference["norms"].items():
                assert math.isclose(row["norms"][field], value, rel_tol=1e-10)
            assert math.isclose(
                row["relative_residual"], reference["relative_residual"], rel_tol=1e-10
            )
            memory.append({"block_mib": block, **checked})
        community = run(
            public / "base",
            public / "adapter",
            public / "native-bf16",
            "--block-mib",
            "1",
            "--summary",
        )
        oracle = json.loads((repository / "studies/conv1d/results/public-after.json").read_text())
        expected = next(
            r["report"] for r in oracle["results"] if r["case"]["name"] == "native-bf16"
        )
        assert community["report"] == expected
        results = {
            "identity": identity,
            "machine": {
                "platform": platform.platform(),
                "logical_cpus": os.cpu_count(),
                "blas_threads": 1,
            },
            "rss_method": (
                "Linux /usr/bin/time %M: fresh isolated installed CLI, "
                "peak resident KiB; %e elapsed seconds"
            ),
            "examples": examples,
            "streaming_reference": reference,
            "streaming": memory,
            "public_bf16": community,
            "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print("Installed example, dense streaming and complete public-report comparisons passed.")


if __name__ == "__main__":
    main()
