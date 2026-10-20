"""Compare complete old/new reports and both terminal views on unchanged inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

CODE = """
import json,sys
from pathlib import Path
import mergeglass
from mergeglass import audit
from mergeglass.cli import render
assert Path(mergeglass.__file__).resolve().parent.parent == Path(sys.argv[1]).resolve()
report=audit(*sys.argv[2:],block_mib=1)
print(json.dumps({'report':report,'terminal':render(report),'summary':render(report,summary_only=True)},allow_nan=False))
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before_src", type=Path)
    parser.add_argument("after_src", type=Path)
    parser.add_argument("repository", type=Path)
    parser.add_argument("public_legacy", type=Path)
    parser.add_argument("native_embeddings", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    cases = []
    for group, root in [
        ("rounding", args.repository / "examples/rounding"),
        ("dora", args.repository / "examples/dora"),
        ("public", args.public_legacy),
    ]:
        for path in sorted(root.iterdir()):
            if path.name in ("base", "adapter"):
                continue
            if (path / "model.safetensors").exists() or (
                path / "model.safetensors.index.json"
            ).exists():
                cases.append((group, path.name, (root / "base", root / "adapter", path)))
    root = args.native_embeddings
    producer = json.loads((root / "producer.json").read_text())
    for name in producer["cases"]:
        cases.append(("embedding", name, (root / "base", root / "adapter", root / name)))
    for adapter in ("transformer-adapter", "transformer-default-save"):
        cases.append(
            (
                "embedding-transformer",
                adapter,
                (root / "transformer-base", root / adapter, root / "transformer-merged"),
            )
        )
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        for group, label, paths in cases:
            outputs = []
            for source in (args.before_src, args.after_src):
                result = subprocess.run(
                    [sys.executable, "-c", CODE, str(source), *map(str, paths)],
                    cwd=directory,
                    env={**os.environ, "PYTHONPATH": str(source), "OPENBLAS_NUM_THREADS": "1"},
                    check=True,
                    capture_output=True,
                )
                assert not result.stderr, result.stderr.decode()
                outputs.append(result.stdout)
            assert outputs[0] == outputs[1], (group, label)
            report = json.loads(outputs[0])["report"]
            rows.append(
                {
                    "group": group,
                    "case": label,
                    "checked_tensors": report["summary"]["checked_tensors"],
                    "complete_reports_and_terminal_views_sha256": hashlib.sha256(
                        outputs[0]
                    ).hexdigest(),
                }
            )
    payload = {
        "method": (
            "Exact UTF-8 equality of complete serialized reports and both per-layer/summary "
            "terminal views in independent before/after source processes"
        ),
        "source_modules": {
            role: {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((source / "mergeglass").glob("*.py"))
            }
            for role, source in [("before", args.before_src), ("after", args.after_src)]
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Complete old/new reports and terminal views match on {len(rows)} unchanged cases")


if __name__ == "__main__":
    main()
