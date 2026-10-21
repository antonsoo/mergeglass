"""Compare complete old-example reports and terminal output across two sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROGRAM = r"""
import hashlib, json, pathlib, sys
source = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source))
import mergeglass
from mergeglass import audit, audit_dora_norms
from mergeglass.cli import render
assert pathlib.Path(mergeglass.__file__).resolve().parent == source/'mergeglass'
results=[]
for item in json.loads(sys.argv[2]):
    base, adapter, merged = [pathlib.Path(item[k]) for k in ['base','adapter','merged']]
    report = audit(base,adapter,merged,block_mib=1)
    entry={'case':item['case'],'report':report,'terminal':render(report),
           'summary_terminal':render(report,summary_only=True)}
    if item['partial']:
        partial=audit_dora_norms(adapter,merged,block_mib=1)
        entry.update(partial=partial,partial_terminal=render(partial),
                     partial_summary_terminal=render(partial,summary_only=True))
    results.append(entry)
print(json.dumps({'cases':results,'runtime_sha256':{
 p.name:hashlib.sha256(p.read_bytes()).hexdigest()
 for p in sorted((source/'mergeglass').glob('*.py'))}},allow_nan=False))
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_src", type=Path)
    parser.add_argument("candidate_src", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    examples = Path(__file__).resolve().parents[2] / "examples"
    cases = []
    for configuration in sorted(examples.glob("*/adapter/adapter_config.json")):
        group = configuration.parent.parent
        if group.name == "embedding-dora":
            continue
        base = group / "base"
        if not base.is_dir():
            continue
        for merged in sorted(group.iterdir()):
            if (
                merged.is_dir()
                and merged.name not in ("adapter", "base")
                and list(merged.glob("*.safetensors"))
            ):
                cases.append(
                    {
                        "case": f"{group.name}/{merged.name}",
                        "base": str(base),
                        "adapter": str(configuration.parent),
                        "merged": str(merged),
                        "partial": group.name == "dora",
                    }
                )
    assert cases
    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    observed = []
    for source in [args.baseline_src, args.candidate_src]:
        result = subprocess.run(
            [sys.executable, "-I", "-c", PROGRAM, str(source.resolve()), json.dumps(cases)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        observed.append(json.loads(result.stdout))
    assert observed[0]["cases"] == observed[1]["cases"], (
        "A complete legacy report or render changed"
    )
    receipt = {
        "comparison": (
            "complete decoded JSON and complete per-layer/summary terminal output; "
            "no excluded fields"
        ),
        "caseCount": len(cases),
        "partialCases": sum(item["partial"] for item in cases),
        "runtimeSha256": {
            "baseline": observed[0]["runtime_sha256"],
            "candidate": observed[1]["runtime_sha256"],
        },
        "cases": [
            {
                "case": item["case"],
                "completeResultSha256": hashlib.sha256(
                    json.dumps(item, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
            for item in observed[1]["cases"]
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {"completeLegacyReportsMatch": len(cases), "partialCases": receipt["partialCases"]}
        )
    )


if __name__ == "__main__":
    main()
