"""Replay norm-only observations against the independently verified full reports.

This process supplies only adapter and merged paths to the installed auditor.
It does not read any base checkpoint or import a model framework.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path

import mergeglass
from mergeglass import audit_dora_norms

DIRECTION_FIELDS = {
    "zero_numerical_reference_rows",
    "direction_comparable_rows",
    "direction_cosine_range",
    "mean_direction_cosine",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("public", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    baseline = here / "rows" / "dense-oracle.json.gz"
    verification = json.loads((here / "rows" / "verification.json").read_text())
    expected_hash = next(
        row["sha256"] for row in verification["receipts"] if row["file"] == baseline.name
    )
    assert sha(baseline) == expected_hash
    original = json.loads(gzip.decompress(baseline.read_bytes()))
    implementation = Path(mergeglass.__file__).parent
    frozen = {p: sha(p) for p in [Path(__file__), *sorted(implementation.glob("*.py"))]}
    cases = []
    for group, root in [("native", args.native), ("public", args.public)]:
        for prior in original[group]:
            case = prior["case"]
            adapter = root / ("transformer-adapter" if case == "transformer-merged" else "adapter")
            report = audit_dora_norms(adapter, root / case, block_mib=1)
            assert report["inputs"] == {
                role: prior["inputs"][role] for role in ("adapter", "merged")
            }
            assert report["schema"] == "mergeglass.dora-norms"
            assert [layer["weight"] for layer in report["layers"]] == [
                layer["weight"] for layer in prior["layers"]
            ]
            for before, after in zip(prior["layers"], report["layers"], strict=True):
                assert after["dora_rows"] == {
                    key: value
                    for key, value in before["dora_rows"].items()
                    if key not in DIRECTION_FIELDS
                }, (group, case, after["weight"])
            cases.append({"group": group, "case": case, "report": report})
            print(json.dumps({"completed": len(cases), "case": case}), flush=True)
    assert all(sha(path) == value for path, value in frozen.items())
    result = {
        "method": "exact equality with norm fields of dense-oracle-verified complete audits",
        "baseline_source_commit": verification["source_commit"],
        "baseline_sha256": expected_hash,
        "script_sha256": frozen[Path(__file__)],
        "auditor_sources": {p.name: frozen[p] for p in implementation.glob("*.py")},
        "versions": {key: importlib.metadata.version(key) for key in ("numpy", "safetensors")},
        "cases": cases,
        "matrices": sum(len(case["report"]["layers"]) for case in cases),
        "rows": sum(case["report"]["summary"]["norm_observed_rows"] for case in cases),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
