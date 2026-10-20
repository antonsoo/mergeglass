"""Exercise an isolated installed wheel against retained source reports.

Run with that environment's python -I; no Torch, PEFT or source PYTHONPATH.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import mergeglass
from mergeglass import AuditError, audit
from mergeglass.cli import render


def digest(report: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest()


def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-m", "mergeglass", *arguments], capture_output=True, text=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("native", type=Path)
    parser.add_argument("public", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    source = {
        name: json.loads((args.evidence / f"{name}.json").read_text())
        for name in ("native", "public")
    }
    modules = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(mergeglass.__file__).parent.glob("*.py"))
    }
    assert modules == source["native"]["auditor_modules"] == source["public"]["auditor_modules"]
    assert sys.flags.isolated
    for module in ("torch", "peft", "transformers"):
        assert importlib.util.find_spec(module) is None
    records = []
    for group, root in [("native", args.native), ("public", args.public)]:
        producer = source[group]["producer"]
        cases = (
            producer["cases"]
            if group == "native"
            else [
                {"base": "base", "adapter": "adapter", "merged": row["name"]}
                for row in producer["cases"]
            ]
        )
        expected = {row["case"]: row for row in source[group]["results"]}
        factor_hashes = {}
        for item in cases:
            paths = [str(root / item[key]) for key in ("base", "adapter", "merged")]
            label = item["adapter"] if item.get("expect_refusal") else item["merged"]
            if item.get("expect_refusal"):
                try:
                    audit(*paths, block_mib=1)
                except AuditError as exc:
                    assert str(exc) == expected[label]["refusal"]
                else:
                    raise AssertionError("Malformed adapter was accepted")
                records.append({"group": group, "case": label, "result": "expected refusal"})
                continue
            report = audit(*paths, block_mib=1)
            report_hash = digest(report)
            assert report_hash == expected[label]["complete_report_sha256"], (group, label)
            factor_hashes[label] = digest({"layers": report["layers"]})
            cli_checked = label in (
                "omit-head-f32",
                "wrong-counter-f32",
                "llama-41-merged",
                "omit-full-modules-f32",
                "correct-bf16",
            )
            if cli_checked:
                result = command([*paths, "--block-mib", "1", "--summary", "--json", "-"])
                assert result.returncode == 0 and not result.stderr, result.stderr
                assert digest(json.loads(result.stdout)) == report_hash
                result = command([*paths, "--block-mib", "1", "--summary"])
                assert result.returncode == 0 and not result.stderr
                assert result.stdout == render(report, summary_only=True) + "\n"
            records.append(
                {
                    "group": group,
                    "case": label,
                    "report_sha256": report_hash,
                    "cli_checked": cli_checked,
                }
            )
        assert (
            factor_hashes["correct-f32"]
            == factor_hashes["omit-head-f32" if group == "native" else "omit-full-modules-f32"]
        )
        if group == "native":
            assert factor_hashes["correct-f32"] == factor_hashes["wrong-counter-f32"]
    paths = [str(args.native / name) for name in ("base", "adapter", "correct-f32")]
    protected = []
    for role, path in [
        ("base", args.native / "base/model.safetensors"),
        ("adapter config", args.native / "adapter/adapter_config.json"),
        ("adapter weights", args.native / "adapter/adapter_model.safetensors"),
        ("merged", args.native / "correct-f32/model.safetensors"),
    ]:
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        result = command([*paths, "--json", str(path)])
        assert result.returncode == 2
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        protected.append(role)
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "report.json"
        out.write_text("previous report\n")
        malformed = [
            str(args.native / name) for name in ("base", "incomplete-adapter", "correct-f32")
        ]
        assert command([*malformed, "--json", str(out)]).returncode == 2
        assert out.read_text() == "previous report\n"
    receipt = {
        "method": (
            "isolated installed API equality with complete source reports; real CLI JSON "
            "and text; input protection and prior-report preservation"
        ),
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mergeglass", "numpy", "safetensors")
        },
        "torch_peft_transformers_absent": True,
        "auditor_modules": modules,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": records,
        "identical_factor_measurements_in_full_module_controls": True,
        "protected_input_roles": protected,
        "prior_report_preserved_on_refusal": True,
    }
    args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(
        f"Verified {len(records)} installed native/public cases "
        f"on Python {platform.python_version()}"
    )


if __name__ == "__main__":
    main()
