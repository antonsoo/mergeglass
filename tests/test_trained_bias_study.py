"""The retained oracle must refuse incomplete evidence, even under python -O."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/trained-bias"


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "producer"
    root.mkdir()
    with tarfile.open(STUDY / "artifacts/modern.tar.gz") as archive:
        for member in archive.getmembers():
            data = archive.extractfile(member)
            assert data is not None
            dest = root / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data.read())
    return root


def run(root: Path, output: Path, *, optimized: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            *(["-O"] if optimized else []),
            str(STUDY / "verify.py"),
            str(root),
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def rehash(root: Path) -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            data = path.read_bytes()
            entries.append(
                {
                    "file": path.relative_to(root).as_posix(),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    (root / "manifest.json").write_text(json.dumps(entries))


@pytest.mark.parametrize("optimized", [False, True])
def test_complete_retained_matrix_is_verified(
    corpus: Path, tmp_path: Path, optimized: bool
) -> None:
    output = tmp_path / "verified.json.gz"
    result = run(corpus, output, optimized=optimized)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(gzip.decompress(output.read_bytes()))
    assert receipt["state"] == "verified"
    assert receipt["measured_cases"] == receipt["expected_measured_cases"] == 36
    assert receipt["refused_cases"] == receipt["expected_refused_cases"] == 0
    assert receipt["bias_entries_checked"] == 1422


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-case",
        "wrong-precision",
        "incorrect-native-bias",
        "unlisted-file",
        "duplicate-identity",
    ],
)
def test_forged_complete_receipt_is_refused_under_optimization(
    corpus: Path, tmp_path: Path, mutation: str
) -> None:
    expected = ""
    if mutation == "missing-case":
        shutil.rmtree(corpus / "all/reloaded-bf16-safe-true")
        rehash(corpus)
        expected = "merge cases"
    elif mutation == "wrong-precision":
        shutil.copyfile(
            corpus / "all/reloaded-f32-safe-true/model.safetensors",
            corpus / "all/reloaded-bf16-safe-true/model.safetensors",
        )
        rehash(corpus)
        expected = "Stored precision differs"
    elif mutation == "incorrect-native-bias":
        path = corpus / "all/reloaded-f32-safe-false/model.safetensors"
        data = load_file(path)
        data["first.bias"] += np.float32(1)
        save_file(data, path)
        rehash(corpus)
        expected = "Native merge has incorrect bias targets"
    elif mutation == "unlisted-file":
        (corpus / "not-fingerprinted.txt").write_text("unverified producer output")
        expected = "exactly every producer file"
    else:
        path = corpus / "manifest.json"
        data = json.loads(path.read_text())
        data.append(data[0])
        path.write_text(json.dumps(data))
        expected = "Duplicate manifest"
    output = tmp_path / "incorrect-pass.json.gz"
    result = run(corpus, output)
    assert result.returncode == 2
    assert expected in result.stderr, result.stderr
    assert not output.exists()
