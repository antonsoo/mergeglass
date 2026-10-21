from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file
from test_audit import fixture

from mergeglass import AuditError, audit, audit_dora_norms
from mergeglass.cli import main, render
from mergeglass.storage import Tensor


def embedding_fixture(tmp_path: Path, *, alpha: float = 2) -> tuple[Path, Path, Path]:
    base, adapter, merged = fixture(tmp_path)
    config = json.loads((adapter / "adapter_config.json").read_text())
    config.update(r=2, lora_alpha=alpha)
    (adapter / "adapter_config.json").write_text(json.dumps(config))
    # Rank-two update with a coordinate-wise closed form, no matmul oracle.
    token = np.arange(37, dtype=np.float32)
    dimension = np.arange(11, dtype=np.float32)
    a = np.stack([token, np.ones(37, dtype=np.float32)]) / 16
    b = np.stack([np.ones(11, dtype=np.float32), 2 * dimension], axis=1) / 16
    delta = (token[:, None] + 2 * dimension[None, :]) / 256 * (alpha / 2)
    save_file({"tokens.weight": np.ones((37, 11), dtype=np.float32)}, base / "model.safetensors")
    save_file(
        {
            "base_model.model.tokens.lora_embedding_A": a,
            "base_model.model.tokens.lora_embedding_B": b,
        },
        adapter / "adapter_model.safetensors",
    )
    save_file({"tokens.weight": 1 + delta}, merged / "model.safetensors")
    return base, adapter, merged


@pytest.mark.parametrize("alpha", [-2, 0, 2, 4])
def test_embedding_rows_are_token_coordinates_and_keep_signed_scales(
    tmp_path: Path, alpha: float
) -> None:
    base, adapter, merged = embedding_fixture(tmp_path, alpha=alpha)
    report = audit(base, adapter, merged)
    layer = report["layers"][0]
    assert layer["weight_layout"] == "embedding"
    assert layer["shape"] == [37, 11]
    assert layer["scale"] == alpha / 2
    assert layer["relative_residual"] == (0 if alpha else None)
    assert layer["projection_gain"] == (pytest.approx(1) if alpha else None)
    assert layer["changed_elements"] == (406 if alpha else 0)
    assert layer["reference_merge_mismatches"] == 0
    assert "rows are token indices" in render(report)


def test_largest_embedding_residual_keeps_original_token_and_dimension(tmp_path: Path) -> None:
    base, adapter, merged = embedding_fixture(tmp_path)
    data = load_file(merged / "model.safetensors")
    data["tokens.weight"][35, 9] += 0.25
    save_file(data, merged / "model.safetensors")
    witness = audit(base, adapter, merged)["layers"][0]["largest_residual"]
    assert witness["row"] == 35
    assert witness["column"] == 9
    assert witness["intended_delta"] == (35 + 2 * 9) / 256
    assert witness["residual"] == 0.25


def test_omitted_embeddings_cannot_be_hidden_by_a_successful_linear_merge(tmp_path: Path) -> None:
    base, adapter, merged = embedding_fixture(tmp_path)
    for directory, name, tensors in [
        (base, "model", {"projection.weight": np.ones((2, 11), dtype=np.float32)}),
        (merged, "model", {"projection.weight": np.full((2, 11), 1.25, dtype=np.float32)}),
        (
            adapter,
            "adapter_model",
            {
                "base_model.model.projection.lora_A.weight": np.ones((2, 11), dtype=np.float32),
                "base_model.model.projection.lora_B.weight": np.full(
                    (2, 2), 0.125, dtype=np.float32
                ),
            },
        ),
    ]:
        data = load_file(directory / f"{name}.safetensors")
        data.update(tensors)
        if directory == merged:
            data["tokens.weight"].fill(1)
        save_file(data, directory / f"{name}.safetensors")
    report = audit(base, adapter, merged)
    linear, tokens = report["layers"]
    assert linear["relative_residual"] == 0
    assert "weight_layout" not in linear
    assert tokens["status"] == "no-observed-update"
    assert tokens["unchanged_nonzero_update_fraction"] == 1
    assert tokens["relative_residual"] == 1
    assert report["summary"]["no_observed_update_tensors"] == 1


@pytest.mark.parametrize(
    "bad", ["mixed", "incomplete", "transposed", "missing_dora_magnitude", "extra", "nonfinite"]
)
def test_embedding_evidence_cannot_be_ambiguous_or_silently_ignored(
    tmp_path: Path, bad: str
) -> None:
    base, adapter, merged = embedding_fixture(tmp_path)
    path = adapter / "adapter_model.safetensors"
    data = load_file(path)
    if bad == "mixed":
        data["base_model.model.tokens.lora_A.weight"] = np.ones((2, 11), dtype=np.float32)
        message = "Mixed Linear and Embedding"
    elif bad == "incomplete":
        del data["base_model.model.tokens.lora_embedding_B"]
        message = "Incomplete"
    elif bad == "transposed":
        data["base_model.model.tokens.lora_embedding_A"] = data[
            "base_model.model.tokens.lora_embedding_A"
        ].T.copy()
        message = "shape mismatch"
    elif bad == "missing_dora_magnitude":
        config_path = adapter / "adapter_config.json"
        config = json.loads(config_path.read_text())
        config["use_dora"] = True
        config_path.write_text(json.dumps(config))
        message = "Missing DoRA magnitude"
    elif bad == "extra":
        data["base_model.model.tokens.weight"] = np.ones((37, 11), dtype=np.float32)
        message = "Unsupported adapter tensor"
    else:
        data["base_model.model.tokens.lora_embedding_A"][1, -1] = np.nan
        message = "Nonfinite weights"
    save_file(data, path)
    with pytest.raises(AuditError, match=message):
        audit(base, adapter, merged)
    if bad == "missing_dora_magnitude":
        with pytest.raises(AuditError, match="Missing DoRA magnitude"):
            audit_dora_norms(adapter, merged)


def test_default_embedding_save_must_prove_its_snapshot_matches_the_original(
    tmp_path: Path,
) -> None:
    base, adapter, merged = embedding_fixture(tmp_path)
    path = adapter / "adapter_model.safetensors"
    data = load_file(path)
    key = "base_model.model.tokens.base_layer.weight"
    data[key] = np.ones((37, 11), dtype=np.float16)
    save_file(data, path)
    layer = audit(base, adapter, merged)["layers"][0]
    assert layer["relative_residual"] == 0
    assert layer["base_snapshot"]["dtype"] == "F16"
    data[key][36, 10] += 0.5
    save_file(data, path)
    with pytest.raises(AuditError, match="snapshot differs"):
        audit(base, adapter, merged)


def test_vocabulary_wide_a_is_streamed_under_the_fixed_factor_limit(
    tmp_path: Path, monkeypatch: Any
) -> None:
    base, adapter, merged = embedding_fixture(tmp_path)
    rows = 100_003
    a = np.ones((2, rows), dtype=np.float32) / 8
    b = np.ones((3, 2), dtype=np.float32) / 8
    save_file({"tokens.weight": np.ones((rows, 3), dtype=np.float32)}, base / "model.safetensors")
    save_file(
        {"tokens.weight": np.full((rows, 3), 1 + 1 / 32, dtype=np.float32)},
        merged / "model.safetensors",
    )
    save_file(
        {
            "base_model.model.tokens.lora_embedding_A": a,
            "base_model.model.tokens.lora_embedding_B": b,
        },
        adapter / "adapter_model.safetensors",
    )
    original = Tensor.floats
    reads = []

    def read(self: Tensor, start: int = 0, count: int | None = None) -> Any:
        if self.name.endswith("lora_embedding_A"):
            assert count is not None and count < rows
            reads.append((start, count))
        return original(self, start, count)

    monkeypatch.setattr(Tensor, "floats", read)
    small = audit(base, adapter, merged, block_mib=1, max_adapter_mib=1)
    assert len(reads) > 2
    assert sum(count for _, count in reads) == a.size
    large = audit(base, adapter, merged, block_mib=16, max_adapter_mib=1)
    assert small["layers"] == large["layers"]
    assert small["layers"][0]["relative_residual"] == 0


def test_embedding_b_has_its_own_decoded_size_limit_and_report_is_preserved(tmp_path: Path) -> None:
    base, adapter, merged = embedding_fixture(tmp_path)
    columns = 70_000
    save_file(
        {"tokens.weight": np.ones((2, columns), dtype=np.float32)}, base / "model.safetensors"
    )
    save_file(
        {"tokens.weight": np.ones((2, columns), dtype=np.float32)}, merged / "model.safetensors"
    )
    save_file(
        {
            "base_model.model.tokens.lora_embedding_A": np.ones((2, 2), dtype=np.float32),
            "base_model.model.tokens.lora_embedding_B": np.ones((columns, 2), dtype=np.float32),
        },
        adapter / "adapter_model.safetensors",
    )
    report = tmp_path / "report.json"
    report.write_text("earlier report")
    assert (
        main(
            [str(base), str(adapter), str(merged), "--max-adapter-mib", "1", "--json", str(report)]
        )
        == 2
    )
    assert report.read_text() == "earlier report"
    with pytest.raises(AuditError, match="B exceeds"):
        audit(base, adapter, merged, max_adapter_mib=1)
