from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file
from test_audit import fixture

from mergeglass import AuditError, audit, audit_dora_norms
from mergeglass.cli import render


def embedding_dora(tmp_path: Path) -> tuple[Path, Path, Path]:
    base, adapter, merged = fixture(tmp_path)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 1,
                "lora_alpha": 1,
                "use_dora": True,
                "target_modules": ["tokens"],
            }
        )
    )
    # Each base column has norm 5. Zero directional product, signed magnitudes.
    save_file({"tokens.weight": np.array([[3.0, 0, 3], [4, 5, 4]])}, base / "model.safetensors")
    save_file(
        {
            "base_model.model.tokens.lora_embedding_A": np.zeros((1, 2)),
            "base_model.model.tokens.lora_embedding_B": np.zeros((3, 1)),
            "base_model.model.tokens.lora_magnitude_vector": np.array([10.0, -5, 0]),
        },
        adapter / "adapter_model.safetensors",
    )
    save_file({"tokens.weight": np.array([[6.0, 0, 0], [8, -5, 0]])}, merged / "model.safetensors")
    return base, adapter, merged


def test_signed_column_magnitudes_change_weights_with_zero_factor_product(tmp_path: Path) -> None:
    paths = embedding_dora(tmp_path)
    layer = audit(*paths)["layers"][0]
    assert layer["shape"] == [2, 3] and layer["dora_normalization_axis"] == 0
    assert layer["coordinate_axes"] == {"row": "token_index", "column": "embedding_dimension"}
    assert layer["relative_residual"] == 0 and layer["projection_gain"] == pytest.approx(1)
    assert layer["changed_elements"] == layer["intended_nonzero_elements"] == 5
    assert layer["reference_merge_mismatches"] == 0
    assert "dora_rows" not in layer
    columns = layer["dora_columns"]
    assert columns["columns"] == 3
    assert columns["column_norm_factor_range"] == [1, 1]
    assert columns["direction_cosine_range"] == [1, 1]
    assert columns["zero_magnitude_columns"] == columns["zero_saved_columns"] == 1
    assert columns["negative_magnitude_columns"] == 1
    assert columns["direction_comparable_columns"] == 2
    assert "full-column norm factor" in render(audit(*paths))


def test_column_witness_keeps_the_original_token_index(tmp_path: Path) -> None:
    base, adapter, merged = embedding_dora(tmp_path)
    path = merged / "model.safetensors"
    tensors = load_file(path)
    tensors["tokens.weight"][1, 2] = 0.25
    save_file(tensors, path)
    layer = audit(base, adapter, merged)["layers"][0]
    witness = layer["largest_residual"]
    assert (witness["row"], witness["column"]) == (1, 2)
    assert witness["intended_delta"] == -4 and witness["residual"] == 0.25
    assert layer["dora_columns"]["zero_magnitude_nonzero_saved_columns"] == 1


def test_no_base_does_not_confuse_column_norm_with_token_direction(tmp_path: Path) -> None:
    base, adapter, merged = embedding_dora(tmp_path)
    path = merged / "model.safetensors"
    data = load_file(path)
    data["tokens.weight"] *= -1
    save_file(data, path)
    full = audit(base, adapter, merged)
    partial = audit_dora_norms(adapter, merged)
    assert full["layers"][0]["dora_columns"]["direction_cosine_range"] == [-1, -1]
    assert partial["layers"][0]["dora_columns"]["column_norm_factor_range"] == [1, 1]
    assert partial["summary"]["norm_observed_rows"] == 0
    assert partial["summary"]["norm_observed_columns"] == 3
    assert "direction_cosine_range" not in partial["layers"][0]["dora_columns"]
    text = render(partial)
    assert "3 embedding columns" in text and "AXIS" in text and "column" in text
    assert "direction cosine" not in text


@pytest.mark.parametrize("magnitude", [np.ones(2), np.ones((1, 3))])
def test_embedding_magnitude_is_one_value_per_dimension(
    tmp_path: Path, magnitude: np.ndarray
) -> None:
    base, adapter, merged = embedding_dora(tmp_path)
    path = adapter / "adapter_model.safetensors"
    data = load_file(path)
    data["base_model.model.tokens.lora_magnitude_vector"] = magnitude
    save_file(data, path)
    with pytest.raises(AuditError, match="magnitude shape must match the embedding dimensions"):
        audit(base, adapter, merged)


def test_undefined_full_vocabulary_direction_refuses_instead_of_adding_epsilon(
    tmp_path: Path,
) -> None:
    base, adapter, merged = embedding_dora(tmp_path)
    save_file({"tokens.weight": np.zeros((2, 3))}, base / "model.safetensors")
    with pytest.raises(AuditError, match="Undefined Embedding DoRA direction.*column 0"):
        audit(base, adapter, merged)
    # With no original base, only the supplied saved column lengths are known.
    assert audit_dora_norms(adapter, merged)["layers"][0]["dora_columns"]["columns"] == 3


def test_float64_inputs_remain_read_only_across_both_passes(tmp_path: Path) -> None:
    paths = embedding_dora(tmp_path)
    files = [p for folder in paths for p in folder.iterdir() if p.is_file()]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for p in files:
        p.chmod(0o444)
    assert audit(*paths)["layers"][0]["relative_residual"] == 0
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


@pytest.mark.parametrize("saved", [1e308, 1e-308])
def test_partial_column_lengths_retain_factors_outside_float64_range(
    tmp_path: Path, saved: float
) -> None:
    base, adapter, merged = embedding_dora(tmp_path)
    save_file({"tokens.weight": np.full((4, 1), saved)}, merged / "model.safetensors")
    save_file(
        {
            "base_model.model.tokens.lora_embedding_A": np.zeros((1, 4)),
            "base_model.model.tokens.lora_embedding_B": np.zeros((1, 1)),
            "base_model.model.tokens.lora_magnitude_vector": np.array([1e308]),
        },
        adapter / "adapter_model.safetensors",
    )
    columns = audit_dora_norms(adapter, merged)["layers"][0]["dora_columns"]
    if saved == 1e308:
        assert columns["column_norm_factor_range"] == [2, 2]
        assert columns["largest_nonzero_log2_norm_error"]["saved_column_norm"] is None
    else:
        assert columns["column_norm_factor_range"] == [None, None]
        assert columns["log2_column_norm_factor_range"][0] < -2000


def test_norms_include_tokens_outside_the_current_audit_block(tmp_path: Path) -> None:
    base, adapter, merged = embedding_dora(tmp_path)
    tokens = np.arange(5003) % 3 - 1
    dimensions = np.arange(1, 8)
    a = tokens[None, :].astype(np.float64) / 8
    b = dimensions[:, None].astype(np.float64) / 16
    factors = np.array([0.0, -0.5, 0.75, 1.25, 1.5, 2, 3])
    # Three exact rational direction groups, counted independently of any block.
    counts = np.bincount(tokens + 1, minlength=3)
    groups = 1 + np.array([-1.0, 0, 1])[:, None] * dimensions[None, :] / 128
    lengths = np.sqrt((counts[:, None] * groups**2).sum(axis=0))
    save_file({"tokens.weight": np.ones((5003, 7), dtype=np.float32)}, base / "model.safetensors")
    save_file(
        {
            "base_model.model.tokens.lora_embedding_A": a,
            "base_model.model.tokens.lora_embedding_B": b,
            "base_model.model.tokens.lora_magnitude_vector": lengths * factors,
        },
        adapter / "adapter_model.safetensors",
    )
    expected = ((1 + tokens[:, None] * dimensions[None, :] / 128) * factors).astype(np.float32)
    save_file({"tokens.weight": expected}, merged / "model.safetensors")
    for block_mib in [1, 16]:
        layer = audit(base, adapter, merged, block_mib=block_mib)["layers"][0]
        assert layer["changed_elements"] == layer["intended_nonzero_elements"] == 5003 * 7
        assert layer["reference_merge_mismatches"] == 0
        assert layer["relative_residual"] < 1e-13
        assert layer["dora_columns"]["column_norm_factor_range"] == pytest.approx([1, 1])
