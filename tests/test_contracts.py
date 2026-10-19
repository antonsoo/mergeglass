from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import save_file
from test_audit import change_config, fixture

from mergeglass import AuditError, audit
from mergeglass.adapter import _override
from mergeglass.cli import main
from mergeglass.storage import Checkpoint, cast


def test_rank_alpha_patterns_first_match_and_rslora(tmp_path: Path) -> None:
    base, adapter, merged = fixture(tmp_path)
    change_config(
        adapter,
        r=8,
        lora_alpha=32,
        use_rslora=True,
        rank_pattern={"linear": 1},
        alpha_pattern={"linear": 2},
    )
    save_file(
        {
            "linear.weight": np.full((2, 3), 1.25, dtype=np.float32),
            "control": np.arange(4, dtype=np.int64),
        },
        merged / "model.safetensors",
    )
    result = audit(base, adapter, merged)["layers"][0]
    assert result["scale"] == 2
    assert result["rank"] == 1
    assert result["relative_residual"] == 0
    assert _override("model.layers.0.q_proj", {"q_proj": 2, "layers.0.q_proj": 4}, 8) == 2
    # Dots retain PEFT's regex meaning, even in this limited pattern alphabet.
    assert _override("model.foo_bar", {"foo.bar": 3}, 8) == 3


def test_off_diagonal_cancellation_is_not_a_nonzero_update(tmp_path: Path) -> None:
    base, adapter, merged = fixture(tmp_path)
    change_config(adapter, r=2)
    save_file(
        {
            "base_model.model.linear.lora_A.weight": np.ones((2, 3), dtype=np.float32),
            "base_model.model.linear.lora_B.weight": np.array([[1, -1], [2, -2]], dtype=np.float32),
        },
        adapter / "adapter_model.safetensors",
    )
    (merged / "model.safetensors").write_bytes((base / "model.safetensors").read_bytes())
    layer = audit(base, adapter, merged)["layers"][0]
    assert layer["status"] == "zero-update"
    assert layer["norms"]["delta"] == 0


def test_bf16_raw_decoding_uses_native_safetensors_bytes() -> None:
    path = Path(__file__).resolve().parents[1] / "examples/rounding/merged-bf16"
    tensor = Checkpoint(path).tensors["large.weight"]
    values = tensor.floats()
    np.testing.assert_array_equal(cast(values, "BF16"), values)
    assert tensor.dtype == "BF16"


def test_index_traversal_and_nonstring_shards_refuse(tmp_path: Path) -> None:
    base, _adapter, _merged = fixture(tmp_path)
    (base / "model.safetensors").unlink()
    index = base / "model.safetensors.index.json"
    for value in ["../other.safetensors", ["not a filename"], "/tmp/a.safetensors"]:
        index.write_text(json.dumps({"weight_map": {"linear.weight": value}}))
        with pytest.raises(AuditError):
            Checkpoint(base)


def test_duplicate_config_key_refuses(tmp_path: Path) -> None:
    base, adapter, merged = fixture(tmp_path)
    (adapter / "adapter_config.json").write_text('{"peft_type":"LORA","r":1,"r":2}')
    with pytest.raises(AuditError, match="Duplicate"):
        audit(base, adapter, merged)


def test_quantized_config_refuses(tmp_path: Path) -> None:
    base, adapter, merged = fixture(tmp_path)
    (base / "config.json").write_text('{"quantization_config":{"quant_method":"gptq"}}')
    with pytest.raises(AuditError, match="Quantized"):
        audit(base, adapter, merged)


def test_input_change_during_read_refuses(tmp_path: Path, monkeypatch: Any) -> None:
    base, adapter, merged = fixture(tmp_path)
    original = Checkpoint.fingerprints

    def tamper(checkpoint: Checkpoint) -> list[dict[str, Any]]:
        data = original(checkpoint)
        if checkpoint.source == base:
            # A concurrent training/save job changes the original base after
            # its fingerprint but before the other inputs finish hashing.
            path = base / "model.safetensors"
            path.write_bytes(path.read_bytes() + b"changed")
        return data

    monkeypatch.setattr(Checkpoint, "fingerprints", tamper)
    with pytest.raises(AuditError, match="changed during audit"):
        audit(base, adapter, merged)


def test_scaled_norm_preserves_tiny_nonzero_reference(tmp_path: Path) -> None:
    base, adapter, merged = fixture(tmp_path)
    save_file(
        {
            "base_model.model.linear.lora_A.weight": np.full((1, 3), 1e-25, dtype=np.float32),
            "base_model.model.linear.lora_B.weight": np.full((2, 1), 1e-25, dtype=np.float32),
        },
        adapter / "adapter_model.safetensors",
    )
    (merged / "model.safetensors").write_bytes((base / "model.safetensors").read_bytes())
    layer = audit(base, adapter, merged)["layers"][0]
    assert layer["norms"]["delta"] > 0
    assert layer["status"] == "no-observed-update"
    assert layer["relative_residual"] == 1


def test_multiple_blocks_match_an_exact_binary_fraction_reference(tmp_path: Path) -> None:
    base, adapter, merged = fixture(tmp_path)
    rows, cols = 257, 513
    row_term = np.arange(rows, dtype=np.float32) % 7 - 3
    col_term = np.arange(cols, dtype=np.float32) % 5
    a = np.stack([np.ones(cols, dtype=np.float32), col_term]) / 16
    b = np.stack([row_term, np.ones(rows, dtype=np.float32)], axis=1) / 16
    # Closed form rather than a second matrix multiply. All values are exact
    # binary fractions and survive FP32 addition to this base.
    expected_delta = (row_term[:, None] + col_term[None, :]) / 256
    save_file(
        {"linear.weight": np.ones((rows, cols), dtype=np.float32)}, base / "model.safetensors"
    )
    save_file({"linear.weight": 1 + expected_delta}, merged / "model.safetensors")
    save_file(
        {"base_model.model.linear.lora_A.weight": a, "base_model.model.linear.lora_B.weight": b},
        adapter / "adapter_model.safetensors",
    )
    change_config(adapter, r=2, lora_alpha=2)
    small = audit(base, adapter, merged, block_mib=1)
    large = audit(base, adapter, merged, block_mib=16)
    assert small["layers"] == large["layers"]
    layer = small["layers"][0]
    assert layer["relative_residual"] == 0
    assert layer["changed_elements"] == int(np.count_nonzero(expected_delta))


def test_hf_snapshot_links_work_and_cannot_be_overwritten_by_json(tmp_path: Path) -> None:
    base, adapter, merged = fixture(tmp_path)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    original = (base / "model.safetensors").read_bytes()
    blob = blobs / "tensor-data"
    blob.write_bytes(original)
    (base / "model.safetensors").unlink()
    (base / "shard.safetensors").symlink_to(blob)
    (base / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"linear.weight": "shard.safetensors", "control": "shard.safetensors"}}
        )
    )
    assert audit(base, adapter, merged)["layers"][0]["relative_residual"] == 0
    assert main([str(base), str(adapter), str(merged), "--json", str(blob)]) == 2
    assert blob.read_bytes() == original
