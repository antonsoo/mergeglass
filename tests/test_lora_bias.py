from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file
from test_audit import change_config, fixture

from mergeglass import AuditError, audit
from mergeglass.cli import render


def bias_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    base, adapter, merged = fixture(tmp_path, alpha=2)
    change_config(adapter, lora_bias=True)
    for path, value in [(base, [1, -1]), (merged, [1.25, -1.5])]:
        tensors = load_file(path / "model.safetensors")
        tensors["linear.bias"] = np.array(value, dtype=np.float32)
        save_file(tensors, path / "model.safetensors")
    factors = load_file(adapter / "adapter_model.safetensors")
    factors["base_model.model.linear.lora_B.bias"] = np.array([0.125, -0.25], dtype=np.float32)
    save_file(factors, adapter / "adapter_model.safetensors")
    return base, adapter, merged


@pytest.mark.parametrize(
    "fault,gain,residual", [(None, 1, 0), ("omitted", 0, 1), ("unscaled", 0.5, 0.5)]
)
def test_bias_fault_cannot_hide_behind_correct_matrix(
    tmp_path: Path, fault: str | None, gain: float, residual: float
) -> None:
    base, adapter, merged = bias_fixture(tmp_path)
    if fault:
        tensors = load_file(merged / "model.safetensors")
        tensors["linear.bias"] = np.array(
            [1, -1] if fault == "omitted" else [1.125, -1.25], dtype=np.float32
        )
        save_file(tensors, merged / "model.safetensors")
    result = audit(base, adapter, merged)
    assert result["layers"][0]["relative_residual"] == 0
    assert result["summary"]["unadapted_changed_tensors"] == 0
    bias = result["lora_biases"][0]
    assert bias["shape"] == [2]
    assert bias["projection_gain"] == pytest.approx(gain)
    assert bias["relative_residual"] == pytest.approx(residual)
    assert bias["reference_merge_mismatches"] == (2 if fault else 0)
    assert "index" in bias["largest_residual"] and "column" not in bias["largest_residual"]
    for summary_only in [False, True]:
        text = render(result, summary_only=summary_only)
        assert "LoRA B biases: 1 vectors / 2 entries" in text
        assert "separate from matrix counts" in text


@pytest.mark.parametrize(
    "fault", ["absent", "wrong-shape", "missing-factor", "undeclared", "not-bool", "dora"]
)
def test_bias_contract_requires_complete_declared_evidence(tmp_path: Path, fault: str) -> None:
    base, adapter, merged = bias_fixture(tmp_path)
    if fault in {"absent", "wrong-shape"}:
        tensors = load_file(base / "model.safetensors")
        if fault == "absent":
            # Keep base/merged inventories equal so the adapter-specific check
            # must refuse the missing original bias, not just a missing key.
            del tensors["linear.bias"]
            target = load_file(merged / "model.safetensors")
            del target["linear.bias"]
            save_file(target, merged / "model.safetensors")
        else:
            tensors["linear.bias"] = np.zeros((2, 1), dtype=np.float32)
        save_file(tensors, base / "model.safetensors")
    elif fault == "missing-factor":
        tensors = load_file(adapter / "adapter_model.safetensors")
        del tensors["base_model.model.linear.lora_B.bias"]
        save_file(tensors, adapter / "adapter_model.safetensors")
    else:
        change_config(
            adapter,
            **(
                {"use_dora": True}
                if fault == "dora"
                else {"lora_bias": "true" if fault == "not-bool" else False}
            ),
        )
    with pytest.raises(AuditError):
        audit(base, adapter, merged)


def test_zero_bias_keeps_unavailable_ratios_and_old_matrix_counts(tmp_path: Path) -> None:
    base, adapter, merged = bias_fixture(tmp_path)
    factors = load_file(adapter / "adapter_model.safetensors")
    factors["base_model.model.linear.lora_B.bias"].fill(0)
    save_file(factors, adapter / "adapter_model.safetensors")
    tensors = load_file(merged / "model.safetensors")
    tensors["linear.bias"] = np.array([1, -1], dtype=np.float32)
    save_file(tensors, merged / "model.safetensors")
    result = audit(base, adapter, merged)
    assert result["lora_biases"][0]["status"] == "zero-update"
    assert result["lora_biases"][0]["projection_gain"] is None
    assert result["summary"]["adapted_tensors"] == 1
    assert result["summary"]["adapted_elements"] == 6
    json.dumps(result, allow_nan=False)


def test_bias_only_update_is_not_described_as_a_zero_adapter(tmp_path: Path) -> None:
    base, adapter, merged = bias_fixture(tmp_path)
    factors = load_file(adapter / "adapter_model.safetensors")
    factors["base_model.model.linear.lora_B.weight"].fill(0)
    save_file(factors, adapter / "adapter_model.safetensors")
    tensors = load_file(merged / "model.safetensors")
    tensors["linear.weight"] = load_file(base / "model.safetensors")["linear.weight"]
    save_file(tensors, merged / "model.safetensors")
    result = audit(base, adapter, merged)
    assert result["summary"]["intended_nonzero_elements"] == 0
    assert result["lora_biases"][0]["projection_gain"] == pytest.approx(1)
    for summary_only in [False, True]:
        text = render(result, summary_only=summary_only)
        assert "zero across all adapted matrices" in text
        assert "zero across all adapted weights" not in text
