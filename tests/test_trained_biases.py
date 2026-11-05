from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file
from test_audit import change_config, fixture

from mergeglass import AuditError, audit, audit_dora_norms
from mergeglass.cli import render


def bias_fixture(tmp_path: Path, mode: str = "all") -> tuple[Path, Path, Path]:
    base, adapter, merged = fixture(tmp_path, alpha=3)
    change_config(adapter, bias=mode)
    for directory, bias in [(base, [0.25, -0.5]), (merged, [0.75, 0.125])]:
        path = directory / "model.safetensors"
        data = load_file(path)
        data["linear.bias"] = np.array(bias, dtype=np.float32)
        data["head.bias"] = np.array([0.5], dtype=np.float32)
        if directory == merged and mode == "all":
            data["head.bias"] += 0.25
        save_file(data, path)
    path = adapter / "adapter_model.safetensors"
    data = load_file(path)
    data["base_model.model.linear.base_layer.bias"] = np.array([0.75, 0.125], dtype=np.float32)
    if mode == "all":
        data["base_model.model.head.bias"] = np.array([0.75], dtype=np.float32)
    save_file(data, path)
    return base, adapter, merged


@pytest.mark.parametrize("mode,count", [("all", 3), ("lora_only", 2)])
def test_bias_target_is_absolute_and_does_not_use_alpha(
    tmp_path: Path, mode: str, count: int
) -> None:
    report = audit(*bias_fixture(tmp_path, mode))
    assert report["layers"][0]["scale"] == 3
    assert report["layers"][0]["relative_residual"] == 0
    assert report["summary"]["trained_bias_target_mismatches"] == 0
    assert report["summary"]["trained_bias_elements"] == count
    assert report["summary"]["unadapted_changed_tensors"] == 0
    assert all(row["status"] == "matches-trained-bias" for row in report["trained_biases"])
    assert "0 target mismatches" in render(report, summary_only=True)
    assert "without LoRA scaling" in render(report)
    assert "saved_modules" not in report
    assert "lora_biases" not in report


def test_omitting_bias_preserves_every_matrix_but_has_exact_saved_coordinates(
    tmp_path: Path,
) -> None:
    base, adapter, merged = bias_fixture(tmp_path)
    data = load_file(merged / "model.safetensors")
    original = load_file(base / "model.safetensors")
    for name in ["linear.bias", "head.bias"]:
        data[name] = original[name]
    save_file(data, merged / "model.safetensors")
    report = audit(base, adapter, merged)
    assert report["layers"][0]["relative_residual"] == 0
    assert report["summary"]["trained_bias_target_mismatches"] == 3
    assert report["summary"]["trained_bias_unchanged_expected_change_elements"] == 3
    row = next(row for row in report["trained_biases"] if row["weight"] == "linear.bias")
    assert row["first_mismatch"] == {"flat_index": 0, "cast_target": 0.75, "merged": 0.25}
    assert row["norms"]["error"] == pytest.approx(np.hypot(0.5, 0.625))
    assert "3 target mismatches" in render(report)


@pytest.mark.parametrize("mode", ["all", "lora_only"])
def test_missing_trained_value_cannot_be_reconstructed_from_original_base(
    tmp_path: Path, mode: str
) -> None:
    base, adapter, merged = bias_fixture(tmp_path, mode)
    path = adapter / "adapter_model.safetensors"
    data = load_file(path)
    del data["base_model.model.linear.base_layer.bias"]
    save_file(data, path)
    with pytest.raises(AuditError, match="Missing trained base bias: linear.bias"):
        audit(base, adapter, merged)


@pytest.mark.parametrize("location", ["base", "adapter", "merged"])
def test_nonfinite_bias_refuses_entire_audit(tmp_path: Path, location: str) -> None:
    inputs = bias_fixture(tmp_path)
    directory = inputs[{"base": 0, "adapter": 1, "merged": 2}[location]]
    path = directory / (
        "adapter_model.safetensors" if location == "adapter" else "model.safetensors"
    )
    data = load_file(path)
    key = "base_model.model.linear.base_layer.bias" if location == "adapter" else "linear.bias"
    data[key][0] = np.nan
    save_file(data, path)
    with pytest.raises(AuditError, match="Nonfinite weights"):
        audit(*inputs)


@pytest.mark.parametrize("kind", ["duplicate", "extra", "shape", "integer", "missing-merged"])
def test_bad_payload_never_produces_a_partial_success(tmp_path: Path, kind: str) -> None:
    base, adapter, merged = bias_fixture(tmp_path)
    path = adapter / "adapter_model.safetensors"
    data = load_file(path)
    key = "base_model.model.linear.base_layer.bias"
    message = ""
    if kind == "duplicate":
        data["base_model.model.linear.bias"] = data[key]
        message = "Ambiguous trained base bias aliases"
    elif kind == "extra":
        data["base_model.model.unknown.bias"] = data[key]
        message = "Unsupported adapter tensor"
    elif kind == "shape":
        data[key] = np.zeros(3, dtype=np.float32)
        message = "shape/dtype mismatch"
    elif kind == "integer":
        data[key] = data[key].astype(np.int32)
        message = "shape/dtype mismatch"
    else:
        values = load_file(merged / "model.safetensors")
        del values["linear.bias"]
        save_file(values, merged / "model.safetensors")
        message = "inventories differ"
    save_file(data, path)
    with pytest.raises(AuditError, match=message):
        audit(base, adapter, merged)


def test_literal_legacy_key_is_accepted_without_suffix_guessing(tmp_path: Path) -> None:
    base, adapter, merged = bias_fixture(tmp_path, "lora_only")
    path = adapter / "adapter_model.safetensors"
    data = load_file(path)
    data["base_model.model.linear.bias"] = data.pop("base_model.model.linear.base_layer.bias")
    save_file(data, path)
    assert audit(base, adapter, merged)["summary"]["trained_bias_target_mismatches"] == 0


@pytest.mark.parametrize("mode", ["all", "lora_only"])
def test_biasless_model_has_no_invented_bias_targets(tmp_path: Path, mode: str) -> None:
    inputs = fixture(tmp_path)
    change_config(inputs[1], bias=mode)
    assert "trained_biases" not in audit(*inputs)


@pytest.mark.parametrize(
    "option,value",
    [("lora_bias", True), ("modules_to_save", ["head"]), ("target_parameters", ["experts"])],
)
def test_unverified_combination_refuses(tmp_path: Path, option: str, value: object) -> None:
    inputs = bias_fixture(tmp_path)
    change_config(inputs[1], **{option: value})
    with pytest.raises(AuditError, match="combined with"):
        audit(*inputs)


def test_no_base_mode_cannot_authorize_skipping_bias_payload(tmp_path: Path) -> None:
    _, adapter, merged = bias_fixture(tmp_path)
    with pytest.raises(AuditError, match="complete base/adapter/merged audit"):
        audit_dora_norms(adapter, merged)
