from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file
from test_audit import change_config, fixture

from mergeglass import AuditError, audit, audit_dora_norms
from mergeglass.cli import main, render


def full_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    base, adapter, merged = fixture(tmp_path)
    change_config(adapter, modules_to_save=["classifier", "score"])
    original = {
        "model.classifier.dense.weight": np.ones((2, 3), dtype=np.float32),
        "model.classifier.dense.bias": np.zeros(2, dtype=np.float32),
        "model.classifier.seen": np.array(2**60, dtype=np.int64),
    }
    trained = {
        "model.classifier.dense.weight": np.full((2, 3), -0.25, dtype=np.float32),
        "model.classifier.dense.bias": np.array([0.25, -0.5], dtype=np.float32),
        "model.classifier.seen": np.array(2**60 + 1, dtype=np.int64),
    }
    for path, payload in [(base, original), (merged, trained)]:
        data = load_file(path / "model.safetensors")
        save_file(data | payload, path / "model.safetensors")
    data = load_file(adapter / "adapter_model.safetensors")
    data.update({"base_model.model." + name: value for name, value in trained.items()})
    save_file(data, adapter / "adapter_model.safetensors")
    return base, adapter, merged


def test_full_state_is_absolute_and_integer_buffer_retains_all_bits(tmp_path: Path) -> None:
    report = audit(*full_fixture(tmp_path))
    assert report["summary"]["saved_module_target_mismatches"] == 0
    assert report["summary"]["unadapted_changed_tensors"] == 0
    assert report["layers"][0]["projection_gain"] == 1
    assert report["unmatched_saved_module_selectors"] == ["score"]
    rows = report["saved_modules"]
    assert all(row["status"] == "matches-saved-module" for row in rows)
    assert [row["expected_change_elements"] for row in rows] == [2, 6, 1]
    assert rows[-1]["norms"] is None


def test_missing_full_module_is_visible_even_when_every_lora_weight_is_correct(
    tmp_path: Path,
) -> None:
    base, adapter, merged = full_fixture(tmp_path)
    data = load_file(merged / "model.safetensors")
    original = load_file(base / "model.safetensors")
    for name in data:
        if name.startswith("model.classifier."):
            data[name] = original[name]
    save_file(data, merged / "model.safetensors")
    report = audit(base, adapter, merged)
    assert report["layers"][0]["relative_residual"] == 0
    assert report["summary"]["saved_module_target_mismatches"] == 9
    assert report["summary"]["unadapted_changed_tensors"] == 0
    assert [row["unchanged_expected_change_elements"] for row in report["saved_modules"]] == [
        2,
        6,
        1,
    ]
    witness = report["saved_modules"][-1]["first_mismatch"]
    assert witness == {"flat_index": 0, "cast_target": 2**60 + 1, "merged": 2**60}
    assert "9 target mismatches" in render(report)
    assert "9 target mismatches" in render(report, summary_only=True)


@pytest.mark.parametrize("case", ["absent", "shape-changed"])
def test_added_head_and_resized_weights_have_no_invented_original_values(
    tmp_path: Path, case: str
) -> None:
    base, adapter, merged = full_fixture(tmp_path)
    path = base / "model.safetensors"
    data = load_file(path)
    name = "model.classifier.dense.weight"
    if case == "absent":
        del data[name]
    else:
        data[name] = np.ones((1, 3), dtype=np.float32)
    save_file(data, path)
    row = next(
        row for row in audit(base, adapter, merged)["saved_modules"] if row["weight"] == name
    )
    assert row["base_comparison_unavailable"] == case
    assert row["changed_from_base_elements"] is None
    assert row["expected_change_elements"] is None
    assert row["unchanged_expected_change_elements"] is None
    assert row["target_mismatches"] == 0


@pytest.mark.parametrize("location", ["adapter", "merged", "reshaped-base"])
def test_nonfinite_full_state_cannot_hide_outside_lora_matrices(
    tmp_path: Path, location: str
) -> None:
    base, adapter, merged = full_fixture(tmp_path)
    directory = adapter if location == "adapter" else merged if location == "merged" else base
    path = directory / (
        "adapter_model.safetensors" if location == "adapter" else "model.safetensors"
    )
    data = load_file(path)
    name = ("base_model.model." if location == "adapter" else "") + "model.classifier.dense.weight"
    data[name] = np.full(
        (1, 1) if location == "reshaped-base" else (2, 3), np.nan, dtype=np.float32
    )
    save_file(data, path)
    with pytest.raises(AuditError, match="Nonfinite weights"):
        audit(base, adapter, merged)


@pytest.mark.parametrize("case", ["partial", "extra", "missing-merged", "integer-dtype", "shape"])
def test_full_state_requires_complete_exact_payload_and_merged_inventory(
    tmp_path: Path, case: str
) -> None:
    base, adapter, merged = full_fixture(tmp_path)
    path = adapter / "adapter_model.safetensors"
    data = load_file(path)
    key = "base_model.model.model.classifier.dense.bias"
    if case == "partial":
        del data[key]
        message = "Incomplete saved module"
    elif case == "extra":
        data["base_model.model.unknown.weight"] = np.zeros(1, dtype=np.float32)
        message = "Unsupported adapter tensor"
    elif case == "integer-dtype":
        key = "base_model.model.model.classifier.seen"
        data[key] = np.array(1, dtype=np.int32)
        message = "Integer/bool dtype"
    elif case == "shape":
        data[key] = np.zeros(3, dtype=np.float32)
        message = "shape mismatch"
    else:
        merged_data = load_file(merged / "model.safetensors")
        del merged_data["model.classifier.dense.bias"]
        save_file(merged_data, merged / "model.safetensors")
        message = "inventories differ"
    save_file(data, path)
    with pytest.raises(AuditError, match=message):
        audit(base, adapter, merged)


@pytest.mark.parametrize("value", [False, "classifier", [""], [1], ["classifier.*"]])
def test_saved_module_configuration_is_not_truthiness_or_regex_matching(
    tmp_path: Path, value: Any
) -> None:
    base, adapter, merged = full_fixture(tmp_path)
    change_config(adapter, modules_to_save=value)
    with pytest.raises(AuditError, match="literal module paths"):
        audit(base, adapter, merged)


def test_overlapping_scopes_or_lora_targets_are_ambiguous(tmp_path: Path) -> None:
    base, adapter, merged = full_fixture(tmp_path)
    change_config(adapter, modules_to_save=["classifier", "dense"])
    with pytest.raises(AuditError, match="Overlapping"):
        audit(base, adapter, merged)
    change_config(adapter, modules_to_save=["classifier", "linear"])
    path = adapter / "adapter_model.safetensors"
    data = load_file(path)
    data["base_model.model.linear.weight"] = np.full((2, 3), 1.125, dtype=np.float32)
    save_file(data, path)
    with pytest.raises(AuditError, match="overlaps a full saved module"):
        audit(base, adapter, merged)


def test_base_less_norm_mode_cannot_silently_ignore_a_full_module(tmp_path: Path) -> None:
    _, adapter, merged = full_fixture(tmp_path)
    with pytest.raises(AuditError, match="complete base/adapter/merged audit"):
        audit_dora_norms(adapter, merged)


def test_boolean_scalar_state_is_exact_and_has_no_float_norm(tmp_path: Path) -> None:
    base, adapter, merged = full_fixture(tmp_path)
    for directory, stem, name, value in [
        (base, "model", "model.classifier.enabled", False),
        (merged, "model", "model.classifier.enabled", False),
        (adapter, "adapter_model", "base_model.model.model.classifier.enabled", True),
    ]:
        path = directory / f"{stem}.safetensors"
        data = load_file(path)
        data[name] = np.array(value, dtype=np.bool_)
        save_file(data, path)
    row = next(
        row
        for row in audit(base, adapter, merged)["saved_modules"]
        if row["weight"] == "model.classifier.enabled"
    )
    assert row["target_mismatches"] == row["unchanged_expected_change_elements"] == 1
    assert row["norms"] is None
    assert row["first_mismatch"] == {"flat_index": 0, "cast_target": True, "merged": False}


def test_full_state_casting_and_zero_target_have_explicit_numeric_semantics(tmp_path: Path) -> None:
    base, adapter, merged = full_fixture(tmp_path)
    path = adapter / "adapter_model.safetensors"
    data = load_file(path)
    key = "base_model.model.model.classifier.dense.weight"
    data[key].fill(1.0001)
    data["base_model.model.model.classifier.dense.bias"].fill(0)
    save_file(data, path)
    path = merged / "model.safetensors"
    data = load_file(path)
    data["model.classifier.dense.weight"] = np.ones((2, 3), dtype=np.float16)
    data["model.classifier.dense.bias"] = np.zeros(2, dtype=np.float16)
    save_file(data, path)
    bias, weight, _ = audit(base, adapter, merged)["saved_modules"]
    assert weight["target_mismatches"] == 0
    assert weight["norms"]["target_cast_drift"] > 0
    assert weight["norms"]["error"] == 0
    assert bias["relative_error"] is None
    assert bias["norms"]["error"] == 0
    assert (
        main(
            [
                str(base),
                str(adapter),
                str(merged),
                "--json",
                str(adapter / "adapter_model.safetensors"),
            ]
        )
        == 2
    )
