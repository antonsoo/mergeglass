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
    [("lora_bias", True), ("target_parameters", ["experts"])],
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


def saved_head_fixture(
    tmp_path: Path, mode: str = "all", *, nested: bool = False
) -> tuple[Path, Path, Path]:
    base, adapter, merged = bias_fixture(tmp_path, mode)
    change_config(adapter, modules_to_save=["head"])
    module = "head.dense" if nested else "head"
    for directory, weight, bias in [(base, 0.25, 0.5), (merged, 0.75, -0.125)]:
        path = directory / "model.safetensors"
        data = load_file(path)
        data.pop("head.bias")
        data[module + ".weight"] = np.full((1, 2), weight, dtype=np.float32)
        data[module + ".bias"] = np.array([bias], dtype=np.float32)
        save_file(data, path)
    path = adapter / "adapter_model.safetensors"
    data = load_file(path)
    data.pop("base_model.model.head.bias", None)
    data[f"base_model.model.{module}.weight"] = np.full((1, 2), 0.75, dtype=np.float32)
    data[f"base_model.model.{module}.bias"] = np.array([-0.125], dtype=np.float32)
    if mode == "all":
        suffix = "dense.bias" if nested else "bias"
        data[f"base_model.model.head.modules_to_save.{'default.' if nested else ''}{suffix}"] = (
            np.array([-0.125], dtype=np.float32)
        )
        data[f"base_model.model.head.original_module.{suffix}"] = np.array([0.5], dtype=np.float32)
    save_file(data, path)
    return base, adapter, merged


@pytest.mark.parametrize("mode", ["all", "lora_only"])
@pytest.mark.parametrize("nested", [False, True])
def test_full_saved_head_and_trained_bias_have_separate_complete_measurements(
    tmp_path: Path, mode: str, nested: bool
) -> None:
    report = audit(*saved_head_fixture(tmp_path, mode, nested=nested))
    summary = report["summary"]
    assert summary["trained_bias_tensors"] == 1
    assert summary["trained_bias_elements"] == 2
    assert summary["trained_bias_target_mismatches"] == 0
    assert summary["saved_module_tensors"] == 2
    assert summary["saved_module_elements"] == 3
    assert summary["saved_module_target_mismatches"] == 0
    assert summary["unadapted_changed_tensors"] == 0
    assert report["layers"][0]["relative_residual"] == 0
    if mode == "all":
        aliases = report["saved_bias_aliases"]
        assert len(aliases) == summary["saved_bias_alias_tensors"] == 2
        assert {a["reference_kind"] for a in aliases} == {"original-base", "saved-full-module"}
        assert "Validated 2 redundant" in render(report, summary_only=True)
    else:
        assert "saved_bias_aliases" not in report


@pytest.mark.parametrize("alias", ["original_module", "modules_to_save"])
@pytest.mark.parametrize("fault", ["value", "nonfinite", "shape", "integer"])
def test_redundant_saved_bias_copies_are_validated_not_ignored(
    tmp_path: Path, alias: str, fault: str
) -> None:
    inputs = saved_head_fixture(tmp_path)
    path = inputs[1] / "adapter_model.safetensors"
    data = load_file(path)
    key = f"base_model.model.head.{alias}.bias"
    if fault == "value":
        data[key] += 1
        reason = "Redundant bias copy differs"
    elif fault == "nonfinite":
        data[key][0] = np.nan
        reason = "Nonfinite weights"
    elif fault == "shape":
        data[key] = np.zeros(2, dtype=np.float32)
        reason = "Redundant bias copy shape mismatch"
    else:
        data[key] = data[key].astype(np.int32)
        reason = "Unsupported redundant bias copy layout"
    save_file(data, path)
    with pytest.raises(AuditError, match=reason):
        audit(*inputs)


def test_missing_canonical_full_target_is_not_recovered_from_a_wrapper_copy(tmp_path: Path) -> None:
    inputs = saved_head_fixture(tmp_path)
    path = inputs[1] / "adapter_model.safetensors"
    data = load_file(path)
    del data["base_model.model.head.bias"]
    save_file(data, path)
    with pytest.raises(AuditError, match="Incomplete saved module"):
        audit(*inputs)


def test_bias_copy_cannot_occupy_another_full_saved_tensors_key(tmp_path: Path) -> None:
    inputs = saved_head_fixture(tmp_path)
    base, adapter, merged = inputs
    for directory in [base, merged]:
        path = directory / "model.safetensors"
        data = load_file(path)
        data["head.original_module.bias"] = np.zeros(1, dtype=np.float32)
        save_file(data, path)
    # This key is already the claimed redundant copy of head.bias. It cannot
    # also be authorized as the canonical target of a distinct saved tensor.
    with pytest.raises(AuditError, match="overlaps a full saved tensor"):
        audit(base, adapter, merged)


def test_multiple_named_bias_copies_are_not_assumed_to_be_one_active_adapter(
    tmp_path: Path,
) -> None:
    inputs = saved_head_fixture(tmp_path, nested=True)
    path = inputs[1] / "adapter_model.safetensors"
    data = load_file(path)
    data["base_model.model.head.modules_to_save.other.dense.bias"] = data[
        "base_model.model.head.modules_to_save.default.dense.bias"
    ].copy()
    save_file(data, path)
    with pytest.raises(AuditError, match="Ambiguous saved-module bias copies"):
        audit(*inputs)


def test_literal_named_copy_is_compared_to_the_known_full_target(tmp_path: Path) -> None:
    inputs = saved_head_fixture(tmp_path, nested=True)
    path = inputs[1] / "adapter_model.safetensors"
    data = load_file(path)
    data["base_model.model.head.modules_to_save.trained_v2.dense.bias"] = data.pop(
        "base_model.model.head.modules_to_save.default.dense.bias"
    )
    save_file(data, path)
    report = audit(*inputs)
    assert report["summary"]["saved_module_target_mismatches"] == 0
    assert report["summary"]["saved_bias_alias_tensors"] == 2


def test_one_named_copy_cannot_be_claimed_for_two_full_saved_biases(tmp_path: Path) -> None:
    base, adapter, merged = saved_head_fixture(tmp_path, nested=True)
    for directory in [base, merged]:
        path = directory / "model.safetensors"
        data = load_file(path)
        data["head.default.dense.bias"] = data["head.dense.bias"].copy()
        save_file(data, path)
    path = adapter / "adapter_model.safetensors"
    data = load_file(path)
    data["base_model.model.head.default.dense.bias"] = data[
        "base_model.model.head.dense.bias"
    ].copy()
    save_file(data, path)
    # The same spelling could mean the legacy copy of default.dense.bias or
    # the named-slot copy of dense.bias. Equal values do not resolve its identity.
    with pytest.raises(AuditError, match="maps to multiple saved tensors"):
        audit(base, adapter, merged)


def test_saved_bias_updates_erased_by_cast_remain_visible_without_target_mismatches(
    tmp_path: Path,
) -> None:
    inputs = bias_fixture(tmp_path, "lora_only")
    base, adapter, merged = inputs
    original = load_file(base / "model.safetensors")["linear.bias"]
    target = np.nextafter(original, np.float32(np.inf))
    data = load_file(adapter / "adapter_model.safetensors")
    data["base_model.model.linear.base_layer.bias"] = target
    save_file(data, adapter / "adapter_model.safetensors")
    after = load_file(merged / "model.safetensors")
    after["linear.bias"] = original.astype(np.float16)
    save_file(after, merged / "model.safetensors")
    report = audit(*inputs)
    row = report["trained_biases"][0]
    assert row["target_mismatches"] == 0
    assert row["expected_change_elements"] == 0
    assert row["unchanged_expected_change_elements"] == 0
    assert row["saved_change_elements"] == 2
    assert row["unchanged_saved_change_elements"] == 2
    assert row["cast_erased_change_elements"] == 2
    assert row["first_cast_erased_change"] == {
        "flat_index": 0,
        "base": 0.25,
        "saved_target": 0.25 + 2**-25,
        "cast_base": 0.25,
        "cast_target": 0.25,
        "merged": 0.25,
        "reference_cast_erased": True,
    }
    assert row["first_unchanged_saved_change"] == row["first_cast_erased_change"]
    assert report["summary"]["trained_bias_unchanged_saved_change_elements"] == 2
    assert "Saved bias changes unchanged from cast base: 2 / 2" in render(report)
    assert "Reference dtype cast erases 2 saved bias changes" in render(report, summary_only=True)
    # A reference cast can erase a change even if an unrelated final edit
    # moves that coordinate away from the base. The two counts stay distinct.
    after["linear.bias"][0] = np.float16(0.5)
    save_file(after, merged / "model.safetensors")
    edited = audit(*inputs)["trained_biases"][0]
    assert edited["target_mismatches"] == 1
    assert edited["cast_erased_change_elements"] == 2
    assert edited["unchanged_saved_change_elements"] == 1
    assert edited["first_cast_erased_change"]["flat_index"] == 0
    assert edited["first_cast_erased_change"]["merged"] == 0.5
    assert edited["first_unchanged_saved_change"]["flat_index"] == 1


def test_source_change_counts_do_not_subtract_opposite_extreme_finite_values(
    tmp_path: Path,
) -> None:
    base, adapter, merged = bias_fixture(tmp_path)
    for directory, value in [(base, -1e308), (merged, 1e308)]:
        path = directory / "model.safetensors"
        data = load_file(path)
        data["linear.bias"] = np.array([value, 0.125], dtype=np.float64)
        save_file(data, path)
    path = adapter / "adapter_model.safetensors"
    data = load_file(path)
    data["base_model.model.linear.base_layer.bias"] = np.array([1e308, 0.125], dtype=np.float64)
    save_file(data, path)
    row = next(
        r for r in audit(base, adapter, merged)["trained_biases"] if r["weight"] == "linear.bias"
    )
    assert row["target_mismatches"] == 0
    assert row["saved_change_elements"] == 1
    assert row["unchanged_saved_change_elements"] == row["cast_erased_change_elements"] == 0


def test_older_bias_report_can_still_render_without_inventing_saved_change_counts(
    tmp_path: Path,
) -> None:
    report = audit(*bias_fixture(tmp_path))
    fields = (
        "saved_change_elements",
        "unchanged_saved_change_elements",
        "cast_erased_change_elements",
    )
    for field in fields:
        del report["summary"]["trained_bias_" + field]
        for row in report["trained_biases"]:
            del row[field]
    rendered = render(report)
    assert "0 target mismatches" in rendered
    assert "Saved bias changes unchanged" not in rendered
