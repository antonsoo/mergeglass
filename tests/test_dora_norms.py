from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file
from test_audit import change_config, fixture
from test_dora import dora_fixture

from mergeglass import AuditError, audit, audit_dora_norms
from mergeglass.cli import main, render


def test_saved_base_snapshot_requires_the_actual_original_base(tmp_path: Path) -> None:
    base, adapter, merged = dora_fixture(
        tmp_path,
        np.array([[3.0, 4]]),
        np.zeros((1, 2)),
        np.zeros((1, 1)),
        np.array([10.0]),
        np.array([[6.0, 8]]),
    )
    path = adapter / "adapter_model.safetensors"
    data = load_file(path)
    data["base_model.model.linear.base_layer.weight"] = np.array([[3.0, 4]])
    save_file(data, path)
    assert audit(base, adapter, merged)["layers"][0]["relative_residual"] == 0
    with pytest.raises(AuditError, match="snapshots require an original base"):
        audit_dora_norms(adapter, merged)


def test_no_base_observation_keeps_norms_and_leaves_direction_unknown(tmp_path: Path) -> None:
    base, adapter, merged = dora_fixture(
        tmp_path,
        np.tile([3.0, 4], (5, 1)),
        np.zeros((1, 2)),
        np.zeros((5, 1)),
        np.array([10.0, -10, 0, 5, 10]),
        np.array([[3.0, 4], [6, 8], [3, 4], [0, 0], [0, 10]]),
    )
    full = audit(base, adapter, merged)
    shutil.rmtree(base)
    partial = audit_dora_norms(adapter, merged)
    expected = {
        key: value
        for key, value in full["layers"][0]["dora_rows"].items()
        if key != "zero_numerical_reference_rows" and "direction" not in key
    }
    assert partial["layers"][0]["dora_rows"] == expected
    assert expected["row_norm_factor_range"] == [0.5, 1.0]
    assert expected["zero_magnitude_nonzero_saved_rows"] == 1
    assert expected["nonzero_magnitude_zero_saved_rows"] == 1
    assert partial["inputs"].keys() == {"adapter", "merged"}
    assert partial["schema"] == "mergeglass.dora-norms"
    assert not {"projection_gain", "relative_residual", "status"} & partial["layers"][0].keys()
    text = render(partial)
    assert "Partial observations" in text and "unmeasured" in text
    assert "direction cosine" not in text and "GAIN" not in text
    json.dumps(partial, allow_nan=False)


def test_matching_norm_cannot_prove_direction_or_an_update(tmp_path: Path) -> None:
    base, adapter, merged = dora_fixture(
        tmp_path,
        np.array([[3.0, 4]]),
        np.zeros((1, 2)),
        np.zeros((1, 1)),
        np.array([5.0]),
        np.array([[-3.0, -4]]),
    )
    assert audit_dora_norms(adapter, merged)["layers"][0]["dora_rows"]["row_norm_factor_range"] == [
        1.0,
        1.0,
    ]
    assert audit(base, adapter, merged)["layers"][0]["dora_rows"]["direction_cosine_range"] == [
        -1.0,
        -1.0,
    ]


@pytest.mark.parametrize("saved", [1e308, 1e-308])
def test_partial_observations_retain_norms_outside_ordinary_float64_range(
    tmp_path: Path, saved: float
) -> None:
    _, adapter, merged = dora_fixture(
        tmp_path,
        np.ones((1, 4)),
        np.zeros((1, 4)),
        np.zeros((1, 1)),
        np.array([1e308]),
        np.full((1, 4), saved),
    )
    report = audit_dora_norms(adapter, merged)
    rows = report["layers"][0]["dora_rows"]
    if saved == 1e308:
        assert rows["row_norm_factor_range"] == [2.0, 2.0]
        assert rows["largest_nonzero_log2_norm_error"]["saved_row_norm"] is None
    else:
        assert rows["row_norm_factor_range"] == [None, None]
        assert rows["log2_row_norm_factor_range"][0] < -2000
    json.dumps(report, allow_nan=False)


def test_without_a_base_no_direction_is_invented_for_zero_base(tmp_path: Path) -> None:
    base, adapter, merged = dora_fixture(
        tmp_path,
        np.zeros((1, 2)),
        np.zeros((1, 2)),
        np.zeros((1, 1)),
        np.zeros(1),
        np.zeros((1, 2)),
    )
    with pytest.raises(AuditError, match="Undefined DoRA direction"):
        audit(base, adapter, merged)
    result = audit_dora_norms(adapter, merged)["layers"][0]["dora_rows"]
    assert result["row_norm_factor_range"] is None
    assert result["zero_saved_rows"] == result["zero_magnitude_rows"] == 1
    assert "zero_numerical_reference_rows" not in result


@pytest.mark.parametrize("location", ["A", "B", "magnitude", "mapped", "unmapped"])
def test_nonfinite_inputs_refuse_partial_results(tmp_path: Path, location: str) -> None:
    _, adapter, merged = dora_fixture(
        tmp_path,
        np.ones((2, 3)),
        np.ones((1, 3)),
        np.ones((2, 1)),
        np.ones(2),
        np.ones((2, 3)),
    )
    path = (
        adapter / "adapter_model.safetensors"
        if location in {"A", "B", "magnitude"}
        else merged / "model.safetensors"
    )
    tensors = load_file(path)
    if location == "unmapped":
        tensors["another.weight"] = np.array([np.inf])
    else:
        key = (
            "base_model.model.linear.lora_magnitude_vector"
            if location == "magnitude"
            else f"base_model.model.linear.lora_{location}.weight"
            if location in {"A", "B"}
            else "linear.weight"
        )
        tensors[key].flat[-1] = np.nan
    save_file(tensors, path)
    with pytest.raises(AuditError, match="Nonfinite weights"):
        audit_dora_norms(adapter, merged)


def test_plain_lora_has_no_saved_norm_target(tmp_path: Path) -> None:
    _, adapter, merged = fixture(tmp_path)
    with pytest.raises(AuditError, match="require a saved DoRA magnitude"):
        audit_dora_norms(adapter, merged)


def test_partial_observations_keep_unmapped_tensors_unknown(tmp_path: Path) -> None:
    _, adapter, merged = dora_fixture(
        tmp_path,
        np.ones((2, 3)),
        np.ones((1, 3)),
        np.ones((2, 1)),
        np.ones(2),
        np.ones((2, 3)),
    )
    tensors = load_file(merged / "model.safetensors")
    tensors["control"] = np.array([123], dtype=np.int64)
    save_file(tensors, merged / "model.safetensors")
    report = audit_dora_norms(adapter, merged)
    assert report["summary"]["unmapped_tensors"] == 1
    assert "unadapted_changes" not in report
    assert "have no base comparison" in render(report, summary_only=True)


def test_cli_json_and_input_protection(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, adapter, merged = dora_fixture(
        tmp_path,
        np.ones((2, 3)),
        np.ones((1, 3)),
        np.ones((2, 1)),
        np.ones(2),
        np.ones((2, 3)),
    )
    arguments = ["--dora-norms", str(adapter), str(merged), "--json"]
    assert main([*arguments, "-"]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "mergeglass.dora-norms"
    target = adapter / "adapter_model.safetensors"
    original = target.read_bytes()
    assert main([*arguments, str(target)]) == 2
    assert target.read_bytes() == original
    external = tmp_path / "external.safetensors"
    target.rename(external)
    target.symlink_to(external)
    assert main([*arguments, str(external)]) == 2
    assert external.read_bytes() == original


def test_flag_named_checkpoint_after_separator_keeps_full_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base, adapter, merged = fixture(tmp_path)
    base.rename(tmp_path / "--dora-norms")
    monkeypatch.chdir(tmp_path)
    assert main(["--json", "-", "--", "--dora-norms", str(adapter), str(merged)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "base" in report["inputs"] and "schema" not in report


def test_metadata_refusals_apply_without_a_base(tmp_path: Path) -> None:
    _, adapter, merged = dora_fixture(
        tmp_path,
        np.ones((2, 3)),
        np.ones((1, 3)),
        np.ones((2, 1)),
        np.ones(2),
        np.ones((2, 3)),
    )
    change_config(adapter, fan_in_fan_out=True)
    with pytest.raises(AuditError, match="Unsupported adapter option"):
        audit_dora_norms(adapter, merged)
