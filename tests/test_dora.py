from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import save_file
from test_audit import change_config, fixture

from mergeglass import AuditError, audit
from mergeglass.cli import main, render


def dora_fixture(
    tmp_path: Path,
    original: np.ndarray[Any, Any],
    a: np.ndarray[Any, Any],
    b: np.ndarray[Any, Any],
    magnitude: np.ndarray[Any, Any],
    after: np.ndarray[Any, Any],
) -> tuple[Path, Path, Path]:
    base, adapter, merged = fixture(tmp_path)
    change_config(adapter, r=a.shape[0], lora_alpha=a.shape[0], use_dora=True)
    save_file({"linear.weight": original}, base / "model.safetensors")
    save_file({"linear.weight": after}, merged / "model.safetensors")
    save_file(
        {
            "base_model.model.linear.lora_A.weight": a,
            "base_model.model.linear.lora_B.weight": b,
            "base_model.model.linear.lora_magnitude_vector": magnitude,
        },
        adapter / "adapter_model.safetensors",
    )
    return base, adapter, merged


def test_magnitude_only_update_uses_complete_rows_and_signed_magnitudes(tmp_path: Path) -> None:
    # The row norms are both 5. No matrix product or reference implementation
    # is needed to establish the expected result, including the negative sign.
    original = np.array([[3.0, 4, 0], [0, 4, -3]])
    after = np.array([[6.0, 8, 0], [0, -4, 3]])
    paths = dora_fixture(
        tmp_path, original, np.zeros((1, 3)), np.zeros((2, 1)), np.array([10.0, -5]), after
    )
    layer = audit(*paths)["layers"][0]
    assert layer["adapter_variant"] == "dora"
    assert layer["adapter_dtypes"]["magnitude"] == "F64"
    assert layer["changed_elements"] == 4
    assert layer["intended_nonzero_elements"] == 4
    assert layer["norms"]["delta"] == pytest.approx(np.sqrt(125))
    assert layer["relative_residual"] == 0
    assert layer["projection_gain"] == pytest.approx(1)
    assert layer["reference_merge_mismatches"] == 0
    text = render(audit(*paths))
    assert "learned magnitudes" in text
    assert "complete weight update" in text


def test_directional_product_can_be_nonzero_with_no_complete_update(tmp_path: Path) -> None:
    original = np.array([[3.0, 4, 0], [3, 4, 0]])
    paths = dora_fixture(
        tmp_path,
        original,
        np.array([[3.0, 4, 0]]),
        np.ones((2, 1)),
        np.array([5.0, 5]),
        original,
    )
    layer = audit(*paths)["layers"][0]
    assert layer["status"] == "zero-update"
    assert layer["intended_nonzero_elements"] == 0
    assert layer["relative_residual"] is None
    assert layer["unchanged_nonzero_update_fraction"] is None


def test_zero_alpha_keeps_the_magnitude_update(tmp_path: Path) -> None:
    original = np.array([[3.0, 4]])
    base, adapter, merged = dora_fixture(
        tmp_path, original, np.ones((1, 2)), np.ones((1, 1)), np.array([10.0]), 2 * original
    )
    change_config(adapter, lora_alpha=0)
    layer = audit(base, adapter, merged)["layers"][0]
    assert layer["scale"] == 0
    assert layer["status"] == "observed-update"
    assert layer["norms"]["delta"] == 5
    assert layer["relative_residual"] == 0


def test_zero_magnitude_is_a_complete_update_to_zero(tmp_path: Path) -> None:
    original = np.array([[3.0, 4]])
    paths = dora_fixture(
        tmp_path, original, np.ones((1, 2)), np.ones((1, 1)), np.zeros(1), np.zeros_like(original)
    )
    layer = audit(*paths)["layers"][0]
    assert layer["status"] == "observed-update"
    assert layer["norms"]["delta"] == 5
    assert layer["relative_residual"] == 0


def test_reference_merge_keeps_weights_lost_by_delta_subtract_add(tmp_path: Path) -> None:
    paths = dora_fixture(
        tmp_path,
        np.ones((1, 1)),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.array([1e-100]),
        np.array([[1e-100]]),
    )
    layer = audit(*paths)["layers"][0]
    assert layer["largest_residual"]["intended_delta"] == -1
    assert layer["largest_residual"]["reference_merged"] == 1e-100
    assert layer["reference_merge_mismatches"] == 0


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_direction_normalization_does_not_square_raw_weights(tmp_path: Path, scale: float) -> None:
    original = np.array([[3 * scale, 4 * scale]])
    paths = dora_fixture(
        tmp_path,
        original,
        np.zeros((1, 2)),
        np.zeros((1, 1)),
        np.array([10 * scale]),
        2 * original,
    )
    layer = audit(*paths)["layers"][0]
    assert layer["norms"]["delta"] == pytest.approx(5 * scale, rel=1e-14, abs=0)
    assert layer["projection_gain"] == pytest.approx(1)
    assert layer["relative_residual"] < 1e-14


def test_undefined_direction_refuses_and_preserves_existing_report(tmp_path: Path) -> None:
    original = np.array([[3.0, 4], [1, 1]])
    paths = dora_fixture(
        tmp_path,
        original,
        np.array([[-1.0, -1]]),
        np.ones((2, 1)),
        np.zeros(2),
        np.zeros_like(original),
    )
    with pytest.raises(AuditError, match="Undefined DoRA direction at linear, row 1"):
        audit(*paths)
    destination = tmp_path / "report.json"
    destination.write_text("previous result")
    assert main([*(str(p) for p in paths), "--json", str(destination)]) == 2
    assert destination.read_text() == "previous result"


def test_undefined_direction_retains_its_original_row_across_blocks(tmp_path: Path) -> None:
    b = np.zeros((70, 1))
    b[35] = -1
    paths = dora_fixture(
        tmp_path,
        np.ones((70, 513)),
        np.ones((1, 513)),
        b,
        np.ones(70),
        np.ones((70, 513)),
    )
    with pytest.raises(AuditError, match="Undefined DoRA direction at linear, row 35"):
        audit(*paths, block_mib=1)


@pytest.mark.parametrize(
    "magnitude",
    [np.ones((2, 1)), np.ones(1), np.array([np.nan, 1]), np.array([1, 2], dtype=np.int32)],
)
def test_incomplete_or_nonfinite_magnitude_cannot_become_an_audit(
    tmp_path: Path, magnitude: np.ndarray[Any, Any]
) -> None:
    paths = dora_fixture(
        tmp_path,
        np.ones((2, 3)),
        np.ones((1, 3)),
        np.ones((2, 1)),
        magnitude,
        np.ones((2, 3)),
    )
    with pytest.raises(AuditError, match="magnitude|Nonfinite"):
        audit(*paths)


@pytest.mark.parametrize("flag", [None, 1, "true", [], {}])
def test_dora_flag_requires_a_boolean(tmp_path: Path, flag: Any) -> None:
    base, adapter, merged = fixture(tmp_path)
    change_config(adapter, use_dora=flag)
    with pytest.raises(AuditError, match="use_dora must be boolean"):
        audit(base, adapter, merged)


def test_magnitude_payload_without_dora_config_is_not_silently_ignored(tmp_path: Path) -> None:
    paths = dora_fixture(
        tmp_path, np.ones((2, 3)), np.ones((1, 3)), np.ones((2, 1)), np.ones(2), np.ones((2, 3))
    )
    change_config(paths[1], use_dora=False)
    with pytest.raises(AuditError, match="Unsupported adapter tensor"):
        audit(*paths)


def test_missing_magnitude_is_not_reconstructed_from_the_base(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    change_config(paths[1], use_dora=True)
    with pytest.raises(AuditError, match="Missing DoRA magnitude vector: linear"):
        audit(*paths)


def test_ordinary_lora_keeps_its_original_measurements(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    report = audit(*paths)
    change_config(paths[1], use_dora=False)
    explicit = audit(*paths)
    assert explicit["layers"] == report["layers"]
    assert explicit["summary"] == report["summary"]
    assert explicit["layers"][0]["adapter_variant"] == "lora"
    assert "magnitude" not in explicit["layers"][0]["adapter_dtypes"]
    json.dumps(explicit, allow_nan=False)
    # Schema-1 reports saved before this extension still render as plain LoRA.
    del explicit["layers"][0]["adapter_variant"]
    assert "saved LoRA merge audit" in render(explicit)


def test_full_rows_separate_norm_loss_wrong_sign_and_direction_errors(tmp_path: Path) -> None:
    # 3-4-5 triangles give independent lengths and direction cosines. The
    # five saved rows have a half norm, reversed sign, unexpected nonzero,
    # unexpected zero and rotated direction, respectively.
    original = np.tile([3.0, 4.0], (5, 1))
    after = np.array([[3.0, 4], [6, 8], [3, 4], [0, 0], [0, 10]])
    paths = dora_fixture(
        tmp_path,
        original,
        np.zeros((1, 2)),
        np.zeros((5, 1)),
        np.array([10.0, -10, 0, 5, 10]),
        after,
    )
    report = audit(*paths)
    rows = report["layers"][0]["dora_rows"]
    assert rows["rows"] == 5
    assert rows["negative_magnitude_rows"] == 1
    assert rows["zero_magnitude_rows"] == rows["zero_saved_rows"] == 1
    assert rows["nonzero_magnitude_zero_saved_rows"] == 1
    assert rows["zero_magnitude_nonzero_saved_rows"] == 1
    assert rows["norm_comparable_rows"] == rows["direction_comparable_rows"] == 3
    assert rows["row_norm_factor_range"] == pytest.approx([0.5, 1])
    assert rows["log2_row_norm_factor_range"] == pytest.approx([-1, 0], abs=1e-14)
    assert rows["mean_absolute_log2_row_norm_factor"] == pytest.approx(1 / 3)
    assert rows["largest_nonzero_log2_norm_error"]["row"] == 0
    assert rows["direction_cosine_range"] == pytest.approx([-1, 1])
    assert rows["mean_direction_cosine"] == pytest.approx(0.8 / 3)
    for summary_only in [False, True]:
        text = render(report, summary_only=summary_only)
        assert "norm factor (saved / |m|): 0.5..1; 3 nonzero row pairs" in text
        assert "direction cosine: -1..1; 3 nonzero row pairs" in text
        assert "Nonzero magnitudes saved as zero rows: 1" in text
    json.dumps(report, allow_nan=False)


def test_all_zero_targets_keep_missing_ratios_and_directions_explicit(tmp_path: Path) -> None:
    paths = dora_fixture(
        tmp_path,
        np.array([[3.0, 4]]),
        np.zeros((1, 2)),
        np.zeros((1, 1)),
        np.zeros(1),
        np.zeros((1, 2)),
    )
    rows = audit(*paths)["layers"][0]["dora_rows"]
    assert rows["norm_comparable_rows"] == rows["direction_comparable_rows"] == 0
    assert rows["row_norm_factor_range"] is None
    assert rows["direction_cosine_range"] is None
    assert rows["zero_magnitude_rows"] == rows["zero_saved_rows"] == 1
    assert rows["zero_magnitude_nonzero_saved_rows"] == 0
    assert rows["nonzero_magnitude_zero_saved_rows"] == 0


def test_factored_norm_survives_an_unrepresentable_saved_row_length(tmp_path: Path) -> None:
    original = np.full((1, 4), 1e308)
    paths = dora_fixture(
        tmp_path, original, np.zeros((1, 4)), np.zeros((1, 1)), np.array([1e308]), original
    )
    report = audit(*paths)
    rows = report["layers"][0]["dora_rows"]
    assert rows["row_norm_factor_range"] == pytest.approx([2, 2])
    assert rows["direction_cosine_range"] == pytest.approx([1, 1])
    worst = rows["largest_nonzero_log2_norm_error"]
    assert worst["saved_row_norm"] is None
    assert worst["saved_row_norm_scale"] == 1e308
    assert worst["saved_row_norm_scaled_length"] == 2
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("original,magnitude,sign", [(1.0, 5e-324, 1), (5e-324, 1e308, -1)])
def test_log_norm_ratio_survives_unrepresentable_factors(
    tmp_path: Path, original: float, magnitude: float, sign: int
) -> None:
    paths = dora_fixture(
        tmp_path,
        np.array([[original]]),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.array([magnitude]),
        np.array([[original]]),
    )
    report = audit(*paths)
    rows = report["layers"][0]["dora_rows"]
    assert rows["row_norm_factor_range"] == [None, None]
    assert all(sign * value > 1073 for value in rows["log2_row_norm_factor_range"])
    assert rows["direction_cosine_range"] == [1, 1]
    assert "2^(" in render(report)
    json.dumps(report, allow_nan=False)


def test_largest_row_identity_survives_a_block_boundary(tmp_path: Path) -> None:
    original = np.ones((70, 513))
    after = original.copy()
    after[35] *= 2
    paths = dora_fixture(
        tmp_path, original, np.zeros((1, 513)), np.zeros((70, 1)), np.full(70, np.sqrt(513)), after
    )
    rows = audit(*paths, block_mib=1)["layers"][0]["dora_rows"]
    assert rows["largest_nonzero_log2_norm_error"]["row"] == 35
    assert rows["largest_nonzero_log2_norm_error"]["row_norm_factor"] == pytest.approx(2)


def test_existing_dora_reports_render_without_optional_row_diagnostics(tmp_path: Path) -> None:
    paths = dora_fixture(
        tmp_path,
        np.array([[3.0, 4]]),
        np.zeros((1, 2)),
        np.zeros((1, 1)),
        np.array([5.0]),
        np.array([[3.0, 4]]),
    )
    report = audit(*paths)
    del report["layers"][0]["dora_rows"]
    assert "saved DoRA merge audit" in render(report)
    assert "DoRA full-row norm factor" not in render(report)
