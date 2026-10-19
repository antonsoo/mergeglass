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
