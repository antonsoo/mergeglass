from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file
from test_audit import change_config

from mergeglass import AuditError, audit
from mergeglass.cli import main


def conv1d_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    base, adapter, merged = (tmp_path / name for name in ("base", "adapter", "merged"))
    for root in (base, adapter, merged):
        root.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "gpt2"}))
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 2, "fan_in_fan_out": True})
    )
    # Exactly representable scalar products, with asymmetric square and
    # rectangular Conv1D weights plus an ordinary Linear head in one adapter.
    weights, factors, targets = {}, {}, {}
    for path, ins, outs in [
        ("transformer.h.0.attn.c_proj", 3, 3),
        ("transformer.h.0.mlp.c_fc", 3, 5),
        ("lm_head", 3, 7),
    ]:
        a = np.arange(2 * ins, dtype=np.float64).reshape(2, ins) / 8
        b = np.arange(2 * outs, dtype=np.float64).reshape(outs, 2) / 4
        product = b @ a
        if path != "lm_head":
            product = product.T
        weights[path + ".weight"] = np.ones_like(product)
        targets[path + ".weight"] = np.ascontiguousarray(1 + product)
        factors[f"base_model.model.{path}.lora_A.weight"] = a
        factors[f"base_model.model.{path}.lora_B.weight"] = b
    save_file(weights, base / "model.safetensors")
    save_file(factors, adapter / "adapter_model.safetensors")
    save_file(targets, merged / "model.safetensors")
    return base, adapter, merged


@pytest.mark.parametrize("flag", [False, True])
def test_mixed_shapes_use_per_module_layout_and_keep_native_coordinates(
    tmp_path: Path, flag: bool
) -> None:
    base, adapter, merged = conv1d_fixture(tmp_path)
    change_config(adapter, fan_in_fan_out=flag)
    values = load_file(merged / "model.safetensors")
    weight = "transformer.h.0.mlp.c_fc.weight"
    values[weight][1, 4] += 0.5
    save_file(values, merged / "model.safetensors")
    report = audit(base, adapter, merged)
    for row in report["layers"]:
        assert (
            row["weight_layout"] == "conv1d"
            if row["weight"] != "lm_head.weight"
            else ("weight_layout" not in row)
        )
        assert row["norms"]["residual"] == (0.5 if row["weight"] == weight else 0)
        if row["weight"] == weight:
            assert row["coordinate_axes"] == {"row": "input", "column": "output"}
            assert row["largest_residual"]["row"] == 1
            assert row["largest_residual"]["column"] == 4
            assert row["largest_residual"]["residual"] == 0.5


def test_transposed_square_control_changes_every_weight_but_is_wrong(tmp_path: Path) -> None:
    base, adapter, merged = conv1d_fixture(tmp_path)
    values = load_file(merged / "model.safetensors")
    key = "transformer.h.0.attn.c_proj.weight"
    values[key] = values[key].T.copy()
    save_file(values, merged / "model.safetensors")
    layer = next(row for row in audit(base, adapter, merged)["layers"] if row["weight"] == key)
    assert layer["changed_elements"] == 9
    assert layer["reference_merge_mismatches"] == 6
    assert layer["relative_residual"] > 0.1


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"fan_in_fan_out": "true"}, "must be boolean"),
        (
            {"auto_mapping": {"parent_library": "transformers.models.imagegpt.modeling_imagegpt"}},
            "Conflicting Conv1D",
        ),
    ],
)
def test_uncertain_or_unsupported_layouts_refuse(
    tmp_path: Path, updates: dict[str, Any], message: str
) -> None:
    base, adapter, merged = conv1d_fixture(tmp_path)
    change_config(adapter, **updates)
    with pytest.raises(AuditError, match=message):
        audit(base, adapter, merged)


def test_bare_file_cannot_use_a_global_true_flag_as_a_layout_map(tmp_path: Path) -> None:
    base, adapter, merged = conv1d_fixture(tmp_path)
    with pytest.raises(AuditError, match="supply standard model metadata"):
        audit(base / "model.safetensors", adapter, merged)


def test_conv1d_dora_refuses_even_with_complete_magnitudes(tmp_path: Path) -> None:
    base, adapter, merged = conv1d_fixture(tmp_path)
    change_config(adapter, use_dora=True)
    factors = load_file(adapter / "adapter_model.safetensors")
    for path, outs in [("lm_head", 7), ("transformer.h.0.attn.c_proj", 3)]:
        factors[f"base_model.model.{path}.lora_magnitude_vector"] = np.ones(outs)
    save_file(factors, adapter / "adapter_model.safetensors")
    with pytest.raises(AuditError, match="Conv1D DoRA"):
        audit(base, adapter, merged)


def test_remote_custom_model_cannot_reuse_the_standard_layout_table(tmp_path: Path) -> None:
    base, adapter, merged = conv1d_fixture(tmp_path)
    (base / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "auto_map": {"AutoModel": "custom.Model"}})
    )
    with pytest.raises(AuditError, match="Custom model auto_map"):
        audit(base, adapter, merged)


def test_failed_layout_resolution_preserves_an_existing_report(tmp_path: Path) -> None:
    base, adapter, merged = conv1d_fixture(tmp_path)
    output = tmp_path / "report.json"
    output.write_text("keep the previous report")
    assert (
        main([str(base / "model.safetensors"), str(adapter), str(merged), "--json", str(output)])
        == 2
    )
    assert output.read_text() == "keep the previous report"
