"""Small hand-calculated controls complement the retained native producer study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from mergeglass import AuditError, audit

MODULE = "model.layers.0.mlp.experts"


def fixture(root: Path, *, both: bool = False, omitted: bool = False) -> tuple[Path, ...]:
    base, adapter, merged = (root / name for name in ("base", "adapter", "merged"))
    for path in (base, adapter, merged):
        path.mkdir()
    config = {
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
        "transformers_version": "5.20.1",
        "num_local_experts": 2,
        "num_hidden_layers": 1,
        "hidden_size": 3,
        "moe_intermediate_size": 2,
    }
    (base / "config.json").write_text(json.dumps(config))
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "peft_version": "0.21.0",
                "r": 2,
                "lora_alpha": 4,
                # Native wrapping follows parameter registration, even when this is reversed.
                "target_parameters": ["down_proj", "gate_up_proj"] if both else ["gate_up_proj"],
                "target_modules": [],
                "bias": "none",
            }
        )
    )
    a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32)
    b = np.array([[k, 2 * k, 3 * k, 4 * k] for k in range(1, 5)], dtype=np.float32)
    module = MODULE + (".base_layer" if both else "")
    factors = {
        f"base_model.model.{module}.lora_A.weight": a,
        f"base_model.model.{module}.lora_B.weight": b,
    }
    if both:
        factors[f"base_model.model.{MODULE}.lora_A.weight"] = np.ones((4, 2), np.float32)
        factors[f"base_model.model.{MODULE}.lora_B.weight"] = np.zeros((3, 4), np.float32)
    save_file(factors, adapter / "adapter_model.safetensors")
    originals, updates = {}, {}
    for expert in range(2):
        for part, offset in (("gate_proj", 0), ("up_proj", 2)):
            name = f"{MODULE}.{expert}.{part}.weight"
            originals[name] = np.ones((2, 3), dtype=np.float32)
            # Hand arithmetic: e0 uses B columns 0,2 and e1 uses columns 1,3.
            delta = np.array(
                [
                    [2 * k, 6 * k, 0] if expert == 0 else [8 * k, 8 * k, 4 * k]
                    for k in range(offset + 1, offset + 3)
                ],
                dtype=np.float32,
            )
            updates[name] = originals[name] + (0 if omitted and expert == 1 else delta)
        name = f"{MODULE}.{expert}.down_proj.weight"
        originals[name] = np.ones((3, 2), dtype=np.float32)
        updates[name] = originals[name].copy()
    save_file(originals, base / "model.safetensors")
    save_file(updates, merged / "model.safetensors")
    return base, adapter, merged


def update(path: Path, **fields: Any) -> None:
    config = json.loads(path.read_text())
    config.update(fields)
    path.write_text(json.dumps(config))


def test_interleaved_experts_and_gate_up_offsets_match_hand_arithmetic(tmp_path: Path) -> None:
    report = audit(*fixture(tmp_path))
    assert report["summary"]["adapted_tensors"] == 4
    assert report["summary"]["adapted_elements"] == 24
    assert report["summary"]["logical_expert_parameters"] == 1
    assert report["unadapted_changes"] == []
    for row in report["layers"]:
        assert row["relative_residual"] == 0
        assert row["projection_gain"] == pytest.approx(1)
        assert row["reference_merge_mismatches"] == 0
        assert row["expert_index"] == int(row["weight"].split(".")[-3])
        assert row["parameter_output_rows"] == (
            [0, 2] if ".gate_proj." in row["weight"] else [2, 4]
        )
        assert row["largest_residual"]["parameter_output_row"] == row["parameter_output_rows"][0]


def test_omitted_expert_keeps_its_saved_coordinates(tmp_path: Path) -> None:
    report = audit(*fixture(tmp_path, omitted=True))
    assert report["summary"]["expert_no_observed_update_tensors"] == 2
    for row in report["layers"]:
        assert row["projection_gain"] == pytest.approx(0 if row["expert_index"] == 1 else 1)
        assert row["relative_residual"] == (1 if row["expert_index"] == 1 else 0)


def test_reversed_selection_binds_outer_down_and_inner_gate_up(tmp_path: Path) -> None:
    report = audit(*fixture(tmp_path, both=True))
    assert report["summary"]["logical_expert_parameters"] == 2
    assert report["summary"]["expert_zero_update_tensors"] == 2
    for row in report["layers"]:
        is_down = row["expert_parameter"].endswith("down_proj")
        assert (".base_layer." in row["factor_tensors"]["A"]) is not is_down
        assert row["status"] == ("zero-update" if is_down else "observed-update")
        assert row["projection_gain"] == (None if is_down else pytest.approx(1))


@pytest.mark.parametrize(
    "fields",
    [
        {"peft_version": "0.20.0"},
        {"target_parameters": ["unknown_proj"]},
        {"target_parameters": [".*gate_up_proj"]},
        {"lora_dropout": 0.1},
        {"r": 3},
    ],
)
def test_unknown_parameter_contract_refuses(tmp_path: Path, fields: dict[str, Any]) -> None:
    base, adapter, merged = fixture(tmp_path)
    update(adapter / "adapter_config.json", **fields)
    with pytest.raises(AuditError):
        audit(base, adapter, merged)


def test_missing_native_up_matrix_refuses(tmp_path: Path) -> None:
    base, adapter, merged = fixture(tmp_path)
    path = base / "model.safetensors"
    tensors = load_file(path)
    del tensors[f"{MODULE}.1.up_proj.weight"]
    save_file(tensors, path)
    path = merged / "model.safetensors"
    tensors = load_file(path)
    del tensors[f"{MODULE}.1.up_proj.weight"]
    save_file(tensors, path)
    with pytest.raises(AuditError, match="Missing or malformed native saved expert"):
        audit(base, adapter, merged)


def test_saved_module_cannot_hide_expert_overlap(tmp_path: Path) -> None:
    base, adapter, merged = fixture(tmp_path)
    update(adapter / "adapter_config.json", modules_to_save=["experts.0"])
    path = adapter / "adapter_model.safetensors"
    tensors = load_file(path)
    for name, value in load_file(base / "model.safetensors").items():
        if f"{MODULE}.0." in name:
            tensors["base_model.model." + name] = value
    save_file(tensors, path)
    with pytest.raises(AuditError, match="overlaps a full saved module"):
        audit(base, adapter, merged)
