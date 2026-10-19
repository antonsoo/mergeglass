from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import save_file

from mergeglass import AuditError, audit
from mergeglass.cli import main
from mergeglass.storage import Checkpoint, cast


def fixture(tmp_path: Path, *, alpha: float = 1, rs: bool = False) -> tuple[Path, Path, Path]:
    base, adapter, merged = (tmp_path / name for name in ["base", "adapter", "merged"])
    for path in [base, adapter, merged]:
        path.mkdir()
    config = {"peft_type": "LORA", "r": 1, "lora_alpha": alpha, "bias": "none", "use_rslora": rs}
    (adapter / "adapter_config.json").write_text(json.dumps(config))
    save_file(
        {
            "linear.weight": np.ones((2, 3), dtype=np.float32),
            "control": np.arange(4, dtype=np.int64),
        },
        base / "model.safetensors",
    )
    save_file(
        {
            "base_model.model.linear.lora_A.weight": np.ones((1, 3), dtype=np.float32),
            "base_model.model.linear.lora_B.weight": np.full((2, 1), 0.125, dtype=np.float32),
        },
        adapter / "adapter_model.safetensors",
    )
    save_file(
        {
            "linear.weight": np.full((2, 3), 1 + 0.125 * alpha, dtype=np.float32),
            "control": np.arange(4, dtype=np.int64),
        },
        merged / "model.safetensors",
    )
    return base, adapter, merged


def change_config(adapter: Path, **updates: Any) -> None:
    path = adapter / "adapter_config.json"
    config = json.loads(path.read_text())
    config.update(updates)
    path.write_text(json.dumps(config))


def test_dense_hand_calculated_update_and_complete_inventory(tmp_path: Path) -> None:
    inputs = fixture(tmp_path)
    report = audit(*inputs)
    layer = report["layers"][0]
    assert layer["relative_residual"] == 0
    assert layer["norms"]["delta"] == pytest.approx(np.sqrt(6) / 8)
    assert layer["changed_elements"] == 6
    assert layer["reference_merge_mismatches"] == 0
    assert layer["unchanged_nonzero_update_fraction"] == 0
    assert layer["unchanged_reference_energy_fraction"] == 0
    assert report["summary"]["checked_tensors"] == 2
    assert len(report["inputs"]["adapter"]) == 2


def test_copied_base_is_observed_absence_not_a_rounding_diagnosis(tmp_path: Path) -> None:
    base, adapter, merged = fixture(tmp_path)
    (merged / "model.safetensors").write_bytes((base / "model.safetensors").read_bytes())
    layer = audit(base, adapter, merged)["layers"][0]
    assert layer["status"] == "no-observed-update"
    assert layer["relative_residual"] == 1
    assert layer["unchanged_nonzero_update_fraction"] == 1
    assert layer["unchanged_reference_energy_fraction"] == pytest.approx(1)
    assert layer["projection_gain"] == 0
    assert layer["cosine_similarity"] is None


def test_zero_scale_is_a_zero_update_with_undefined_relative_error(tmp_path: Path) -> None:
    layer = audit(*fixture(tmp_path, alpha=0))["layers"][0]
    assert layer["status"] == "zero-update"
    assert layer["relative_residual"] is None
    assert layer["unchanged_nonzero_update_fraction"] is None
    assert layer["unchanged_reference_energy_fraction"] is None
    assert layer["projection_gain"] is None
    assert layer["cosine_similarity"] is None


@pytest.mark.parametrize("alpha", [-0.5, -1, -2])
def test_signed_alpha_is_part_of_the_intended_reference(tmp_path: Path, alpha: float) -> None:
    layer = audit(*fixture(tmp_path, alpha=alpha))["layers"][0]
    assert layer["alpha"] == alpha
    assert layer["scale"] == alpha
    assert layer["changed_elements"] == 6
    assert layer["relative_residual"] == 0
    assert layer["projection_gain"] == pytest.approx(1)
    assert layer["cosine_similarity"] == pytest.approx(1)


def test_signed_alpha_pattern_override_is_preserved(tmp_path: Path) -> None:
    base, adapter, merged = fixture(tmp_path)
    change_config(adapter, alpha_pattern={"linear": -2})
    save_file(
        {
            "linear.weight": np.full((2, 3), 0.75, dtype=np.float32),
            "control": np.arange(4, dtype=np.int64),
        },
        merged / "model.safetensors",
    )
    layer = audit(base, adapter, merged)["layers"][0]
    assert layer["alpha"] == -2
    assert layer["relative_residual"] == 0
    assert layer["projection_gain"] == pytest.approx(1)


@pytest.mark.parametrize("updates", [{"r": -1}, {"r": 0.5}, {"rank_pattern": {"linear": -1}}])
def test_signed_coefficients_keep_rank_validation(tmp_path: Path, updates: dict[str, Any]) -> None:
    base, adapter, merged = fixture(tmp_path)
    change_config(adapter, **updates)
    with pytest.raises(AuditError, match="positive integer"):
        audit(base, adapter, merged)


@pytest.mark.parametrize("gain", [-1, 0.5, 1, 2])
def test_signed_gain_distinguishes_complete_but_wrong_updates(tmp_path: Path, gain: float) -> None:
    base, adapter, merged = fixture(tmp_path)
    save_file(
        {
            "linear.weight": np.full((2, 3), 1 + gain * 0.125, dtype=np.float32),
            "control": np.arange(4, dtype=np.int64),
        },
        merged / "model.safetensors",
    )
    layer = audit(base, adapter, merged)["layers"][0]
    assert layer["changed_elements"] == 6
    assert layer["projection_gain"] == pytest.approx(gain)
    assert layer["cosine_similarity"] == pytest.approx(np.sign(gain))
    assert layer["relative_residual"] == pytest.approx(abs(1 - gain))


@pytest.mark.parametrize("magnitude", [1e-200, 1e200])
def test_alignment_does_not_square_raw_tiny_or_large_updates(
    tmp_path: Path, magnitude: float
) -> None:
    base, adapter, merged = fixture(tmp_path)
    save_file({"linear.weight": np.zeros((2, 3))}, base / "model.safetensors")
    save_file(
        {
            "base_model.model.linear.lora_A.weight": np.ones((1, 3)),
            "base_model.model.linear.lora_B.weight": np.full((2, 1), magnitude),
        },
        adapter / "adapter_model.safetensors",
    )
    # Opposite rows cancel in the inner product, despite every entry changing.
    update = np.array([[magnitude] * 3, [-magnitude] * 3])
    save_file({"linear.weight": update}, merged / "model.safetensors")
    layer = audit(base, adapter, merged)["layers"][0]
    assert layer["projection_gain"] == pytest.approx(0, abs=1e-15)
    assert layer["cosine_similarity"] == pytest.approx(0, abs=1e-15)
    assert layer["observed_to_intended_norm"] == pytest.approx(1)
    assert layer["relative_residual"] == pytest.approx(np.sqrt(2))


def test_alignment_across_rescaled_blocks_agrees_with_dense_reference(tmp_path: Path) -> None:
    base, adapter, merged = fixture(tmp_path)
    rng = np.random.default_rng(782)
    a = rng.integers(-10, 11, (1, 257)).astype(np.float64) / 16
    b = rng.integers(-10, 11, (700, 1)).astype(np.float64) / 16
    b[:200] *= 2**-20
    b[500:] *= 2**20
    delta = b @ a
    observed = 1.5 * delta + rng.integers(-10, 11, delta.shape) / 16
    save_file({"linear.weight": np.zeros_like(delta)}, base / "model.safetensors")
    save_file({"linear.weight": observed}, merged / "model.safetensors")
    save_file(
        {
            "base_model.model.linear.lora_A.weight": a,
            "base_model.model.linear.lora_B.weight": b,
        },
        adapter / "adapter_model.safetensors",
    )
    product = float(np.sum(observed * delta))
    expected_gain = product / float(np.sum(delta * delta))
    expected_cosine = product / float(np.linalg.norm(observed)) / float(np.linalg.norm(delta))
    for blocks in (1, 16):
        layer = audit(base, adapter, merged, block_mib=blocks)["layers"][0]
        assert layer["projection_gain"] == pytest.approx(expected_gain, rel=1e-13)
        assert layer["cosine_similarity"] == pytest.approx(expected_cosine, rel=1e-13)


@pytest.mark.parametrize("model_type", ["gpt2", "openai-gpt", "imagegpt"])
@pytest.mark.parametrize("witness", ["base-config", "adapter-mapping"])
def test_global_linear_flag_cannot_hide_known_square_conv1d(
    tmp_path: Path, model_type: str, witness: str
) -> None:
    base, adapter, merged = fixture(tmp_path)
    module = "transformer.h.0.attn.c_proj"
    a = np.array([[1, 2]], dtype=np.float32)
    b = np.array([[0.125], [0.5]], dtype=np.float32)
    save_file({module + ".weight": np.ones((2, 2), dtype=np.float32)}, base / "model.safetensors")
    save_file({module + ".weight": 1 + (b @ a).T.copy()}, merged / "model.safetensors")
    save_file(
        {
            f"base_model.model.{module}.lora_A.weight": a,
            f"base_model.model.{module}.lora_B.weight": b,
        },
        adapter / "adapter_model.safetensors",
    )
    change_config(adapter, fan_in_fan_out=False)
    if witness == "base-config":
        (base / "config.json").write_text(json.dumps({"model_type": model_type}))
        base_input = base
    else:
        library_name = "openai" if model_type == "openai-gpt" else model_type
        change_config(
            adapter,
            auto_mapping={
                "parent_library": f"transformers.models.{library_name}.modeling_{library_name}"
            },
        )
        base_input = base / "model.safetensors"
    with pytest.raises(AuditError, match="Conv1D.*per-module orientation"):
        audit(base_input, adapter, merged)


def test_known_conv1d_architecture_keeps_its_linear_head_supported(tmp_path: Path) -> None:
    base, adapter, merged = fixture(tmp_path)
    (base / "config.json").write_text(json.dumps({"model_type": "gpt2"}))
    change_config(
        adapter,
        auto_mapping={"parent_library": "transformers.models.gpt2.modeling_gpt2"},
    )
    # Ordinary Linear paths must remain usable even when the same architecture
    # also contains Conv1D modules. The native study checks actual lm_head exports.
    report = audit(base, adapter, merged)
    assert report["layers"][0]["relative_residual"] == 0


def test_unchanged_entry_count_and_reference_energy_are_different(tmp_path: Path) -> None:
    base, adapter, merged = fixture(tmp_path)
    a = np.array([[2**-20, 0.25]], dtype=np.float32)
    b = np.ones((2, 1), dtype=np.float32)
    save_file({"linear.weight": np.ones((2, 2), dtype=np.float32)}, base / "model.safetensors")
    save_file({"linear.weight": (1 + b @ a).astype(np.float16)}, merged / "model.safetensors")
    save_file(
        {
            "base_model.model.linear.lora_A.weight": a,
            "base_model.model.linear.lora_B.weight": b,
        },
        adapter / "adapter_model.safetensors",
    )
    report = audit(base, adapter, merged)
    layer = report["layers"][0]
    expected = (2**-20) ** 2 / ((2**-20) ** 2 + 0.25**2)
    assert layer["unchanged_nonzero_update_fraction"] == 0.5
    assert layer["unchanged_reference_energy_fraction"] == pytest.approx(expected, rel=1e-13)
    assert report["summary"]["unchanged_reference_energy_fraction"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "magnitude,multiple,expected", [(1e-200, 2, 0.2), (1e200, 2, 0.2), (1.5e308, 1, 0.5)]
)
def test_global_energy_fraction_weights_layers_without_raw_squares(
    tmp_path: Path, magnitude: float, multiple: float, expected: float
) -> None:
    base, adapter, merged = fixture(tmp_path)
    original = {"missed.weight": np.zeros((1, 1)), "present.weight": np.zeros((1, 1))}
    after = {
        "missed.weight": np.zeros((1, 1)),
        "present.weight": np.array([[magnitude * multiple]]),
    }
    factors = {}
    for name, multiplier in [("missed", 1), ("present", multiple)]:
        factors[f"base_model.model.{name}.lora_A.weight"] = np.ones((1, 1))
        factors[f"base_model.model.{name}.lora_B.weight"] = np.array([[magnitude * multiplier]])
    save_file(original, base / "model.safetensors")
    save_file(after, merged / "model.safetensors")
    save_file(factors, adapter / "adapter_model.safetensors")
    report = audit(base, adapter, merged)
    assert report["summary"]["unchanged_nonzero_update_fraction"] == 0.5
    assert report["summary"]["unchanged_reference_energy_fraction"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "option,value",
    [
        ("fan_in_fan_out", True),
        ("bias", "all"),
        ("modules_to_save", ["lm_head"]),
        ("target_parameters", ["experts"]),
        ("init_lora_weights", "pissa"),
        ("future_variant", None),
        ("lora_bias", True),
        ("rank_pattern", {"linear$": 1}),
        ("r", True),
        ("lora_alpha", float("nan")),
    ],
)
def test_unsupported_configuration_refuses(tmp_path: Path, option: str, value: Any) -> None:
    base, adapter, merged = fixture(tmp_path)
    change_config(adapter, **{option: value})
    with pytest.raises(AuditError):
        audit(base, adapter, merged)


def test_unadapted_integer_changes_are_retained_exactly(tmp_path: Path) -> None:
    base, adapter, merged = fixture(tmp_path)
    save_file(
        {
            "linear.weight": np.full((2, 3), 1.125, dtype=np.float32),
            "control": np.array([0, 1, 2, 2**62 + 3], dtype=np.int64),
        },
        merged / "model.safetensors",
    )
    report = audit(base, adapter, merged)
    assert report["summary"]["unadapted_changed_elements"] == 1
    assert report["unadapted_changes"][0]["first_change"]["merged"] == 2**62 + 3


def test_conversion_is_separate_from_adapter_update(tmp_path: Path) -> None:
    base, adapter, merged = fixture(tmp_path)
    original = np.full((2, 3), 1.0001, dtype=np.float32)
    save_file(
        {"linear.weight": original, "control": np.arange(4, dtype=np.int64)},
        base / "model.safetensors",
    )
    save_file(
        {
            "linear.weight": (original + 0.125).astype(np.float16),
            "control": np.arange(4, dtype=np.int64),
        },
        merged / "model.safetensors",
    )
    layer = audit(base, adapter, merged)["layers"][0]
    assert layer["norms"]["base_cast_drift"] > 0
    assert layer["relative_residual"] == 0


@pytest.mark.parametrize("kind", ["missing", "shape", "nonfinite", "incomplete", "extra"])
def test_incomplete_or_invalid_evidence_refuses(tmp_path: Path, kind: str) -> None:
    base, adapter, merged = fixture(tmp_path)
    if kind in {"missing", "shape", "nonfinite", "extra"}:
        tensors = {
            "linear.weight": np.ones((2, 3), dtype=np.float32),
            "control": np.arange(4, dtype=np.int64),
        }
        if kind == "missing":
            del tensors["control"]
        if kind == "shape":
            tensors["linear.weight"] = np.ones((3, 2), dtype=np.float32)
        if kind == "nonfinite":
            tensors["linear.weight"][0, 0] = np.nan
        if kind == "extra":
            tensors["extra"] = np.zeros(1, dtype=np.float32)
        save_file(tensors, merged / "model.safetensors")
    else:
        save_file(
            {"base_model.model.linear.lora_A.weight": np.ones((1, 3), dtype=np.float32)},
            adapter / "adapter_model.safetensors",
        )
    with pytest.raises(AuditError):
        audit(base, adapter, merged)


def test_bf16_ties_subnormals_and_double_rounding() -> None:
    values = np.array(
        [
            1.00390625,
            1.00390625 + 2**-50,
            1.01171875,
            -1.00390625 - 2**-50,
            2**-134,
            3 * 2**-134,
            -0.0,
        ]
    )
    expected = [1, 1.0078125, 1.015625, -1.0078125, 0, 2**-132, -0.0]
    np.testing.assert_array_equal(cast(values, "BF16"), expected)
    assert np.signbit(cast(values, "BF16")[-1])


def test_sharded_checkpoint_inventory_and_index_validation(tmp_path: Path) -> None:
    base, adapter, merged = fixture(tmp_path)
    checkpoint = Checkpoint(base)
    weights = {
        name: tensor.raw().reshape(tensor.shape) for name, tensor in checkpoint.tensors.items()
    }
    (base / "model.safetensors").unlink()
    mapping = {}
    for index, (name, value) in enumerate(weights.items()):
        shard = f"part-{index}.safetensors"
        save_file({name: value}, base / shard)
        mapping[name] = shard
    index_path = base / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": mapping}))
    assert audit(base, adapter, merged)["layers"][0]["relative_residual"] == 0
    mapping["missing"] = "part-0.safetensors"
    index_path.write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(AuditError, match="missing tensors"):
        audit(base, adapter, merged)


def test_duplicate_safetensors_header_is_not_silently_overwritten(tmp_path: Path) -> None:
    header = (
        b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'
        b'"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    )
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    with pytest.raises(AuditError, match="Duplicate"):
        Checkpoint(path)


def test_cli_refusal_preserves_earlier_report_and_inputs(tmp_path: Path, capsys: Any) -> None:
    base, adapter, merged = fixture(tmp_path)
    report = tmp_path / "report.json"
    report.write_text("earlier result")
    original = (base / "model.safetensors").read_bytes()
    assert (
        main([str(base), str(adapter), str(merged), "--json", str(base / "model.safetensors")]) == 2
    )
    assert (base / "model.safetensors").read_bytes() == original
    change_config(adapter, use_dora=True)
    assert main([str(base), str(adapter), str(merged), "--json", str(report)]) == 2
    assert report.read_text() == "earlier result"
    assert capsys.readouterr().err


def test_cli_json_is_parseable_and_does_not_include_machine_paths(
    tmp_path: Path, capsys: Any
) -> None:
    paths = fixture(tmp_path)
    assert main([*(str(p) for p in paths), "--json", "-"]) == 0
    text = capsys.readouterr().out
    assert json.loads(text)["schema_version"] == 1
    assert str(tmp_path) not in text
