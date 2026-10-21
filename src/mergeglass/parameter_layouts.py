"""Explicit saved ParamWrapper layouts for one verified native producer ABI.

No target is chosen by its numeric fit. The fixed Qwen3MoeExperts parameter
registration order determines PEFT's nested wrapper names in these versions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .storage import AuditError, Checkpoint

LITERAL = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*\Z")
SAVED_EXPERT = re.compile(
    r"(model\.layers\.(0|[1-9][0-9]{0,8})\.mlp\.experts)\."
    r"(0|[1-9][0-9]{0,8})\.(gate_proj|up_proj|down_proj)\.weight\Z"
)
PARAMETER_ORDER = ("gate_up_proj", "down_proj")
LIBRARY = "transformers.models.qwen3_moe.modeling_qwen3_moe"


@dataclass(frozen=True)
class ExpertMatrix:
    weight: str
    expert: int
    output_start: int
    rows: int


@dataclass(frozen=True)
class ExpertLayout:
    parameter: str
    factor_module: str
    experts: int
    outputs: int
    inputs: int
    matrices: tuple[ExpertMatrix, ...]
    evidence: tuple[str, ...]


def parameter_layouts(config: dict[str, Any], base: Checkpoint) -> dict[str, ExpertLayout]:
    value = config.get("target_parameters")
    if value is None or value == []:
        return {}
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not LITERAL.fullmatch(item) for item in value)
    ):
        raise AuditError("target_parameters requires a nonempty list of literal parameter paths")
    selectors: list[str] = list(dict.fromkeys(value))
    if (
        base.config.get("model_type") != "qwen3_moe"
        or base.config.get("architectures") != ["Qwen3MoeForCausalLM"]
        or base.config.get("transformers_version") != "5.20.1"
        or config.get("peft_version") != "0.21.0"
    ):
        raise AuditError(
            "Expert parameters require declared standard Qwen3MoeForCausalLM "
            "Transformers 5.20.1 / PEFT 0.21.0 metadata"
        )
    if base.config.get("auto_map") or base.config.get("quantization_config"):
        raise AuditError("Custom or quantized expert model declarations are unsupported")
    mapping = config.get("auto_mapping")
    if mapping is not None and (
        not isinstance(mapping, dict)
        or mapping.get("parent_library") != LIBRARY
        or mapping.get("base_model_class") != "Qwen3MoeForCausalLM"
    ):
        raise AuditError("Conflicting expert adapter auto_mapping metadata")
    for field in (
        "use_dora",
        "lora_bias",
        "fan_in_fan_out",
        "lora_dropout",
        "exclude_modules",
        "layers_to_transform",
        "layers_pattern",
    ):
        if config.get(field):
            raise AuditError(f"Unsupported expert parameter option: {field}")
    dimensions = {}
    for field in ("num_local_experts", "hidden_size", "moe_intermediate_size", "num_hidden_layers"):
        item = base.config.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise AuditError(f"Expert model {field} must be a positive integer")
        dimensions[field] = item
    experts = dimensions["num_local_experts"]
    if base.config.get("num_experts", experts) != experts:
        raise AuditError("Conflicting expert-count aliases in model metadata")
    hidden = dimensions["hidden_size"]
    intermediate = dimensions["moe_intermediate_size"]
    modules: set[str] = set()
    for name in base.tensors:
        match = SAVED_EXPERT.fullmatch(name)
        if match:
            if int(match[2]) >= dimensions["num_hidden_layers"]:
                raise AuditError(f"Expert parameter exceeds declared layer count: {name}")
            if int(match[3]) >= experts:
                raise AuditError(f"Saved expert exceeds declared expert count: {name}")
            modules.add(match[1])
    result = {}
    matched: set[str] = set()
    for module in sorted(modules):
        selected = []
        for short_name in PARAMETER_ORDER:
            name = module + "." + short_name
            matches = [s for s in selectors if name == s or name.endswith("." + s)]
            if matches:
                matched.update(matches)
                selected.append((name, short_name))
        for index, (name, short_name) in enumerate(selected):
            outputs, inputs = (
                (2 * intermediate, hidden)
                if short_name == "gate_up_proj"
                else (hidden, intermediate)
            )
            # The native HF saver expands runtime 3-D parameters into ordinary
            # expert Linear weights, splitting gate_up_proj into gate and up.
            # Follow those exact keys; never invent a contiguous saved 3-D view.
            parts = (
                (("gate_proj", 0, intermediate), ("up_proj", intermediate, intermediate))
                if short_name == "gate_up_proj"
                else (("down_proj", 0, hidden),)
            )
            matrices = []
            for expert in range(experts):
                for part, start, rows in parts:
                    weight = f"{module}.{expert}.{part}.weight"
                    tensor = base.tensors.get(weight)
                    if tensor is None or tensor.shape != (rows, inputs):
                        raise AuditError(
                            f"Missing or malformed native saved expert matrix: {weight}"
                        )
                    matrices.append(ExpertMatrix(weight, expert, start, rows))
            depth = len(selected) - index - 1
            factor_module = module + ".base_layer" * depth
            result[factor_module] = ExpertLayout(
                name,
                factor_module,
                experts,
                outputs,
                inputs,
                tuple(matrices),
                (
                    "base/config.json:model_type=qwen3_moe; architectures=Qwen3MoeForCausalLM",
                    "base/config.json:transformers_version=5.20.1",
                    "adapter_config.json:peft_version=0.21.0",
                    "Qwen3MoeExperts registration order: gate_up_proj, down_proj; "
                    "PEFT nests later selected parameters outside earlier ones",
                    f"declared dimensions: experts={experts}, outputs={outputs}, inputs={inputs}",
                    "native HF serialization: per-expert Linear weights; gate_up_proj splits "
                    "into gate_proj then up_proj output rows",
                ),
            )
    if set(selectors) != matched:
        missing = sorted(set(selectors) - matched)
        raise AuditError(
            "Unsupported or unobserved expert parameter selectors: " + ", ".join(missing)
        )
    return result
