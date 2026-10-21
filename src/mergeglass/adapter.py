"""Explicit PEFT Linear/Embedding layouts; no guessed key matching."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .parameter_layouts import ExpertLayout, parameter_layouts
from .storage import FLOAT_DTYPES, AuditError, Checkpoint, Tensor

# These fields select variants, add other trained parameters, change the base,
# or require architecture information that a checkpoint cannot establish.
UNSUPPORTED = {
    "use_qalora",
    "use_bdlora",
    "layer_replication",
    "trainable_token_indices",
    "alora_invocation_tokens",
    "arrow_config",
    "megatron_config",
    "loftq_config",
    "eva_config",
    "corda_config",
    "velora_config",
    "monteclora_config",
    "kasa_config",
    "ensure_weight_tying",
    "use_mora",
    "use_rslora_bias",
    "lora_extra_parameters",
}
KNOWN = UNSUPPORTED | {
    "peft_type",
    "peft_version",
    "task_type",
    "base_model_name_or_path",
    "revision",
    "inference_mode",
    "r",
    "lora_alpha",
    "lora_dropout",
    "bias",
    "target_modules",
    "modules_to_save",
    "exclude_modules",
    "layers_to_transform",
    "layers_pattern",
    "rank_pattern",
    "alpha_pattern",
    "use_rslora",
    "use_dora",
    "lora_bias",
    "target_parameters",
    "fan_in_fan_out",
    "init_lora_weights",
    "auto_mapping",
    "megatron_core",
    "runtime_config",
    "qalora_group_size",
    "use_lora_bias",
    "tie_word_embeddings",
    "init_lora_weights_params",
    "lora_ga_config",
    "loraplus_lr_ratio",
    "use_structured_lora",
    "mica_config",
}
PAIR = re.compile(r"base_model\.model\.(.+)\.lora_([AB])\.weight\Z")
LORA_B_BIAS = re.compile(r"base_model\.model\.(.+)\.lora_B\.bias\Z")
EMBEDDING_PAIR = re.compile(r"base_model\.model\.(.+)\.lora_embedding_([AB])\Z")
BASE_SNAPSHOT = re.compile(r"base_model\.model\.(.+)\.base_layer\.weight\Z")
MAGNITUDE = re.compile(r"base_model\.model\.(.+)\.lora_magnitude_vector\Z")
LITERAL_PATH = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*\Z")
CONV1D_PATH = re.compile(
    r"(?:^|\.)h\.[0-9]+\.(?:(?:attn|crossattention)\.(?:c_attn|q_attn|c_proj)"
    r"|mlp\.(?:c_fc|c_proj))\Z"
)
STANDARD_CONV1D_PATH = re.compile(
    r"(?:transformer\.)?h\.[0-9]+\.(?:attn\.(?:c_attn|c_proj)"
    r"|crossattention\.(?:c_attn|q_attn|c_proj)"
    r"|mlp\.(?:c_fc|c_proj))\Z"
)
CONV1D_LIBRARIES = {
    "transformers.models.gpt2.modeling_gpt2": "gpt2",
    "transformers.models.openai.modeling_openai": "openai-gpt",
    "transformers.models.imagegpt.modeling_imagegpt": "imagegpt",
}
CONV1D_MODELS = frozenset(CONV1D_LIBRARIES.values())


def _conv1d_evidence(config: dict[str, Any], base: Checkpoint, path: str) -> tuple[str, ...]:
    """Resolve standard GPT layouts from declared metadata, never a numeric fit.

    PEFT dispatch mutates its *global* flag at each target. An ordinary head
    can leave it false even though earlier Conv1D modules need transposition.
    """
    mapping = config.get("auto_mapping")
    library = mapping.get("parent_library") if isinstance(mapping, dict) else None
    model_type = base.config.get("model_type")
    mapped_type = CONV1D_LIBRARIES.get(library) if isinstance(library, str) else None
    known_model = isinstance(model_type, str) and model_type in CONV1D_MODELS
    evidence = []
    if (known_model or mapped_type) and CONV1D_PATH.search(path):
        if known_model and library is not None and mapped_type is None:
            raise AuditError(f"Unrecognized Conv1D adapter library at {path}: {library}")
        if model_type is not None and mapped_type is not None and model_type != mapped_type:
            raise AuditError(f"Conflicting Conv1D model metadata at {path}")
        if base.config.get("auto_map"):
            raise AuditError(f"Custom model auto_map cannot establish a Conv1D layout: {path}")
        if not STANDARD_CONV1D_PATH.fullmatch(path):
            raise AuditError(f"Unsupported nonstandard Conv1D module path: {path}")
        kind = model_type if known_model else mapped_type
        if (".crossattention." in path or path.endswith(".q_attn")) and kind != "gpt2":
            raise AuditError(f"Unsupported Conv1D path for {kind}: {path}")
        if known_model:
            evidence.append(f"{base.source.name}/config.json:model_type={model_type}")
        if mapped_type:
            evidence.append(f"adapter_config.json:auto_mapping.parent_library={library}")
    return tuple(evidence)


def _number(value: Any, field: str, *, rank: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError(f"{field} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise AuditError(f"{field} exceeds the numeric range") from exc
    if not math.isfinite(number):
        raise AuditError(f"{field} must be a finite number")
    if rank and (value < 1 or int(value) != value):
        raise AuditError(f"{field} must be a positive integer")
    return number


def _pattern(config: dict[str, Any], field: str) -> dict[str, float]:
    patterns = config.get(field, {})
    if not isinstance(patterns, dict):
        raise AuditError(f"{field} must be an object")
    result = {}
    for key, value in patterns.items():
        if not LITERAL_PATH.fullmatch(key):
            raise AuditError(f"Only literal module paths are supported in {field}: {key}")
        result[key] = _number(value, field, rank=field == "rank_pattern")
    return result


def _override(path: str, patterns: dict[str, float], fallback: float) -> float:
    # Match PEFT's get_pattern_key, including its regex interpretation of dots.
    # Our accepted characters exclude quantifiers/groups and bound regex work.
    return next(
        (value for key, value in patterns.items() if re.match(rf"(.*\.)?({key})$", path)),
        fallback,
    )


@dataclass(frozen=True)
class Pair:
    module: str
    weight: str
    a: Tensor
    b: Tensor
    rank: int
    alpha: float
    scale: float
    magnitude: Tensor | None = None
    embedding: bool = False
    base_snapshot: Tensor | None = None
    conv1d_evidence: tuple[str, ...] = ()
    bias: Tensor | None = None
    expert_layout: ExpertLayout | None = None

    @property
    def transposed(self) -> bool:
        return self.embedding or bool(self.conv1d_evidence)


@dataclass(frozen=True)
class SavedTensor:
    module: str
    weight: str
    tensor: Tensor


@dataclass(frozen=True)
class SavedModules:
    tensors: tuple[SavedTensor, ...] = ()
    unmatched_selectors: tuple[str, ...] = ()


def _saved_selectors(config: dict[str, Any]) -> list[str]:
    value = config.get("modules_to_save")
    if value is None:
        return []
    if not isinstance(value, list) or any(
        not isinstance(path, str) or not LITERAL_PATH.fullmatch(path) for path in value
    ):
        raise AuditError("modules_to_save must be a list of literal module paths or null")
    return list(dict.fromkeys(value))


def saved_modules(adapter: Checkpoint, base: Checkpoint, merged: Checkpoint) -> SavedModules:
    """Map full saved state to exact names, including explicitly added weights.

    PEFT selects modules with literal string suffixes, then strips its wrapper
    and adapter name when saving their complete state_dict. Checkpoint tensors
    establish observable module prefixes, not an architecture or tied aliases.
    """
    selectors = _saved_selectors(adapter.config)
    if not selectors:
        return SavedModules()
    names = base.tensors.keys() | merged.tensors.keys()
    prefixes = {name[:index] for name in names for index, char in enumerate(name) if char == "."}
    modules = sorted(path for path in prefixes if any(path.endswith(s) for s in selectors))
    for index, path in enumerate(modules):
        if any(other.startswith(path + ".") for other in modules[index + 1 :]):
            raise AuditError(f"Overlapping modules_to_save scopes are unsupported: {path}")
    result = []
    for module in modules:
        for name in sorted(name for name in names if name.startswith(module + ".")):
            key = "base_model.model." + name
            if key not in adapter.tensors:
                raise AuditError(f"Incomplete saved module; missing adapter tensor: {key}")
            result.append(SavedTensor(module, name, adapter.tensors[key]))
    if not result:
        raise AuditError("No saved module tensors found for modules_to_save")
    return SavedModules(
        tuple(result), tuple(s for s in selectors if not any(m.endswith(s) for m in modules))
    )


def adapter_pairs(
    adapter: Checkpoint,
    base: Checkpoint,
    *,
    weight_role: str = "base",
    full_weights: frozenset[str] = frozenset(),
) -> list[Pair]:
    config = adapter.config
    if config.get("peft_type") != "LORA":
        raise AuditError("Only PEFT LORA adapters are supported")
    if _saved_selectors(config) and not full_weights:
        raise AuditError("Full saved modules require the complete base/adapter/merged audit")
    unknown = config.keys() - KNOWN
    if unknown:
        raise AuditError("Unknown adapter configuration fields: " + ", ".join(sorted(unknown)))
    for key in UNSUPPORTED:
        if config.get(key):
            raise AuditError(f"Unsupported adapter option: {key}")
    for key in (
        "use_lora_bias",
        "tie_word_embeddings",
        "init_lora_weights_params",
        "lora_ga_config",
        "loraplus_lr_ratio",
        "use_structured_lora",
        "mica_config",
    ):
        if config.get(key):
            raise AuditError(f"Unsupported adapter option: {key}")
    if config.get("bias", "none") != "none":
        raise AuditError("Trained bias parameters are unsupported")
    init = config.get("init_lora_weights", True)
    if init is not True and init is not False and init != "gaussian":
        raise AuditError("Base-mutating or variant LoRA initialization is unsupported")
    if config.get("runtime_config"):
        raise AuditError("Saved runtime_config is unsupported")
    r = _number(config.get("r"), "r", rank=True)
    alpha = _number(config.get("lora_alpha"), "lora_alpha")
    rs = config.get("use_rslora", False)
    if not isinstance(rs, bool):
        raise AuditError("use_rslora must be boolean")
    dora = config.get("use_dora", False)
    if not isinstance(dora, bool):
        raise AuditError("use_dora must be boolean")
    lora_bias = config.get("lora_bias", False)
    if not isinstance(lora_bias, bool):
        raise AuditError("lora_bias must be boolean")
    if lora_bias and dora:
        raise AuditError("LoRA B bias combined with DoRA is unsupported")
    fan = config.get("fan_in_fan_out", False)
    if not isinstance(fan, bool):
        raise AuditError("fan_in_fan_out must be boolean")
    ranks = _pattern(config, "rank_pattern")
    alphas = _pattern(config, "alpha_pattern")
    parameters = parameter_layouts(config, base)
    modules: dict[str, dict[str, Tensor]] = {}
    for name, tensor in sorted(adapter.tensors.items()):
        if name in full_weights:
            continue
        match = PAIR.fullmatch(name)
        if match is None:
            embedding_match = EMBEDDING_PAIR.fullmatch(name)
            if embedding_match is not None:
                path, side = embedding_match.group(1), "embedding_" + embedding_match.group(2)
            else:
                snapshot_match = BASE_SNAPSHOT.fullmatch(name)
                magnitude_match = MAGNITUDE.fullmatch(name)
                bias_match = LORA_B_BIAS.fullmatch(name)
                if bias_match is not None and lora_bias:
                    path, side = bias_match.group(1), "bias"
                elif snapshot_match is not None:
                    path, side = snapshot_match.group(1), "base_snapshot"
                elif magnitude_match is not None and dora:
                    path, side = magnitude_match.group(1), "magnitude"
                else:
                    raise AuditError(f"Unsupported adapter tensor: {name}")
        else:
            path, side = match.groups()
        modules.setdefault(path, {})[side] = tensor
    if parameters.keys() - modules.keys():
        raise AuditError(
            "Missing expert parameter factors: "
            + ", ".join(sorted(parameters.keys() - modules.keys()))
        )
    result = []
    for path, sides in modules.items():
        parameter = parameters.get(path)
        embedding = bool({"embedding_A", "embedding_B"} & sides.keys())
        if embedding and {"A", "B"} & sides.keys():
            raise AuditError(f"Mixed Linear and Embedding factors at the same module: {path}")
        if embedding and lora_bias:
            raise AuditError(f"LoRA B bias with Embedding factors is unsupported: {path}")
        conv1d_evidence = _conv1d_evidence(config, base, path)
        if conv1d_evidence and embedding:
            raise AuditError(f"Embedding factors conflict with Conv1D model metadata: {path}")
        if conv1d_evidence and dora:
            raise AuditError(f"Conv1D DoRA is unsupported: {path}")
        if fan and not conv1d_evidence and not embedding:
            # In a mixed standard model, the ordinary head is still Linear.
            mapping = config.get("auto_mapping")
            library = mapping.get("parent_library") if isinstance(mapping, dict) else None
            model_type = base.config.get("model_type")
            known_model = (isinstance(model_type, str) and model_type in CONV1D_MODELS) or (
                isinstance(library, str) and library in CONV1D_LIBRARIES
            )
            if not known_model or path != "lm_head":
                raise AuditError(
                    f"Unsupported adapter option: fan_in_fan_out=True without a supported "
                    f"per-module layout at {path}; supply standard model metadata"
                )
        a_key, b_key = ("embedding_A", "embedding_B") if embedding else ("A", "B")
        if not {a_key, b_key} <= sides.keys():
            raise AuditError(f"Incomplete A/B pair: {path}")
        magnitude = sides.get("magnitude")
        if dora and magnitude is None:
            raise AuditError(f"Missing DoRA magnitude vector: {path}")
        a, b = sides[a_key], sides[b_key]
        if parameter and (
            embedding or magnitude is not None or sides.get("base_snapshot") is not None
        ):
            raise AuditError(f"Unsupported expert parameter factor variant at {path}")
        weight = parameter.parameter if parameter else path + ".weight"
        if parameter is None and weight not in base.tensors:
            raise AuditError(f"No exact {weight_role} tensor for adapter module: {weight}")
        shape = (parameter.outputs, parameter.inputs) if parameter else base.tensors[weight].shape
        pattern_path = parameter.parameter if parameter else path
        rank = int(_override(pattern_path, ranks, r))
        layer_alpha = _override(pattern_path, alphas, alpha)
        transposed = embedding or bool(conv1d_evidence)
        if parameter:
            if a.shape != (rank * parameter.experts, parameter.inputs) or b.shape != (
                parameter.outputs,
                rank * parameter.experts,
            ):
                raise AuditError(f"Expert adapter/{weight_role}/rank shape mismatch at {weight}")
        elif (
            len(shape) != 2
            or min(shape) == 0
            or a.shape != (rank, shape[0] if transposed else shape[1])
            or b.shape != (shape[1] if transposed else shape[0], rank)
        ):
            raise AuditError(f"Adapter/{weight_role}/rank shape mismatch at {path}")
        if magnitude is not None:
            axis = 1 if embedding else 0
            if magnitude.shape != (shape[axis],):
                role = "embedding dimensions" if embedding else "output rows"
                raise AuditError(f"DoRA magnitude shape must match the {role} at {path}")
        bias = sides.get("bias")
        if lora_bias:
            if bias is None:
                raise AuditError(f"Missing LoRA B bias: {path}")
            bias_weight = base.tensors.get(path + ".bias")
            expected = (shape[1] if transposed else shape[0],)
            if bias_weight is None:
                raise AuditError(f"No exact {weight_role} bias tensor for adapter module: {path}")
            if bias.shape != expected or bias_weight.shape != expected:
                raise AuditError(f"Adapter/{weight_role} bias shape mismatch at {path}")
            if bias_weight.dtype not in FLOAT_DTYPES:
                raise AuditError(f"Adapted {weight_role} bias is not floating point: {path}")
        snapshot = sides.get("base_snapshot")
        if snapshot is not None and (snapshot.shape != shape or snapshot.dtype not in FLOAT_DTYPES):
            raise AuditError(f"Base snapshot shape/dtype mismatch at {path}")
        scale = layer_alpha / (math.sqrt(rank) if rs else rank)
        result.append(
            Pair(
                parameter.parameter if parameter else path,
                weight,
                a,
                b,
                rank,
                layer_alpha,
                scale,
                magnitude,
                embedding,
                snapshot,
                conv1d_evidence,
                bias,
                parameter,
            )
        )
    if not result:
        raise AuditError("Adapter contains no supported A/B pairs")
    return result
