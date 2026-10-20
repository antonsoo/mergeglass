"""Explicit PEFT Linear/Embedding layouts; no guessed key matching."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .storage import FLOAT_DTYPES, AuditError, Checkpoint, Tensor

# These fields select variants, add other trained parameters, change the base,
# or require architecture information that a checkpoint cannot establish.
UNSUPPORTED = {
    "use_qalora",
    "use_bdlora",
    "lora_bias",
    "fan_in_fan_out",
    "modules_to_save",
    "layer_replication",
    "trainable_token_indices",
    "target_parameters",
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
    "exclude_modules",
    "layers_to_transform",
    "layers_pattern",
    "rank_pattern",
    "alpha_pattern",
    "use_rslora",
    "use_dora",
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
EMBEDDING_PAIR = re.compile(r"base_model\.model\.(.+)\.lora_embedding_([AB])\Z")
BASE_SNAPSHOT = re.compile(r"base_model\.model\.(.+)\.base_layer\.weight\Z")
MAGNITUDE = re.compile(r"base_model\.model\.(.+)\.lora_magnitude_vector\Z")
LITERAL_PATH = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*\Z")
CONV1D_PATH = re.compile(
    r"(?:^|\.)h\.[0-9]+\.(?:(?:attn|crossattention)\.(?:c_attn|q_attn|c_proj)"
    r"|mlp\.(?:c_fc|c_proj))\Z"
)
CONV1D_MODELS = ("gpt2", "openai-gpt", "imagegpt")
CONV1D_LIBRARIES = (
    "transformers.models.gpt2.modeling_gpt2",
    "transformers.models.openai.modeling_openai",
    "transformers.models.imagegpt.modeling_imagegpt",
)


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


def adapter_pairs(
    adapter: Checkpoint, base: Checkpoint, *, weight_role: str = "base"
) -> list[Pair]:
    config = adapter.config
    if config.get("peft_type") != "LORA":
        raise AuditError("Only PEFT LORA adapters are supported")
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
    ranks = _pattern(config, "rank_pattern")
    alphas = _pattern(config, "alpha_pattern")
    modules: dict[str, dict[str, Tensor]] = {}
    for name, tensor in sorted(adapter.tensors.items()):
        match = PAIR.fullmatch(name)
        if match is None:
            embedding_match = EMBEDDING_PAIR.fullmatch(name)
            if embedding_match is not None:
                path, side = embedding_match.group(1), "embedding_" + embedding_match.group(2)
            else:
                snapshot_match = BASE_SNAPSHOT.fullmatch(name)
                magnitude_match = MAGNITUDE.fullmatch(name)
                if snapshot_match is not None:
                    path, side = snapshot_match.group(1), "base_snapshot"
                elif magnitude_match is not None and dora:
                    path, side = magnitude_match.group(1), "magnitude"
                else:
                    raise AuditError(f"Unsupported adapter tensor: {name}")
        else:
            path, side = match.groups()
        modules.setdefault(path, {})[side] = tensor
    result = []
    for path, sides in modules.items():
        embedding = bool({"embedding_A", "embedding_B"} & sides.keys())
        if embedding and {"A", "B"} & sides.keys():
            raise AuditError(f"Mixed Linear and Embedding factors at the same module: {path}")
        if embedding and dora:
            raise AuditError(f"Embedding DoRA is unsupported: {path}")
        mapping = config.get("auto_mapping")
        library = mapping.get("parent_library") if isinstance(mapping, dict) else None
        if (
            base.config.get("model_type") in CONV1D_MODELS or library in CONV1D_LIBRARIES
        ) and CONV1D_PATH.search(path):
            # PEFT mutates the global config flag as it dispatches each target.
            # A final Linear head can hide earlier Conv1D targets. Square weights
            # pass the Linear shape check despite requiring a transposed product.
            raise AuditError(
                f"Unsupported Conv1D target indicated by model metadata: {path}; "
                "saved fan_in_fan_out is not a per-module orientation map"
            )
        a_key, b_key = ("embedding_A", "embedding_B") if embedding else ("A", "B")
        if not {a_key, b_key} <= sides.keys():
            raise AuditError(f"Incomplete A/B pair: {path}")
        magnitude = sides.get("magnitude")
        if dora and magnitude is None:
            raise AuditError(f"Missing DoRA magnitude vector: {path}")
        a, b = sides[a_key], sides[b_key]
        weight = path + ".weight"
        if weight not in base.tensors:
            raise AuditError(f"No exact {weight_role} tensor for adapter module: {weight}")
        shape = base.tensors[weight].shape
        rank = int(_override(path, ranks, r))
        layer_alpha = _override(path, alphas, alpha)
        if (
            len(shape) != 2
            or min(shape) == 0
            or a.shape != (rank, shape[0] if embedding else shape[1])
            or b.shape != (shape[1] if embedding else shape[0], rank)
        ):
            raise AuditError(f"Adapter/{weight_role}/rank shape mismatch at {path}")
        if magnitude is not None and magnitude.shape != (shape[0],):
            raise AuditError(f"DoRA magnitude shape must match the output rows at {path}")
        snapshot = sides.get("base_snapshot")
        if snapshot is not None and (snapshot.shape != shape or snapshot.dtype not in FLOAT_DTYPES):
            raise AuditError(f"Base snapshot shape/dtype mismatch at {path}")
        scale = layer_alpha / (math.sqrt(rank) if rs else rank)
        result.append(
            Pair(path, weight, a, b, rank, layer_alpha, scale, magnitude, embedding, snapshot)
        )
    if not result:
        raise AuditError("Adapter contains no supported A/B pairs")
    return result
