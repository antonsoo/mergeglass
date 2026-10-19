"""The deliberately narrow PEFT Linear LoRA contract; no guessed key matching."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .storage import AuditError, Checkpoint, Tensor

# These fields select variants, add other trained parameters, change the base,
# or require architecture information that a checkpoint cannot establish.
UNSUPPORTED = {
    "use_dora",
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
LITERAL_PATH = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*\Z")


def _number(value: Any, field: str, *, rank: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AuditError(f"{field} must be a finite number")
    if value < (1 if rank else 0) or (rank and int(value) != value):
        raise AuditError(f"{field} must be {'a positive integer' if rank else 'nonnegative'}")
    return float(value)


def _pattern(config: dict[str, Any], field: str) -> dict[str, float]:
    patterns = config.get(field) or {}
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


def adapter_pairs(adapter: Checkpoint, base: Checkpoint) -> list[Pair]:
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
    ranks = _pattern(config, "rank_pattern")
    alphas = _pattern(config, "alpha_pattern")
    modules: dict[str, dict[str, Tensor]] = {}
    for name, tensor in sorted(adapter.tensors.items()):
        match = PAIR.fullmatch(name)
        if match is None:
            raise AuditError(f"Unsupported adapter tensor: {name}")
        path, side = match.groups()
        modules.setdefault(path, {})[side] = tensor
    result = []
    for path, sides in modules.items():
        if sides.keys() != {"A", "B"}:
            raise AuditError(f"Incomplete A/B pair: {path}")
        a, b = sides["A"], sides["B"]
        weight = path + ".weight"
        if weight not in base.tensors:
            raise AuditError(f"No exact base tensor for adapter module: {weight}")
        shape = base.tensors[weight].shape
        rank = int(_override(path, ranks, r))
        layer_alpha = _override(path, alphas, alpha)
        if (
            len(shape) != 2
            or min(shape) == 0
            or a.shape != (rank, shape[1])
            or b.shape != (shape[0], rank)
        ):
            raise AuditError(f"Adapter/base/rank shape mismatch at {path}")
        scale = layer_alpha / (math.sqrt(rank) if rs else rank)
        result.append(Pair(path, weight, a, b, rank, layer_alpha, scale))
    if not result:
        raise AuditError("Adapter contains no supported A/B pairs")
    return result
