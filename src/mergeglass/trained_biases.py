"""Absolute PEFT trained base biases, kept separate from scaled LoRA B bias."""

from __future__ import annotations

from dataclasses import dataclass

from .adapter import EMBEDDING_PAIR, PAIR, SavedModules, SavedTensor
from .storage import FLOAT_DTYPES, AuditError, Checkpoint, Tensor


def bias_mode(adapter: Checkpoint) -> str:
    mode = adapter.config.get("bias", "none")
    if mode not in ("none", "all", "lora_only"):
        raise AuditError("bias must be none, all or lora_only")
    return str(mode)


@dataclass(frozen=True)
class BiasAlias:
    tensor: Tensor
    reference: Tensor
    reference_kind: str


@dataclass(frozen=True)
class TrainedBiases:
    tensors: tuple[SavedTensor, ...] = ()
    aliases: tuple[BiasAlias, ...] = ()


def trained_biases(adapter: Checkpoint, base: Checkpoint, full: SavedModules) -> TrainedBiases:
    """Require every declared observable bias; never replace a missing trained value.

    Accepted names are literal saved base keys or the exact base_layer wrapper
    of an observed A/B module. No architecture is imported or inferred. The
    all mode conservatively refuses other 'bias' substrings because PEFT saves
    those too, but they need not represent affine bias vectors.
    """
    mode = bias_mode(adapter)
    if mode == "none":
        return TrainedBiases()
    if adapter.config.get("lora_bias"):
        raise AuditError("Trained base biases combined with LoRA B bias are unsupported")
    if adapter.config.get("target_parameters"):
        raise AuditError("Trained base biases combined with expert parameters are unsupported")
    modules = set()
    for name in adapter.tensors:
        match = PAIR.fullmatch(name) or EMBEDDING_PAIR.fullmatch(name)
        if match is not None:
            modules.add(match.group(1))
    expected = []
    full_weights = {saved.weight for saved in full.tensors}
    for name, tensor in sorted(base.tensors.items()):
        if name in full_weights:
            continue
        module, _, suffix = name.rpartition(".")
        if mode == "lora_only" and module not in modules:
            continue
        if (mode == "all" and "bias" not in name) or (mode == "lora_only" and suffix != "bias"):
            continue
        if suffix != "bias" or not module or len(tensor.shape) != 1:
            raise AuditError(f"Unsupported trained base bias layout: {name}")
        if tensor.dtype not in FLOAT_DTYPES:
            raise AuditError(f"Trained base bias must be floating point: {name}")
        keys = ["base_model.model." + name]
        if module in modules:
            keys.append("base_model.model." + module + ".base_layer.bias")
        found = [adapter.tensors[key] for key in keys if key in adapter.tensors]
        if not found:
            raise AuditError(
                f"Missing trained base bias: {name} (bias={mode}); "
                "the original base cannot recover an omitted trained value"
            )
        if len(found) != 1:
            raise AuditError(f"Ambiguous trained base bias aliases: {name}")
        saved = found[0]
        if saved.shape != tensor.shape or saved.dtype not in FLOAT_DTYPES:
            raise AuditError(f"Trained base bias shape/dtype mismatch: {name}")
        expected.append(SavedTensor(module, name, saved))
    aliases = []
    seen_aliases: set[str] = set()
    full_keys = {replacement.tensor.name for replacement in full.tensors}
    if mode == "all":
        for replacement in full.tensors:
            if not replacement.weight.endswith(".bias"):
                continue
            suffix = replacement.weight[len(replacement.module) + 1 :]
            prefix = "base_model.model." + replacement.module
            active_prefix = prefix + ".modules_to_save."
            active = []
            for key in adapter.tensors:
                if not key.startswith(active_prefix):
                    continue
                tail = key[len(active_prefix) :]
                if tail == suffix:
                    active.append(key)
                elif tail.endswith("." + suffix):
                    slot = tail[: -(len(suffix) + 1)]
                    # Native nested modules can retain the adapter slot. Only
                    # one path component is a slot; no suffix guessing across
                    # arbitrary module depth is authorized.
                    if slot and "." not in slot:
                        active.append(key)
            if len(active) > 1:
                raise AuditError(f"Ambiguous saved-module bias copies: {replacement.weight}")
            candidates: list[tuple[str, str, Tensor | None]] = [
                (key, "saved-full-module", replacement.tensor) for key in active
            ]
            candidates.append(
                (
                    f"{prefix}.original_module.{suffix}",
                    "original-base",
                    base.tensors.get(replacement.weight),
                )
            )
            for name, kind, reference in candidates:
                copy = adapter.tensors.get(name)
                if copy is None:
                    continue
                if name in full_keys:
                    raise AuditError(f"Redundant bias key overlaps a full saved tensor: {name}")
                if reference is None:
                    raise AuditError(f"No original base to validate redundant bias copy: {name}")
                if (
                    copy.dtype not in FLOAT_DTYPES
                    or reference.dtype not in FLOAT_DTYPES
                    or len(copy.shape) != 1
                ):
                    raise AuditError(f"Unsupported redundant bias copy layout: {name}")
                if name in seen_aliases:
                    raise AuditError(f"Redundant bias copy maps to multiple saved tensors: {name}")
                seen_aliases.add(name)
                aliases.append(BiasAlias(copy, reference, kind))
    return TrainedBiases(tuple(expected), tuple(aliases))
