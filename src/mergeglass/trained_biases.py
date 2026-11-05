"""Absolute PEFT trained base biases, kept separate from scaled LoRA B bias."""

from __future__ import annotations

from .adapter import EMBEDDING_PAIR, PAIR, SavedTensor
from .storage import FLOAT_DTYPES, AuditError, Checkpoint


def bias_mode(adapter: Checkpoint) -> str:
    mode = adapter.config.get("bias", "none")
    if mode not in ("none", "all", "lora_only"):
        raise AuditError("bias must be none, all or lora_only")
    return str(mode)


def trained_biases(adapter: Checkpoint, base: Checkpoint) -> tuple[SavedTensor, ...]:
    """Require every declared observable bias; never replace a missing trained value.

    Accepted names are literal saved base keys or the exact base_layer wrapper
    of an observed A/B module. No architecture is imported or inferred. The
    all mode conservatively refuses other 'bias' substrings because PEFT saves
    those too, but they need not represent affine bias vectors.
    """
    mode = bias_mode(adapter)
    if mode == "none":
        return ()
    if adapter.config.get("modules_to_save"):
        raise AuditError("Trained base biases combined with modules_to_save are unsupported")
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
    for name, tensor in sorted(base.tensors.items()):
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
    return tuple(expected)
