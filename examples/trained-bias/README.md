# Synthetic trained-base-bias merge controls

These tiny original models were trained and exported through PEFT 0.21.2.
`base` is the pre-training model; `adapter` uses `bias="all"` and two LoRA
matrices. `reloaded-f32-safe-false` is the native saved merge.

`omitted-bias-f32` restores all three trained bias vectors to their original
values. `scaled-bias-f32` multiplies their absolute trained values by three.
Both authored faults preserve every other saved tensor byte. They each produce
twelve bias target mismatches while retaining both correct LoRA matrices.

`historical-base`, `incomplete-historical-adapter` and
`historical-in-memory-f32` are a separate PEFT 0.18.1 `lora_only` run. The
unchanged adapter export omitted both trained biases. Auditing it refuses
because those values are unavailable, even though the in-memory merge had them.
Do not mix these historical inputs with the modern base.

Run commands, independent checks, full native archives and producer details:
[trained-bias study](../../studies/trained-bias/README.md).

All models, inputs, training targets and faults are synthetic; no pretrained
weights or quality claims. Original artifacts use the repository's MIT license,
Copyright (c) 2026 Anthony Mercer.
