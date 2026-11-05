# Synthetic native bias precision control

The base is BF16-exact and stored in FP32. After synthetic training, each
original bias is set to the adjacent FP32 value in its learned direction.
This is an explicit post-training control; the trained LoRA factors remain
unchanged. PEFT 0.21.2 produces the actual saved merges.

`merged-f32-safe-false` retains all twelve saved bias changes.
`merged-bf16-safe-false` leaves all twelve at the cast original base. Both
match the absolute adapter targets after the appropriate dtype cast, so both
have zero bias target mismatches. Inspect the separate saved-change and
reference-cast-erasure counts and their exact coordinate witnesses.

[Run commands, complete precision matrix, before/after audit and independent oracle](../../studies/trained-bias/precision.md).

All models, inputs, targets and controls are synthetic. Original artifacts use
the repository's MIT license, Copyright (c) 2026 Anthony Mercer.
