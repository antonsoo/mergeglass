# Synthetic nested full-head omission

These are original native PEFT 0.21.2 artifacts using `bias="all"` together
with `modules_to_save=["head"]`, rsLoRA and rank/alpha overrides.

`reloaded-f32-safe-false` is the native saved merge. `omitted-head-f32` is an
authored control that restores the complete head to its original state while
preserving both LoRA matrices and all ten other trained bias values. Its 27
full-state target mismatches include an integer buffer above `2**53`.

`conflicting-modules_to_save-adapter` changes only a redundant nested bias
copy; its canonical full head remains present. Auditing refuses the conflicting
artifact instead of silently ignoring that payload.

[Native study, run commands, independent byte checks and limitations](../../studies/trained-bias-heads/README.md).
Models, training inputs and targets are synthetic. Original artifacts use the
repository's MIT license, Copyright (c) 2026 Anthony Mercer.
