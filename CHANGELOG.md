# Changelog

## 0.1.0 - 2026-10-19

Initial local source version. Compares original, PEFT Linear LoRA and merged
safetensors artifacts with bounded dense blocks. Measures nonzero updates that
remain unchanged, relative weight residuals, base dtype drift and unrelated
tensor edits. Supports BF16, F16, F32, F64, rsLoRA, rank/alpha overrides and
indexed Hugging Face shards. Retains artifact hashes and concrete residual
coordinates in JSON. Includes native PEFT examples and independent numeric replay.
Signed projection gain and cosine similarity distinguish update magnitude from
direction, checked against repeated, reversed, wrong-adapter and rsLoRA controls.

No registry release or hosted service accompanies this version.
