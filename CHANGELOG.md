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
Refuses known GPT-2/OpenAI GPT/ImageGPT Conv1D paths when model metadata exposes
them, including mixed targets hidden by a global `fan_in_fan_out=False` flag.
Measures the reference squared norm at unchanged positions per layer and across
the checkpoint, so equal-count and magnitude-weighted fractions stay distinct.
Accepts native finite signed global/per-layer alpha coefficients, with positive
integer ranks and independently checked rsLoRA exports in three storage dtypes.
Adds an aggregate CLI view with complete JSON coverage and a real public-adapter
terminal capture. The installed public BF16 audit measures 61.24 MiB peak RSS.

No registry release or hosted service accompanies this version.
