# Changelog

## Unreleased

Audit the declared PEFT 0.21.0 / Transformers 5.20.1 Qwen3 MoE parameter layout.
Follow nested wrappers and interleaved B columns into actual saved per-expert
gate/up/down matrices, retaining physical keys and logical expert coordinates.
Stream saved rows with B plus one expert A slice under the decoded-factor cap.
Keep zero references, omitted nonzero updates and wrong gains distinct; reject
unknown layouts and full-module overlap. Retain 44 native precision/fault saves,
original parameter-identity witnesses, 37,920 independent Decimal entry checks
and runnable omission/order controls. No registry release.

Audit PEFT `lora_bias: true` on Linear and supported Conv1D targets, using
`base.bias + lora_B.bias * scale` as a separate vector reference. Retain gain,
residual, rounding, unchanged-update counts and original one-dimensional
coordinates without mixing them into matrix totals. Require complete declared
payloads and existing floating base biases; keep `bias: all/lora_only`,
Embedding combinations and DoRA combinations unsupported. Include 32 native
precision/control cases, a bias-only trained adapter, an independent stdlib
reader/Decimal oracle and a runnable omission example. No registry release.

Audit standard GPT-2, OpenAI GPT and ImageGPT Conv1D LoRA with `(B @ A).T`,
resolving each supported module from declared model metadata rather than PEFT's
global orientation flag. Retain saved input/output coordinates and layout
evidence. Mix Linear heads, embeddings and full saved modules; stream A columns
and retain B under its decoded-factor limit. Refuse missing/conflicting layout
evidence, remote custom-model declarations, unsupported paths and Conv1D DoRA.
Include 48 native precision/control cases, a pinned public 48-matrix adapter,
dense and scalar references, explicit base-serialization equality, an installed
16.8-million-weight memory replay and a runnable native wrong-transpose example.

Audit PEFT `modules_to_save` as complete absolute replacement state beside
LoRA/DoRA factors. Check every floating, integer and bool value in blocks,
including nested module biases and scalar buffers above `2**53`. Accept only
explicit saved tensors for added heads or resized full modules; retain null
original-base measurements where comparison is unavailable. Refuse incomplete
or overlapping scopes, factor/module overlap, unexplained payloads and
nonfinite state. Keep ordinary factor reports unchanged. Include native trained
classifiers, tied/resized Llama embeddings, the unchanged public SmolLM2 adapter,
lost-module and wrong-counter controls, independent dense checks and real forwards.

Audit native token-embedding LoRA factors with `(B @ A).T`, retaining token-row
coordinates. Mix Embedding and Linear LoRA, including rsLoRA and rank/alpha
overrides. Stream vocabulary-wide A columns and retain only embedding B;
apply the decoded-factor limit accordingly. Validate saved base-layer snapshots
against the supplied original base instead of ignoring them. Refuse mixed
factor identities, Embedding DoRA, changed snapshots and unrecognized payloads.
Include native wrong-orientation, omitted/reversed embedding and sharded Llama
controls, an independent scalar reference and a vocabulary-scale memory replay.

Add `--dora-norms ADAPTER MERGED` and `audit_dora_norms` for base-independent
row-length observations. Use a separate partial-report schema; leave reference
directions, complete updates and unmapped tensor comparisons unmeasured.
Validate all adapter factors and scan all floating inputs for finite values,
without decoding a whole A matrix. Retain checkpoint input protection.

Add full-row DoRA norm factors, signed-reference direction cosines and explicit
zero-row counts. Preserve logarithmic ratios when ordinary factors cannot fit
float64, with a concrete original-row witness for the largest nonzero log-ratio
error. Keep ordinary LoRA measurements and older report rendering compatible.
Include 3-4-5 triangle, zero/sign, block-boundary and extreme-range controls,
plus independent dense checks of 622,359 native/public rows.

Audit native ordinary Linear DoRA adapters using their learned output-row
magnitudes and the complete normalized weight update. Existing residual, gain,
rounding and unchanged-entry measurements now apply to that full update.
Reject missing/malformed magnitudes and undefined zero directions. Preserve
bounded row processing, including magnitude reads, with no added runtime dependencies.
Include native FP32/F16/BF16 merges, deliberately omitted/incorrect magnitudes,
a sharded Transformers save, an independent dense oracle and scalar Decimal checks.

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
