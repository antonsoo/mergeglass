# What a saved-weight audit establishes

Mergeglass reads three local artifacts: the original base, one saved PEFT LoRA
adapter, and the actual merged checkpoint. It never loads a model class or writes
weights. An audit requires matching complete base/merged tensor inventories and
an exact A/B pair for every saved adapter module.

For each ordinary Linear weight with `A: [rank, in]`, `B: [out, rank]`:

```text
delta       = (float64(B) @ float64(A)) * scale
cast_base   = round_to_merged_dtype(base)
observed    = float64(merged) - float64(cast_base)
residual    = observed - delta
cast_drift  = float64(cast_base) - float64(base)
```

`scale = alpha / rank`, or `alpha / sqrt(rank)` for rsLoRA. Rank and alpha
overrides follow PEFT's first-matching-key rule. The supported pattern alphabet
is letters, digits, underscores and dots. Dots retain PEFT's regular-expression
meaning; more complex regex patterns refuse the audit. Recorded A/B shapes must
agree with the resolved rank and exact base tensor shape.

The reference is binary64 numerical arithmetic, not an exact real product. The
producer may multiply factors or accumulate updates in another dtype or order.
Nonzero counts refer to this declared numerical reference. Very small products
can underflow or cancel in binary64. This is not a proof about arbitrary real
numbers.

## Measurements

| JSON field | Definition |
| --- | --- |
| `changed_elements` | Entries where the saved merge differs numerically from `cast_base` |
| `intended_nonzero_elements` | Entries where `delta != 0` |
| `unchanged_nonzero_update_fraction` | Entries with nonzero `delta` and no observed change, divided by nonzero entries |
| `relative_residual` | `norm(residual) / norm(delta)` |
| `observed_to_intended_norm` | `norm(observed) / norm(delta)`; may exceed 1 |
| `norms.base_cast_drift` | Difference caused by base dtype conversion alone |
| `reference_merge_mismatches` | Entries differing from `round_to_merged_dtype(base + delta)` |
| `reference_merge_overflows` | Nonfinite entries in that reference cast |
| `largest_residual` | An original row/column with base, cast base, delta, merged and residual values |

All norms are Frobenius norms. Scaled sums of squares avoid losing a small norm
solely by squaring its entries. Ratios with a zero reference norm are JSON null,
not zero or infinity. Overflow of measured norms/ratios refuses the audit.
Reference-cast overflow is counted; an overflowing example is null in JSON.
Nonfinite saved input values always refuse the whole audit.

| Status | Numeric evidence |
| --- | --- |
| `zero-update` | The binary64 product has no nonzero entries |
| `no-observed-update` | The product is nonzero; no merged entry differs from the cast base |
| `observed-update` | At least one entry differs from the cast base |

`observed-update` is not a correctness verdict. A wrong sign or scale can change
every weight while leaving a large residual. `no-observed-update` cannot
distinguish rounding, an omitted adapter, a wrong input base, or a copied file.
The controlled examples supply the cause externally through their producer;
the auditor itself does not infer it.

`reference_merge_mismatches` is not a pass/fail count. For example, native FP32
matmul followed by FP32 addition can disagree at a rounding boundary with a
binary64 product followed by one cast. Generation quality requires a separate
model/task evaluation, regardless of these measurements.

## Dtype conversion

Supported floating storage types are BF16, F16, F32 and F64. F16/F32 casts use
NumPy's nearest-even conversion. The BF16 reference rounds binary64 directly
to its 7-bit fraction, including subnormals and overflow, with ties to even.
It deliberately avoids a float32 intermediate, which can double-round.
Signed zero differences are not counted as numeric changes.

Unadapted floating tensors are checked against the base cast to their merged
dtype. Unadapted integer/bool tensors require identical dtypes and values;
integer comparisons preserve values above `2**53`. Each changed tensor is
listed with its count and one original flat-index example. Missing, extra,
shape-incompatible or unsupported-dtype tensors refuse the entire audit.

## Coverage and inputs

Saved adapter keys must have exactly this form:

```text
base_model.model.<module>.lora_A.weight
base_model.model.<module>.lora_B.weight
              |
              +--> <module>.weight in base and merged checkpoints
```

There is no suffix search or architecture import. Embedding/conv/Conv1D LoRA,
DoRA, quantized checkpoints, extra trained biases, modules-to-save, aLoRA,
base-mutating initializers and other variant payloads refuse. Unknown adapter
configuration fields also refuse. This version recognizes the saved configuration
of PEFT 0.21.0; other versions can work only within that explicit field contract.

Coverage is the saved A/B pair inventory. If both factors of a target module
are absent from the adapter, the file alone does not establish that they were
supposed to exist. `target_modules` is not used to reconstruct an unseen model
architecture. Likewise, the base model name in metadata is not an identity
certificate. Supply the original weights used by your merge.

A checkpoint is a single safetensors file or a directory containing
`model.safetensors` / `model.safetensors.index.json`. Adapters are directories
with `adapter_config.json` and `adapter_model.safetensors` or its index.
Indexed inventories must match actual shard contents exactly. Relative shard
paths cannot contain `..`; ordinary HF snapshot symlinks into blob storage work.
Pickle checkpoints and remote model IDs are unsupported. Metadata is read as
data; `auto_mapping`, `auto_map`, and model code are never imported.

The Rust safetensors reader validates offsets/shapes/layout. JSON duplicate
members refuse before metadata can be silently overwritten. Headers/config/index
JSON have a 16 MiB limit. Each input file is SHA-256 hashed, and file identity,
size and modification metadata are checked for changes during the audit.
These checks detect ordinary concurrent saves; they are not a transactional
filesystem snapshot or authentication scheme.

## Memory and time

Only A is decoded in full; B, base, merged, delta and residual are row blocks.
Unadapted tensors use flat blocks. `--block-mib` is a working-block target, not
a hard process RSS cap: at least one complete matrix row is needed, and NumPy,
BLAS, Python, headers and A consume additional memory. The default decoded A
limit is 512 MiB per module and can be changed with `--max-adapter-mib`.
There is never a full-layer dense delta allocation in the auditor.

Arithmetic costs approximately `O(out * in * rank)` per adapted matrix, and
every base/merged tensor is read even when unadapted. Hashing makes a further
sequential pass over each artifact. Large checkpoints therefore need enough
local I/O time and disk space for the already saved three inputs.

Reports retain file names, sizes, hashes and full tensor paths, but no automatic
machine-specific absolute paths or complete tensor payloads. Names supplied in
an artifact can themselves contain private information.
