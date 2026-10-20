# What a saved-weight audit establishes

Mergeglass reads three local artifacts: the original base, one saved PEFT LoRA/DoRA
adapter, and the actual merged checkpoint. It never loads a model class or writes
weights. An audit requires every original base tensor in the merged inventory,
an exact A/B pair for each saved factor module and complete explicit state for
each observed `modules_to_save` scope. Additional merged tensors are accepted
only as exact full-module targets saved in the adapter.

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

Ranks must be positive integers. Alpha coefficients may be finite positive,
zero or negative numbers, globally and in supported per-layer overrides. Their
sign is part of the intended reference. The [native signed-scale cases](../studies/signed-scale/README.md)
retain valid PEFT exports that the original nonnegative-only check refused.

The reference is binary64 numerical arithmetic, not an exact real product. The
producer may multiply factors or accumulate updates in another dtype or order.
Nonzero counts refer to this declared numerical reference. Very small products
can underflow or cancel in binary64. This is not a proof about arbitrary real
numbers.

## Token-embedding LoRA

Embedding factors have native parameter keys without a `.weight` suffix:

```text
base_model.model.<module>.lora_embedding_A   [rank, vocabulary]
base_model.model.<module>.lora_embedding_B   [dimensions, rank]
                 |
                 +--> <module>.weight      [vocabulary, dimensions]
```

Their update is `(float64(B) @ float64(A)).T * scale`. Weight rows are token
indices; columns are embedding dimensions. The report adds
`weight_layout: "embedding"` and `reference_product: "(B @ A).T"` to these
layers. Every count, gain, norm and residual uses that orientation. A largest
residual's coordinates refer to the saved embedding weight, never the
untransposed product. Ordinary Linear report fields remain unchanged.

Linear and Embedding LoRA can coexist, including supported rank/alpha overrides
and rsLoRA. A module with both key families refuses even when square shapes
would otherwise fit. Embedding DoRA is outside this contract; its normalization
must not be interpreted as Linear output-row normalization.

PEFT's default save can also include
`base_model.model.<module>.base_layer.weight` beside the factors. Every such
snapshot must match the supplied original weight, rounded to the snapshot's
floating dtype, at every coordinate. Shape, nonfinite or value disagreements
refuse the entire audit. Reports retain each snapshot's key and dtype, and all
input files are hashed as usual. The snapshot does not replace the supplied
original base. A standalone snapshot without factors and arbitrary undeclared
full weights remain unsupported. Base-independent DoRA observations
refuse snapshots because their relationship to an original base is unmeasured.

Factor-adapted embedding shapes must still match the original base. Explicit
full saved modules have the separate replacement contract below. Neither path
reconstructs tied-weight aliases or infers architecture from token IDs.
Saved coordinates do not identify human-readable tokens
without a separately verified tokenizer. [Native merge controls, a sharded
Llama save and memory measurements](../studies/embeddings/README.md).

## Full saved modules

PEFT's `modules_to_save` contains literal module suffixes. Its native save strips
the training wrapper and adapter name, leaving exact tensor keys:

```text
adapter: base_model.model.<module>.<state path>
                         |
merged:                  +--> <module>.<state path>
```

These tensors are **absolute trained state**, including parameters and buffers.
No A/B product or additive update is inferred for them. For floating state:

```text
target           = round_to_merged_dtype(saved_adapter_state)
target_mismatches = count(merged != target)
norms.error      = norm(merged - target)
relative_error   = norms.error / norm(target), or null when target is zero
```

`norms.target_cast_drift` measures `target - saved_adapter_state` separately.
Equality is numeric; signed zeros do not count as mismatches. Integer/bool
state requires the target and merged dtype to match exactly, without a float
conversion. Its mismatch counts and witnesses preserve all bits; error norms
and relative error are null.

The resolver follows PEFT's literal `endswith` module-selection rule against
observable tensor prefixes from the base and merged inventories. It then
requires an exact saved adapter tensor for **every** observed tensor under each
resolved scope. There is no fuzzy tensor-name mapping. Extra payloads outside
those scopes refuse, as do overlapping scopes or a full scope overlapping a
factor module. The literal-path alphabet excludes wildcards and regex syntax.
Selectors with no observed prefix are retained in
`unmatched_saved_module_selectors`; their architecture existence is unknown.
Native sequence classifiers can name both `classifier` and `score` while only
one has saved state. At least one full saved tensor must resolve when selectors
are declared.

Full modules may explicitly add a merged tensor absent from the original file,
or replace a tensor with a different shape. Their saved targets are still
checked at every coordinate. `base_comparison_unavailable` records `absent`,
`shape-changed`, `dtype-category-changed` or `integer-dtype-changed` as applicable.
Original dtype/shape metadata is retained, while original-change counts are
null. No tied alias, original head value or resized token row is invented.
An incompatible original floating tensor is still scanned for finite values.
Every original tensor must remain present in the merged artifact.

Where base shapes and dtype categories permit comparison, the base is rounded
to the merged dtype. `changed_from_base_elements`, `expected_change_elements`
and `unchanged_expected_change_elements` count observed changes, required target
changes and required changes still equal to that base. This is separate from
the LoRA unchanged-update fraction. The `saved_modules` list and its summary
counts are separate from factor `layers` and unadapted changes. Existing reports
without full modules retain their original fields and measurements.

The complete audit checks these targets. Base-independent DoRA norm mode
explicitly refuses full saved modules instead of silently omitting them.
[Native/public full-state study](../studies/saved-modules/README.md).

## Linear DoRA

For `use_dora: true`, every saved A/B pair also needs the native
`base_model.model.<module>.lora_magnitude_vector` tensor of shape `[out]`.
Its floating dtype is recorded alongside A and B. Missing, extra, malformed or
nonfinite magnitude evidence refuses; it is never substituted with a base norm.
Plain LoRA rejects magnitude payloads rather than silently ignoring them.

DoRA separates the direction and magnitude of each output row. The reference is:

```text
direction[row]        = base_float64[row] + scale * (B_float64 @ A_float64)[row]
reference_weight[row] = magnitude_float64[row] * direction[row] / norm(direction[row])
delta                = reference_weight - base_float64
reference_merge      = round_to_merged_dtype(reference_weight)
```

The row convention follows the `[out, in]` storage of ordinary Linear weights;
it is not normalization over output columns. The implementation first divides
each row by its largest absolute entry, normalizes the scaled row, then
multiplies by its magnitude. This avoids overflow from raw squares or
`magnitude / norm`, and underflow for tiny but nonzero directions. All operations
remain binary64; arithmetic order can differ from a direct division.
Finite signed or zero magnitudes retain their saved value. An all-zero direction
refuses with its original row number, including when the magnitude is zero.

Every existing count, norm, gain and residual uses this **complete** `delta`.
The low-rank product can be zero while learned magnitudes change the weights;
a nonzero product can also normalize back to an unchanged weight. Consequently,
the DoRA update is generally not low rank. Reports identify `adapter_variant`
as `lora` or `dora`; the latter adds `adapter_dtypes.magnitude`. These are
additive schema-version-1 fields.

Normalization uses the **supplied original base**, before its storage conversion.
The observed update still subtracts the dtype-cast base and base-cast drift stays
separate. A producer that normalizes a differently rounded or already modified
base can disagree with this reference. Mergeglass does not infer its arithmetic
sequence from the saved result. It casts `reference_weight` directly for the
reference-merge comparison, avoiding a subtract/add reconstruction of that weight.

The scope is one ordinary Linear DoRA adapter, optionally with rsLoRA and the
supported rank/alpha overrides. Embedding/conv/Conv1D DoRA, quantized bases,
multiple sequential adapters and other variants remain outside this contract.
The formula follows [PEFT's DoRA implementation](https://github.com/huggingface/peft/blob/v0.21.0/src/peft/tuners/lora/variants.py)
and [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353).
[Native artifacts and independent calculations](../studies/dora/README.md)
establish the implemented numerical scope, not task quality.

## Measurements

| JSON field | Definition |
| --- | --- |
| `changed_elements` | Entries where the saved merge differs numerically from `cast_base` |
| `intended_nonzero_elements` | Entries where `delta != 0` |
| `unchanged_nonzero_update_fraction` | Entries with nonzero `delta` and no observed change, divided by nonzero entries |
| `norms.unchanged_reference` | Norm of `delta` restricted to entries where merged equals cast base |
| `unchanged_reference_energy_fraction` | Squared reference norm at unchanged entries, divided by the full squared reference norm |
| `relative_residual` | `norm(residual) / norm(delta)` |
| `observed_to_intended_norm` | `norm(observed) / norm(delta)`; may exceed 1 |
| `projection_gain` | `dot(observed, delta) / dot(delta, delta)`; signed least-squares scale |
| `cosine_similarity` | `dot(observed, delta) / (norm(observed) * norm(delta))`; direction agreement |
| `norms.base_cast_drift` | Difference caused by base dtype conversion alone |
| `reference_merge_mismatches` | Entries differing from the declared reference merge (direct `reference_weight` cast for DoRA) |
| `reference_merge_overflows` | Nonfinite entries in that reference cast |
| `largest_residual` | An original row/column with base, cast base, delta, merged and residual values |

All norms are Frobenius norms. Scaled sums of squares avoid losing a small norm
solely by squaring its entries. Ratios with a zero reference norm are JSON null,
not zero or infinity. Overflow of measured norms/ratios refuses the audit.
Reference-cast overflow is counted; an overflowing example is null in JSON.
Nonfinite saved input values always refuse the whole audit.

Gain is 1 for an exact intended update, 2 for an exact doubled update, -1 for a
reversed update, and 0 for no observed update or an orthogonal change. Cosine
separates a change in scale from a change in direction: it is 1 for a positive
multiple and -1 for a negative multiple. A zero reference makes both undefined;
a zero observed change with a nonzero reference has gain 0 and undefined cosine.
The inner product scales each operand separately before multiplication, avoiding
raw squared weights that could overflow or underflow. These are geometric
measurements, not diagnoses: rounding and multiple simultaneous errors can
also affect them. See the [native controls](../studies/merge-controls/README.md).

Unchanged-entry counts weight every nonzero entry equally. The reference energy
fraction weights them by squared update magnitude. For the same mask `M` where
`merged == cast_base`, this is `sum(delta[M]**2) / sum(delta**2)`. The checkpoint
summary combines squared norms across all adapted matrices; it is not an average
of per-layer percentages. The implementation uses separately scaled sums rather
than raw squares, including when a global unnormalized norm would overflow.
A zero reference produces null. Very small numerical fractions can underflow
binary64, while the separate raw norms and nonzero counts remain available.

This fraction describes where the reference update's squared norm resides.
It does not measure retained observed energy: changed entries can undershoot,
overshoot or reverse direction. It also does not weight entries by a model's
functional sensitivity or measure generation quality.

| Status | Numeric evidence |
| --- | --- |
| `zero-update` | The binary64 product is zero and no merged entries change |
| `change-with-zero-reference` | The binary64 product is zero but merged entries change |
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
listed with its count and one original flat-index example. Missing original
tensors, unexplained additions, unadapted shape changes and unsupported dtypes
refuse the entire audit. Explicit full-module targets follow the contract above.

## Coverage and inputs

Saved factor keys must have exactly this form (DoRA also requires the magnitude key above):

```text
base_model.model.<module>.lora_A.weight
base_model.model.<module>.lora_B.weight
              |
              +--> <module>.weight in base and merged checkpoints
```

Factor weights are never suffix-searched and no architecture is imported.
Conv/Conv1D LoRA, quantized checkpoints, extra trained biases outside full saved
modules, aLoRA,
base-mutating initializers and other variant payloads refuse. Unknown adapter
configuration fields also refuse. This version recognizes the saved configuration
of PEFT 0.21.0; other versions can work only within that explicit field contract.

A saved `fan_in_fan_out=False` flag does not establish every module's layout.
PEFT can overwrite this global flag while loading mixed Conv1D and Linear
targets; square Conv1D matrices still pass a Linear shape check. When base
`model_type` or adapter `auto_mapping.parent_library` identifies GPT-2,
OpenAI GPT or ImageGPT, their known attention/MLP Conv1D paths explicitly refuse.
Ordinary Linear heads on the same architectures remain supported. This is a
guard against contradictory metadata, not a universal module-type detector.
Missing metadata, custom models and other architectures cannot be certified
from tensor shapes. The supported plain-Linear contract remains a prerequisite.
See the [six native orientation cases](../studies/orientation/README.md).

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

Linear adapters retain A and read B rows. Embedding adapters retain B and read
the corresponding A columns in bounded slices, one per rank component. The
vocabulary-wide A is never decoded in full. DoRA magnitudes, optional base
snapshots, base, merged, delta and residual are read in blocks.
Unadapted tensors use flat blocks. `--block-mib` is a working-block target, not
a hard process RSS cap: at least one complete matrix row is needed, and NumPy,
BLAS, Python, headers and the retained factor consume additional memory. The
default decoded fixed-factor limit is 512 MiB per module (Linear A, Embedding B)
and can be changed with `--max-adapter-mib`.
There is never a full-layer dense delta allocation in the auditor.

Arithmetic costs approximately `O(out * in * rank)` per adapted matrix, and
every base/merged tensor is read even when unadapted. Hashing makes a further
sequential pass over each artifact. Large checkpoints therefore need enough
local I/O time and disk space for the already saved three inputs.

Reports retain file names, sizes, hashes and full tensor paths, but no automatic
machine-specific absolute paths or complete tensor payloads. Names supplied in
an artifact can themselves contain private information.
