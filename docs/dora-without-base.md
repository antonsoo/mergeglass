# Inspect DoRA row norms when the training base is unavailable

An adapter card may name a model without identifying the exact original
weights. Substituting another revision cannot establish what was merged.
Ordinary Linear DoRA nevertheless saves one useful base-independent quantity:
each output row's target length is `abs(saved magnitude)`.

```sh
mergeglass --dora-norms ./saved-adapter ./merged-model --summary --json norms.json
```

The inputs are only the saved adapter and merged safetensors. The command
does not find, download or reconstruct a base. For the pinned public DoRA
inputs in [the native study](../studies/dora/README.md), the local BF16 merge
has 155,520 observed rows across 210 adapted matrices:

```text
DoRA full-row norm factor (saved / |m|): 0.9986923..1.0011716
Nonzero magnitudes saved as zero rows: 0; zero magnitudes saved nonzero: 0.
Checkpoint: 272 tensors; 62 unmapped tensor(s) have no base comparison.
```

Those near-unit factors do not describe how well the small update survived.
The complete audit of the same artifact finds 32.48% of nonzero intended
entries unchanged. That comparison requires an original base and a declared
numerical reference. This partial report does not supply those measurements.

## A necessary condition with an explicit limit

Suppose a direction is `[3, 4]` and the saved magnitude is 5. Saving `[-3, -4]`
has the correct norm but the opposite direction. With no original base, the
norm observation is one; the correct reference direction remains unknown.
The [hand-calculated control](../tests/test_dora_norms.py) retains this case.

| Observation | No-base report | Complete audit |
| --- | --- | --- |
| Saved full-row norm / abs(m) | Measured | Measured |
| Zero targets saved nonzero, and conversely | Counted | Counted |
| Reference full-row directions | Unknown | Compared with declared reference |
| Intended update, gain, residual, erased entries | Unknown | Measured |
| Unmapped tensors equal to cast base | Unknown | Compared |
| Actual model quality or original training provenance | Unknown | Unknown |

There is no tolerance that certifies correctness. Storage rounding can shift
the norm of a correctly merged row; a wrong direction can preserve it exactly.
Zero rows receive explicit counts and no fabricated norm ratios. Negative
magnitudes use their absolute value for length and remain counted separately.
The [factored and logarithmic representation](dora-rows.md#range-zeros-and-interpretation)
also works for extreme finite weights.

## API and schema

```python
from mergeglass import audit_dora_norms

report = audit_dora_norms("saved-adapter", "merged-model", block_mib=16)
assert report["schema"] == "mergeglass.dora-norms"
for layer in report["layers"]:
    print(layer["weight"], layer["dora_rows"]["row_norm_factor_range"])
```

`schema_version` is 1 within this separate `mergeglass.dora-norms` schema.
`inputs` fingerprints only the adapter and merged files. `summary` identifies
the observed matrices and rows, the checkpoint's tensor count, and the unmapped
count. Layers retain full tensor identities, shapes, storage dtypes, norms,
zero counts and original-row witnesses. They contain no reference direction,
gain, residual, update count or correct/incorrect status. Existing complete
audit reports retain their schema and measurements.

The usual supported Linear DoRA configuration, exact A/B/magnitude inventory,
shape, checkpoint and metadata checks still apply. Every adapter tensor and
every floating checkpoint tensor is scanned for finite values, including
unmapped floats; malformed evidence refuses the whole observation. A/B are
read in bounded slices and their product is never formed. Magnitudes and
checkpoint rows are also read in blocks. `--block-mib` is a workspace target;
one complete row, headers and interpreter overhead can exceed it.

The API performs no model execution and needs only NumPy and safetensors.
Exit 0 means observations completed; exit 2 means invalid/unsupported evidence
or an I/O failure. `--summary` changes only the terminal view. JSON remains
complete and must be written outside the inputs, including their symlink
targets. `--max-adapter-mib` belongs to complete audits and is not an option
here because no whole A matrix is decoded.

## Native and installed replay

The [artifact receipt](../studies/dora/no-base/verification.json) identifies the
clean source archive and the wheel built from it. All eight runtime Python
files match source `abcbaf691111b506b95579ad97cd5b539de4e684` and both isolated
installations. Python 3.10 with the minimum NumPy/safetensors versions and
Python 3.15 with current pinned versions each pass the source archive's 102
checks and two additional extreme-range controls. Neither auditor installation
contains PyTorch or PEFT.

The [installed replay](../studies/dora/no-base/norm-observations.json.gz) matches
every norm field of the independently checked complete audits: 16 native and
public cases, 877 matrices, 622,359 rows. It passes only adapter and merged
paths to the auditor. The shared row refactor also preserves every complete
row observation exactly; a separate dense PyTorch replay checks that code
extracted from the wheel against the original weights.

On the native 8192 x 4096, rank-8 control, norm factors are exactly one for
6,144 nonzero row pairs; the other 2,048 targets and saved rows are zero.
[Fresh installed processes](../studies/dora/no-base/streaming.json.gz) measured
38.48 / 40.36 / 53.16 MiB peak RSS at 1 / 16 / 64 MiB block targets. These are
local process measurements during concurrent inference, not a hard memory
cap or throughput benchmark. No base file is an input to those observations.

With the native and public inputs from the earlier study, reproduce the
installed norm comparison using a Python environment with Mergeglass installed:

```sh
OPENBLAS_NUM_THREADS=1 python studies/dora/norms_without_base.py \
  /tmp/mergeglass-dora/native /tmp/mergeglass-dora/public /tmp/norm-observations.json
OPENBLAS_NUM_THREADS=1 python studies/dora/streaming.py \
  /tmp/mergeglass-dora/streaming --dora-norms --out /tmp/norm-memory.json
```
