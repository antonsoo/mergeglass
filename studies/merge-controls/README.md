# Changed weights can contain the wrong update

Eight native PEFT controls use the same saved, synthetic two-layer Llama and
adapter as [the framework study](../native-peft/README.md). They exercise actual
`PeftModel.from_pretrained` and `merge_and_unload` calls. All base and merged
checkpoints have the same 21 tensors; four matrices contain 3,072 adapted entries.
All 17 unadapted tensors match the cast base in every case.

| Native FP32 control | Changed adapted entries | Projection gain | Relative residual |
| --- | ---: | ---: | ---: |
| Original adapter, one merge | 3,072 / 3,072 | about 1 | 0.000000062 to 0.000000090 |
| Reload adapter onto already merged weights, then merge again | 3,072 / 3,072 | about 2 | about 1 |
| `scale_layer(-1)` before merging | 3,072 / 3,072 | about -1 | about 2 |
| rsLoRA factors loaded with `use_rslora=False` | 3,072 / 3,072 | 0.5 or 0.707107 | 0.5 or 0.292893 |
| Different trained adapter on the same base | 3,072 / 3,072 | -0.016612 to 0.010144 | 1.3297 to 1.3927 |
| Base saved without loading an adapter | 0 / 3,072 | 0 | 1 |

The rows are deliberately constructed controls, not a survey of failures in
deployed models. The producer establishes their causes. The auditor receives
only the original base, the original saved adapter and each saved candidate;
it measures geometric differences without inferring which pipeline caused them.

For a reference update `D` and observed change `O`, projection gain is
`dot(O, D) / dot(D, D)`: the least-squares multiplier of `D` that best fits `O`.
Cosine similarity records agreement in direction, independently of magnitude.
The wrong adapter in this study has cosine from -0.017449 to 0.010424, despite
changing every adapted entry. An exact reversed update has cosine -1; an exact
doubled update has cosine 1. A zero observed update has undefined cosine.
Rounding affects these measurements too; there is no universal pass threshold.

The rsLoRA control illustrates a per-layer distinction: the rank-4 targets have
gain 0.5, while the rank-2 target has gain `1 / sqrt(2)`. A single expected scale
for every layer would miss this pattern.

## `safe_merge` is a finite-value check

Two further artifacts merge FP32 factors into BF16 base weights, once with
`safe_merge=False` and once with `True`. In PEFT 0.21.0's ordinary
[Linear merge implementation](https://github.com/huggingface/peft/blob/v0.21.0/src/peft/tuners/lora/layer.py),
the safe path casts the product to the base dtype before adding it; the unsafe
path adds the FP32 product directly. Both saved checkpoints are finite, and
both change 3,059 of 3,072 adapted entries. However, **755 saved entries differ
between the two artifacts**, spread across all four adapted matrices.

The flag's finite-value check does not promise identical rounding, recovery of
small updates or generation quality. This is why a reference-merge mismatch
count is not a correctness verdict and a saved artifact needs to be measured
separately from the producer options used to create it.

## Reproduce

Use the pinned producer environment from the [native study](../native-peft/README.md).
Generate its transformer artifacts first, then use new output directories:

```sh
PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 /tmp/mergeglass-producer/bin/python studies/merge-controls/produce.py /tmp/mergeglass-transformer /tmp/mergeglass-controls
OPENBLAS_NUM_THREADS=1 /tmp/mergeglass-producer/bin/python studies/merge-controls/verify.py /tmp/mergeglass-controls /tmp/mergeglass-controls-verification.json
```

The [complete receipt](results/verification.json) retains source and artifact
hashes, versions, all per-layer reports, and exact cross-artifact differences.
The verifier checks every file against its producer receipt and uses the
independent dense PyTorch oracle, including PEFT's own rank/alpha resolution.
Projection, cosine, all norms, residual coordinates and exact counts are checked;
1 MiB and 16 MiB audit settings must agree. Lightweight regression cases also
cover cancelling projections, changing block scales, and tiny/large updates
whose raw squares would underflow/overflow binary64.
An independent producer run in another output directory reproduced all 129
artifact files and the complete producer receipt byte for byte with these pins.

The second adapter trains for eight AdamW steps on different seeded synthetic
tokens. No pretrained weights, external model code, or third-party weights are
needed or included. This study does not evaluate language or task quality.
