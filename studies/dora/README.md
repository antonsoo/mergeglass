# Inspect complete DoRA updates, including learned magnitudes

The original auditor refused `use_dora=True`. Supporting it requires a different
reference from ordinary LoRA: the saved magnitude scales the normalized sum of
the base and directional update. Treating the factors alone as the intended
update would miss magnitude-only changes and can accept a plausible but wrong
merged checkpoint.

The [baseline CLI replay](results/before.json), run from source commit
`068b150`, refuses all 12 native artifact/control cases with exit 2 and
`Unsupported adapter option: use_dora`. The extension completes these audits.
The implementation is now available in public source. [All changed public files](results/publication.json)
match the verified local commit. No registry release or deployment accompanies it.

## What the native producer establishes

`produce.py` uses the pinned [PEFT study environment](../native-peft/requirements.txt).
No pretrained weights or remote model code are used. A small rectangular
Linear network trains on seeded random regression inputs for 16 AdamW steps.
It uses DoRA, rsLoRA, a rank override and an alpha override. Its original base
was rounded to BF16 before training so later base conversions are exact.
The saved adapters are FP32. Two explicit post-training controls zero one B
factor and another module's magnitude vector; the remaining module is unchanged
after training.

| FP32 artifact | Matrix | Changed / total | Projection gain | Relative residual |
| --- | --- | ---: | ---: | ---: |
| Native DoRA merge | `direction` | 96 / 96 | 1.000000005 | 7.388e-8 |
| Native load as plain LoRA | `direction` | 96 / 96 | 0.9918016 | 0.2301606 |
| Native DoRA merge | `magnitude_only` | 40 / 40 | 1.000000011 | 8.916e-8 |
| Native load as plain LoRA | `magnitude_only` | 0 / 40 | 0 | 1 |
| Native DoRA merge | `zero_magnitude` | 32 / 32 | 1 | 0 |
| Halved magnitudes | `magnitude_only` | 40 / 40 | 2.428684 | 1.437825 |

Halving magnitudes does not halve the complete update: subtracting the original
base is part of that update. The last row makes this distinction observable.
The incorrect plain-LoRA artifact uses PEFT's own config-override loading and
merge methods; the audit still receives the unchanged DoRA adapter.

Six correct native saves cover FP32/F16/BF16 bases and both `safe_merge` modes.
Three plain-LoRA controls discard magnitudes in those dtypes. Halved magnitudes
and a copied base supply two further controls. A separate two-layer Llama trains
for eight steps on synthetic token IDs and saves four actual HF shards, with
four DoRA-adapted matrices. All 12 cases and 37 per-matrix audits are retained in
[the complete native verification](results/verification.json), including
producer losses, exact input hashes, original coordinates and independent norms.
The [small checked-in examples](../../examples/dora/) contain a subset of the
unchanged native teaching files; no third-party weights are included.

Forward outputs are a separate check in the producer receipt. On its 64x8
synthetic input, the maximum absolute unmerged/merged output difference is
`4.806e-7` for FP32, `0.001953125` for F16 and `0.0078125` for BF16, over 1,536
output positions per case. Both safe modes give the same result here. This is
a numerical regression-network comparison, not a task evaluation.

## Independent arithmetic

`verify.py` loads complete tensors with `safetensors.torch`, resolves rank and
alpha through PEFT, and uses dense binary64 PyTorch products, direct row norms
and `magnitude / norm` scaling. An independent adjacent-bit-pattern grid rounds
the reference directly from binary64. The auditor instead uses NumPy row blocks and
normalizes scaled rows before multiplying by magnitude. Norms, gain, cosine,
residual, unchanged-reference energy and concrete worst coordinates must agree
within `1e-8` relative or `1e-15` absolute tolerance. Counts and reference casts
agree exactly in these cases. Both block settings retain the same measurements.

The framework cast itself needs a separate check. On the pinned CPU runtime,
PyTorch's F64-to-BF16/F16 results match an F32 intermediate and can double-round.
For example, `0x1.0100000000001p+0` is above the BF16 midpoint after 1, yet the
framework cast returns 1. The direct cast returns 1.0078125. The
[exhaustive midpoint replay](results/rounding.json) checks every adjacent finite
pair, both signs and each midpoint's binary64 neighbours:

| Storage dtype | Boundary cases | Auditor disagreements | Independent grid disagreements | PyTorch cast disagreements |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 195,840 | 0 | 0 | 65,280 |
| F16 | 190,464 | 0 | 0 | 63,488 |

These deliberately chosen cases are not an estimated failure rate in models.
The independent DoRA checker uses the validated direct-cast grid. On the public
BF16 reference below, the framework cast differs from that direct cast at 833
of 106,168,320 positions. Reference-cast mismatch counts retain the declared
arithmetic; neither count is a universal merge-correctness verdict.

`decimal_oracle.py` independently evaluates scalar dot products, row norms and
the full weight update at 80-digit Decimal precision. Its 36 constructed cases
cover 4,212 coordinates, signed/zero directional scales, signed/zero magnitudes,
correct rounded weights, halved magnitudes and base copies.
[The retained results](results/decimal.json) compare counts exactly and measured
norms with their numerical tolerance. The ordinary regression examples also
check exact 3-4-5 row geometry, a nonzero directional product that normalizes to
no update, undefined directions, and a reference weight lost by a delta
subtract/add reconstruction.

The full reference contract and base-conversion boundary are in
[docs/arithmetic.md](../../docs/arithmetic.md#linear-dora).
The algorithm follows [PEFT's native DoRA variant](https://github.com/huggingface/peft/blob/v0.21.0/src/peft/tuners/lora/variants.py)
and the [DoRA paper](https://arxiv.org/abs/2402.09353).

## Unchanged public DoRA artifacts

The public replay fetches only pinned safetensors and metadata, verifies every
download against [its size/hash manifest](public-sources.json), and never runs
model-repository code. The eight input files total 827,402,904 bytes. Weights
remain outside the repository. Retrieved 2026-10-19:

| Role | Repository | Pinned revision |
| --- | --- | --- |
| Selected base | `HuggingFaceTB/SmolLM2-135M-Instruct` | [`12fd25f`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac) |
| Saved adapter | `ehyoussef/smollm2-135m-function-dora-adapter` | [`8ce5526`](https://huggingface.co/ehyoussef/smollm2-135m-function-dora-adapter/tree/8ce55268048940efdc497bf9ea6d2b45d9eb2ab1) |
| Published merge | `ehyoussef/smollm2-135m-function-dora` | [`a13a47e`](https://huggingface.co/ehyoussef/smollm2-135m-function-dora/tree/a13a47e9821e45cf23d13911c250e1c2b4d20c03) |

The adapter declares DoRA, rank 16, alpha 32 and PEFT 0.18.0. It names the base
but pins no training revision; its card leaves training and evaluation details
unspecified. The separately published merged checkpoint is therefore evidence
to compare, not an assumed ground-truth pairing. Local native merges use the
unchanged adapter and the selected pinned base, with the same PEFT 0.21.0
environment as the synthetic checks. The `local-as-lora-f32` control deliberately
loads it with `use_dora=False`; it is not a claim about how the author merged.

All four audits check 272 tensors, including 210 adapted matrices and
106,168,320 intended nonzero entries. All 62 unadapted tensors match their cast
bases. The [full independent dense replay](results/public.json) retains all 840
matrix reports, source file hashes and local producer settings. A separate
complete comparison finds the published FP32 checkpoint and the local FP32
merge equal at every named tensor value, with identical dtypes: 272 tensors,
134,515,008 entries. Their file containers differ because the local save is
sharded. This establishes numeric agreement for the supplied artifacts, without
reconstructing their training history.

| Saved checkpoint | Unchanged nonzero entries | Share unchanged | Reference squared norm at unchanged positions | Per-matrix relative residual range |
| --- | ---: | ---: | ---: | ---: |
| Unchanged published FP32 artifact | 553 | 0.000521% | 1.38e-13% | 3.42e-6 to 2.52e-5 |
| Local native DoRA FP32 | 553 | 0.000521% | 1.38e-13% | 3.42e-6 to 2.52e-5 |
| Local native DoRA BF16 | 34,484,820 | 32.48% | 4.58% | 0.0947 to 0.501 |
| Controlled native load as LoRA, FP32 | 589 | 0.000555% | 0.0000242% | 0.114 to 0.220 |

Both near-complete changed-entry counts and gain near 1 can hide a missing
magnitude contribution. The controlled plain-LoRA merge has gains from 0.9508
to 0.9880 but a substantial residual. Conversely, differing from
a single-round reference cast is not a failed merge: the ordinary native FP32
DoRA merge differs at 53,300,601 cast entries while its relative residuals
remain small. Producer arithmetic and the mathematical reference are separate.
None of these weight-space quantities establishes function-calling accuracy,
task quality or the original training base revision.

To reproduce this optional larger study, use the source checkout and the same
separate producer environment. Downloads are explicit study steps; the audit
runtime does not fetch anything:

```sh
python studies/community/fetch.py /tmp/mergeglass-dora-public --sources studies/dora/public-sources.json
OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=src /tmp/mergeglass-producer/bin/python studies/dora/public.py /tmp/mergeglass-dora-public /tmp/mergeglass-dora-public.json
OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=src /tmp/mergeglass-producer/bin/python studies/dora/rounding.py /tmp/mergeglass-dora-rounding.json
```

## Row streaming with an exact closed form

`streaming.py --prepare` makes a separate, explicitly constructed native PEFT
merge with 8,192 output rows, 4,096 input columns and rank 8. The base is all
ones and each A/B entry is 1/32, so every directional product entry is 1/128.
Each direction row has norm 64.5. Repeating magnitudes `0, 32, 65, -65` produce
constant merged rows `0, 1/2, 65/64, -65/64`; their complete updates are
`-1, -1/2, 1/64, -129/64`. Fractions establish the full norm independently.
Varying the magnitude every row also exercises magnitude reads across block
boundaries, including blocks that do not align with that four-row cycle.

Every one of 33,554,432 entries is checked, with zero residual and zero reference
cast mismatches at 1/16/64 MiB workspace targets. A complete binary64 delta would
occupy 256 MiB. Fresh lightweight auditor processes measure memory separately
from the dense native producer. The target is not a hard RSS cap; interpreter,
A, metadata and BLAS also consume memory. Measurements describe this shared
WSL2 host with 14 logical CPUs and one BLAS/OMP thread, not isolated throughput.

The [installed-wheel measurement](results/streaming.json) records:

| Block target | Peak process RSS | Elapsed audit time |
| --- | ---: | ---: |
| 1 MiB | 38.22 MiB | 2.319 s |
| 16 MiB | 61.63 MiB | 1.315 s |
| 64 MiB | 134.69 MiB | 1.104 s |

Each setting uses a fresh Python 3.15 worker. Timing includes validation,
arithmetic and hashes, and excludes interpreter startup and input preparation.
The native producer runs in a separate process so its dense allocations do not
contaminate the auditor's RSS measurement. These are warm-cache local results.
[Clean source and two installed Python environments](../../docs/dora-verification.md)
also verify the wheel's actual module bytes and replay the public artifacts.

## Reproduce

The light auditor needs only its normal environment. The native producer and
independent dense oracle use the separate pinned Python 3.12 environment from
[the original PEFT study](../native-peft/README.md#reproduce). Here
`/tmp/mergeglass-producer/bin/python` refers to that environment:

```sh
PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 /tmp/mergeglass-producer/bin/python studies/dora/produce.py /tmp/mergeglass-dora-native
OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=src /tmp/mergeglass-producer/bin/python studies/dora/verify.py /tmp/mergeglass-dora-native /tmp/mergeglass-dora-verification.json
OPENBLAS_NUM_THREADS=1 uv run python studies/dora/decimal_oracle.py /tmp/mergeglass-dora-decimal /tmp/mergeglass-dora-decimal.json
PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 /tmp/mergeglass-producer/bin/python studies/dora/streaming.py /tmp/mergeglass-dora-streaming --prepare
OPENBLAS_NUM_THREADS=1 uv run python studies/dora/streaming.py /tmp/mergeglass-dora-streaming
```

Use new output directories. The producer is deliberately heavyweight; importing
PyTorch, PEFT or Transformers is never an audit-runtime step. Seeds and one
torch thread make the examples reproducible in the pinned environment, while
the receipts always identify the exact bytes actually checked.

## Limits

The capability covers one saved ordinary Linear DoRA adapter. It retains the
existing rsLoRA and literal-path override rules. It does not add quantized,
embedding, convolutional or Conv1D DoRA, infer unseen module types, reconstruct
a missing magnitude or normalize an undefined zero direction. Signed magnitudes
are values in the saved adapter, not promises about how a trainer constrained
them. Tiny complete updates can cancel in binary64. Original-base identity,
training provenance, inference equivalence and model quality remain separate
claims.
