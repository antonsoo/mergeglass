# Saved token-embedding updates

PEFT stores Embedding LoRA as `lora_embedding_A: [rank, vocabulary]` and
`lora_embedding_B: [dimensions, rank]`. The saved weight is
`[vocabulary, dimensions]`: its update is `(B @ A).T * scale`. The original
Mergeglass implementation refused these native keys, including checkpoints
that combine Embedding and Linear adapters.

The extension measures their updates in saved-weight coordinates and validates
optional saved base-layer snapshots. It does not import a model or tokenizer.
The key layout is described in [PEFT's checkpoint guide](https://huggingface.co/docs/peft/en/developer_guides/checkpoint);
the transpose follows its [0.21.0 Embedding implementation](https://github.com/huggingface/peft/blob/v0.21.0/src/peft/tuners/lora/layer.py).

## Native controls, before and after

[`produce.py`](produce.py) trains a small mixed Embedding/Linear model on
**synthetic IDs and random regression targets**. Rank/alpha overrides, rsLoRA
and a negative Linear alpha exercise the saved metadata. The original base is
exactly representable in BF16, separating conversion drift from merge rounding.
It saves native PEFT FP32/F16/BF16 merges with both safe-merge modes, then
separate controls that zero or negate only embedding A. The Linear merge stays
correct in those controls. A square embedding permits a shape-compatible
untransposed-product control.

| FP32 artifact, `square` embedding | Changed / intended nonzero entries | Gain | Relative residual |
| --- | ---: | ---: | ---: |
| Correct native merge | 121 / 121 | 1.00000002 | 6.72e-8 |
| Embedding omitted | 0 / 121 | 0 | 1 |
| Embedding reversed | 121 / 121 | -1.00000002 | 2.00000002 |
| Transpose omitted | 121 / 121 | 0.118199 | 1.328007 |

These controls establish their cause through the producer. The auditor only
reports numerical observations; it does not diagnose an incident or estimate
model quality. The fixture is not a pretrained model or a measured failure rate.

A separately trained one-layer Llama uses native Transformers module names and
six checkpoint shards. Both a factor-only adapter and its **default PEFT save**
are checked. The latter contains an embedding `base_layer.weight` snapshot:
every saved entry must equal the supplied original base rounded to the
snapshot dtype. The copy is not ignored or used as a substitute original.

The unchanged prior source refuses all 15 cases. The new source accepts them
and agrees with an independent dense PyTorch replay of every changed/nonzero
count, unchanged-entry energy, norm, gain, cosine and largest-residual witness.
An additional 80-digit Decimal calculation checks **6,864 embedding
coordinates** without a matrix multiplication. It sums scalar products in token
order and then applies the declared binary64 scale.

Complete receipts: [before](results/before.json), [after](results/verification.json).
Both include the exact native input sizes and SHA-256 hashes, producer/verifier
identities and auditor source hashes. Tiny native files for the example controls
are [bundled unchanged](../../examples/embeddings/README.md); the Llama and
vocabulary-scale checkpoints are regenerated locally, not redistributed.
All 38 native output files also [reproduce byte-for-byte](results/reproduction.json).
Across 11 unchanged legacy inputs, including the two full public SmolLM2 merges,
all measured fields, input fingerprints and unrelated changes have
[exact decoded equality](results/legacy.json) with the previous auditor.

## Vocabulary-scale memory

[`streaming.py`](streaming.py) makes a native PEFT BF16 merge of a synthetic
250,003-token, 128-dimensional, rank-8 embedding: **32,000,384 weights**. Every
base value is one. A and B are explicit binary-fraction controls, not trained
weights; the intended entry is `(token_index % 33 - 16) / 4096`.

An independent integer-residue calculation counts every nonzero, changed and
unchanged entry and calculates all squared norms from exact BF16 spacing. It
agrees at both block sizes, including the slight imbalance from the partial
last group of 33 token indices. The native BF16 merge leaves 23,272,832 of
31,030,656 nonzero intended entries unchanged. That is a property of this
constructed control, not a prediction for a trained vocabulary.

| Block target | First installed run RSS / time | Rebuilt-wheel run RSS / time |
| --- | ---: | ---: |
| 1 MiB | 44.6 MiB / 7.56 s | 38.4 MiB / 8.55 s |
| 16 MiB | 64.8 MiB / 4.86 s | 64.6 MiB / 7.60 s |

Each entry is one fresh installed-auditor process on this 14-vCPU WSL2
Linux host with one BLAS thread, Python 3.10.21, NumPy 1.24.0 and safetensors
0.6.2. No Torch is imported by the audit processes. A full binary64 update would
occupy 244.14 MiB. A full decoded A would occupy 15.26 MiB, exceeding the chosen
**1 MiB fixed-factor limit**; streaming A columns permits the audit while B
needs only 8 KiB. The block target is not a hard RSS limit. Python, NumPy, BLAS,
headers, one complete weight row and the retained B also consume memory.
[First measurements](results/streaming.json) and
[rebuilt-wheel repeat](results/streaming-repeat.json) retain identical original
input and auditor-module hashes. Timing and RSS vary between processes; the
table is not a statistically estimated performance guarantee.

## Reproduce

Use the separate [native-producer environment](../native-peft/README.md) and its
pinned requirements. It needs Torch/PEFT/Transformers; the auditor does not.
With the environment's Python executable substituted for `python`:

```sh
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python studies/embeddings/produce.py /tmp/embedding-native
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python studies/embeddings/verify.py /tmp/embedding-native /tmp/embedding-verification.json
```

Set `PYTHONPATH` to an unchanged prior checkout's `src` and add `--before` to
retain its refusals. Python's hash seed fixes PEFT's set-based target-module
list order in configuration JSON; it does not determine training randomness.

To reproduce memory observations, first install the built wheel in a clean
Python environment. `--python` selects that installed environment. The driver
starts isolated processes and does not use a source `PYTHONPATH`:

```sh
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python studies/embeddings/streaming.py produce /tmp/embedding-large
OPENBLAS_NUM_THREADS=1 python studies/embeddings/streaming.py measure /tmp/embedding-large --python /path/to/installed/bin/python --out /tmp/embedding-memory.json
```

## Boundaries

- Ordinary Embedding LoRA only; Embedding DoRA's different normalization is
  explicitly refused. Linear DoRA remains supported.
- Native full base-layer snapshots require matching original values. Changed,
  nonfinite or shape-incompatible snapshots refuse. Snapshot-only modules and
  arbitrary saved full weights are not interpreted as adapters.
- Both factor key families at one module are ambiguous and refuse, including
  square shapes. Unknown payloads and adapter variants still refuse.
- Vocabulary resizing, missing tied-weight aliases and different base/merged
  inventories remain unsupported. No architecture or tokenizer identity is
  inferred from tensor shapes or module names.
- This work is local source for review. It has no source push, registry release
  or hosted application. The new wheel and source distribution are checked
  independently from the working checkout; see [package evidence](results/installed.json).
