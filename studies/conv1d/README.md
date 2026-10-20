# Native GPT Conv1D merges, with saved-layout evidence

The source auditor previously refused standard GPT Conv1D LoRA. That refusal
protected against an earlier, reproduced error: a mixed Linear/Conv1D adapter
can save `fan_in_fan_out=False` even though individual Conv1D matrices require
transposition. The [historical study](../orientation/README.md) retains that
failure. This extension resolves supported standard module layouts from declared
model metadata, records the evidence and audits the actual saved weights.
It never chooses an orientation by a better residual.

The change is local source work. No registry release or deployment accompanies
it. [Input contract and limits](../../docs/conv1d.md).

## Native synthetic producer

[produce.py](produce.py) constructs six native Transformers profiles:

| Profile | Adapted state |
| --- | --- |
| GPT-2 mixed | Four Conv1D projections, Linear head, token embedding |
| OpenAI GPT mixed | Four Conv1D projections, Linear head |
| ImageGPT mixed | Four Conv1D projections, Linear head, token embedding |
| GPT-2 cross-attention | Seven Conv1D projections, Linear head, token embedding |
| GPT-2 backbone | Four Conv1D projections with bare `h.N.*` names |
| GPT-2 saved head | Four Conv1D projections, embedding, absolute full trained head |

Models, token IDs and training are synthetic. Each profile trains for five AdamW
steps, with a fixed seed and one Torch thread. Base parameters are rounded to
BF16 and stored in FP32 before training to keep later base conversions exact.
The adapter uses rsLoRA, a rank override and a signed alpha override. It retains
native per-module type, rank, scale and orientation before unloading.

Each profile has six native merges: FP32, FP16 and BF16, each with
`safe_merge=False` and `True`. Two explicit faulty conversions retain the other
updates but transpose or omit the square attention output projection. Native
sharded saves exercise the ordinary checkpoint reader. There are **48 saved
artifact cases: 36 native merges and 12 authored controls**.

| Small GPT-2 square projection | Changed / 256 | Relative residual |
| --- | ---: | ---: |
| Native FP32 | 256 | 4.41e-8 |
| Wrong orientation | 256 | 1.4809 |
| Omitted projection | 0 | 1 |

The exact native FP32 number is retained in the complete report; the table
rounds it. All native FP32 layers across the six profiles have relative
residual below 9.46e-8. These are numerical observations, not quality scores,
deployment incident rates or universal thresholds.

The [before report](results/before.json), made with installed revision
`af4179b9fe36051b7a92219524bee0903b042213`, records 48 refusals. The
[after report](results/after.json) checks all 48 cases. Both retain every input
file hash, the producer receipt and the installed auditor's module hashes.
The [original six cases](results/original-six.json) also remain usable under
the new support, including the Linear-only controls.

## Independent arithmetic

[verify.py](verify.py) uses complete dense Torch matrices and the per-module
orientation captured by native PEFT, independently of the auditor's metadata
resolver. It checks every changed/nonzero/unchanged count, all norms, projection
gain, cosine, cast drift, reference mismatches and largest-residual coordinates.
Every unadapted tensor must equal its cast base. Explicit saved heads must equal
their cast adapter targets. It additionally computes all **18,368 transposed
reference entries** with separate 80-digit scalar Decimal sums.

The rounding oracle searches a table of adjacent representable FP16/BF16 values
and applies even-bit tie breaking. It reuses neither the auditor's exponent
rounding implementation nor Torch's half conversion. The latter can convert
through FP32: the small ImageGPT FP16 head contains one such direct-rounding
disagreement in each safe-merge mode. Native saved values and numerical reference
mismatches remain distinct observations. Floating metric comparisons use
relative tolerance `2e-10` and absolute tolerance `1e-13`; counts agree exactly.

## Unchanged public adapter

The larger replay uses these public artifacts, pinned before download:

| Artifact | Repository | Revision |
| --- | --- | --- |
| Original GPT-2 | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e) | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| Adapter | [palsp/gpt2-lora](https://huggingface.co/palsp/gpt2-lora/tree/4235d59235a7269b34119195e63e1424d2a530e1) | `4235d59235a7269b34119195e63e1424d2a530e1` |

Both model cards declare MIT. No third-party weights are redistributed here.
The adapter's training dataset and original base revision are unknown. Native
merges are produced locally; no claim is made about a separately published
merged artifact or its generation quality.

The historical base file has 160 tensors with bare GPT2Model names; native
LM-head saves use `transformer.*` names and omit twelve nonpersistent causal-mask
buffers. [public_produce.py](public_produce.py) first saves the unadapted base
with that writer. It requires every value in all 148 retained tensors to equal
its downloaded original and records the exact name map plus omitted buffers.
The raw download remains hashed separately. The native original-base save is
the audit input; **no fuzzy renaming or buffer omission happens in the auditor**.
The downloaded adapter configuration and safetensors remain unchanged.

All 48 adapted matrices contain nonzero reference updates, covering **84,934,656
entries**. All 100 unadapted tensors match their cast bases in every merge:

| Native saved merge | Unchanged nonzero reference entries | Share unchanged | Reference squared norm at unchanged entries |
| --- | ---: | ---: | ---: |
| FP32 | 97 | 0.0001142% | 1.41e-16% |
| FP16 | 804,849 | 0.9476% | 0.00008732% |
| BF16 | 6,327,161 | 7.4494% | 0.03703% |

An explicit wrong-orientation control transposes only the twelve square
attention output updates. Those projections' relative residuals are
1.4117-1.4156; correct FP32 layers range from 4.33e-7 to 1.62e-6. Other adapted
matrices and all unadapted tensors retain the native FP32 result.

[public_verify.py](public_verify.py) checks all four complete dense saved-weight
comparisons and three 80-digit scalar coordinates per matrix per case,
including the reported worst residual. It uses the same independent adjacent-
value rounding check. Across the complete public matrices, Torch conversion
would disagree with direct reference rounding at 5,079 FP16 and 667 BF16 entries.
Those differences are retained rather than treated as auditor failures.
[Before refusals](results/public-before.json), [after measurements](results/public-after.json).

## Memory

[streaming_produce.py](streaming_produce.py) constructs a native GPT-2 with a
32,768-by-512 adapted MLP output projection, rank eight, and explicitly seeded
nonzero B. This is synthetic initialization with no training. It saves the
original, adapter and native FP32 merge, then computes an independent dense
Torch reference in the producer process.

Its 16,777,216 weights require 128 MiB for one full binary64 delta. Decoded A is
2 MiB, decoded B is 32 KiB. The installed auditor succeeds with a **1 MiB factor
limit** because it retains B and streams A columns. All counts and norms agree
with the dense reference; 397 nonzero reference entries are unchanged.

[check_install.py](check_install.py) launches an isolated installed CLI in fresh
processes, outside the checkout. Linux `/usr/bin/time %M` measures peak resident
KiB; `%e` measures elapsed seconds. The producer's dense allocation is in a
different process. On this 14-logical-CPU WSL2 Linux host, one BLAS thread:

| Installed audit | Block target | Peak RSS | Elapsed |
| --- | ---: | ---: | ---: |
| Native 16,777,216-weight case | 1 MiB | 36.8 MiB | 0.92 s |
| Same case | 16 MiB | 56.0 MiB | 0.61 s |
| Complete public BF16 checkpoint | 1 MiB | 36.8 MiB | 5.91 s |

These are single local measurements, not hard caps or portable benchmarks.
The [installed receipt](results/installed.json) retains machine/runtime identity,
all three runnable example reports, the dense streaming reference, both memory
runs and the complete public BF16 report. All eight installed runtime modules
match source byte hashes; importing the auditor loads no Torch. The complete
installed public report equals the independently checked source report.

## Reproduce

Use the separate Python 3.12 native environment in the
[native-producer guide](../native-peft/README.md), with its exact requirements.
The audit runtime itself still needs only NumPy and safetensors. Output
directories below must be new.

```sh
PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 /tmp/mergeglass-producer/bin/python studies/conv1d/produce.py /tmp/conv1d-native
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 /tmp/mergeglass-producer/bin/python studies/conv1d/verify.py /tmp/conv1d-native /tmp/conv1d-native.json
```

The public producer downloads only the pinned files above; it does not execute
remote model code. It saves native local merges and an authored control:

```sh
HF_HUB_DISABLE_PROGRESS_BARS=1 /tmp/mergeglass-producer/bin/python studies/conv1d/public_produce.py /tmp/conv1d-public
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 /tmp/mergeglass-producer/bin/python studies/conv1d/public_verify.py /tmp/conv1d-public /tmp/conv1d-public.json
```

For the before reports, install revision
`af4179b9fe36051b7a92219524bee0903b042213` in a separate environment and run the
current verifiers with `--before`, without `PYTHONPATH=src`. They require complete
refusals, not unavailable measurements filled in as successes.

To reproduce installed/memory checks, build and install a wheel in a fresh venv,
then pass that venv's interpreter to the checker:

```sh
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 /tmp/mergeglass-producer/bin/python studies/conv1d/streaming_produce.py /tmp/conv1d-streaming
uv build
uv venv /tmp/conv1d-installed --python 3.12
uv pip install --python /tmp/conv1d-installed/bin/python dist/mergeglass-0.1.0-py3-none-any.whl
python studies/conv1d/check_install.py /tmp/conv1d-installed/bin/python /tmp/conv1d-streaming /tmp/conv1d-public /tmp/conv1d-installed.json
```

The native layout rules are checked against pinned upstream
[PEFT dispatch and merge implementation](https://github.com/huggingface/peft/blob/v0.21.0/src/peft/tuners/lora/layer.py),
[GPT-2](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/gpt2/modeling_gpt2.py),
[OpenAI GPT](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/openai/modeling_openai.py)
and [ImageGPT](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/imagegpt/modeling_imagegpt.py).
Tensor shapes and metadata remain declarations about a supported layout, not
proof of an unseen architecture, training provenance or inference equivalence.
