# Complete trained modules beside low-rank factors

PEFT's [`modules_to_save`](https://huggingface.co/docs/peft/en/package_reference/lora)
trains and saves complete modules such as classifier heads. Its
[native save implementation](https://github.com/huggingface/peft/blob/v0.21.0/src/peft/utils/save_and_load.py)
removes the training wrapper and adapter name from those tensor keys. These are
absolute replacement values, not another A/B product. A converter can merge all
low-rank updates correctly while discarding this state.

The previous source at `4a99422` refuses these native artifacts. The implementation
at `2d0ddb0` checks complete saved modules alongside LoRA/DoRA factors without
loading an architecture. [Field contract](../../docs/arithmetic.md#full-saved-modules).

## An unchanged public adapter

[Pinned sources](sources.json) identify the original
[Kibalama/smollm2-finetune-python-2 adapter](https://huggingface.co/Kibalama/smollm2-finetune-python-2/tree/8612b1f0feaeb3d34005848813d13387a13a901f)
and the supplied SmolLM2-135M base. The adapter has 210 rank-16 factor pairs,
plus complete `[49152, 576]` token-embedding and language-head weights:
56,623,104 full-module values. The adapter names the base repository but does
not pin its training revision. This study establishes the exact supplied inputs,
not the identity of its original training checkpoint or training data.

The native producer loads only local files with built-in Transformers code,
uses FP32 adapters and calls `merge_and_unload(safe_merge=False)`. PEFT explicitly
sets `tie_word_embeddings=False` after merging the independently trained token
table and head. Both appear in the sharded merged safetensors. The original base
omits the tied `lm_head.weight` alias: its absolute saved target is fully checked,
while a comparison with an original head stays unavailable. No alias is invented.

| Saved artifact | Full-module target mismatches | LoRA matrices checked |
| --- | ---: | ---: |
| Native FP32 merge | 0 / 56,623,104 | 210 |
| Native FP16 merge | 0 / 56,623,104 | 210 |
| Native BF16 merge | 0 / 56,623,104 | 210 |
| FP32 control restoring both original full modules | 39,791,249 / 56,623,104 | 210 |

In the control, the head has 28,311,551 target mismatches and the token table has
11,479,698. All 106,168,320 LoRA weight measurements remain identical to the
native FP32 merge. This is an explicitly authored faulty conversion, **not an
observed upstream failure**. Both full modules use the original base values in
that control; the original head is obtained from the native model at runtime,
not inferred by the auditor from a missing file.

Twelve deterministic token-ID probe positions provide a separate observation:
the control changes 7 greedy next-token choices relative to the native FP32
merge, with maximum logit difference 13.5939064. Correct native merges change
0, 0 and 2 choices relative to their unmerged FP32/F16/BF16 adapter models.
These are numerical probes, not decoded language tasks, quality scores or a
generation benchmark. Exact IDs, native dtypes, warnings, output fingerprints
and forward observations are retained in [public evidence](results/public.json).
[Before-source refusals](results/public-before.json) retain the original behavior.
Third-party weights are not redistributed.

The actual saved checkpoints are then reloaded through built-in Transformers
code and compared with fresh native PEFT merges. All four cases match in all
273 state tensors (162,826,560 elements), retain untied input/output weights,
and produce identical logits at all 589,824 compared coordinates with the same
SDPA attention path. This separately checks serialization and reload behavior;
it does not infer it from weight-space norms. [Reload receipts](results/reloaded.json)
retain exact producer, script and logit hashes.

## Native classifiers, counters and vocabulary changes

`produce.py` trains a small nested classifier from scratch on labelled synthetic
data. PEFT saves the full head's two matrices, two biases and a scalar integer
buffer, beside its rank-2 projection factors. The saved counter is `2**60 + 3`.

| Case | Full-state mismatches | Original-base comparison |
| --- | ---: | --- |
| Native FP32/F16/BF16 classifier merges | 0 | Available for every full tensor |
| Original classifier restored, correct LoRA retained | 64 | All 64 required changes absent |
| Counter reduced by one, correct float weights retained | 1 | Exact integer comparison |
| Native sharded BERT sequence classifier | 0 | `classifier` observed; `score` selector unmatched |
| Native Llama with a tied 37-token base | 0 | Original head absent; token table comparable |
| Native Llama resized from 37 to 41 tokens | 0 | Original head absent; old token-table shape incompatible |
| Native FP32/F16/BF16 Linear DoRA with a full classifier | 0 | Independent full-state and DoRA references |
| Derived adapter with one trained bias deleted | Refusal | Incomplete saved module |

On 96 synthetic classifier probes, restoring the original head changes 47
predicted classes; maximum logit difference is 5.359664. This makes the lost
state's effect observable without claiming task quality. The bundled
[small examples](../../examples/saved-modules/README.md) retain the original
FP32 files, including the one-count error that float64 would hide.
[Native evidence](results/native.json) and [before-source refusals](results/before.json).

## Independent numerical verification

`verify.py` reads the original safetensors through Torch independently from the
auditor's storage reader. It checks every full tensor and every adapted matrix
with dense operations: exact mismatch/change counts, target/cast/error norms,
complete factor updates, gain, residuals and original-coordinate witnesses.
It also checks all unadapted tensors and complete observed inventories.
The auditor uses bounded NumPy blocks; the independent verifier intentionally
allocates complete tensors and runs only in the optional producer environment.

A naive `torch.float64 -> torch.float16` reference disagreed on rounding counts.
For example, `-0x1.a2a000fc7dd0ep-4` rounds directly to `-0.10223388671875`
with CPython's `struct.pack('e')`, while Torch returns `-0.1021728515625` through
an intermediate midpoint. The verifier now checks the interval between native
`nextafter` neighbors with exact ties-to-even, then checks every repaired value
against rational distances; every repaired FP16 value is also checked with
CPython's independent half conversion. Across the public factor reference,
6,369 provisional FP16 casts and 873 provisional BF16 casts require repair.
This preserves the declared direct-rounding contract; it does not make the
auditor a native PEFT arithmetic emulator.

## Installed artifacts and bounded memory

The wheel is built from a source distribution made from clean archived source
`06b47e7`, then installed in isolated Python 3.10.21/NumPy 1.24.0/safetensors
0.6.2 and Python 3.15.0/NumPy 2.5.4/safetensors 0.8.0 environments. Neither has
Torch, PEFT or Transformers. Both replay all 16 native/public cases, including
the expected incomplete-module refusal, and pass all 141 packaged-source tests.
CLI JSON and API reports agree exactly within each environment; real terminal
output, input protection and preservation of an earlier report are checked too.
[Artifact identities](results/installed.json), [Python 3.10](results/installed-310.json),
[Python 3.15](results/installed-315.json).

Python 3.15 reproduces the complete source reports exactly. With NumPy 1.24,
every discrete field remains exact; 1,544 floating report fields differ by at
most `8.881784197001252e-16`. The driver checks every field against the
hash-verified complete source report, with relative tolerance `2e-11` and
absolute tolerance `2e-14` for floating values only. It records actual drift
separately. This is library arithmetic variation, not a saved-weight mismatch.
The dropped-head controls' complete factor measurements remain byte-identical
to their respective correct merges within both environments.

A fresh installed Python 3.15 CLI checks all 273 public BF16 tensors, including
56,623,104 full-state values, across 840,855,823 input-file bytes with **39.95 MiB
peak process RSS** at a 1 MiB block setting. GNU time measures 40,904 KiB and
9.90 seconds including interpreter startup and input hashes. This is one
warm-cache observation on the shared 14-vCPU WSL2 host, with one OpenBLAS/OMP
thread; it is not a hard memory cap or isolated throughput benchmark.
[Complete method, input identities and report match](results/memory.json).

## Reproduction

The optional native producer environment is pinned in the retained receipts:
Python 3.12, PyTorch 2.14.1+cpu, PEFT 0.21.0, Transformers 5.20.1,
NumPy 2.5.4 and safetensors 0.8.0. These are verification dependencies, not
auditor runtime dependencies. Use a new temporary output directory:

```sh
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python studies/saved-modules/produce.py /tmp/saved-modules-native
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python studies/saved-modules/verify.py /tmp/saved-modules-native /tmp/saved-modules-native.json
```

For the public case, download exactly the two files in each source entry of
`sources.json` into `/tmp/saved-modules-public/base` and `adapter`. Immutable
Hugging Face URLs have the form
`https://huggingface.co/<repository>/resolve/<revision>/<file>`. The producer
refuses inputs with a different size or SHA-256 before loading the model:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python studies/saved-modules/public.py /tmp/saved-modules-public
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python studies/saved-modules/verify.py /tmp/saved-modules-public /tmp/saved-modules-public.json --public
```

Set `PYTHONPATH` to the unchanged prior checkout's `src` and add `--before` to
retain baseline refusals. Python's hash seed stabilizes PEFT configuration list
ordering. No network is used by either producer after the pinned files exist.

Public verification can be resumed with `--case correct-f16`, for example.
The retained public receipt combines four such runs only after checking their
producer, verifier and source-module identities and complete case coverage.
`reload.py ROOT CASE OUT` independently checks a saved model against a fresh
native merge. `legacy.py --help` describes the 26 unchanged old-input comparisons.
Run `installed.py --help` with an isolated installed Python for package replay;
the optional reference cache preserves complete source reports and their
retained hashes when comparing different NumPy versions.

## Boundaries

- Targets are explicit complete saved module state. Arbitrary undeclared full
  tensors, partial modules, overlapping scopes and unexplained merged additions
  refuse. Existing original tensors cannot disappear.
- The resolver follows PEFT's literal suffix selection for observable module
  prefixes, then maps tensor names exactly. It imports no architecture and does
  not infer tied aliases, tokenizer identity or missing module state.
- An absent or incompatible original base tensor has null original-change
  measurements. Its full saved target is still checked at every coordinate.
  Incompatible original floating state is scanned for finite values.
- Integer/bool state requires exact target/merged dtypes. Its norms are null;
  there is no precision-losing float conversion. Floating targets are rounded
  directly to merged dtype; cast drift and mismatch error remain distinct.
- Base-independent DoRA norm mode refuses full modules. It cannot silently
  report a factor-only observation for an adapter with additional trained state.
- No architecture execution, quality estimate, training identity certification,
  registry release or hosted application is part of the lightweight audit.

The complete feature and evidence are published in the repository's `main`
branch. [Immutable source-byte comparisons](results/publication.json) cover all
40 changed files at `68fc880`; package and model-reload checks remain separate.
