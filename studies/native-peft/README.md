# Native PEFT artifacts and a separate dense oracle

The small checked-in [example](../../examples/rounding/) is produced by
`produce.py` using PEFT 0.21.0, PyTorch 2.14.1+cpu and its native
`save_pretrained` / `merge_and_unload(safe_merge=False)` methods. It is **not a
pretrained or production model**. A four-matrix teaching network trains for 12
SGD steps on seeded random inputs. After training, one B factor is multiplied
by 0.0001, another by 100, and another set to zero. Those explicit controls
isolate small, large and zero updates. No frequency in deployed models is inferred.

The original base entries are rounded to BF16 before training, then stored in
FP32. Their later dtype conversion is exact, so the small-update comparison does
not confuse base conversion with adapter merging. Adapter factors remain FP32
in each native FP32/F16/BF16 merge. An unchanged base copy and a base with one
unrelated edit are separate controls.

| Teaching module | FP32 changed / 64 | BF16 changed / 64 | Interpretation in this constructed producer |
| --- | ---: | ---: | --- |
| `small` | 31 | 0 | Nonzero update; BF16 merge retains no numeric change |
| `large` | 64 | 62 | Most entries change; residual quantifies the difference |
| `zero` | 0 | 0 | Zero reference product; relative ratios are undefined |

`transformer.py` independently constructs a random two-layer Llama through
Transformers, trains it for eight AdamW steps on seeded synthetic token IDs,
and saves native HF shards. Its four adapted matrices include rsLoRA, one
per-layer rank override and an alpha override. Three saved merges exercise
FP32, F16 and BF16. This checks a real framework export, not language quality.

The [complete verification](results/verification.json) retains all eight
artifact/control cases, all per-layer measurements, input file hashes and both
producer receipts. The public, externally trained adapter is checked
[separately](../community/README.md).

## Independent measurements

The auditor uses NumPy and bounded row blocks. `verify.py` uses complete tensors
from `safetensors.torch`, binary64 PyTorch matmul, PEFT's own `LoraConfig` and
`get_pattern_key`, and independent saved-weight comparisons. It checks resolved
ranks/scales, every norm, exact changed/nonzero/unchanged counts, unadapted edits
and concrete largest-residual coordinates. Counts must agree exactly; floating
measurements use relative tolerance `1e-8` and absolute tolerance `1e-15`.
It also compares 1 MiB and 16 MiB audit settings with those tolerances. The
separate regression case uses exact binary fractions across many blocks.

The native merger is not the mathematical oracle. In particular,
[PEFT's Linear implementation at v0.21.0](https://github.com/huggingface/peft/blob/v0.21.0/src/peft/tuners/lora/layer.py)
can cast a CPU half-precision factor product back to its original dtype, and
`safe_merge=True` can cast a delta before addition. Mergeglass measures actual
saved artifacts and does not equate those operation sequences with one ideal
binary64 addition and cast.

## Reproduce

The audit runtime does not require these heavyweight dependencies. Use a
separate Python 3.12 environment and the exact
[producer requirements](requirements.txt):

```sh
uv venv /tmp/mergeglass-producer --python 3.12
uv pip install --python /tmp/mergeglass-producer/bin/python torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /tmp/mergeglass-producer/bin/python -r studies/native-peft/requirements.txt
uv pip install --python /tmp/mergeglass-producer/bin/python --no-deps .
PYTHONHASHSEED=0 /tmp/mergeglass-producer/bin/python studies/native-peft/produce.py /tmp/mergeglass-teaching
PYTHONHASHSEED=0 /tmp/mergeglass-producer/bin/python studies/native-peft/transformer.py /tmp/mergeglass-transformer
OPENBLAS_NUM_THREADS=1 /tmp/mergeglass-producer/bin/python studies/native-peft/verify.py /tmp/mergeglass-teaching /tmp/mergeglass-transformer /tmp/mergeglass-verification.json
```

Producer output directories must be new. Fixed seeds, one torch thread and
`PYTHONHASHSEED=0` retain numerical and configuration ordering controls.
Exact file hashes can change across producer/library/platform versions;
the verifier always checks the receipt belonging to the files it reads.
Generated model-card boilerplate is discarded; adapter configuration and
safetensors remain unchanged native outputs.
