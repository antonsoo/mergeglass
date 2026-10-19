# Identical named weights, different attention paths

The same pinned SmolLM2 artifacts and 146 prefix positions from the
[forward study](README.md) were run through eager attention and PyTorch's scaled
dot-product attention (SDPA). Each corresponding profile has the same input IDs,
versions, dtype, adapter factors, evaluation mode and named in-memory tensor
values. The attention implementation setting is the controlled difference.

| Profile | FP32 changed greedy choices / 146 | FP16 changed choices / 146 | BF16 changed choices / 146 |
| --- | ---: | ---: | ---: |
| Base only | 0 | 0 | 4 |
| Unmerged adapter | 0 | 1 | 2 |
| Saved merge | 0 | 0 | 5 |

For the BF16 unmerged adapter, mean probability total variation between attention
paths is 0.02025768; maximum absolute logit difference is 0.78125. The
[complete cross-path receipt](results/attention.json) retains all nine profiles,
all 54 per-probe comparisons, and concrete changed token IDs. The separate
[eager](results/forward.json) and [SDPA](results/sdpa.json) receipts retain every
profile's outputs and the merge-versus-unmerged comparisons within that path.

Even the merge comparison depends on the selected path: on these inputs,
eager attention produces 0, 1 and 6 changed choices for FP32, FP16 and BF16;
SDPA produces 0, 0 and 5. These counts are numerical observations on six short
authored texts. They do not establish which choices are better, accuracy rates,
performance, or behavior on another runtime.

## What is held equal

The producer fingerprints the actual loaded `state_dict` before each profile's
forward passes. The digest covers sorted tensor names, dtypes, shapes and every
raw byte, including named weights/buffers and adapter factors. Corresponding
eager/SDPA digests must agree before their outputs are compared. This checks
loaded numerical values rather than relying only on identical input filenames.
It does not claim that the two programs or every non-tensor attribute are equal;
the attention implementation is explicitly different.

Both runs use CPU, one thread, evaluation mode with dropout disabled,
no KV cache and identical token IDs. Every forward repeats bit for bit. All 54
eager arrays also reproduce the independent earlier process, before fingerprints
were added. No weights or complete logits are redistributed.

The comparison verifier imports only NumPy and the standard library. It checks
all 108 array hashes and independently recalculates all 96 within-path
measurements, including summaries and concrete token changes, before calculating
the 54 cross-path measurements. NumPy 1.24.0 and 2.5.4 agree within relative
tolerance `1e-10` and absolute tolerance `1e-15`; counts and identities are exact.
This is separate from the producer's PyTorch float64 measurement code.

The setting names the Transformers attention path. This study does not identify
every lower-level SDPA kernel, certify mathematical equivalence of implementations,
test GPU execution or evaluate task quality. Saved-weight equality and runtime
numerical behavior are separate observations.

## Reproduce

Use the same [pinned producer environment](../native-peft/README.md) and public
native merge artifacts. Keep the two output directories separate:

```sh
HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 /tmp/mergeglass-producer/bin/python studies/forward/run.py /tmp/mergeglass-community /tmp/mergeglass-eager /tmp/eager.json --attention eager
HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 /tmp/mergeglass-producer/bin/python studies/forward/run.py /tmp/mergeglass-community /tmp/mergeglass-sdpa /tmp/sdpa.json --attention sdpa
OPENBLAS_NUM_THREADS=1 uv run --no-dev python studies/forward/attention.py /tmp/eager.json /tmp/sdpa.json /tmp/mergeglass-eager /tmp/mergeglass-sdpa /tmp/attention.json
```

The first two commands execute the model. The final command needs only the
lightweight runtime dependencies and the already captured arrays. Exact bit
identity across another platform is not promised; inspect its recorded versions,
settings and state fingerprints before attributing a difference to merging.
