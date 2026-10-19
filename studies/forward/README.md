# Weight counts and actual forward behavior are different measurements

The [saved-weight study](../community/README.md) finds 44.36% of nonzero update
entries unchanged in the native BF16 merge of a pinned public SmolLM2 adapter.
This separate study loads those exact saved artifacts into the actual built-in
Transformers model and compares their outputs with an unmerged PEFT adapter.
It does not convert a weight-space percentage into an accuracy claim.

Six [authored text probes](probes.json) contain 146 shared prefix positions.
Each position receives exactly the same observed tokens in every profile.
The measurements cover every vocabulary logit at every position; they do not
roll out free-running completions.

| Comparison on these 146 prefix positions | Changed greedy next-token choices | Maximum absolute logit difference | Mean probability total variation |
| --- | ---: | ---: | ---: |
| FP32 saved merge versus unmerged FP32 adapter | 0 | 0.00007153 | 0.000001612 |
| FP16 saved merge versus unmerged FP16 base + FP32 adapter factors | 1 | 0.46875 | 0.00326107 |
| BF16 saved merge versus unmerged BF16 base + FP32 adapter factors | 6 | 2.375 | 0.02337044 |
| Unmerged FP16 runtime versus unmerged FP32 runtime | 0 | 0.5329638 | 0.00307394 |
| Unmerged BF16 runtime versus unmerged FP32 runtime | 7 | 5.392186 | 0.02583185 |
| Unmerged FP32 adapter versus base-only FP32 | 15 | 2.679738 | 0.05079504 |
| Unmerged FP16 adapter versus base-only FP16 | 14 | 2.671875 | 0.05109721 |
| Unmerged BF16 adapter versus base-only BF16 | 11 | 2.6875 | 0.05612345 |

Total variation is `sum(abs(softmax(candidate) - softmax(reference))) / 2`,
computed in float64 from the actual returned logits. It is between 0 and 1;
the mean weights every prefix position equally. The greedy counts compare
argmax token IDs. Neither metric says which answer would be better. Prefix
positions in the same short text are correlated, and these six authored inputs
are not a sampled benchmark. No accuracy, perplexity, significance or general
frequency is inferred.

Both saved-weight and activation arithmetic matter. An unmerged PEFT layer
computes the base result and the adapter branch separately, then combines their
outputs. A saved merge first changes the stored weights and subsequently runs
the model with those weights. Dtypes, addition order and matrix kernels can
affect the outputs even for an ordinary mathematically equivalent LoRA update.
The runtime-dtype row also changes more than the merge operation; it remains
separate from the merge comparisons.

## Input and execution identity

The table above uses eager attention. A separate [matched attention comparison](attention.md)
runs the same named in-memory weights through SDPA, checks their identities and
retains the differing outputs. The runtime path needs to remain explicit when
interpreting a merge comparison.

The [receipt](results/forward.json) retains exact probe text and token IDs,
the revision/hash of the original tokenizer, all producer versions and settings,
per-probe measurements, concrete changed top-1 token IDs, and hashes/shapes of
all 54 full-logit arrays. Large logits and third-party weights stay in throwaway
storage. The original model/adapter and native merged checkpoint hashes must
match the previously recorded public merge receipt before any model executes.

The additional [tokenizer source](tokenizer-source.json) is pinned to the same
SmolLM2 base revision and SHA-256 checked. It is parsed by the installed
`tokenizers` library; model code comes from the pinned installed Transformers
package. `local_files_only=True` and `trust_remote_code=False` prevent model
downloads or model-supplied executable code. Fetching the tokenizer is an
explicit study preparation step, outside the Mergeglass audit runtime.

Every profile uses CPU, one torch/OpenBLAS thread, `eval()`, `inference_mode()`,
eager attention, no KV cache, and identical IDs without added special tokens.
The wrapped profiles keep FP32 adapter factors. No chat template is inferred
from the adapter. These choices make the comparison explicit; they do not
establish instruction quality or behavior under another execution kernel.

Every forward is repeated and must return bit-identical logits. All 54 eager
arrays also reproduce across independent processes, including after adding
in-memory fingerprints. A separate NumPy verifier checks all 48 per-probe
comparison measurements and their summaries. This establishes observed reproducibility on this
pinned runtime and host, rather than cross-platform numerical identity.

## Reproduce

Prepare the existing [public native merges](../community/README.md), then use
the same [pinned producer environment](../native-peft/README.md):

```sh
HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 /tmp/mergeglass-producer/bin/python studies/forward/run.py /tmp/mergeglass-community /tmp/mergeglass-forward /tmp/mergeglass-forward.json
```

The first run explicitly retrieves the pinned tokenizer if it is absent from
the output directory. After that file is cached, model loading and inference
operate only on local files. `HF_HUB_OFFLINE` constrains Hub loaders; the explicit
tokenizer preparation fetch remains visible in the script. The output directory
stores roughly 250 MiB of logits, plus the tokenizer; use throwaway storage.
Compare the new small JSON with the retained receipt using tolerances suitable
for another platform. Check its versions/settings before interpreting differences.
