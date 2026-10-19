# An unchanged public adapter, merged in FP32 and BF16

This study uses the public
[hfm8tr/smollm2-135m-smoltalk-lora](https://huggingface.co/hfm8tr/smollm2-135m-smoltalk-lora/tree/fb39c7012c3b125f94d2bb1a093254994053f4ca)
adapter and the configured
[HuggingFaceTB/SmolLM2-135M base](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2).
Their exact revisions, file sizes and SHA-256 values are pinned in
[sources.json](sources.json). The four original files total 270,921,059 bytes.
They are fetched unchanged to local throwaway storage and are not redistributed.
Reports retain a few scalar weight values at the largest-residual coordinates;
they contain no complete tensor or checkpoint payload.

The author's card describes a learning project trained on 2,000 SmolTalk
everyday-conversation samples. Those training claims and original base revision
are **not independently verified here**. The adapter config records PEFT
0.19.1, rank 8, alpha 16 and q_proj/v_proj targets. The base card declares
Apache-2.0; the retrieved adapter card declares no license. Our numeric reports
and source do not relicense either upstream artifact.

## Actual local merges

`merge.py` hash-checks the downloaded files, then uses built-in Transformers
Llama code and PEFT 0.21.0 locally, with `trust_remote_code=False`, offline mode
and `local_files_only=True`. It makes two new merges with FP32 adapter factors:
one FP32 base and one BF16 base, each through
`merge_and_unload(safe_merge=False)`. The resulting HF safetensors shards are
the actual audit inputs. This is not a claim about a third party's deployed merge.

| Measurement | FP32 merge | BF16 merge |
| --- | ---: | ---: |
| Complete checkpoint tensors checked | 272 | 272 |
| Adapted matrices | 60 | 60 |
| Nonzero reference update entries | 13,271,040 | 13,271,040 |
| Entries unchanged from the dtype-cast base | 105 | 5,886,980 |
| Share of nonzero updates with no observed change | 0.000791% | 44.36% |
| Reference squared norm at unchanged positions | 1.24e-13% | 15.05% |
| Unadapted tensors differing from cast base | 0 / 212 | 0 / 212 |
| Per-layer relative residual range | 0.00000184–0.00001187 | 0.1212–0.5761 |

These are counts and norms over saved weights. Small entries count equally
with large ones in the 44.36% count fraction. Those unchanged entries contain
15.05% of the complete reference update's squared norm; this is summed across
matrices, not averaged from per-layer percentages. Changed positions can still
undershoot, overshoot or reverse direction, so neither percentage measures
retained observed energy, accuracy loss or downstream quality. No matrix disappears
entirely in either merge. The dense reference is numerical binary64, and
producer matmul/addition precision can differ.

The [full receipt](results/verification.json) records every layer, largest-residual
coordinate, original input hash, native merged-shard hash and exact library
version. The independent dense PyTorch oracle checks the same quantities as the
[controlled native study](../native-peft/README.md), including all unadapted
weights and different audit block settings. It uses PEFT's configuration resolver
for scale, not the auditor's scale parser.

The [separate forward study](../forward/README.md) runs the exact saved merges
and unmerged adapter on six authored texts. It records actual logit and token
differences while keeping numerical equivalence separate from task accuracy.

## Reproduce

Use the separate environment from the
[native study](../native-peft/README.md). Fetching requires internet access;
merging, verifying and the audit runtime operate on local files only.

```sh
python3 studies/community/fetch.py /tmp/mergeglass-community
OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /tmp/mergeglass-producer/bin/python studies/community/merge.py /tmp/mergeglass-community
OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 /tmp/mergeglass-producer/bin/python studies/community/verify.py /tmp/mergeglass-community /tmp/mergeglass-community-verification.json
```

The saved merged checkpoints need additional disk space. `fetch.py` reuses only
hash-verified cached files. Output weights remain outside the repository. If
upstream files disappear, the source and recorded hashes remain inspectable;
the downloaded weights are not bundled as a fallback.
