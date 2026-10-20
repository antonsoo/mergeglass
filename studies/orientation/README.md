# A global orientation flag can hide a transposed target

This is the retained study of the original error and its initial refusal guard.
The current source now [audits these standard Conv1D layouts](../conv1d/README.md)
with explicit metadata evidence, including mixed targets. The historical
measurements below describe the earlier versions; they are not current refusals.

A native PEFT adapter can contain both a Transformers `Conv1D` target and an
ordinary `Linear` head while saving `fan_in_fan_out=False`. The flag is global;
each constructed module retains its own orientation. For a square Conv1D weight,
the Linear dimension check passes even though its reference update needs a
transpose. Treating the flag as a complete orientation witness therefore
produces a misleading residual for a valid merge.

The producer constructs tiny GPT-2, OpenAI GPT and ImageGPT models, selects
`transformer.h.0.attn.c_proj` and `lm_head`, gives B seeded nonzero values, then
saves native PEFT adapters and native merged checkpoints. There is no training
or pretrained model in this study. All three saved global flags are false;
all three actual `c_proj` modules require a transposed delta.

| Mixed native model | Linear-assumed relative residual for `c_proj` | Dense reference with native orientation | Corrected audit |
| --- | ---: | ---: | --- |
| GPT-2 | 1.339205 | 0.0000001004 | Refused |
| OpenAI GPT | 1.474574 | 0.0000001375 | Refused |
| ImageGPT | 1.355879 | 0.0000000842 | Refused |

The original audit accepted each artifact and used the wrong orientation. The
[before receipt](results/before.json) records those reports with auditor module
hashes. The [after receipt](results/after.json) records the explicit refusal and
the independent dense measurements. Three additional adapters target only the
ordinary `lm_head` on the same model types; those remain accepted with relative
residuals below `0.000000146`. This avoids rejecting an entire architecture when
only its ordinary Linear weights are adapted.

The correction uses available base `model_type` or adapter
`auto_mapping.parent_library` as evidence for known Conv1D module paths. It
does not import model code, transpose a product based on a better numerical fit,
or claim universal architecture detection. A tensor shape, global flag, absent
configuration or arbitrary custom class cannot establish all module types. Plain
Linear targets remain the required input contract.

The relevant upstream definitions are
[PEFT's native target dispatch](https://github.com/huggingface/peft/blob/v0.21.0/src/peft/tuners/lora/layer.py),
[GPT-2](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/gpt2/modeling_gpt2.py),
[OpenAI GPT](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/openai/modeling_openai.py)
and [ImageGPT](https://github.com/huggingface/transformers/blob/v5.20.1/src/transformers/models/imagegpt/modeling_imagegpt.py).
The historical source guard covered their known attention and MLP Conv1D paths, including
cross-attention paths. It does not infer a tensor's implementation from a name
without the corresponding model metadata.

## Reproduce

Use the [pinned native-producer environment](../native-peft/README.md), with
the current package installed. Use a new output directory:

```sh
PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 /tmp/mergeglass-producer/bin/python studies/orientation/produce.py /tmp/mergeglass-orientation
OPENBLAS_NUM_THREADS=1 /tmp/mergeglass-producer/bin/python studies/orientation/verify.py /tmp/mergeglass-orientation /tmp/mergeglass-orientation-verification.json
```

The verifier checks all producer file hashes and computes complete float64 PyTorch
products using the per-module orientations retained from native PEFT modules.
By default it now checks six supported native audits. Use `--refused` with the
historical guard revision to require the three mixed-target refusals and three
accepted controls.
To reproduce the old behavior, install source revision
`a28007649836d644be50cb2946079f1f837c4eb5` in a separate environment and run the
current verifier with `--before`. No third-party weights or model code are
downloaded or executed. Runtime inspection uses only local artifact bytes.
