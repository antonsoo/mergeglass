# Auditing GPT Conv1D projections

A Transformers `Conv1D` is a dense projection whose stored matrix uses input
rows and output columns. It is distinct from PyTorch's convolutional `Conv1d`.
The native PEFT factors still describe output-by-input products:

| Module | Saved weight | A | B | Intended update |
| --- | --- | --- | --- | --- |
| Linear | `[out, in]` | `[rank, in]` | `[out, rank]` | `(B @ A) * scale` |
| GPT Conv1D | `[in, out]` | `[rank, in]` | `[out, rank]` | `(B @ A).T * scale` |
| Embedding | `[tokens, dimensions]` | `[rank, tokens]` | `[dimensions, rank]` | `(B @ A).T * scale` |

A square weight passes either shape check, but the two updates can point in
different directions. Changing every entry therefore does not establish the
correct orientation. Try the native, explicitly authored control:

```sh
uv run --no-dev mergeglass examples/conv1d/base examples/conv1d/adapter examples/conv1d/wrong-square-orientation
```

Replace the final input with `examples/conv1d/correct-f32-safe-false` or
`examples/conv1d/omitted-square` to compare. These are synthetic models trained
for five steps on random token IDs, followed by native PEFT merges or explicit
faulty conversions. They measure no deployed failure rate or task quality.

## What establishes the layout

The auditor recognizes these standard Transformers models from the original
base's `config.json:model_type`, or the adapter's exact
`auto_mapping.parent_library`. All files remain local; no architecture or model
code is loaded by the auditor.

| Model type | Adapter parent library |
| --- | --- |
| `gpt2` | `transformers.models.gpt2.modeling_gpt2` |
| `openai-gpt` | `transformers.models.openai.modeling_openai` |
| `imagegpt` | `transformers.models.imagegpt.modeling_imagegpt` |

Supported module paths are `h.N.attn.c_attn`, `h.N.attn.c_proj`,
`h.N.mlp.c_fc`, and `h.N.mlp.c_proj`, optionally prefixed by `transformer.`.
For GPT-2, `h.N.crossattention.c_attn`, `q_attn`, and `c_proj` are also supported
with the same optional prefix. `N` is a decimal layer index. An ordinary
`lm_head` in these models retains Linear orientation even when the saved global
flag is true. Embedding factors identify their separate layout by their native
key family.

`fan_in_fan_out` must be boolean. It is not treated as a map of individual
modules: PEFT can mutate it as it dispatches each target. A known standard
Conv1D path is transposed with either flag value. No transpose is selected by
comparing residuals or picking a better numeric fit.

Conflicting model/library evidence, an unrecognized adapter library beside a
known Conv1D model, remote custom-model `auto_map` declarations and nonstandard
paths refuse. A true flag without a supported layout also refuses. This table
describes the declared standard architectures; metadata cannot independently
prove the absence of custom runtime modifications. With no recognized Conv1D
evidence and a false flag, ordinary Linear factors retain their existing input
contract; their implementation is not certified from a square shape.

## Reports and memory

Each Conv1D layer adds these fields:

```json
{
  "weight_layout": "conv1d",
  "reference_product": "(B @ A).T",
  "layout_evidence": ["base/config.json:model_type=gpt2"],
  "coordinate_axes": {"row": "input", "column": "output"}
}
```

Norms, projection gain, unchanged entries, reference rounding and the largest
residual all use the saved orientation. Residual coordinates are input/output
indices in the original weight. JSON fingerprints identify the configuration
files supplying the declared evidence. Existing ordinary Linear fields remain
unchanged.

The auditor retains decoded B and streams columns of A into saved input-row
blocks. `--max-adapter-mib` limits decoded B for these layers. The default
512 MiB limit and `--block-mib` workspace target are not process RSS caps.
An actual native 32,768-by-512 case checks every entry with a 1 MiB block and
1 MiB factor limit although decoded A alone is 2 MiB.
[Measurements and independent reference](../studies/conv1d/README.md#memory).

## Original-base serialization

The base and merged inventories must match exact tensor names, apart from
explicit full saved modules. The auditor does not rename weights or discard
buffers. A native load/save can change historical checkpoint namespaces and
nonpersistent buffers while retaining parameter values.

For example, the pinned public GPT-2 file has bare `h.N.*` names and twelve
causal-mask buffers. The current native LM-head save has `transformer.h.N.*`
names and omits those buffers. The public study first saves the **unadapted**
base with the same native writer, checks all 148 retained tensors against the
downloaded checkpoint and records both inventories. Only then is that native
original-base save used for the audit. The downloaded adapter is unchanged.
The raw download remains a separately hashed input to the study.

Use the original base from before adapter loading, preserve its dtype, and
verify any required reserialization. A model name, a matching shape, or a low
residual does not establish the training base's identity.

Conv1D DoRA, arbitrary custom/transposed modules, convolutional adapters,
quantization and extra trained biases remain outside this extension. Signed
alpha, rsLoRA, supported rank/alpha overrides and explicit full saved modules
keep their existing contracts. Matching weights do not certify inference
equivalence or model quality.
