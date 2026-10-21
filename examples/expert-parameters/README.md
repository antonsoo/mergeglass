# Native expert matrices with an authored omitted expert

These small safetensors files come from six optimizer steps on a locally
constructed synthetic Qwen3 MoE model using PEFT 0.21.0 and Transformers 5.20.1.
They contain three experts, two logical adapted parameters and nine physical
saved expert matrices. They do not represent a useful trained language model.

```sh
uv run mergeglass examples/expert-parameters/base examples/expert-parameters/adapter examples/expert-parameters/correct-f32-safe-false
uv run mergeglass examples/expert-parameters/base examples/expert-parameters/adapter examples/expert-parameters/omitted-expert --json /tmp/expert-omission.json
uv run mergeglass examples/expert-parameters/base examples/expert-parameters/adapter examples/expert-parameters/contiguous-B
```

The omission copies expert 1's gate/up runtime slices back from the original
base, then saves through Transformers. Both resulting physical weights have
no observed update: gain 0 and relative residual 1. Other saved tensor payloads
are exactly the native correct merge. The second fault incorrectly groups B
columns by expert instead of following the native rank-first interleaving.
Every input is generated locally; no third-party model weights are included.

[Producer, parameter identity, complete precision cases, independent calculation
and limitations](../../studies/expert-parameters/README.md).
