# Native DoRA merge controls

These small files come from PEFT `save_pretrained` / `merge_and_unload`, with
labelled synthetic regression training and explicit post-training controls.
They are a subset of the artifacts listed in [producer.json](producer.json).
The full producer, independent oracle and complete receipt are retained in
[the DoRA study](../../studies/dora/README.md).

| Directory | Actual artifact |
| --- | --- |
| `base` | Original FP32 weights, rounded to BF16 before training |
| `adapter` | Unchanged saved DoRA A/B factors and learned magnitudes |
| `correct-f32-safe-false` | Native DoRA merge, FP32 base and adapters |
| `correct-bf16-safe-false` | Native DoRA merge, BF16 base and FP32 adapters |
| `as-lora-f32` | Same adapter loaded with `use_dora=False`, then natively merged |
| `omitted` | Base saved without any adapter |

The `direction` matrix uses its trained factors and magnitudes. After training,
`magnitude_only` has B explicitly zeroed while retaining its learned magnitudes;
`zero_magnitude` has its magnitudes explicitly zeroed while retaining A/B.
`control` is unadapted. These controls are not claims about training outcomes
in a real application. Signed magnitudes are covered separately by the scalar
oracle and exact examples.

From the repository root, compare:

```sh
uv run --no-dev mergeglass examples/dora/base examples/dora/adapter examples/dora/as-lora-f32
uv run --no-dev mergeglass examples/dora/base examples/dora/adapter examples/dora/correct-f32-safe-false
```

Every retained binary is ours, produced from synthetic inputs with no downloaded
model. The native config's `auto_mapping` is inert metadata; the auditor never
imports the model class.
