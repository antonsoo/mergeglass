# A trained head lost during conversion

These are unchanged native PEFT outputs from synthetic classifier training.
The base has an 8 x 8 projection, a nested 8 -> 5 -> 3 classifier and an integer
buffer. The projection uses rank-2 LoRA; `modules_to_save=["classifier"]` saves
the complete trained head. No pretrained weights or task benchmark is included.

```sh
uv run --no-dev mergeglass examples/saved-modules/base examples/saved-modules/adapter examples/saved-modules/omit-head-f32
```

| Merged directory | What was saved | Full-state target mismatches |
| --- | --- | ---: |
| `correct-f32` | Native `merge_and_unload()` | 0 |
| `omit-head-f32` | Correct native LoRA; original classifier restored | 64 |
| `wrong-counter-f32` | Correct float weights; integer counter reduced by one | 1 |

The last counter is above `2**60`; comparing it through float64 would hide the
one-count error. All three directories have identical LoRA weights. The wrong
artifacts are explicit authored conversion controls, not observed incidents.

`producer.json` retains the full producer's manifest; this directory includes
only the small FP32 subset listed above. [Producer, complete native/public study
and independent checks](../../studies/saved-modules/README.md).
