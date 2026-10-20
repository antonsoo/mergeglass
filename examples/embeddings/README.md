# Native token-embedding controls

These are unchanged small artifacts from
[`studies/embeddings/produce.py`](../../studies/embeddings/produce.py).
Training IDs and regression targets are synthetic. Wrong merges are explicit
post-training controls, not observed user incidents. [Producer and selected
input hashes](producer.json).

```sh
uv run --no-dev mergeglass examples/embeddings/base examples/embeddings/adapter examples/embeddings/wrong-orientation-f32
```

Replace the last input with:

| Directory | Saved artifact |
| --- | --- |
| `correct-f32-safe-false` | Native FP32 PEFT merge |
| `correct-bf16-safe-false` | Native BF16 PEFT merge with FP32 adapter factors |
| `omit-embeddings-bf16` | Native BF16 merge after zeroing only embedding A; Linear merge retained |
| `wrong-orientation-f32` | Native FP32 merge with the square embedding replaced by an untransposed update |

The `square` matrix has 11 token rows and 11 embedding dimensions, so the wrong
orientation has a valid shape. Its changed-entry count alone cannot establish
fidelity. `tokens` has 37 rows and 11 dimensions, and `projection` is ordinary
Linear LoRA. The control weight remains unadapted.

These artifacts support numerical inspection, not generation or quality
claims. [Complete native cases and independent references](../../studies/embeddings/README.md).
