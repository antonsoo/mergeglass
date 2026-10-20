# Correct matrices, omitted bias

These are unchanged native PEFT outputs from the synthetic `affine` profile
in [the LoRA B-bias study](../../studies/lora-bias/README.md), plus two explicitly
authored faulty conversions. The model is a small three-layer affine network;
six optimizer steps update the selected adapters. No pretrained model or
training-quality claim is involved.

```sh
uv run mergeglass examples/lora-bias/base examples/lora-bias/adapter examples/lora-bias/omitted-bias
```

Both matrix updates remain those of the correct native merger. All ten
nonzero intended bias updates are absent: both vectors have gain 0 and relative
residual 1. Switch the final path to `correct-f32-safe-false` for the actual
native result, or `unscaled-bias` for a conversion that forgot the scale.
Its gains are about 1/3 and -1 because the selected modules have scales 3 and -1.

The bias numbers are separate from the matrix totals. The correct FP32 result
has two differences from the single-rounding bias reference; native float32
operation order differs from that reference. A completed audit is an observation,
not a success certificate for the model or its predictions.
