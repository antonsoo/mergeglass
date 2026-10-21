# Column norms cannot establish token-coordinate fidelity

These unchanged saved files come from native PEFT 0.21.0. A small mixed
Embedding/Linear model trains on seeded synthetic IDs and random regression
targets. Explicit post-training magnitudes add zero, negative and magnitude-only
controls. This is not a pretrained model or an evaluation of model quality.
[Producer, complete checks and limits](../../studies/embedding-dora/README.md).

Run from the repository root; the auditor needs only NumPy and safetensors:

```sh
uv run --no-dev mergeglass examples/embedding-dora/base examples/embedding-dora/adapter examples/embedding-dora/rotated-token-rows-f32 --summary
```

The rotated embedding token rows retain the native column value multisets.
Their column norm factors stay near one, but their complete-reference direction
cosines and update residuals change. Norm-only observations leave that unknown:

```sh
uv run --no-dev mergeglass --dora-norms examples/embedding-dora/adapter examples/embedding-dora/rotated-token-rows-f32 --summary
```

Compare a shape-compatible normalization error with the native saved merge:

```sh
uv run --no-dev mergeglass examples/embedding-dora/base examples/embedding-dora/adapter examples/embedding-dora/wrong-row-axis-f32
uv run --no-dev mergeglass examples/embedding-dora/base examples/embedding-dora/adapter examples/embedding-dora/correct-f32-safe-false
```

`reset-embedding-magnitudes-f32` is another native merge after replacing only
embedding magnitudes with their directional column norms. Its Linear layer
retains DoRA. The magnitude-only embedding has an exactly zero factor product
and 115 nonzero complete-reference updates; resetting its magnitudes leaves
all 115 unobserved. These deliberate controls do not estimate real incident rates.

The [manifest](manifest.json) records hashes for all seven original native files.
The adapter configuration and safetensors are unchanged. No tokenizer is supplied;
saved row numbers identify only the synthetic token coordinates used here.
