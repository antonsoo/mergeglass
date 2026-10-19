# Contributing

Use Python 3.10+ and uv. Run `uv sync --locked`, then:

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv build
```

Changes to scale interpretation, dtype conversion or artifact inventories need
an independent numeric example and a native producer replay. The
[PEFT study](studies/native-peft/README.md) describes the separate, heavier
environment. Do not add torch or model-loading dependencies to the audit runtime.

Preserve refusals for unsupported variants, uncertain tensor identities and
incomplete checkpoints. Do not turn a weight-space statistic into a claim about
generation quality or the reason an adapter is absent.

Reports should carry original input hashes and declared arithmetic. Native
fixtures must identify their producer and label constructed training data and
controls. Do not commit downloaded model weights, large benchmark tensors,
machine-specific paths, or generated package build output.

