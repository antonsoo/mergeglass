# Local verification, 2026-11-04

The runtime implementation is commit `f2c3fc1`; the fresh source archive and
installed packages were made from `6996f0e`, which includes the native study,
examples and README. Later verification documentation does not change runtime
code. No push, publication or registry release was performed.

## Fresh source archive

Extracted `git archive HEAD` into a new directory with no existing environment.
Ran these commands there:

```sh
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv build --out-dir /tmp/mergeglass-packages
```

Ruff: clean; 123 Python files formatted. Mypy: clean for 11 source files.
Pytest: 201 passed on Python 3.15.1. Both wheel and sdist built successfully.
The project lockfile and runtime dependencies did not change.

## Installed artifacts outside the checkout

Created an empty Python 3.10.21 environment and installed the wheel. Confirmed
that imports resolve into its `site-packages`. Ran the existing rounding
quickstart and the new correct, omitted and scaled-bias CLI examples from an
unrelated working directory. Bias mismatch counts were 0, 12 and 12. The
incomplete historical example exited 2 with the missing-trained-value error
and produced no JSON report.

Replayed both extracted native archives with the wheel's Python and the study
verifier: 149 file identities, 2854 bias-entry comparisons, 73 measured cases
and ten expected refusals. All independent checks agreed. Ran the complete
archived test suite against that installed wheel: 201 passed on Python 3.10.21
with NumPy 2.2.6 and safetensors 0.8.0.

Separately installed the sdist in an empty Python 3.15.1 environment and ran the
omission CLI outside the checkout. It reported twelve bias mismatches, as
expected. This environment used NumPy 2.5.4 and safetensors 0.8.0.

Machine-readable receipts include producer/runtime versions and package
identities: [wheel](results/wheel-installation.json),
[sdist and package SHA-256](results/sdist-installation.json).
The retained oracle reports and manifests remain the primary numerical evidence.

## Visual and scope review

Captured the real omission CLI output and rendered it as
[trained-bias.png](../../docs/assets/trained-bias.png). Opened and inspected the
image: complete output, readable columns, no clipped lines. This is a terminal
artifact; no website or browser/offline delivery claim is made.

Native training data/models and fault cases are synthetic. Only the stated
producer versions and layouts were replayed. There is no public-adapter,
production-incident-rate or task-quality result in this extension. Unsupported
combined bias modes and module layouts are described in the study.
