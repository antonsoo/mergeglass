# Local DoRA source and installed-artifact verification

The implementation at `59b73911a8a918840d19d12773f52d10623f206b` adds native
ordinary Linear DoRA audit support. The previous source at `068b150` refused
all 12 native DoRA cases. The [study](../studies/dora/README.md) retains those
before/after imports, small runnable artifacts and independent calculations.

The implementation was verified locally before publication. The subsequent
[source publication receipt](../studies/dora/results/publication.json) compares
all changed files at public commit `37aa24e` byte for byte. No tag, package release,
hosted service or site deployment was performed. The public model inputs were read as pinned
evidence, not published by this work. Quantized and non-Linear adapters remain
outside the supported arithmetic contract; numerical agreement does not certify
task quality or training provenance.

## Clean source and package bytes

A `git archive` of the implementation commit was extracted to a fresh directory.
These commands completed there:

```sh
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv build
```

Lint, formatting and strict type checking passed. All 82 tests passed. The build
created both `dist/mergeglass-0.1.0-py3-none-any.whl` and
`dist/mergeglass-0.1.0.tar.gz`. The [machine-readable receipt](../studies/dora/results/installed.json)
retains the source archive hash, exact command arguments, check-log hashes,
installed module hashes and artifact identities. `$WORK` in that receipt denotes
the throwaway validation directory and `$REPO` the original checkout.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Wheel | 23,254 | `83793acde8235fd3ee14f1ad6803db30c04ccc560f554f3a821bf43d991af339` |
| Source distribution | 1,109,684 | `02ee44f38a8f0babc5482ca333ab0d3c9351a62ae7f2b8749a89453b5eb12bee` |

These hashes identify the clean implementation snapshot. The subsequent
verification-documentation commit does not change runtime modules.

## Actual installed execution

Two fresh environments installed that wheel, with no editable source link:

| Python | NumPy | safetensors | Installed tests | Native example replays | Decimal coordinates |
| --- | --- | --- | ---: | ---: | ---: |
| 3.10.21 | 1.24.0 | 0.6.2 | 82 | 4 | 4,212 |
| 3.15.0 | 2.5.4 | 0.8.0 | 82 | 4 | 4,212 |

The test command was the installed interpreter's `-m pytest -q` against the
archived tests, run outside the checkout with `PYTHONPATH` empty. Every imported
Mergeglass module was inside the environment's `site-packages`, and every `.py`
hash matched its wheel member. PyTorch, PEFT and Transformers were absent.

Both environments replayed correct FP32/BF16 DoRA examples, the omitted-magnitude
control and copied base, comparing complete reports against the dense native
oracle receipt. They also ran the original rounding quickstart, summary/file
JSON and `--block-mib 4 --json -` workflows. The 80-digit scalar study completed
independently in each environment.

The installed Python 3.15 CLI additionally audited the unchanged public FP32
checkpoint and the local BF16 public-adapter merge. Both complete reports agree
with the independent dense receipt: 272 tensors checked, including 210 adapted
matrices. The exact changed/unchanged counts and input fingerprints agree.

The same installed environment ran the native 33,554,432-entry streaming case
in fresh workers. At the default 16 MiB block target it measured 61.63 MiB peak
process RSS, with zero residual against the exact rational control.
[All block settings, input hashes and local timing](../studies/dora/results/streaming.json)
are retained; the block target is not a hard memory cap.

## Existing behavior and the visible example

[Eighteen existing LoRA replays](../studies/dora/results/lora-compatibility.json)
retain exactly equal decoded measurements, coordinates, summaries and input
hashes before/after the extension, excluding only the added `adapter_variant`
field. These include native F32/F16/BF16 saves, repeated/reversed/wrong-adapter
controls and the prior public SmolLM2 case. Existing schema-1 LoRA reports also
still render without the new field.

The [DoRA terminal capture](assets/dora.png) renders the unedited output in
[dora.txt](assets/dora.txt). The actual installed CLI reproduces that text byte
for byte. The screenshot was opened and inspected: all table columns and the
command are legible and contained. This is a CLI capture, not a web-app workflow.

The optional study also checks 386,304 adjacent-bit-pattern midpoint cases for
F16/BF16 casts. The auditor and independent direct-cast grid have zero
disagreements. The pinned CPU PyTorch conversion behaves like an F32 intermediate
on those cases, so the dense oracle uses the direct grid when checking the
auditor's declared reference. No numeric comparison tolerance was loosened.

## Repository handoff

Suggested GitHub description: "Audit saved LoRA and DoRA merges against original
weights, with streaming numerical evidence."

Suggested topics: `lora`, `dora`, `peft`, `safetensors`, `model-merging`,
`bfloat16`, `numerical-analysis`.

Implementation history at verification time:

```text
59b7391 feat: audit complete native Linear DoRA weight updates
068b150 docs: retain clean builds and installed public audit verification
```

Build command: `uv build`. Outputs: `dist/` wheel and source distribution.
There is no static-site build or deployment in this CLI repository.
