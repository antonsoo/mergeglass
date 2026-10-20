# Conv1D source and artifact verification

Implementation commit: `318cb53`. Native/public evidence and runnable example
commit: `9abce44897f3c0d136d48aa71b577fb2293cd176`.
The [clean-source receipt](../studies/conv1d/results/clean-source.json) records
the full verified source commit, Git archive digest and package hashes.

| Claim | Evidence |
| --- | --- |
| Supported standard GPT Conv1D layouts | 48 native saved cases; six profiles, both safe-merge modes, three storage dtypes, wrong-orientation/omission controls |
| Public adapter coverage | 48 adapted matrices and 100 unadapted tensors, four local saved artifact cases, pinned original inputs |
| Independent arithmetic | Complete dense Torch comparisons, 18,368 native scalar Decimal entries, public sampled coordinates, independent adjacent-value rounding |
| Installed behavior | Eight runtime modules match source; complete public BF16 report equals the source report; three native examples work outside the checkout |
| Streaming | Native 16,777,216-weight case agrees with a separate dense reference; 1 MiB block and retained-factor limits succeed; about 37 MiB process peak RSS |
| Clean source checks | Python 3.10.21 and 3.15.0: locked install, ruff check/format, mypy, 150 tests and build all pass |
| Package contents | Both wheels match source runtime hashes and are byte-identical; both sdists retain exact native example bytes and have identical file payloads |
| Archive wheel consumer | Fresh Python 3.10 wheel install runs the wrong-orientation example with isolated imports outside the checkout |
| Screenshot | [Actual CLI output](assets/conv1d.png), inspected for readable names, measurements and limitations |
| Publication | Local commits only; no push, registry release, tag or hosted update |

Build command: `uv build`. Output directory: `dist/`, containing the wheel and
source distribution. There is no website build or deployment in this change.
The two compressed sdists have distinct archive hashes despite equal file
payloads; their separate digests are preserved. The wheels are byte-identical.

The [complete replay](../studies/conv1d/README.md) records commands, producer
versions, independent checks and the native-base serialization. The
[input guide](conv1d.md) lists supported paths and metadata. Unknown/custom
layouts, convolutional adapters, Conv1D DoRA and extra trained biases remain
unsupported. Base and merged tensor names must match; original-base provenance
and task quality are not certified. Memory and timing are single local runs.

Suggested repository description:
`Inspect which LoRA and DoRA updates survive in saved merged safetensors.`
Suggested topics: `lora`, `dora`, `peft`, `safetensors`, `model-merging`, `bfloat16`,
`model-auditing`.

```text
9abce44 test: retain native and public GPT layout audit evidence
318cb53 feat: audit standard GPT Conv1D LoRA in saved coordinates
```

The final documentation commit contains this verification handoff; no runtime
code changes follow the archived implementation.
