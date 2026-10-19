# Local source and installed artifact verification

This historical snapshot's [machine-readable receipt](verification.json) identifies source commit
`ea93d6811eeb141d0ba4e85f976842bb03d5770b`, its git archive, all 47 archived
file hashes, six runtime module hashes and the built artifacts. It predates the
signed projection measurements and [native merge controls](../studies/merge-controls/README.md).
Those later changes have their own verification and artifact identities.

## What was exercised

| Boundary | Observed result |
| --- | --- |
| Fresh archive, no existing virtualenv | README's two-command quickstart works; JSON report written to the local filesystem |
| Fresh source install, Python 3.15 | Ruff, formatting, strict mypy, 37 tests and wheel/sdist build pass |
| Installed wheel, Python 3.10.21 | 37 tests pass with NumPy 1.24.0 and safetensors 0.6.2 |
| Installed wheel, Python 3.15.0 | NumPy 2.5.4 / safetensors 0.8.0; example, direct-rounding oracle and large-tensor benchmark pass |
| Runtime dependency boundary | Isolated installed-package imports find no torch in either lightweight environment |
| Fresh pinned producer install, Python 3.12.12 | Documented separate CPU-torch then PyPI installation succeeds |
| Native producer reproduction | All 8 teaching files and all 50 tiny-Llama files reproduce byte for byte |
| Installed package, native dense oracle | All 8 constructed/native artifact cases agree; report bytes match source-run receipts |
| Installed package, public adapter | Both native merge cases agree with dense PyTorch; report bytes match source-run receipts |
| Minimum dependencies, public adapter | BF16 counts and norms agree with the dense oracle at a different block setting |
| BF16 midpoint oracle | 195,840 cases each on minimum/current dependencies, zero disagreements |
| Independent wheel builds | Original checkout and fresh-archive wheels are byte-identical; the wheel also builds from the source distribution |
| Screenshot | Real unedited CLI output, rendered locally and visually inspected; aligned, legible and labelled as a controlled example |

The source check commands were:

```sh
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv build
```

The fresh quickstart first ran `uv sync --locked --no-dev`, followed by the
README's exact example command. Wheel consumers were installed into separate
environments with `uv pip install ... dist/mergeglass-0.1.0-py3-none-any.whl`.
Isolated imports verified actual site-packages origins and every runtime module's
bytes, preventing the editable checkout from satisfying an artifact check.

The native [replay](../studies/native-peft/README.md),
[public study](../studies/community/README.md),
[rounding oracle](../studies/rounding/README.md) and
[memory measurement](../studies/streaming/README.md) have their own exact commands
and receipts. The public original files are revision-pinned and SHA-256 checked.
They remain outside the repository; reports retain a few scalar residual examples.
No inference-quality result is claimed.

## Artifact identities

These are the artifacts built from the verified source commit above:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `mergeglass-0.1.0-py3-none-any.whl` | 18,665 | `a5491e33d09e27b189bb6f2035e92a51c66a1171fab1c8f1c1656e3f35ecf8d5` |
| `mergeglass-0.1.0.tar.gz` | 300,668 | `a4f983794dfdf27bf3d90cc21fa5f40ad9bdcdb2263ba501085927f446614fcf` |

The source git-archive SHA-256 is
`760c5cb8e60c73dd6b6b66a92756a84dc88c76d25a0647bb959a4430b9d12768`.
Later source-distribution builds include these additional verification documents
and the installed benchmark receipt, so their archive hashes can differ. The
unchanged runtime modules and README leave the wheel bytes reproducible.

Build with `uv build`; artifacts appear in `dist/`. This CLI has no static-site
build or browser application. Its [terminal screenshot](assets/rounding.png)
uses the [captured output](assets/rounding.txt) from the bundled synthetic control.

## Publication boundary

At this snapshot, local commits, clean installations and native artifacts were verified. No Git
remote, push, tag, registry release or hosted demo was created. The CI workflow
is present, but this is a local verification result, not a GitHub Actions claim.

Suggested repository description:

> Inspect which LoRA updates survive saved safetensors merges, with bounded memory and reproducible evidence.

Suggested topics: `lora`, `peft`, `safetensors`, `model-merging`, `bfloat16`, `python`.
