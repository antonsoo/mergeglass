# Installed verification of the public summary build

This receipt belongs to source
[`e0eca864fd03f4dce907eec9e622940d07e5a2f8`](https://github.com/antonsoo/mergeglass/tree/e0eca864fd03f4dce907eec9e622940d07e5a2f8),
which includes finite signed alpha support, the native Conv1D guard,
reference-energy measurements and the complete-audit summary view.
The [machine-readable receipt](final-verification.json) retains the source
archive hash, artifact hashes and exact installed module identities.
Later documentation commits have different source archives; the artifact
identities here belong specifically to this revision.

| Boundary | Observed result |
| --- | --- |
| Public source | Remote main agreed with the recorded full revision |
| Public checkout | Pulled that revision and completed the README quickstart plus summary CLI |
| Fresh archived source | Locked runtime-only quickstart, dev sync, lint/format, strict mypy, 62 cases and build completed |
| Independent builds | Original-checkout and clean-archive wheel/sdist bytes matched exactly |
| Installed content | All ten wheel content files matched in three environments; installer-generated `RECORD` excluded |
| Minimum runtime | Python 3.10.21, NumPy 1.24.0, safetensors 0.6.2, no PyTorch; 62 cases completed |
| Current runtime | Python 3.15.0, NumPy 2.5.4, safetensors 0.8.0, no PyTorch |
| Producer runtime | Python 3.12.12, separately pinned native producer dependencies |
| Signed coefficients | All six native global/per-layer signed-scale cases independently checked against dense PyTorch |
| Target orientation | Three mixed native Conv1D exports refused; three ordinary Linear controls accepted |
| Public summary JSON | All 60 adapted matrices and 272 tensors retained; same-runtime full/summary JSON bytes identical |
| Minimum public JSON | Every non-floating value identical; 1,202 floating scalars agree at relative 1e-12 / absolute 1e-15 |
| Public process memory | 62,708 KiB peak RSS while checking 540,004,492 input-file bytes, PyTorch absent |
| GitHub Actions | Recorded run completed with failure; both jobs have zero executed steps. Local verification is recorded separately |
| Registry / service | No PyPI release or hosted Mergeglass application created |

The greatest relative scalar difference in the cross-runtime public comparison
was `7.486219198679733e-16`. NumPy versions differ; this comparison does not claim
bit-identical floating arithmetic across environments. Counts, original
coordinates, statuses and file hashes agree exactly.

| Built file | Bytes | SHA-256 |
| --- | ---: | --- |
| `mergeglass-0.1.0-py3-none-any.whl` | 21,535 | `bf42c1b7aa2097b23be088dd4bf0196c518d84abd23684e343bf762c321dcc12` |
| `mergeglass-0.1.0.tar.gz` | 627,252 | `1433834e44b334fb712eef48f5a50e5759ee60040d8f0b8c6a3a68ca3127077b` |

The 78-file git archive has SHA-256
`c7d9df56a4fce026355054d2fdda09770a3a9e57fb53daf6624a8f9a9c201e31`.
Build with `uv build` at the pinned revision to reproduce those identities.
The memory result is a single shared-host observation, not a hard process limit.

The [native public study](../studies/community/README.md),
[actual forward measurements](../studies/forward/README.md),
[matched attention comparison](../studies/forward/attention.md),
[signed-scale replays](../studies/signed-scale/README.md) and
[memory method](../studies/streaming/README.md) retain their producer inputs and
reproduction commands. The [earlier publication receipts](publication.md) remain
historical snapshots rather than identities of this later build.
