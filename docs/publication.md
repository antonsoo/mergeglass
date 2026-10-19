# Public source and installed verification at 23ed58c

The [verification receipt](publication.json) identifies source revision
`23ed58c50765fc3720856eb84090a7840ce320c3`, its 58 archived files, runtime module
hashes, clean-build artifacts and three separately installed environments.
This snapshot includes signed update projection and the mixed Conv1D refusal.
Later study/documentation additions have their own source commits; these
artifact identities belong specifically to this archived revision.

| Boundary | Observed result |
| --- | --- |
| Public source | GitHub main and local source agreed at the recorded revision |
| Fresh public clone | README's three-command quickstart worked; the subsequent corrected pull refused the native mixed GPT-2 case with exit 2 |
| Fresh archived source | Quickstart, lint/format, strict mypy, 51 tests and wheel/sdist build completed |
| Independent builds | Archived-source and original-checkout wheel/sdist bytes matched exactly |
| Minimum installed runtime | Python 3.10.21, NumPy 1.24.0, safetensors 0.6.2; 51 contract tests completed |
| Current installed runtime | Python 3.15.0, NumPy 2.5.4, safetensors 0.8.0; no torch runtime dependency |
| Producer installed runtime | Python 3.12.12 with separately pinned torch/PEFT/Transformers dependencies |
| Installed module identity | All six runtime modules matched archived source in all three environments |
| Dense native measurements | Eight original native cases, eight merge controls and both public-adapter merges checked independently |
| Mixed orientation | Three native mixed Conv1D exports refused; three ordinary Linear-head controls accepted |
| Native control reproduction | All 129 files and the complete producer receipt reproduced byte for byte |
| GitHub Actions | Jobs had no executed steps; local checks are the verification result |
| Registry / service | No PyPI release or hosted Mergeglass application created |

The artifact identities are:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `mergeglass-0.1.0-py3-none-any.whl` | 20,144 | `2170d5b0d8af0148cb45ca119f315de738140a71b8a3bc355889477f24bd9481` |
| `mergeglass-0.1.0.tar.gz` | 351,864 | `1f613bc5491ca40c264fb559e9a4c0d3ee95b3caa75755958b36dd330ee9fbbf` |

The git-archive SHA-256 is
`b99d16815efb4f41ef4e09975ea0230e6507163bda1c597f2032bed88ab8ecf7`.
Use `uv build` to produce artifacts in `dist/`; added studies and documentation
can change subsequent source archives, and README changes affect wheel metadata.

The [earlier verification snapshot](verification.md) remains a historical
record of the initial implementation. Neither snapshot treats saved weights,
installed artifacts, public source, CI or generation quality as interchangeable
claims. The [public native study](../studies/community/README.md),
[failure controls](../studies/merge-controls/README.md) and
[orientation study](../studies/orientation/README.md) retain their own evidence.
