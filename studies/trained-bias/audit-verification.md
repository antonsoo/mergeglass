# Follow-up audit and publication evidence

The self-audit found and fixed two errors in our earlier evidence/reporting:

* The native study verifier could announce success after a case was removed,
  and its `assert` checks disappeared under `python -O`. It now requires the
  complete case/precision/control matrix, exact manifest coverage and explicit
  checks. Missing cases, altered payloads, mislabeled precision and duplicate
  identities are rejected, including under optimized Python.
* Saved-state change counts were defined after the reference precision cast.
  A source change erased by that cast could disappear from the displayed
  denominator. Reports now retain source saved-change counts, cast-erasure
  counts and concrete first-value witnesses separately. The existing target
  mismatch and post-cast fields keep their original definitions.

The follow-up also supports trained ordinary biases combined with full saved
modules. It validates redundant wrapper copies separately without counting a
physical target twice. Native nested-module copies, integer buffers above
`2**53`, rsLoRA and DoRA are covered by the [saved-head study](../trained-bias-heads/README.md).
The [public adapter replay](../trained-bias-public/README.md) checks the same
runtime against a real legacy-format adapter and isolates a native loader
compatibility failure.

## Reviewed runtime and packages

The reviewed runtime is `45783928e1e53c615880aa83c5ad22f05cf1c280`.
A fresh `git archive` checkout ran `uv sync --locked`, Ruff lint/format,
`mypy src`, all 230 tests and wheel/sdist builds successfully.
No existing environment was copied into that checkout. Later additions are
study scripts, evidence and documentation; runtime source remains identical.

| Installed package | Interpreter | Evidence |
| --- | --- | --- |
| Wheel | Python 3.10.21, NumPy 2.2.6 | Full 230-test suite outside checkout, four complete native oracle corpora under `-O`, omission/precision/refusal CLI controls |
| Sdist | Python 3.15.1, NumPy 2.5.4 | Installed outside checkout, saved-head and precision-loss CLI controls |

[Wheel receipt](results/audit/wheel.json) and
[sdist receipt](results/audit/sdist.json) record package digests, installed
runtime code hashes, interpreter/library versions and actual observed fields.
These are local installed-artifact results, not package-registry releases.

The installed-wheel [wide-head measurement](../trained-bias-heads/wide-wheel.json)
streams an independently checked 4096-by-4096 full saved head. All 16,777,216
head weights are compared for source changes; target and merged payload hashes
match. Fresh CLI processes peaked at 36,580 KiB with a 1 MiB audit block and
50,672 KiB with a 16 MiB block on the recorded Linux/WSL machine. These are
machine observations, not a cross-machine performance promise. The measurement
ran alongside a small oracle process; elapsed times are not a controlled
benchmark. The dense head alone would require 128 MiB as float64.

## Remote state and CI boundary

Commit `45783928e1e53c615880aa83c5ad22f05cf1c280` was pushed normally to
`origin/main`, and the remote reference was read back. The
[public-byte receipt](results/audit/public-bytes.json) independently downloaded
all 68 changed files relative to the prior remote `c877c79` at the immutable
commit URL and compared each with Git blobs: 1,442,301 bytes matched.
This receipt covers that commit, not later documentation or study additions.

[GitHub Actions run 41962862558](https://github.com/antonsoo/mergeglass/actions/runs/41962862558)
completed without executing checks: the Python 3.10 job was cancelled and the
Python 3.15 job failed, both with empty step lists. This is **not passing hosted
CI**. Local clean-install and installed-package results are separate evidence.
No tag or package-registry release was made.
