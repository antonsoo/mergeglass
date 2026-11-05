# Follow-up audit, 2026-11-04

## Evidence verifier corrections

The initial verifier had two reproducible gaps. Its comparisons were correct
on the retained complete inputs, but these acceptance paths were too weak:

| Mutation | Initial verifier | Corrected verifier |
| --- | --- | --- |
| Remove `all/reloaded-bf16-safe-true` and its manifest entry | Exit 0; 35 measured cases | Exit 2; incomplete case matrix |
| Change a normal FP32 merged bias, refresh its hash, run `python -O` | Exit 0; 36 measured cases | Exit 2; incorrect native bias target |
| Copy FP32 save over a BF16-labeled case and refresh its hash | Precision label was not checked | Exit 2; stored precision differs |
| Add a producer file without a manifest identity | File was not covered | Exit 2; exact inventory required |
| Duplicate a manifest identity | Could inflate identity count | Exit 2; duplicate identity |

The first mutation exposed discovery by whatever files happened to exist.
The second exposed comparisons implemented with Python `assert`, which the
optimized interpreter removes. The verifier now uses explicit runtime checks,
requires all six native precision/safe-merge combinations and all authored
controls in each profile, and checks actual saved dtypes against their labels.
It requires unique manifest paths covering every producer file, and emits
`state: verified` with expected and measured case counts only after all checks.

`tests/test_trained_bias_study.py` runs the real script under both ordinary and
optimized Python, with complete retained bytes and the mutations above. It
requires failures to exit 2 and leave no success receipt. This guards evidence
completeness; it does not substitute for model or numerical validation.

Both original archives were replayed with the strengthened oracle. All original
numerical result rows are exactly unchanged: 73 measured comparisons, ten
expected refusals, and 2854 independently checked bias entries. The additional
receipt fields make the verification boundary explicit. Existing archives and
previously captured native outputs did not need modification.

## Saved changes that vanish in the reference cast

A further native replay exposed a reporting problem in the original runtime:
`unchanged_expected_change_elements` counts changes after casting the target,
but its CLI label described trained changes without that qualification. A
BF16 output that retained none of twelve one-ulp saved bias changes printed
zero changes left at the base. Zero target mismatches were valid: every saved
value matched the correct rounded target.

The [precision audit](precision.md) retains that before output and a twelve-case
native control. New source-change counts and first-coordinate witnesses expose
the difference between saved changes, erasure in the reference dtype cast,
and actual final equality. Existing post-cast measurements keep their original
meaning. Full saved modules receive the same metrics; missing original state
stays unknown and integer identities remain exact. No producer phase or model
quality is inferred from unchanged values.
