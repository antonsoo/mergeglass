# Direct binary64-to-BF16 rounding

The BF16 reference must not first cast a binary64 sum to float32. A value just
above a BF16 midpoint can round to that midpoint in F32; a second ties-to-even
cast can choose the other BF16 neighbor. The auditor's direct conversion uses
binary64 significands and BF16 spacing instead.

`oracle.py` uses a different construction: enumerate every pair of adjacent
positive finite BF16 bit patterns, decode them with `struct`, construct their
exact binary64 midpoint and the immediately adjacent binary64 values, then
select the expected neighbor by bit parity. Repeat for negative values and
add the finite-to-infinity overflow threshold. This includes zero/subnormal,
subnormal/normal and exponent transitions. Zero signs are checked separately.

| Reference check | Result |
| --- | ---: |
| Deliberately chosen boundary cases | 195,840 |
| Direct cast disagreements | 0 |
| Zero sign disagreements | 0 |
| Float32-intermediate disagreements | 65,280 |

[Recorded output](results/oracle.json). The last count describes these adversarial
boundary cases; it is not an estimated frequency in model weights or hardware.
This verifies the auditor's declared nearest-even conversion, not every hardware
kernel or a producer's intermediate arithmetic.

```sh
uv run python studies/rounding/oracle.py /tmp/mergeglass-rounding.json
```

