# Signed alpha is part of the saved reference

PEFT 0.21.0 accepts a negative finite `lora_alpha` or `alpha_pattern` coefficient,
saves it normally, and applies its sign while merging. Mergeglass originally
refused these native exports because its coefficient validator also imposed a
nonnegative constraint. The corrected validator preserves signed finite alpha;
ranks remain positive integers.

The producer uses the existing synthetic
[two-layer Llama factors](../native-peft/README.md), unchanged byte for byte.
It loads them through native PEFT with two explicit configurations:

| Saved configuration | q_proj coefficient | v_proj override | Rank/scale convention |
| --- | ---: | ---: | --- |
| `negative-global` | -8 | +4 | rsLoRA, including rank-2 and rank-4 targets |
| `negative-pattern` | +8 | -4 | rsLoRA, including rank-2 and rank-4 targets |

Each configuration produces native FP32, FP16 and BF16 merged checkpoints.
All six exports were previously refused. The [before receipt](results/before.json)
retains those refusals and actual auditor module hashes; the
[after receipt](results/after.json) retains the accepted reports and independent
dense PyTorch measurements of every layer. The producer receipt includes every
input/output hash, versions, saved coefficients and proof that the adapter
safetensors bytes match the original factors.

The reference follows the saved sign. A native merge that applies its declared
negative coefficient can have projection gain near 1. The
[reversed merge control](../merge-controls/README.md) supplies the original
positive adapter configuration while reversing the runtime coefficient, giving
gain near -1 instead. These are measurements relative to explicit saved inputs;
neither alpha sign nor a weight sign is itself a quality verdict.

The original factors came from labelled synthetic training. The signed overrides
are deliberate controls, not a claim about the frequency or quality of negative
coefficients in deployed adapters. No pretrained weights or external model code
are used in this study.

## Reproduce

Use the [pinned native environment](../native-peft/README.md), with the current
package installed and its transformer artifacts already produced:

```sh
PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 /tmp/mergeglass-producer/bin/python studies/signed-scale/produce.py /tmp/mergeglass-transformer /tmp/mergeglass-signed
OPENBLAS_NUM_THREADS=1 /tmp/mergeglass-producer/bin/python studies/signed-scale/verify.py /tmp/mergeglass-signed /tmp/mergeglass-signed-verification.json
```

Use a new output directory. To replay the old refusal, install the runtime from
source before the signed-coefficient correction in a separate environment and
run the same verifier with `--before`. The recorded module hashes identify that
baseline. Lightweight cases also check fractional negative alpha, signed pattern
overrides and invalid ranks. The unchanged positive public-adapter reports are
replayed separately to check existing numeric behavior.
