"""Independent adjacent-bit-pattern BF16 midpoint oracle, no exponent rounding code."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

from mergeglass.storage import cast


def main() -> None:
    # Every adjacent positive finite BF16 pair, including zero/subnormals.
    # A midpoint of two such dyadic numbers is exactly representable in F64.
    representable = np.array(
        [struct.unpack("<f", struct.pack("<I", bits << 16))[0] for bits in range(0x7F80)],
        dtype=np.float64,
    )
    low, high = representable[:-1], representable[1:]
    midpoint = (low + high) / 2
    ties = np.where(np.arange(len(low)) % 2 == 0, low, high)
    values = np.stack(
        [np.nextafter(midpoint, -np.inf), midpoint, np.nextafter(midpoint, np.inf)], axis=1
    ).ravel()
    expected = np.stack([low, ties, high], axis=1).ravel()
    overflow = float.fromhex("0x1.ffp+127")
    values = np.concatenate(
        [values, [np.nextafter(overflow, 0), overflow, np.nextafter(overflow, np.inf)]]
    )
    expected = np.concatenate([expected, [representable[-1], np.inf, np.inf]])
    values = np.concatenate([values, -values])
    expected = np.concatenate([expected, -expected])
    actual = cast(values, "BF16")
    differences = int(np.count_nonzero(actual != expected))
    zero_sign_differences = int(
        np.count_nonzero((actual == 0) & (np.signbit(actual) != np.signbit(expected)))
    )
    assert differences == 0 and zero_sign_differences == 0
    # This deliberately weaker implementation demonstrates why conversion
    # through F32 cannot serve as the declared direct-F64 reference.
    two_stage = cast(values.astype(np.float32).astype(np.float64), "BF16")
    result = {
        "oracle": "all adjacent finite BF16 bit patterns; exact midpoints and F64 neighbors",
        "cases": len(values),
        "direct_cast_disagreements": differences,
        "zero_sign_disagreements": zero_sign_differences,
        "via_float32_disagreements": int(np.count_nonzero(two_stage != expected)),
    }
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
