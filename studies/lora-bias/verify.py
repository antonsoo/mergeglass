"""Check native bias merges with a separate stdlib byte reader and Decimal oracle.

No Torch, Transformers or PEFT imports. Original tensor payloads are checked
against the producer manifest before any numerical comparison.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import struct
from decimal import Decimal, localcontext
from functools import lru_cache
from pathlib import Path
from typing import Any

from mergeglass import AuditError, audit


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@lru_cache(maxsize=128)
def file_data(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    size = int.from_bytes(raw[:8], "little")
    return json.loads(raw[8 : 8 + size]), raw[8 + size :]


def payload(root: Path, name: str) -> tuple[str, list[int], bytes]:
    index = root / "model.safetensors.index.json"
    filename = (
        json.loads(index.read_text())["weight_map"][name]
        if index.exists()
        else "adapter_model.safetensors"
        if (root / "adapter_model.safetensors").exists()
        else "model.safetensors"
    )
    header, raw = file_data(root / filename)
    spec = header[name]
    start, end = spec["data_offsets"]
    return spec["dtype"], spec["shape"], raw[start:end]


def values(root: Path, name: str) -> tuple[str, list[float]]:
    dtype, shape, raw = payload(root, name)
    if dtype == "BF16":
        result = [
            struct.unpack("<f", (v[0] << 16).to_bytes(4, "little"))[0]
            for v in struct.iter_unpack("<H", raw)
        ]
    else:
        result = [
            float(v[0])
            for v in struct.iter_unpack({"F32": "<f", "F16": "<e", "F64": "<d"}[dtype], raw)
        ]
    assert len(result) == math.prod(shape)
    return dtype, result


def rounded(value: float, dtype: str) -> float:
    if dtype == "F64":
        return value
    if dtype == "F16":
        return float(struct.unpack("<e", struct.pack("<e", value))[0])
    raw = struct.pack("<f", value)
    if dtype == "F32":
        return float(struct.unpack("<f", raw)[0])
    assert dtype == "BF16"
    bits = int.from_bytes(raw, "little")
    high, low = bits >> 16, bits & 0xFFFF
    high += int(low > 0x8000 or low == 0x8000 and high & 1 == 1)
    return float(struct.unpack("<f", (high << 16).to_bytes(4, "little"))[0])


def norm(items: list[float]) -> float:
    return float(sum((Decimal.from_float(v) ** 2 for v in items), Decimal(0)).sqrt())


def close(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, rel_tol=3e-13, abs_tol=1e-15), (actual, expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--before", action="store_true")
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root / "manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        raw = (root / artifact["file"]).read_bytes()
        assert len(raw) == artifact["bytes"] and sha(raw) == artifact["sha256"]
    results = []
    checked = 0
    with localcontext() as context:
        context.prec = 80
        for profile in manifest["cases"]:
            source = root / profile["profile"]
            for merge in profile["merges"]:
                destination = source / merge["name"]
                identity = {"profile": profile["profile"], "merge": merge["name"]}
                try:
                    report = audit(
                        source / "base",
                        source / "adapter",
                        destination,
                        block_mib=1,
                        max_adapter_mib=1,
                    )
                except AuditError as error:
                    assert args.before and str(error) == "Unsupported adapter option: lora_bias"
                    results.append({**identity, "status": "refused", "reason": str(error)})
                    continue
                assert not args.before
                assert len(report["lora_biases"]) == len(profile["layers"])
                assert report["summary"]["unadapted_changed_tensors"] == 0
                if profile["profile"] == "bias-only":
                    assert report["summary"]["intended_nonzero_elements"] == 0
                    assert report["summary"]["lora_bias_intended_nonzero_elements"] > 0
                oracles = []
                for row in report["lora_biases"]:
                    path = row["module"]
                    captured = profile["layers"][path]
                    config = json.loads((source / "adapter/adapter_config.json").read_text())
                    rank = config["rank_pattern"].get(path, config["r"])
                    alpha = config["alpha_pattern"].get(path, config["lora_alpha"])
                    scale = alpha / (math.sqrt(rank) if config["use_rslora"] else rank)
                    assert scale == captured["scale"] == row["scale"]
                    assert rank == captured["rank"] == row["rank"]
                    _, original = values(source / "base", row["weight"])
                    _, factor = values(source / "adapter", f"base_model.model.{path}.lora_B.bias")
                    dtype, after = values(destination, row["weight"])
                    baseline = [rounded(v, dtype) for v in original]
                    delta = [
                        float(Decimal.from_float(v) * Decimal.from_float(scale)) for v in factor
                    ]
                    observed = [a - b for a, b in zip(after, baseline, strict=True)]
                    residual = [o - d for o, d in zip(observed, delta, strict=True)]
                    target = [
                        rounded(float(Decimal.from_float(b) + Decimal.from_float(d)), dtype)
                        for b, d in zip(original, delta, strict=True)
                    ]
                    nonzero = sum(d != 0 for d in delta)
                    changed = sum(a != b for a, b in zip(after, baseline, strict=True))
                    unchanged = sum(
                        d != 0 and a == b for d, a, b in zip(delta, after, baseline, strict=True)
                    )
                    mismatches = sum(a != t for a, t in zip(after, target, strict=True))
                    assert row["intended_nonzero_elements"] == nonzero
                    assert row["changed_elements"] == changed
                    assert row["unchanged_nonzero_update_elements"] == unchanged
                    assert row["reference_merge_mismatches"] == mismatches
                    reference_norm = norm(delta)
                    residual_norm = norm(residual)
                    close(row["norms"]["delta"], reference_norm)
                    close(row["norms"]["observed"], norm(observed))
                    close(row["norms"]["residual"], residual_norm)
                    gain = float(
                        sum(
                            (
                                Decimal.from_float(d) * Decimal.from_float(o)
                                for d, o in zip(delta, observed, strict=True)
                            ),
                            Decimal(0),
                        )
                        / sum((Decimal.from_float(d) ** 2 for d in delta), Decimal(0))
                    )
                    close(row["projection_gain"], gain)
                    close(row["relative_residual"], residual_norm / reference_norm)
                    first = max(range(len(residual)), key=lambda i: abs(residual[i]))
                    assert row["largest_residual"]["index"] == first
                    assert row["largest_residual"]["intended_delta"] == delta[first]
                    if merge["name"] == "omitted-bias":
                        assert row["status"] == "no-observed-update" and gain == 0
                        assert unchanged == nonzero == len(delta)
                    elif merge["name"] == "unscaled-bias":
                        assert math.isclose(gain, 1 / scale, rel_tol=2e-5)
                    else:
                        assert math.isclose(gain, 1, rel_tol=0.03)
                    checked += len(delta)
                    oracles.append(
                        {
                            "weight": row["weight"],
                            "entries": len(delta),
                            "scale": scale,
                            "gain": gain,
                            "relative_residual": residual_norm / reference_norm,
                            "unchanged": unchanged,
                            "reference_merge_mismatches": mismatches,
                        }
                    )
                if merge["name"] in {"omitted-bias", "unscaled-bias"}:
                    # Every matrix and full saved-module tensor keeps its exact
                    # native correct payload. Only targeted biases were edited.
                    for row in report["layers"] + report.get("saved_modules", []):
                        assert payload(destination, row["weight"]) == payload(
                            source / "correct-f32-safe-false", row["weight"]
                        )
                results.append({**identity, "bias_oracles": oracles, "report": report})
    result = {
        "native_manifest_sha256": sha((root / "manifest.json").read_bytes()),
        "producer_versions": manifest["versions"],
        "verified_artifacts": len(manifest["artifacts"]),
        "decimal_precision": 80,
        "bias_scalar_entries_checked": checked,
        "before": args.before,
        "cases": results,
        "verifier_sha256": sha(Path(__file__).read_bytes()),
    }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".gz":
        args.output.write_bytes(gzip.compress(text.encode(), mtime=0))
    else:
        args.output.write_text(text)
    print(f"Verified {len(results)} saved merges; {checked} independent bias scalar entries.")


if __name__ == "__main__":
    main()
