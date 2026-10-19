"""Validated safetensors inventories and bounded, read-only tensor slices.

Rust safetensors validates the file layout. The small duplicate-rejecting JSON
reader retains offsets so BF16 can be decoded without installing a tensor runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from safetensors import safe_open

FloatArray = NDArray[np.float64]
RawArray = NDArray[Any]
FLOAT_DTYPES = {"F64", "F32", "F16", "BF16"}
DTYPES = {
    "F64": "<f8",
    "F32": "<f4",
    "F16": "<f2",
    "BF16": "<u2",
    "I64": "<i8",
    "I32": "<i4",
    "I16": "<i2",
    "I8": "i1",
    "U64": "<u8",
    "U32": "<u4",
    "U16": "<u2",
    "U8": "u1",
    "BOOL": "?",
}
MAX_HEADER = 16 * 1024 * 1024


class AuditError(ValueError):
    """Evidence is incomplete or outside the supported arithmetic contract."""


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"Duplicate JSON member: {key}")
        result[key] = value
    return result


def _nonfinite_json(value: str) -> None:
    raise AuditError(f"Nonfinite JSON number: {value}")


def read_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_HEADER:
        raise AuditError(f"JSON exceeds the 16 MiB metadata limit: {path.name}")
    try:
        result = json.loads(
            path.read_text("utf-8"), object_pairs_hook=_unique_pairs, parse_constant=_nonfinite_json
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise AuditError(f"Invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(result, dict):
        raise AuditError(f"Expected a JSON object in {path.name}")
    return result


@dataclass(frozen=True)
class FileStamp:
    size: int
    mtime_ns: int
    inode: int
    device: int
    ctime_ns: int


def stamp(path: Path) -> FileStamp:
    stat = path.stat()
    return FileStamp(stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev, stat.st_ctime_ns)


@dataclass(frozen=True)
class Tensor:
    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int

    @property
    def elements(self) -> int:
        return math.prod(self.shape)

    def raw(self, start: int = 0, count: int | None = None) -> RawArray:
        """Copy a contiguous flattened slice; no view can retain an entire mapping."""
        if count is None:
            count = self.elements - start
        if start < 0 or count < 0 or start + count > self.elements:
            raise AuditError(f"Slice outside tensor {self.name}")
        dtype = np.dtype(DTYPES[self.dtype])
        with self.path.open("rb") as stream:
            stream.seek(self.offset + start * dtype.itemsize)
            data = stream.read(count * dtype.itemsize)
        if len(data) != count * dtype.itemsize:
            raise AuditError(f"Checkpoint changed or was truncated while reading {self.path.name}")
        return np.frombuffer(data, dtype=dtype)

    def floats(self, start: int = 0, count: int | None = None) -> FloatArray:
        if self.dtype not in FLOAT_DTYPES:
            raise AuditError(f"Expected floating weights, got {self.dtype} at {self.name}")
        data = self.raw(start, count)
        if self.dtype == "BF16":
            values = (data.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        else:
            values = data.astype(np.float64)
        if not np.isfinite(values).all():
            raise AuditError(f"Nonfinite weights at {self.name} in {self.path.name}")
        return values


def cast(values: FloatArray, dtype: str) -> FloatArray:
    """Round to nearest, ties to even. BF16 rounds directly from binary64.

    Going through float32 can double-round a value near a BF16 midpoint.
    Keep the reference independent of that intermediate arithmetic choice.
    """
    if dtype == "BF16":
        bits = values.view(np.uint64)
        magnitude = bits & np.uint64(0x7FFFFFFFFFFFFFFF)
        # BF16 has the float32 exponent range and 7 explicit fraction bits.
        # Divide by the local BF16 spacing, round the integer significand,
        # then rescale. Subnormal spacing remains fixed at 2**-133.
        _fraction, exponent = np.frexp(np.abs(values))
        spacing = np.exp2(np.maximum(exponent - 8, -133).astype(np.float64))
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.rint(values / spacing) * spacing
        maximum = float.fromhex("0x1.fep+127")
        result = np.where(np.abs(result) > maximum, np.copysign(np.inf, result), result)
        # Preserve signed zeros. The arithmetic above already preserves them;
        # copying avoids np.where returning a noncontiguous view to callers.
        return np.asarray(np.where(magnitude == 0, values, result), dtype=np.float64)
    if dtype not in FLOAT_DTYPES:
        raise AuditError(f"Unsupported merge dtype: {dtype}")
    with np.errstate(over="ignore", invalid="ignore"):
        return values.astype(DTYPES[dtype]).astype(np.float64)


def _tensors(path: Path) -> dict[str, Tensor]:
    with path.open("rb") as stream:
        size = stream.read(8)
        if len(size) != 8:
            raise AuditError(f"Truncated safetensors header: {path.name}")
        length = struct.unpack("<Q", size)[0]
        if length > MAX_HEADER:
            raise AuditError(f"Safetensors header exceeds 16 MiB: {path.name}")
        try:
            header = json.loads(
                stream.read(length), object_pairs_hook=_unique_pairs, parse_constant=_nonfinite_json
            )
        except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise AuditError(f"Invalid safetensors header: {path.name}") from exc
    result: dict[str, Tensor] = {}
    try:
        with safe_open(path, framework="numpy") as handle:
            for name in handle.keys():
                descriptor = header[name]
                dtype = descriptor["dtype"]
                if dtype not in DTYPES:
                    raise AuditError(f"Unsupported tensor dtype {dtype}: {name}")
                shape = tuple(handle.get_slice(name).get_shape())
                result[name] = Tensor(
                    name, path, dtype, shape, 8 + length + descriptor["data_offsets"][0]
                )
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError(f"Invalid safetensors file {path.name}: {exc}") from exc
    return result


class Checkpoint:
    """One file or an explicitly indexed Hugging Face checkpoint directory."""

    def __init__(self, source: str | Path, *, adapter: bool = False):
        self.source = Path(source).resolve()
        self.tensors: dict[str, Tensor] = {}
        self.files: dict[Path, FileStamp] = {}
        self.config: dict[str, Any] = {}
        expected: dict[str, Any] | None = None
        if self.source.is_file():
            if adapter:
                raise AuditError("The adapter must be a directory containing adapter_config.json")
            paths = [self.source]
        else:
            if not self.source.is_dir():
                raise AuditError(f"Checkpoint does not exist: {self.source}")
            config_path = self.source / ("adapter_config.json" if adapter else "config.json")
            if config_path.exists():
                self.files[config_path] = stamp(config_path)
                self.config = read_json(config_path)
                if self.config.get("quantization_config") is not None:
                    raise AuditError(
                        "Quantized checkpoints are unsupported; supply floating weights"
                    )
            elif adapter:
                raise AuditError("Missing adapter_config.json")
            stem = "adapter_model" if adapter else "model"
            single = self.source / f"{stem}.safetensors"
            index = self.source / f"{stem}.safetensors.index.json"
            if single.exists() and index.exists():
                raise AuditError(f"Ambiguous checkpoint: both {single.name} and {index.name} exist")
            if index.exists():
                self.files[index] = stamp(index)
                expected = read_json(index).get("weight_map")
                if not isinstance(expected, dict) or not expected:
                    raise AuditError(f"Missing or empty weight_map in {index.name}")
                paths = []
                if not all(isinstance(name, str) for name in expected.values()):
                    raise AuditError("Shard filenames must be strings")
                for name in dict.fromkeys(expected.values()):
                    if not isinstance(name, str) or not name.endswith(".safetensors"):
                        raise AuditError("Shard filenames must be local safetensors paths")
                    relative = Path(name)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise AuditError("Shard path escapes its checkpoint directory")
                    # HF cache snapshots use symlinks to sibling blob storage.
                    # Validate the index's logical path, retain it for evidence,
                    # and let ordinary local file reads follow that snapshot link.
                    path = self.source / relative
                    paths.append(path)
            elif single.is_file():
                paths = [single]
            else:
                raise AuditError(f"Missing {stem}.safetensors or its shard index")
        for path in paths:
            self.files[path] = stamp(path)
            for name, tensor in _tensors(path).items():
                if name in self.tensors:
                    raise AuditError(f"Tensor occurs in multiple shards: {name}")
                if expected is not None and expected.get(name) != str(
                    path.relative_to(self.source)
                ):
                    raise AuditError(f"Shard index disagrees with stored tensor: {name}")
                self.tensors[name] = tensor
        if expected is not None and self.tensors.keys() != expected.keys():
            raise AuditError("Shard index lists missing tensors")
        if not self.tensors:
            raise AuditError("Checkpoint contains no tensors")

    def verify_stable(self) -> None:
        for path, initial in self.files.items():
            if stamp(path) != initial:
                raise AuditError(f"Input changed during audit: {path.name}")

    def fingerprints(self) -> list[dict[str, Any]]:
        result = []
        for path in sorted(self.files):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
            label = str(path.relative_to(self.source)) if self.source.is_dir() else path.name
            result.append(
                {"file": label, "bytes": self.files[path].size, "sha256": digest.hexdigest()}
            )
        self.verify_stable()
        return result
