#!/usr/bin/env python3
# tb_canonical.py
r"""
Tensorized Blockchain canonical serialization
============================================

Project Chimera / Akkurat - Tensorized Blockchain canonicalization layer.

Purpose
-------
A blockchain stack cannot safely hash arbitrary Python objects unless every
module uses the same deterministic byte representation. This module owns that
shared representation for JSON-like objects, dataclasses, NumPy arrays, byte
payloads, floats, schema-versioned payloads, and domain-separated hashes.

Design goals
------------
- Dependency-light: stdlib only, with optional NumPy support when installed.
- Import-safe: no imports from project modules, no side effects at import time.
- Deterministic: canonical JSON uses sorted keys and compact separators.
- Explicit float policy: non-finite floats are rejected by default.
- Array-aware: NumPy arrays are represented by shape, dtype, order, and digest.
- Schema-aware: payloads can be wrapped in deterministic schema envelopes.
- Hash-ready: domain-separated SHA-256 helpers for commitments and manifests.

Public API
----------
- CanonicalConfig
- CanonicalSchemaPayload
- canonical_json_safe(...)
- canonical_dumps(...)
- canonical_bytes(...)
- canonical_hash(...)
- domain_hash(...)
- canonical_array_payload(...)
- canonical_array_bytes(...)
- normalize_schema_payload(...)
- verify_canonical_hash(...)
- merkle_root_hex(...)
- hash_file(...)

This module intentionally does not know about transactions, twins, ledgers,
TensorTrains, or lobes. Higher layers should use this module rather than local
serialization helpers when producing consensus-critical hashes.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import enum
import hashlib
import json
import math
import os
import pathlib
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

try:  # Optional dependency. The stack normally has NumPy, but canonicalization should import without it.
    import numpy as np  # type: ignore
    _NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover - only used in minimal Python environments
    np = None  # type: ignore
    _NUMPY_AVAILABLE = False


# =============================================================================
# Version constants
# =============================================================================

TB_CANONICAL_VERSION = "tb-canonical-v1.0-production"
CANONICAL_JSON_SCHEME = "tb.canonical-json.v1"
CANONICAL_BYTES_SCHEME = "tb.canonical-bytes.v1"
CANONICAL_ARRAY_SCHEME = "tb.canonical-array.v1"
CANONICAL_SCHEMA_ENVELOPE = "tb.schema-envelope.v1"
DEFAULT_HASH_ALGORITHM = "sha256"


# =============================================================================
# Errors / enums / config
# =============================================================================

class CanonicalError(ValueError):
    """Raised when a payload cannot be converted to a canonical representation."""


class NonFiniteFloatPolicy(str, Enum):
    """Policy for NaN / ±Infinity encountered during canonicalization."""

    REJECT = "reject"
    STRING = "string"
    ZERO = "zero"


class BytesEncoding(str, Enum):
    """How raw bytes are represented in canonical JSON-safe payloads."""

    BASE64 = "base64"
    HEX = "hex"
    DIGEST_ONLY = "digest_only"


class ArrayEncoding(str, Enum):
    """How arrays are represented in canonical JSON-safe payloads."""

    DIGEST = "digest"
    VALUES = "values"
    DIGEST_AND_VALUES = "digest_and_values"


class UnknownObjectPolicy(str, Enum):
    """Policy for objects that are not directly JSON/dataclass/array safe."""

    REJECT = "reject"
    STRING = "string"
    DICT = "dict"


@dataclass(frozen=True)
class CanonicalConfig:
    """Canonical serialization policy.

    Defaults are intentionally strict enough for consensus-critical hashing:
    non-finite floats are rejected, unknown objects are rejected, NumPy arrays
    are committed by shape/dtype/digest, and float64 is used as the canonical
    numeric storage dtype for arrays unless a caller overrides it.
    """

    schema: str = CANONICAL_JSON_SCHEME
    nonfinite_float_policy: str = NonFiniteFloatPolicy.REJECT.value
    bytes_encoding: str = BytesEncoding.BASE64.value
    array_encoding: str = ArrayEncoding.DIGEST.value
    canonical_array_dtype: str = "float64"
    array_order: str = "C"
    unknown_object_policy: str = UnknownObjectPolicy.REJECT.value
    sort_keys: bool = True
    ensure_ascii: bool = False
    include_type_tags: bool = True
    include_array_preview: bool = False
    array_preview_items: int = 16
    reject_negative_zero: bool = False
    max_depth: int = 128
    max_file_bytes: int = 512 * 1024 * 1024

    def normalized(self) -> "CanonicalConfig":
        nf = str(self.nonfinite_float_policy or NonFiniteFloatPolicy.REJECT.value).lower().strip()
        if nf not in {x.value for x in NonFiniteFloatPolicy}:
            nf = NonFiniteFloatPolicy.REJECT.value

        be = str(self.bytes_encoding or BytesEncoding.BASE64.value).lower().strip()
        if be not in {x.value for x in BytesEncoding}:
            be = BytesEncoding.BASE64.value

        ae = str(self.array_encoding or ArrayEncoding.DIGEST.value).lower().strip()
        if ae not in {x.value for x in ArrayEncoding}:
            ae = ArrayEncoding.DIGEST.value

        up = str(self.unknown_object_policy or UnknownObjectPolicy.REJECT.value).lower().strip()
        if up not in {x.value for x in UnknownObjectPolicy}:
            up = UnknownObjectPolicy.REJECT.value

        order = str(self.array_order or "C").upper().strip()
        if order not in {"C", "F"}:
            order = "C"

        dtype = str(self.canonical_array_dtype or "float64").lower().strip()
        return CanonicalConfig(
            schema=str(self.schema or CANONICAL_JSON_SCHEME),
            nonfinite_float_policy=nf,
            bytes_encoding=be,
            array_encoding=ae,
            canonical_array_dtype=dtype,
            array_order=order,
            unknown_object_policy=up,
            sort_keys=bool(self.sort_keys),
            ensure_ascii=bool(self.ensure_ascii),
            include_type_tags=bool(self.include_type_tags),
            include_array_preview=bool(self.include_array_preview),
            array_preview_items=int(max(0, self.array_preview_items)),
            reject_negative_zero=bool(self.reject_negative_zero),
            max_depth=int(max(1, self.max_depth)),
            max_file_bytes=int(max(1, self.max_file_bytes)),
        )


DEFAULT_CONFIG = CanonicalConfig().normalized()


@dataclass(frozen=True)
class CanonicalSchemaPayload:
    """Deterministic schema envelope for consensus-critical payloads."""

    schema: str
    version: str
    payload: Any
    domain: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, config: CanonicalConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
        cfg = config.normalized()
        out: Dict[str, Any] = {
            "canonical_envelope": CANONICAL_SCHEMA_ENVELOPE,
            "schema": str(self.schema),
            "version": str(self.version),
            "payload": canonical_json_safe(self.payload, config=cfg),
        }
        if self.domain:
            out["domain"] = str(self.domain)
        if self.metadata:
            out["metadata"] = canonical_json_safe(self.metadata, config=cfg)
        out["payload_hash"] = canonical_hash(out["payload"], config=cfg)
        return out

    def hash(self, *, config: CanonicalConfig = DEFAULT_CONFIG) -> str:
        return canonical_hash(self.to_dict(config=config), config=config)


# =============================================================================
# Low-level helpers
# =============================================================================

def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def _is_negative_zero(value: float) -> bool:
    try:
        return value == 0.0 and math.copysign(1.0, value) < 0.0
    except Exception:
        return False


def normalize_float(value: Any, *, config: CanonicalConfig = DEFAULT_CONFIG) -> Union[int, float, str]:
    """Normalize a scalar float according to the configured float policy."""
    cfg = config.normalized()
    try:
        v = float(value)
    except Exception as exc:
        raise CanonicalError(f"Cannot coerce float value {value!r}: {exc}") from exc

    if not math.isfinite(v):
        policy = NonFiniteFloatPolicy(cfg.nonfinite_float_policy)
        if policy == NonFiniteFloatPolicy.REJECT:
            raise CanonicalError(f"Non-finite float is not canonical: {value!r}")
        if policy == NonFiniteFloatPolicy.ZERO:
            return 0.0
        if math.isnan(v):
            return "NaN"
        return "Infinity" if v > 0 else "-Infinity"

    if cfg.reject_negative_zero and _is_negative_zero(v):
        raise CanonicalError("Negative zero is rejected by this canonical policy.")
    if _is_negative_zero(v):
        v = 0.0

    # JSON can represent integers more compactly and deterministically.
    if v.is_integer() and abs(v) <= 9_007_199_254_740_991:  # JS safe integer boundary for UI compatibility.
        return int(v)
    return float(v)


def _canonical_bytes_payload(data: bytes, *, config: CanonicalConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
    cfg = config.normalized()
    raw = bytes(data)
    enc = BytesEncoding(cfg.bytes_encoding)
    payload: Dict[str, Any] = {
        "__tb_type__": "bytes",
        "schema": CANONICAL_BYTES_SCHEME,
        "length": int(len(raw)),
        "sha256": _sha256_bytes(raw),
    }
    if enc == BytesEncoding.BASE64:
        payload["encoding"] = "base64"
        payload["data"] = base64.b64encode(raw).decode("ascii")
    elif enc == BytesEncoding.HEX:
        payload["encoding"] = "hex"
        payload["data"] = raw.hex()
    else:
        payload["encoding"] = "digest_only"
    return payload


def _dtype_name(dtype: Any) -> str:
    if not _NUMPY_AVAILABLE:
        return str(dtype)
    try:
        return str(np.dtype(dtype))
    except Exception:
        return str(dtype)


def canonical_array_bytes(array: Any, *, config: CanonicalConfig = DEFAULT_CONFIG) -> bytes:
    """Return deterministic bytes for a NumPy-like array.

    The bytes include only the canonical element buffer, not shape/dtype metadata.
    Use canonical_array_payload(...) or canonical_hash(...) when metadata must be
    committed as well.
    """
    if not _NUMPY_AVAILABLE:
        raise CanonicalError("NumPy is required for canonical array bytes.")
    cfg = config.normalized()
    try:
        arr = np.asarray(array)
    except Exception as exc:
        raise CanonicalError(f"Cannot convert object to NumPy array: {exc}") from exc

    if arr.dtype.kind in {"f", "c"}:
        # Complex arrays are represented as real/imag pairs in stable float dtype.
        if arr.dtype.kind == "c":
            arr = np.stack([arr.real, arr.imag], axis=-1)
        arr = np.asarray(arr, dtype=np.dtype(cfg.canonical_array_dtype))
        if not np.isfinite(arr).all():
            policy = NonFiniteFloatPolicy(cfg.nonfinite_float_policy)
            if policy == NonFiniteFloatPolicy.REJECT:
                raise CanonicalError("Non-finite values in array are not canonical.")
            if policy == NonFiniteFloatPolicy.ZERO:
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                # String policy is not valid for raw array bytes; use JSON values instead.
                raise CanonicalError("String non-finite float policy is incompatible with raw array bytes.")
    elif arr.dtype.kind in {"i", "u", "b"}:
        arr = np.asarray(arr)
    elif arr.dtype.kind in {"U", "S", "O"}:
        # Object/string arrays are canonicalized through JSON, then encoded as UTF-8.
        safe = canonical_json_safe(arr.tolist(), config=cfg)
        return canonical_bytes(safe, config=cfg)
    else:
        arr = np.asarray(arr)

    # Native little-endian canonical storage.
    if arr.dtype.byteorder == ">":
        arr = arr.byteswap().newbyteorder("<")
    elif arr.dtype.byteorder == "=":
        if sys.byteorder == "big":
            arr = arr.byteswap().newbyteorder("<")

    arr = np.asfortranarray(arr) if cfg.array_order == "F" else np.ascontiguousarray(arr)
    return arr.tobytes(order=cfg.array_order)


def canonical_array_payload(array: Any, *, config: CanonicalConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Return a JSON-safe canonical representation for a NumPy-like array."""
    if not _NUMPY_AVAILABLE:
        raise CanonicalError("NumPy is required for canonical array payloads.")
    cfg = config.normalized()
    arr_original = np.asarray(array)

    arr_for_meta = arr_original
    if arr_for_meta.dtype.kind == "c":
        arr_for_meta = np.stack([arr_for_meta.real, arr_for_meta.imag], axis=-1)

    raw = canonical_array_bytes(arr_original, config=cfg)
    digest = _sha256_bytes(raw)

    payload: Dict[str, Any] = {
        "__tb_type__": "ndarray",
        "schema": CANONICAL_ARRAY_SCHEME,
        "shape": [int(x) for x in list(arr_for_meta.shape)],
        "original_dtype": _dtype_name(arr_original.dtype),
        "canonical_dtype": _dtype_name(np.asarray(arr_for_meta).dtype),
        "order": cfg.array_order,
        "size": int(arr_for_meta.size),
        "nbytes": int(len(raw)),
        "sha256": digest,
        "encoding": cfg.array_encoding,
    }

    if arr_original.dtype.kind in {"f", "c"}:
        payload["canonical_dtype"] = str(np.dtype(cfg.canonical_array_dtype))
    elif arr_original.dtype.kind in {"U", "S", "O"}:
        payload["canonical_dtype"] = "json-utf8"

    encoding = ArrayEncoding(cfg.array_encoding)
    if encoding in {ArrayEncoding.VALUES, ArrayEncoding.DIGEST_AND_VALUES}:
        if arr_original.dtype.kind == "c":
            values = np.stack([arr_original.real, arr_original.imag], axis=-1).tolist()
        else:
            values = arr_original.tolist()
        payload["values"] = canonical_json_safe(values, config=cfg)

    if cfg.include_array_preview and encoding == ArrayEncoding.DIGEST:
        flat = np.asarray(arr_original).reshape(-1)
        preview = flat[: int(cfg.array_preview_items)].tolist()
        payload["preview"] = canonical_json_safe(preview, config=cfg)

    return payload


def _call_to_dict(obj: Any) -> Any:
    for name in ("to_canonical", "to_dict", "model_dump", "dict", "describe"):
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:
                try:
                    return fn(exclude_none=False)
                except Exception:
                    continue
            except Exception:
                continue
    raise CanonicalError(f"Object of type {type(obj).__name__} has no usable mapping conversion.")


# =============================================================================
# Canonical JSON conversion
# =============================================================================

def canonical_json_safe(
    obj: Any,
    *,
    config: CanonicalConfig = DEFAULT_CONFIG,
    _depth: int = 0,
) -> Any:
    """Convert obj into a deterministic JSON-safe structure.

    The output contains only JSON primitives, lists, and dictionaries, plus
    explicit type-tagged dictionaries for bytes, arrays, paths, decimals, and
    schema envelopes when needed.
    """
    cfg = config.normalized()
    if _depth > cfg.max_depth:
        raise CanonicalError(f"Maximum canonicalization depth exceeded: {cfg.max_depth}")

    if obj is None or isinstance(obj, (str, bool, int)):
        return obj

    if isinstance(obj, float):
        return normalize_float(obj, config=cfg)

    if isinstance(obj, Decimal):
        # Preserve Decimal exactly as a decimal string; do not silently cast to binary float.
        return {
            "__tb_type__": "decimal",
            "value": format(obj, "f"),
        } if cfg.include_type_tags else format(obj, "f")

    if isinstance(obj, Enum):
        return canonical_json_safe(obj.value, config=cfg, _depth=_depth + 1)

    if isinstance(obj, bytes):
        return _canonical_bytes_payload(obj, config=cfg)

    if isinstance(obj, bytearray):
        return _canonical_bytes_payload(bytes(obj), config=cfg)

    if isinstance(obj, memoryview):
        return _canonical_bytes_payload(obj.tobytes(), config=cfg)

    if isinstance(obj, pathlib.Path):
        # Keep paths as strings only. Higher layers decide whether absolute paths are safe.
        return str(obj.as_posix())

    if _NUMPY_AVAILABLE:
        if isinstance(obj, np.generic):
            return canonical_json_safe(obj.item(), config=cfg, _depth=_depth + 1)
        if isinstance(obj, np.ndarray):
            return canonical_array_payload(obj, config=cfg)

    if is_dataclass(obj):
        return canonical_json_safe(asdict(obj), config=cfg, _depth=_depth + 1)

    if isinstance(obj, Mapping):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            key = str(k)
            out[key] = canonical_json_safe(v, config=cfg, _depth=_depth + 1)
        return {k: out[k] for k in sorted(out.keys())} if cfg.sort_keys else out

    if isinstance(obj, (list, tuple)):
        return [canonical_json_safe(v, config=cfg, _depth=_depth + 1) for v in obj]

    if isinstance(obj, set):
        items = [canonical_json_safe(v, config=cfg, _depth=_depth + 1) for v in obj]
        # Sets are unordered; sort their canonical string representation.
        return sorted(items, key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False))

    policy = UnknownObjectPolicy(cfg.unknown_object_policy)
    if policy == UnknownObjectPolicy.STRING:
        return str(obj)
    if policy == UnknownObjectPolicy.DICT:
        try:
            return canonical_json_safe(_call_to_dict(obj), config=cfg, _depth=_depth + 1)
        except Exception:
            return str(obj)
    raise CanonicalError(f"Object of type {type(obj).__name__} is not canonical JSON safe.")


def canonical_dumps(obj: Any, *, config: CanonicalConfig = DEFAULT_CONFIG) -> str:
    """Return canonical JSON text for obj."""
    cfg = config.normalized()
    safe = canonical_json_safe(obj, config=cfg)
    try:
        return json.dumps(
            safe,
            sort_keys=bool(cfg.sort_keys),
            separators=(",", ":"),
            ensure_ascii=bool(cfg.ensure_ascii),
            allow_nan=False,
        )
    except ValueError as exc:
        raise CanonicalError(f"Failed to dump canonical JSON: {exc}") from exc


def canonical_bytes(obj: Any, *, config: CanonicalConfig = DEFAULT_CONFIG) -> bytes:
    """Return UTF-8 canonical JSON bytes for obj."""
    return canonical_dumps(obj, config=config).encode("utf-8")


def canonical_hash(obj: Any, *, config: CanonicalConfig = DEFAULT_CONFIG, algorithm: str = DEFAULT_HASH_ALGORITHM) -> str:
    """Hash obj after canonical JSON serialization."""
    alg = str(algorithm or DEFAULT_HASH_ALGORITHM).lower().strip()
    data = canonical_bytes(obj, config=config)
    if alg != "sha256":
        try:
            h = hashlib.new(alg)
        except Exception as exc:
            raise CanonicalError(f"Unsupported hash algorithm: {algorithm}") from exc
        h.update(data)
        return h.hexdigest()
    return hashlib.sha256(data).hexdigest()


def verify_canonical_hash(
    obj: Any,
    expected_hash: str,
    *,
    config: CanonicalConfig = DEFAULT_CONFIG,
    algorithm: str = DEFAULT_HASH_ALGORITHM,
) -> bool:
    """Constant-time verification of a canonical object hash."""
    actual = canonical_hash(obj, config=config, algorithm=algorithm)
    return _constant_time_equal(actual, str(expected_hash))


def _constant_time_equal(a: str, b: str) -> bool:
    try:
        import hmac
        return hmac.compare_digest(str(a), str(b))
    except Exception:
        return str(a) == str(b)


def domain_hash(
    domain: str,
    payload: Any,
    *,
    config: CanonicalConfig = DEFAULT_CONFIG,
    algorithm: str = DEFAULT_HASH_ALGORITHM,
) -> str:
    """Domain-separated hash used for commitments and protocol subobjects."""
    envelope = {
        "canonical_scheme": CANONICAL_JSON_SCHEME,
        "domain": str(domain),
        "payload": canonical_json_safe(payload, config=config),
    }
    return canonical_hash(envelope, config=config, algorithm=algorithm)


def normalize_schema_payload(
    *,
    schema: str,
    version: str,
    payload: Any,
    domain: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
    config: CanonicalConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """Create a canonical schema-versioned envelope with embedded payload hash."""
    obj = CanonicalSchemaPayload(
        schema=str(schema),
        version=str(version),
        payload=payload,
        domain=str(domain or ""),
        metadata=dict(metadata or {}),
    )
    return obj.to_dict(config=config)


def schema_payload_hash(
    *,
    schema: str,
    version: str,
    payload: Any,
    domain: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
    config: CanonicalConfig = DEFAULT_CONFIG,
) -> str:
    return canonical_hash(
        normalize_schema_payload(schema=schema, version=version, payload=payload, domain=domain, metadata=metadata, config=config),
        config=config,
    )


# =============================================================================
# Merkle helpers
# =============================================================================

def _hex_to_bytes(value: str) -> bytes:
    s = str(value).lower().strip()
    try:
        return bytes.fromhex(s)
    except Exception:
        return hashlib.sha256(s.encode("utf-8", errors="ignore")).digest()


def merkle_leaf_hash(value: Any, *, domain: str = "TB:MERKLE:LEAF:v1", config: CanonicalConfig = DEFAULT_CONFIG) -> str:
    data_hash = canonical_hash(value, config=config)
    return hashlib.sha256(str(domain).encode("utf-8") + b"\x00" + _hex_to_bytes(data_hash)).hexdigest()


def merkle_node_hash(left_hex: str, right_hex: str, *, domain: str = "TB:MERKLE:NODE:v1") -> str:
    return hashlib.sha256(str(domain).encode("utf-8") + b"\x00" + _hex_to_bytes(left_hex) + _hex_to_bytes(right_hex)).hexdigest()


def merkle_root_hex(leaves: Sequence[Any], *, config: CanonicalConfig = DEFAULT_CONFIG) -> str:
    """Return a deterministic domain-separated SHA-256 Merkle root."""
    if not leaves:
        return hashlib.sha256(b"TB:MERKLE:EMPTY:v1").hexdigest()
    level = [merkle_leaf_hash(x, config=config) for x in list(leaves)]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [merkle_node_hash(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def merkle_proof(leaves: Sequence[Any], index: int, *, config: CanonicalConfig = DEFAULT_CONFIG) -> List[Dict[str, str]]:
    """Return a sibling proof for a leaf index."""
    if not leaves:
        raise CanonicalError("Cannot build a Merkle proof for an empty leaf set.")
    idx = int(index)
    if idx < 0 or idx >= len(leaves):
        raise IndexError("Merkle proof index out of range.")
    level = [merkle_leaf_hash(x, config=config) for x in list(leaves)]
    proof: List[Dict[str, str]] = []
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        sibling_idx = idx ^ 1
        side = "right" if idx % 2 == 0 else "left"
        proof.append({"side": side, "hash": level[sibling_idx]})
        idx //= 2
        level = [merkle_node_hash(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return proof


def verify_merkle_proof(leaf: Any, proof: Sequence[Mapping[str, str]], root_hex: str, *, config: CanonicalConfig = DEFAULT_CONFIG) -> bool:
    cur = merkle_leaf_hash(leaf, config=config)
    for item in proof:
        side = str(item.get("side", "")).lower().strip()
        h = str(item.get("hash", ""))
        if side == "left":
            cur = merkle_node_hash(h, cur)
        elif side == "right":
            cur = merkle_node_hash(cur, h)
        else:
            return False
    return _constant_time_equal(cur, str(root_hex))


# =============================================================================
# File helpers
# =============================================================================

def hash_file(
    path: Union[str, os.PathLike[str]],
    *,
    max_bytes: Optional[int] = None,
    algorithm: str = DEFAULT_HASH_ALGORITHM,
) -> Dict[str, Any]:
    """Hash a local file stream with size guard.

    Higher layers must still decide whether the path is allowed. This helper only
    enforces a maximum size and returns a canonical manifest fragment.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(str(p))
    limit = int(DEFAULT_CONFIG.max_file_bytes if max_bytes is None else max_bytes)
    size = int(p.stat().st_size)
    if size > limit:
        raise CanonicalError(f"File exceeds canonical hashing size limit: {size} > {limit}")
    alg = str(algorithm or DEFAULT_HASH_ALGORITHM).lower().strip()
    h = hashlib.new(alg)
    with p.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return {
        "schema": "tb.file-hash.v1",
        "algorithm": alg,
        "size_bytes": size,
        "digest": h.hexdigest(),
        "filename": p.name,
    }


# =============================================================================
# Status / CLI / self-test
# =============================================================================

def canonical_status() -> Dict[str, Any]:
    return {
        "ok": True,
        "version": TB_CANONICAL_VERSION,
        "canonical_json_scheme": CANONICAL_JSON_SCHEME,
        "canonical_array_scheme": CANONICAL_ARRAY_SCHEME,
        "numpy_available": bool(_NUMPY_AVAILABLE),
        "default_config": canonical_json_safe(DEFAULT_CONFIG, config=DEFAULT_CONFIG),
        "python": sys.version.split()[0],
        "ts": now_iso_utc(),
    }


def _self_test() -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG
    a = {"b": 2, "a": [1, 2, 3.0], "z": b"abc"}
    h1 = canonical_hash(a, config=cfg)
    h2 = canonical_hash({"z": b"abc", "a": [1, 2, 3.0], "b": 2}, config=cfg)
    if h1 != h2:
        raise AssertionError("Canonical key ordering failed.")

    try:
        canonical_hash({"bad": float("nan")}, config=cfg)
        raise AssertionError("NaN should be rejected by default.")
    except CanonicalError:
        pass

    envelope = normalize_schema_payload(schema="demo.payload", version="v1", payload=a, domain="self-test", config=cfg)
    if not verify_canonical_hash(envelope, canonical_hash(envelope, config=cfg), config=cfg):
        raise AssertionError("Canonical hash verification failed.")

    leaves = [{"i": 0}, {"i": 1}, {"i": 2}]
    root = merkle_root_hex(leaves, config=cfg)
    proof = merkle_proof(leaves, 1, config=cfg)
    if not verify_merkle_proof(leaves[1], proof, root, config=cfg):
        raise AssertionError("Merkle proof verification failed.")

    array_result: Dict[str, Any] = {"tested": False}
    if _NUMPY_AVAILABLE:
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        p = canonical_array_payload(arr, config=cfg)
        if p["shape"] != [2, 2] or len(p["sha256"]) != 64:
            raise AssertionError("Array canonical payload failed.")
        array_result = {"tested": True, "array_hash": p["sha256"], "shape": p["shape"]}

    return {
        "ok": True,
        "version": TB_CANONICAL_VERSION,
        "hash_stability": h1,
        "schema_hash": canonical_hash(envelope, config=cfg),
        "merkle_root": root,
        "array": array_result,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tensorized Blockchain canonical serialization utility")
    p.add_argument("--mode", choices=["status", "self-test", "hash-json", "hash-file"], default="self-test")
    p.add_argument("--json", default="", help="JSON string to canonicalize/hash in hash-json mode.")
    p.add_argument("--path", default="", help="Path to hash in hash-file mode.")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        if args.mode == "status":
            out = canonical_status()
        elif args.mode == "self-test":
            out = _self_test()
        elif args.mode == "hash-json":
            if not args.json:
                raise ValueError("--json is required for hash-json mode")
            payload = json.loads(args.json)
            out = {
                "ok": True,
                "canonical_json": canonical_dumps(payload),
                "sha256": canonical_hash(payload),
            }
        elif args.mode == "hash-file":
            if not args.path:
                raise ValueError("--path is required for hash-file mode")
            out = {"ok": True, "file": hash_file(args.path)}
        else:
            raise ValueError(f"unknown mode: {args.mode}")
        print(json.dumps(canonical_json_safe(out, config=DEFAULT_CONFIG), indent=2 if args.pretty else None, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        err = {"ok": False, "error": repr(exc), "version": TB_CANONICAL_VERSION}
        print(json.dumps(err, indent=2, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
