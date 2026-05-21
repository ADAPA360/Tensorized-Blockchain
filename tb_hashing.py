#!/usr/bin/env python3
# tb_hashing.py
r"""
Tensorized Blockchain hashing and Merkle utilities
=================================================

Project Chimera / Akkurat - Tensorized Blockchain cryptographic hashing layer.

Purpose
-------
This module owns domain-separated cryptographic hashing, Merkle tree
construction, inclusion proofs, hash manifests, and state-root helpers for the
Tensorized Blockchain stack. It is intentionally dependency-light and safe for
use by low-level modules without creating circular imports.

Design goals
------------
- Domain separation everywhere: no raw SHA-256 concatenation ambiguity.
- Canonical object hashing through tb_canonical when available.
- SHA-256 as the mandatory baseline algorithm.
- Optional BLAKE3 support when the third-party package is installed.
- Deterministic Merkle roots, proofs, and verification.
- Explicit schema/version tags for long-lived ledger commitments.
- Small, auditable API suitable for blocks, transactions, state roots, tensor
  commitments, source manifests, lobe receipts, and checkpoint manifests.

Public API
----------
- HashAlgorithm
- HashDigest
- HashManifestEntry
- HashManifest
- MerkleProofStep
- MerkleProof
- MerkleTree
- hash_bytes(...)
- hash_hex(...)
- hash_object(...)
- domain_hash(...)
- domain_hash_hex(...)
- leaf_hash(...)
- node_hash(...)
- merkle_root(...)
- merkle_root_hex(...)
- merkle_proof(...)
- verify_merkle_proof(...)
- hash_manifest(...)
- state_root_from_mapping(...)
- state_root_from_entries(...)
- hash_file(...)

This module does not know about ledgers, digital twins, TensorTrains, or lobes.
Higher layers should use this module rather than local one-off hashing helpers.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import enum
import hashlib
import hmac
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

try:  # Optional canonical serializer generated earlier in the stack.
    from tb_canonical import (  # type: ignore
        DEFAULT_CONFIG as _CANONICAL_DEFAULT_CONFIG,
        CanonicalConfig as _CanonicalConfig,
        canonical_bytes as _canonical_bytes,
        canonical_dumps as _canonical_dumps,
        canonical_json_safe as _canonical_json_safe,
    )
    _TB_CANONICAL_OK = True
except Exception:  # pragma: no cover - fallback used in isolated deployments
    _TB_CANONICAL_OK = False
    _CANONICAL_DEFAULT_CONFIG = None
    _CanonicalConfig = Any  # type: ignore
    _canonical_bytes = None  # type: ignore
    _canonical_dumps = None  # type: ignore
    _canonical_json_safe = None  # type: ignore

try:  # Optional. SHA-256 remains the required consensus baseline.
    import blake3 as _blake3  # type: ignore
    _BLAKE3_OK = True
except Exception:  # pragma: no cover - optional dependency
    _blake3 = None  # type: ignore
    _BLAKE3_OK = False

try:  # Optional array handling for fallback canonicalization only.
    import numpy as np  # type: ignore
    _NUMPY_OK = True
except Exception:  # pragma: no cover - minimal Python env
    np = None  # type: ignore
    _NUMPY_OK = False


# =============================================================================
# Version constants / domains
# =============================================================================

TB_HASHING_VERSION = "tb-hashing-v1.0-production"
HASH_DOMAIN_PREFIX = b"TBH|v1|"
DEFAULT_HASH_ALGORITHM = "sha256"
DEFAULT_MERKLE_DOMAIN = "tb.merkle.v1"
DEFAULT_MANIFEST_DOMAIN = "tb.manifest.v1"
DEFAULT_STATE_DOMAIN = "tb.state-root.v1"
DEFAULT_FILE_DOMAIN = "tb.file.v1"

EMPTY_MERKLE_SENTINEL = b"<tb-empty-merkle-tree-v1>"
MAX_DOMAIN_LEN = 256
MAX_TAG_LEN = 256


# =============================================================================
# Errors / enums / dataclasses
# =============================================================================

class HashingError(ValueError):
    """Raised for invalid hashing inputs or unsupported algorithms."""


class MerkleError(ValueError):
    """Raised for invalid Merkle tree/proof operations."""


class HashAlgorithm(str, Enum):
    SHA256 = "sha256"
    BLAKE3 = "blake3"


def normalize_algorithm(algorithm: Union[str, HashAlgorithm, None] = None) -> str:
    alg = str(algorithm.value if isinstance(algorithm, HashAlgorithm) else (algorithm or DEFAULT_HASH_ALGORITHM))
    alg = alg.lower().replace("-", "").replace("_", "").strip()
    if alg in {"sha256", "sha2", "sha2256"}:
        return HashAlgorithm.SHA256.value
    if alg in {"blake3", "b3"}:
        if not _BLAKE3_OK:
            raise HashingError("BLAKE3 requested but the 'blake3' package is not installed.")
        return HashAlgorithm.BLAKE3.value
    raise HashingError(f"Unsupported hash algorithm: {algorithm!r}")


@dataclass(frozen=True)
class HashDigest:
    """Portable representation of a cryptographic digest."""

    algorithm: str
    digest_hex: str
    domain: str = ""
    tag: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "algorithm", normalize_algorithm(self.algorithm))
        hx = normalize_hex(self.digest_hex)
        object.__setattr__(self, "digest_hex", hx)
        object.__setattr__(self, "domain", str(self.domain or ""))
        object.__setattr__(self, "tag", str(self.tag or ""))

    @property
    def bytes(self) -> bytes:
        return bytes.fromhex(self.digest_hex)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "algorithm": self.algorithm,
            "digest_hex": self.digest_hex,
        }
        if self.domain:
            out["domain"] = self.domain
        if self.tag:
            out["tag"] = self.tag
        return out


@dataclass(frozen=True)
class HashManifestEntry:
    """Named digest entry inside a deterministic manifest."""

    name: str
    digest_hex: str
    algorithm: str = DEFAULT_HASH_ALGORITHM
    domain: str = ""
    role: str = ""
    size_bytes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": str(self.name),
            "algorithm": normalize_algorithm(self.algorithm),
            "digest_hex": normalize_hex(self.digest_hex),
        }
        if self.domain:
            out["domain"] = str(self.domain)
        if self.role:
            out["role"] = str(self.role)
        if self.size_bytes is not None:
            out["size_bytes"] = int(self.size_bytes)
        if self.metadata:
            out["metadata"] = canonical_json_safe(self.metadata)
        return out


@dataclass(frozen=True)
class HashManifest:
    """Schema-versioned hash manifest with a commitment digest."""

    manifest_id: str
    entries: List[HashManifestEntry]
    algorithm: str = DEFAULT_HASH_ALGORITHM
    domain: str = DEFAULT_MANIFEST_DOMAIN
    version: str = "1"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def payload(self) -> Dict[str, Any]:
        entries = [e.to_dict() for e in self.entries]
        entries.sort(key=lambda x: (str(x.get("name", "")), str(x.get("role", "")), str(x.get("digest_hex", ""))))
        out: Dict[str, Any] = {
            "schema": "tb.hash-manifest.v1",
            "version": str(self.version),
            "manifest_id": str(self.manifest_id),
            "algorithm": normalize_algorithm(self.algorithm),
            "domain": str(self.domain),
            "entries": entries,
        }
        if self.metadata:
            out["metadata"] = canonical_json_safe(self.metadata)
        return out

    def commitment_hex(self) -> str:
        return domain_hash_hex(self.payload(), domain=self.domain, tag="manifest", algorithm=self.algorithm)

    def to_dict(self) -> Dict[str, Any]:
        payload = self.payload()
        payload["manifest_hash"] = self.commitment_hex()
        return payload


@dataclass(frozen=True)
class MerkleProofStep:
    """One step in a Merkle inclusion proof."""

    sibling_hash: str
    sibling_position: str  # "left" or "right" relative to the current path hash.

    def __post_init__(self) -> None:
        pos = str(self.sibling_position).lower().strip()
        if pos not in {"left", "right"}:
            raise MerkleError("sibling_position must be 'left' or 'right'.")
        object.__setattr__(self, "sibling_position", pos)
        object.__setattr__(self, "sibling_hash", normalize_hex(self.sibling_hash))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sibling_hash": self.sibling_hash,
            "sibling_position": self.sibling_position,
        }


@dataclass(frozen=True)
class MerkleProof:
    """Merkle inclusion proof for one leaf."""

    algorithm: str
    domain: str
    leaf_index: int
    leaf_count: int
    leaf_hash: str
    root_hash: str
    steps: List[MerkleProofStep] = field(default_factory=list)
    leaf_tag: str = "leaf"
    node_tag: str = "node"
    duplicate_odd_leaf: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "algorithm", normalize_algorithm(self.algorithm))
        object.__setattr__(self, "domain", str(self.domain or DEFAULT_MERKLE_DOMAIN))
        object.__setattr__(self, "leaf_index", int(self.leaf_index))
        object.__setattr__(self, "leaf_count", int(self.leaf_count))
        object.__setattr__(self, "leaf_hash", normalize_hex(self.leaf_hash))
        object.__setattr__(self, "root_hash", normalize_hex(self.root_hash))
        object.__setattr__(self, "steps", [s if isinstance(s, MerkleProofStep) else MerkleProofStep(**dict(s)) for s in self.steps])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "tb.merkle-proof.v1",
            "algorithm": self.algorithm,
            "domain": self.domain,
            "leaf_index": int(self.leaf_index),
            "leaf_count": int(self.leaf_count),
            "leaf_hash": self.leaf_hash,
            "root_hash": self.root_hash,
            "steps": [s.to_dict() for s in self.steps],
            "leaf_tag": self.leaf_tag,
            "node_tag": self.node_tag,
            "duplicate_odd_leaf": bool(self.duplicate_odd_leaf),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MerkleProof":
        p = dict(payload)
        return cls(
            algorithm=str(p.get("algorithm", DEFAULT_HASH_ALGORITHM)),
            domain=str(p.get("domain", DEFAULT_MERKLE_DOMAIN)),
            leaf_index=int(p.get("leaf_index", 0)),
            leaf_count=int(p.get("leaf_count", 0)),
            leaf_hash=str(p.get("leaf_hash", "")),
            root_hash=str(p.get("root_hash", "")),
            steps=[MerkleProofStep(**dict(x)) for x in list(p.get("steps", []))],
            leaf_tag=str(p.get("leaf_tag", "leaf")),
            node_tag=str(p.get("node_tag", "node")),
            duplicate_odd_leaf=bool(p.get("duplicate_odd_leaf", True)),
        )

    def verify(self, leaf_payload: Any = None, *, leaf_digest_hex: Optional[str] = None) -> bool:
        return verify_merkle_proof(
            leaf_payload,
            self,
            leaf_digest_hex=leaf_digest_hex,
        )


# =============================================================================
# Canonicalization helpers
# =============================================================================

def _fallback_json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise HashingError("Non-finite float cannot be canonicalized.")
        if obj == 0.0:
            return 0.0
        return float(obj)
    if isinstance(obj, bytes):
        return {"__bytes_b64__": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, bytearray):
        return {"__bytes_b64__": base64.b64encode(bytes(obj)).decode("ascii")}
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return _fallback_json_safe(asdict(obj))
    if _NUMPY_OK and isinstance(obj, np.generic):  # type: ignore[union-attr]
        return _fallback_json_safe(obj.item())
    if _NUMPY_OK and isinstance(obj, np.ndarray):  # type: ignore[union-attr]
        arr = np.asarray(obj)
        if not np.all(np.isfinite(arr)):
            raise HashingError("Array contains non-finite values.")
        canon = np.ascontiguousarray(arr)
        return {
            "__ndarray__": True,
            "shape": list(map(int, canon.shape)),
            "dtype": str(canon.dtype),
            "sha256": hashlib.sha256(canon.tobytes(order="C")).hexdigest(),
        }
    if isinstance(obj, Mapping):
        return {str(k): _fallback_json_safe(v) for k, v in sorted(dict(obj).items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple, set)):
        return [_fallback_json_safe(v) for v in list(obj)]
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return _fallback_json_safe(obj.to_dict())
        except Exception:
            pass
    raise HashingError(f"Object of type {type(obj).__name__} is not canonical JSON safe.")


def canonical_json_safe(obj: Any, *, config: Any = None) -> Any:
    if _TB_CANONICAL_OK and _canonical_json_safe is not None:
        try:
            kwargs = {"config": config} if config is not None else {}
            return _canonical_json_safe(obj, **kwargs)  # type: ignore[misc]
        except TypeError:
            return _canonical_json_safe(obj)  # type: ignore[misc]
    return _fallback_json_safe(obj)


def canonical_dumps(obj: Any, *, config: Any = None) -> str:
    if _TB_CANONICAL_OK and _canonical_dumps is not None:
        try:
            kwargs = {"config": config} if config is not None else {}
            return str(_canonical_dumps(obj, **kwargs))  # type: ignore[misc]
        except TypeError:
            return str(_canonical_dumps(obj))  # type: ignore[misc]
    return json.dumps(canonical_json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_bytes(obj: Any, *, config: Any = None) -> bytes:
    if isinstance(obj, bytes):
        return bytes(obj)
    if isinstance(obj, bytearray):
        return bytes(obj)
    if _TB_CANONICAL_OK and _canonical_bytes is not None:
        try:
            kwargs = {"config": config} if config is not None else {}
            return bytes(_canonical_bytes(obj, **kwargs))  # type: ignore[misc]
        except TypeError:
            return bytes(_canonical_bytes(obj))  # type: ignore[misc]
    return canonical_dumps(obj, config=config).encode("utf-8")


# =============================================================================
# Core hashing
# =============================================================================

def _new_hasher(algorithm: Union[str, HashAlgorithm, None] = None):
    alg = normalize_algorithm(algorithm)
    if alg == HashAlgorithm.SHA256.value:
        return hashlib.sha256(), alg
    if alg == HashAlgorithm.BLAKE3.value:
        if not _BLAKE3_OK or _blake3 is None:
            raise HashingError("BLAKE3 requested but unavailable.")
        return _blake3.blake3(), alg  # type: ignore[union-attr]
    raise HashingError(f"Unsupported hash algorithm: {algorithm!r}")


def hash_bytes(data: Union[bytes, bytearray, memoryview], *, algorithm: Union[str, HashAlgorithm, None] = None) -> bytes:
    """Hash raw bytes and return digest bytes."""
    if isinstance(data, memoryview):
        b = data.tobytes()
    else:
        b = bytes(data)
    h, _ = _new_hasher(algorithm)
    h.update(b)
    return h.digest()


def hash_hex(data: Union[bytes, bytearray, memoryview], *, algorithm: Union[str, HashAlgorithm, None] = None) -> str:
    return hash_bytes(data, algorithm=algorithm).hex()


def hash_object(obj: Any, *, algorithm: Union[str, HashAlgorithm, None] = None, config: Any = None) -> str:
    return hash_hex(canonical_bytes(obj, config=config), algorithm=algorithm)


def _frame_part(part: Union[str, bytes], *, max_len: int, label: str) -> bytes:
    if isinstance(part, bytes):
        b = bytes(part)
    else:
        b = str(part or "").encode("utf-8")
    if len(b) > int(max_len):
        raise HashingError(f"{label} too long: {len(b)} > {max_len}")
    return len(b).to_bytes(2, "big") + b


def domain_prefix(domain: str, tag: str = "", *, algorithm: Union[str, HashAlgorithm, None] = None) -> bytes:
    """Return an unambiguous domain-separation prefix."""
    alg = normalize_algorithm(algorithm)
    return (
        HASH_DOMAIN_PREFIX
        + _frame_part(alg, max_len=64, label="algorithm")
        + _frame_part(str(domain or ""), max_len=MAX_DOMAIN_LEN, label="domain")
        + _frame_part(str(tag or ""), max_len=MAX_TAG_LEN, label="tag")
    )


def domain_hash_bytes(
    payload: Any,
    *,
    domain: str,
    tag: str = "",
    algorithm: Union[str, HashAlgorithm, None] = None,
    config: Any = None,
) -> bytes:
    """Domain-separated hash of canonical payload bytes."""
    alg = normalize_algorithm(algorithm)
    body = canonical_bytes(payload, config=config)
    return hash_bytes(domain_prefix(domain, tag, algorithm=alg) + len(body).to_bytes(8, "big") + body, algorithm=alg)


def domain_hash_hex(
    payload: Any,
    *,
    domain: str,
    tag: str = "",
    algorithm: Union[str, HashAlgorithm, None] = None,
    config: Any = None,
) -> str:
    return domain_hash_bytes(payload, domain=domain, tag=tag, algorithm=algorithm, config=config).hex()


def domain_hash(
    payload: Any,
    *,
    domain: str,
    tag: str = "",
    algorithm: Union[str, HashAlgorithm, None] = None,
    config: Any = None,
) -> HashDigest:
    alg = normalize_algorithm(algorithm)
    return HashDigest(algorithm=alg, digest_hex=domain_hash_hex(payload, domain=domain, tag=tag, algorithm=alg, config=config), domain=domain, tag=tag)


# =============================================================================
# Hex / byte normalization
# =============================================================================

def normalize_hex(value: Union[str, bytes, bytearray, HashDigest]) -> str:
    if isinstance(value, HashDigest):
        return value.digest_hex
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    s = str(value or "").lower().strip()
    if s.startswith("0x"):
        s = s[2:]
    s = "".join(s.split())
    if not s:
        raise HashingError("Empty hex digest.")
    if len(s) % 2 != 0:
        raise HashingError(f"Hex digest length must be even, got {len(s)}.")
    try:
        bytes.fromhex(s)
    except Exception as exc:
        raise HashingError(f"Invalid hex digest: {value!r}") from exc
    return s


def digest_bytes(value: Union[str, bytes, bytearray, HashDigest]) -> bytes:
    if isinstance(value, HashDigest):
        return value.bytes
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return bytes.fromhex(normalize_hex(str(value)))


def constant_time_hex_equal(a: Union[str, bytes, bytearray, HashDigest], b: Union[str, bytes, bytearray, HashDigest]) -> bool:
    try:
        return hmac.compare_digest(digest_bytes(a), digest_bytes(b))
    except Exception:
        return False


# =============================================================================
# Merkle utilities
# =============================================================================

def leaf_hash(
    payload: Any,
    *,
    domain: str = DEFAULT_MERKLE_DOMAIN,
    algorithm: Union[str, HashAlgorithm, None] = None,
    leaf_tag: str = "leaf",
    config: Any = None,
) -> str:
    return domain_hash_hex(payload, domain=domain, tag=leaf_tag, algorithm=algorithm, config=config)


def leaf_hash_from_digest(
    digest_hex_value: Union[str, bytes, bytearray, HashDigest],
    *,
    domain: str = DEFAULT_MERKLE_DOMAIN,
    algorithm: Union[str, HashAlgorithm, None] = None,
    leaf_tag: str = "leaf-digest",
) -> str:
    """Commit an existing digest as a Merkle leaf without treating it as raw node bytes."""
    return domain_hash_hex(
        {"digest_hex": normalize_hex(digest_hex_value)},
        domain=domain,
        tag=leaf_tag,
        algorithm=algorithm,
    )


def node_hash(
    left_hash: Union[str, bytes, bytearray, HashDigest],
    right_hash: Union[str, bytes, bytearray, HashDigest],
    *,
    domain: str = DEFAULT_MERKLE_DOMAIN,
    algorithm: Union[str, HashAlgorithm, None] = None,
    node_tag: str = "node",
) -> str:
    left = normalize_hex(left_hash)
    right = normalize_hex(right_hash)
    payload = {
        "left": left,
        "right": right,
    }
    return domain_hash_hex(payload, domain=domain, tag=node_tag, algorithm=algorithm)


def empty_merkle_root(
    *,
    domain: str = DEFAULT_MERKLE_DOMAIN,
    algorithm: Union[str, HashAlgorithm, None] = None,
) -> str:
    return domain_hash_hex(EMPTY_MERKLE_SENTINEL, domain=domain, tag="empty-root", algorithm=algorithm)


class MerkleTree:
    """Deterministic binary Merkle tree with duplicated odd leaves."""

    def __init__(
        self,
        leaves: Sequence[Any],
        *,
        domain: str = DEFAULT_MERKLE_DOMAIN,
        algorithm: Union[str, HashAlgorithm, None] = None,
        leaves_are_hashes: bool = False,
        leaf_tag: str = "leaf",
        node_tag: str = "node",
        duplicate_odd_leaf: bool = True,
        config: Any = None,
    ):
        self.algorithm = normalize_algorithm(algorithm)
        self.domain = str(domain or DEFAULT_MERKLE_DOMAIN)
        self.leaf_tag = str(leaf_tag or "leaf")
        self.node_tag = str(node_tag or "node")
        self.duplicate_odd_leaf = bool(duplicate_odd_leaf)
        self.leaves_are_hashes = bool(leaves_are_hashes)
        self.config = config
        self.original_leaf_count = len(list(leaves))

        if self.leaves_are_hashes:
            level = [normalize_hex(x) for x in list(leaves)]
        else:
            level = [leaf_hash(x, domain=self.domain, algorithm=self.algorithm, leaf_tag=self.leaf_tag, config=self.config) for x in list(leaves)]

        self.leaf_hashes: List[str] = list(level)
        self.levels: List[List[str]] = []
        if not level:
            self.levels = [[empty_merkle_root(domain=self.domain, algorithm=self.algorithm)]]
            return

        self.levels.append(level)
        cur = list(level)
        while len(cur) > 1:
            if len(cur) % 2 == 1:
                if self.duplicate_odd_leaf:
                    cur = cur + [cur[-1]]
                else:
                    cur = cur + [empty_merkle_root(domain=self.domain, algorithm=self.algorithm)]
            nxt: List[str] = []
            for i in range(0, len(cur), 2):
                nxt.append(node_hash(cur[i], cur[i + 1], domain=self.domain, algorithm=self.algorithm, node_tag=self.node_tag))
            self.levels.append(nxt)
            cur = nxt

    @property
    def root_hash(self) -> str:
        return self.levels[-1][0]

    def root(self) -> str:
        return self.root_hash

    def proof(self, index: int) -> MerkleProof:
        if self.original_leaf_count <= 0:
            raise MerkleError("Cannot build proof for empty Merkle tree.")
        idx = int(index)
        if idx < 0 or idx >= self.original_leaf_count:
            raise MerkleError(f"Leaf index out of range: {index}")

        steps: List[MerkleProofStep] = []
        cur_idx = idx
        for level in self.levels[:-1]:
            effective = list(level)
            if len(effective) % 2 == 1:
                effective = effective + ([effective[-1]] if self.duplicate_odd_leaf else [empty_merkle_root(domain=self.domain, algorithm=self.algorithm)])

            if cur_idx % 2 == 0:
                sibling_idx = cur_idx + 1
                sibling_pos = "right"
            else:
                sibling_idx = cur_idx - 1
                sibling_pos = "left"

            sibling_hash = effective[sibling_idx]
            steps.append(MerkleProofStep(sibling_hash=sibling_hash, sibling_position=sibling_pos))
            cur_idx //= 2

        return MerkleProof(
            algorithm=self.algorithm,
            domain=self.domain,
            leaf_index=idx,
            leaf_count=self.original_leaf_count,
            leaf_hash=self.leaf_hashes[idx],
            root_hash=self.root_hash,
            steps=steps,
            leaf_tag=self.leaf_tag,
            node_tag=self.node_tag,
            duplicate_odd_leaf=self.duplicate_odd_leaf,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "tb.merkle-tree.v1",
            "algorithm": self.algorithm,
            "domain": self.domain,
            "leaf_count": int(self.original_leaf_count),
            "root_hash": self.root_hash,
            "leaf_hashes": list(self.leaf_hashes),
            "levels": [list(x) for x in self.levels],
            "leaf_tag": self.leaf_tag,
            "node_tag": self.node_tag,
            "duplicate_odd_leaf": bool(self.duplicate_odd_leaf),
        }


def merkle_root(
    leaves: Sequence[Any],
    *,
    domain: str = DEFAULT_MERKLE_DOMAIN,
    algorithm: Union[str, HashAlgorithm, None] = None,
    leaves_are_hashes: bool = False,
    config: Any = None,
) -> HashDigest:
    alg = normalize_algorithm(algorithm)
    root = MerkleTree(leaves, domain=domain, algorithm=alg, leaves_are_hashes=leaves_are_hashes, config=config).root_hash
    return HashDigest(algorithm=alg, digest_hex=root, domain=domain, tag="merkle-root")


def merkle_root_hex(
    leaves: Sequence[Any],
    *,
    domain: str = DEFAULT_MERKLE_DOMAIN,
    algorithm: Union[str, HashAlgorithm, None] = None,
    leaves_are_hashes: bool = False,
    config: Any = None,
) -> str:
    return merkle_root(leaves, domain=domain, algorithm=algorithm, leaves_are_hashes=leaves_are_hashes, config=config).digest_hex


def merkle_proof(
    leaves: Sequence[Any],
    index: int,
    *,
    domain: str = DEFAULT_MERKLE_DOMAIN,
    algorithm: Union[str, HashAlgorithm, None] = None,
    leaves_are_hashes: bool = False,
    config: Any = None,
) -> MerkleProof:
    tree = MerkleTree(leaves, domain=domain, algorithm=algorithm, leaves_are_hashes=leaves_are_hashes, config=config)
    return tree.proof(index)


def verify_merkle_proof(
    leaf_payload: Any,
    proof: Union[MerkleProof, Mapping[str, Any]],
    *,
    leaf_digest_hex: Optional[Union[str, bytes, bytearray, HashDigest]] = None,
    config: Any = None,
) -> bool:
    p = proof if isinstance(proof, MerkleProof) else MerkleProof.from_dict(proof)
    try:
        if p.leaf_count <= 0 or p.leaf_index < 0 or p.leaf_index >= p.leaf_count:
            return False
        if leaf_digest_hex is not None:
            cur = normalize_hex(leaf_digest_hex)
        else:
            cur = leaf_hash(leaf_payload, domain=p.domain, algorithm=p.algorithm, leaf_tag=p.leaf_tag, config=config)
        if not constant_time_hex_equal(cur, p.leaf_hash):
            return False
        for step in p.steps:
            if step.sibling_position == "left":
                cur = node_hash(step.sibling_hash, cur, domain=p.domain, algorithm=p.algorithm, node_tag=p.node_tag)
            else:
                cur = node_hash(cur, step.sibling_hash, domain=p.domain, algorithm=p.algorithm, node_tag=p.node_tag)
        return constant_time_hex_equal(cur, p.root_hash)
    except Exception:
        return False


# =============================================================================
# Manifests / state roots / file hashing
# =============================================================================

def hash_manifest(
    entries: Sequence[Union[HashManifestEntry, Mapping[str, Any]]],
    *,
    manifest_id: str = "manifest",
    domain: str = DEFAULT_MANIFEST_DOMAIN,
    algorithm: Union[str, HashAlgorithm, None] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> HashManifest:
    alg = normalize_algorithm(algorithm)
    normalized_entries: List[HashManifestEntry] = []
    for item in entries:
        if isinstance(item, HashManifestEntry):
            normalized_entries.append(item)
        else:
            p = dict(item)
            normalized_entries.append(
                HashManifestEntry(
                    name=str(p.get("name", p.get("path", p.get("id", "entry")))),
                    digest_hex=str(p.get("digest_hex", p.get("hash", p.get("sha256", "")))),
                    algorithm=str(p.get("algorithm", alg)),
                    domain=str(p.get("domain", "")),
                    role=str(p.get("role", "")),
                    size_bytes=(None if p.get("size_bytes") is None else int(p.get("size_bytes"))),
                    metadata=dict(p.get("metadata", {})),
                )
            )
    return HashManifest(
        manifest_id=str(manifest_id),
        entries=normalized_entries,
        algorithm=alg,
        domain=str(domain),
        metadata=dict(metadata or {}),
    )


def state_root_from_entries(
    entries: Sequence[Union[HashManifestEntry, Mapping[str, Any]]],
    *,
    state_id: str = "state",
    domain: str = DEFAULT_STATE_DOMAIN,
    algorithm: Union[str, HashAlgorithm, None] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    manifest = hash_manifest(entries, manifest_id=state_id, domain=domain, algorithm=algorithm, metadata=metadata)
    return manifest.commitment_hex()


def state_root_from_mapping(
    mapping: Mapping[str, Any],
    *,
    state_id: str = "state",
    domain: str = DEFAULT_STATE_DOMAIN,
    algorithm: Union[str, HashAlgorithm, None] = None,
    config: Any = None,
) -> str:
    alg = normalize_algorithm(algorithm)
    entries: List[HashManifestEntry] = []
    for key in sorted(mapping.keys(), key=lambda x: str(x)):
        val = mapping[key]
        entries.append(
            HashManifestEntry(
                name=str(key),
                digest_hex=domain_hash_hex(val, domain=domain, tag=f"state-entry:{key}", algorithm=alg, config=config),
                algorithm=alg,
                domain=domain,
                role="state_entry",
            )
        )
    return state_root_from_entries(entries, state_id=state_id, domain=domain, algorithm=alg)


def hash_file(
    path: Union[str, os.PathLike],
    *,
    algorithm: Union[str, HashAlgorithm, None] = None,
    domain: str = DEFAULT_FILE_DOMAIN,
    max_bytes: Optional[int] = None,
    chunk_size: int = 1024 * 1024,
    return_manifest_entry: bool = False,
) -> Union[str, HashManifestEntry]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(str(p))
    size = int(p.stat().st_size)
    if max_bytes is not None and size > int(max_bytes):
        raise HashingError(f"File exceeds max_bytes: {size} > {max_bytes}")
    alg = normalize_algorithm(algorithm)
    h, _ = _new_hasher(alg)
    prefix = domain_prefix(domain, "file-bytes", algorithm=alg)
    h.update(prefix)
    h.update(size.to_bytes(8, "big"))
    with p.open("rb") as f:
        while True:
            chunk = f.read(int(max(1, chunk_size)))
            if not chunk:
                break
            h.update(chunk)
    hx = h.hexdigest()
    if return_manifest_entry:
        return HashManifestEntry(
            name=p.name,
            digest_hex=hx,
            algorithm=alg,
            domain=domain,
            role="file",
            size_bytes=size,
            metadata={"suffix": p.suffix.lower()},
        )
    return hx


# =============================================================================
# Compatibility aliases used by likely higher-level scripts
# =============================================================================

sha256_hex = hash_hex
canonical_hash_hex = hash_object
domain_separated_hash_hex = domain_hash_hex
build_merkle_root = merkle_root_hex
build_merkle_proof = merkle_proof
verify_proof = verify_merkle_proof


# =============================================================================
# Status / self-test / CLI
# =============================================================================

def status_report() -> Dict[str, Any]:
    return {
        "ok": True,
        "version": TB_HASHING_VERSION,
        "default_algorithm": DEFAULT_HASH_ALGORITHM,
        "algorithms": {
            "sha256": True,
            "blake3": bool(_BLAKE3_OK),
        },
        "tb_canonical_available": bool(_TB_CANONICAL_OK),
        "numpy_available": bool(_NUMPY_OK),
    }


def run_self_test() -> Dict[str, Any]:
    payloads = [
        {"tx": 1, "amount": 10.5},
        {"tx": 2, "amount": 11.0},
        {"tx": 3, "amount": 9.75},
        {"tx": 4, "amount": 2.25},
        {"tx": 5, "amount": 0.0},
    ]
    tree = MerkleTree(payloads, domain="tb.selftest.merkle")
    if tree.root_hash != merkle_root_hex(payloads, domain="tb.selftest.merkle"):
        raise AssertionError("Merkle root mismatch.")
    for i, p in enumerate(payloads):
        proof = tree.proof(i)
        if not verify_merkle_proof(p, proof):
            raise AssertionError(f"Merkle proof failed for index {i}.")
        bad = dict(p)
        bad["amount"] = float(bad["amount"]) + 1.0
        if verify_merkle_proof(bad, proof):
            raise AssertionError("Bad Merkle proof unexpectedly verified.")

    root1 = state_root_from_mapping({"b": 2, "a": 1}, state_id="demo")
    root2 = state_root_from_mapping({"a": 1, "b": 2}, state_id="demo")
    if root1 != root2:
        raise AssertionError("State root should be key-order independent.")
    root3 = state_root_from_mapping({"a": 1, "b": 3}, state_id="demo")
    if root1 == root3:
        raise AssertionError("State root should change when values change.")

    manifest = hash_manifest(
        [
            HashManifestEntry(name="alpha", digest_hex=hash_object({"x": 1}), role="object"),
            HashManifestEntry(name="beta", digest_hex=hash_object({"y": 2}), role="object"),
        ],
        manifest_id="selftest",
    )
    md = manifest.to_dict()
    if md["manifest_hash"] != manifest.commitment_hex():
        raise AssertionError("Manifest hash mismatch.")

    return {
        "ok": True,
        "status": status_report(),
        "merkle_root": tree.root_hash,
        "state_root": root1,
        "manifest_hash": manifest.commitment_hex(),
        "proof_steps_for_last_leaf": len(tree.proof(len(payloads) - 1).steps),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tensorized Blockchain hashing utilities")
    p.add_argument("--mode", choices=["status", "self-test", "hash-file"], default="self-test")
    p.add_argument("--path", default="", help="File path for --mode hash-file")
    p.add_argument("--algorithm", default=DEFAULT_HASH_ALGORITHM, choices=["sha256", "blake3"] if _BLAKE3_OK else ["sha256"])
    p.add_argument("--pretty", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        if args.mode == "status":
            out = status_report()
        elif args.mode == "hash-file":
            if not args.path:
                raise ValueError("--path is required for --mode hash-file")
            out = {
                "ok": True,
                "path": str(Path(args.path)),
                "digest_hex": hash_file(args.path, algorithm=args.algorithm),
                "algorithm": normalize_algorithm(args.algorithm),
            }
        else:
            out = run_self_test()
        print(json.dumps(canonical_json_safe(out), indent=2 if args.pretty else None, sort_keys=True, ensure_ascii=False))
        return 0 if bool(out.get("ok", False)) else 2
    except Exception as exc:
        err = {
            "ok": False,
            "error": repr(exc),
            "status": status_report(),
        }
        print(json.dumps(canonical_json_safe(err), indent=2, sort_keys=True, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
