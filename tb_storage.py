#!/usr/bin/env python3
# tb_storage.py
r"""
Tensorized Blockchain safe persistence layer
===========================================

Project Chimera / Akkurat - Tensorized Blockchain storage module.

Purpose
-------
This module provides audited, atomic, path-safe persistence utilities for the
Tensorized Blockchain stack. It is deliberately dependency-light and can be
used by the ledger, tensor runtime, digital twin runtime, lobe substrates, API,
and demo app without importing those high-level modules.

Responsibilities
----------------
- Atomic JSON/text/bytes writes with fsync where available.
- Safe path handling under explicit storage roots.
- JSON sidecar manifests for persisted artifacts.
- Hash verification for stored artifacts.
- Ledger, block, tensor artifact, lobe checkpoint, and demo artifact stores.
- NPZ save/load helpers with manifest sidecars.
- Exportable store index and self-test utilities.

Design boundaries
-----------------
This module does not define transaction, block, tensor, lobe, or twin semantics.
It persists bytes and JSON-like payloads safely and records cryptographic
manifests. Higher-level modules own business logic and schema validation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

try:  # Optional. Used only for NPZ helpers and self-tests.
    import numpy as np  # type: ignore
    _NUMPY_OK = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _NUMPY_OK = False

try:  # Prefer the shared canonical serializer.
    from tb_canonical import canonical_json_safe, canonical_dumps, canonical_bytes  # type: ignore
    _CANONICAL_OK = True
except Exception:  # pragma: no cover
    _CANONICAL_OK = False
    canonical_json_safe = None  # type: ignore
    canonical_dumps = None  # type: ignore
    canonical_bytes = None  # type: ignore

try:  # Prefer the shared hashing layer.
    from tb_hashing import hash_file as _tb_hash_file  # type: ignore
    from tb_hashing import hash_hex as _tb_hash_hex  # type: ignore
    from tb_hashing import hash_object as _tb_hash_object  # type: ignore
    from tb_hashing import domain_hash_hex as _tb_domain_hash_hex  # type: ignore
    from tb_hashing import hash_manifest as _tb_hash_manifest  # type: ignore
    _HASHING_OK = True
except Exception:  # pragma: no cover
    _HASHING_OK = False
    _tb_hash_file = None  # type: ignore
    _tb_hash_hex = None  # type: ignore
    _tb_hash_object = None  # type: ignore
    _tb_domain_hash_hex = None  # type: ignore
    _tb_hash_manifest = None  # type: ignore

try:  # Optional policy integration.
    from tb_config import FileSystemSafetyPolicy as _TBFileSystemSafetyPolicy  # type: ignore
    from tb_config import DEFAULT_MAX_LOCAL_FILE_BYTES as _DEFAULT_MAX_LOCAL_FILE_BYTES  # type: ignore
except Exception:  # pragma: no cover
    _TBFileSystemSafetyPolicy = None  # type: ignore
    _DEFAULT_MAX_LOCAL_FILE_BYTES = 512 * 1024 * 1024


# =============================================================================
# Version constants / defaults
# =============================================================================

TB_STORAGE_VERSION = "tb-storage-v1.0-production"
STORAGE_MANIFEST_SCHEMA = "tb.storage-manifest.v1"
ARTIFACT_RECORD_SCHEMA = "tb.artifact-record.v1"
STORE_INDEX_SCHEMA = "tb.store-index.v1"
DEFAULT_HASH_ALGORITHM = "sha256"
DEFAULT_CHUNK_BYTES = 1024 * 1024
DEFAULT_MAX_ARTIFACT_BYTES = int(_DEFAULT_MAX_LOCAL_FILE_BYTES)
ZERO_HASH_256 = "0" * 64

SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._=-]+")


# =============================================================================
# Errors / enums
# =============================================================================

class StorageError(RuntimeError):
    """Base error for storage operations."""


class PathPolicyError(StorageError):
    """Raised when a path violates the active storage path policy."""


class ArtifactIntegrityError(StorageError):
    """Raised when an artifact hash or manifest does not verify."""


class ArtifactKind(str, Enum):
    JSON = "json"
    TEXT = "text"
    BYTES = "bytes"
    NPZ = "npz"
    LEDGER = "ledger"
    BLOCK = "block"
    TENSOR = "tensor"
    LOBE_CHECKPOINT = "lobe_checkpoint"
    DEMO = "demo"
    MANIFEST = "manifest"


# =============================================================================
# Canonical / hashing fallbacks
# =============================================================================

def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_safe(obj: Any) -> Any:
    if canonical_json_safe is not None:
        try:
            return canonical_json_safe(obj)
        except Exception:
            pass
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if _NUMPY_OK:
        try:
            if isinstance(obj, np.generic):  # type: ignore[attr-defined]
                return _json_safe(obj.item())
            if isinstance(obj, np.ndarray):  # type: ignore[attr-defined]
                arr = np.nan_to_num(np.asarray(obj), nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[union-attr]
                return arr.tolist()
        except Exception:
            pass
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return _json_safe(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "describe") and callable(obj.describe):
        try:
            return _json_safe(obj.describe())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return _json_safe(vars(obj))
        except Exception:
            pass
    return str(obj)


def _canonical_dumps(obj: Any, *, pretty: bool = False) -> str:
    if canonical_dumps is not None and not pretty:
        try:
            return str(canonical_dumps(obj))
        except Exception:
            pass
    return json.dumps(_json_safe(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":") if not pretty else None, indent=2 if pretty else None)


def _canonical_bytes(obj: Any) -> bytes:
    if canonical_bytes is not None:
        try:
            return bytes(canonical_bytes(obj))
        except Exception:
            pass
    return _canonical_dumps(obj).encode("utf-8", errors="ignore")


def _hash_bytes(data: Union[bytes, bytearray, memoryview], *, algorithm: str = DEFAULT_HASH_ALGORITHM) -> str:
    if _tb_hash_hex is not None:
        try:
            return str(_tb_hash_hex(bytes(data), algorithm=algorithm))
        except TypeError:
            try:
                return str(_tb_hash_hex(bytes(data)))
            except Exception:
                pass
        except Exception:
            pass
    import hashlib
    alg = str(algorithm or DEFAULT_HASH_ALGORITHM).lower().replace("-", "")
    if alg != "sha256":
        raise StorageError(f"Unsupported fallback hash algorithm: {algorithm!r}")
    return hashlib.sha256(bytes(data)).hexdigest()


def _hash_object(obj: Any, *, algorithm: str = DEFAULT_HASH_ALGORITHM, domain: str = "tb.storage.object.v1") -> str:
    if _tb_domain_hash_hex is not None:
        try:
            return str(_tb_domain_hash_hex(domain, obj, algorithm=algorithm))
        except TypeError:
            try:
                return str(_tb_domain_hash_hex(domain, obj))
            except Exception:
                pass
        except Exception:
            pass
    if _tb_hash_object is not None:
        try:
            return str(_tb_hash_object(obj, algorithm=algorithm))
        except TypeError:
            try:
                return str(_tb_hash_object(obj))
            except Exception:
                pass
        except Exception:
            pass
    return _hash_bytes(_canonical_bytes({"domain": domain, "payload": _json_safe(obj)}), algorithm=algorithm)


def _hash_file(path: Union[str, os.PathLike[str]], *, algorithm: str = DEFAULT_HASH_ALGORITHM, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> str:
    p = Path(path)
    if _tb_hash_file is not None:
        try:
            val = _tb_hash_file(p, algorithm=algorithm)
            if isinstance(val, str):
                return val
            if hasattr(val, "digest_hex"):
                return str(val.digest_hex)
        except TypeError:
            try:
                val = _tb_hash_file(p)
                if isinstance(val, str):
                    return val
                if hasattr(val, "digest_hex"):
                    return str(val.digest_hex)
            except Exception:
                pass
        except Exception:
            pass
    import hashlib
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(int(chunk_bytes))
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _safe_name(value: Any, *, default: str = "artifact", max_len: int = 160) -> str:
    s = SAFE_NAME_RE.sub("-", str(value or "").strip())
    s = re.sub(r"-+", "-", s).strip("-._")
    if not s:
        s = default
    if s in {".", ".."}:
        s = default
    return s[: int(max_len)]


def _atomic_fsync_dir(path: Path) -> None:
    # Directory fsync is not available on every platform/filesystem. Best effort.
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class StorageDiagnostic:
    code: str
    severity: str
    component: str
    message: str
    recoverable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class StorageManifestEntry:
    path: str
    artifact_kind: str
    size_bytes: int
    hash_hex: str
    hash_algorithm: str = DEFAULT_HASH_ALGORITHM
    mime_type: str = "application/octet-stream"
    role: str = "artifact"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "StorageManifestEntry":
        e = dataclasses.replace(self)
        e.path = str(e.path).replace("\\", "/")
        e.artifact_kind = str(e.artifact_kind or ArtifactKind.BYTES.value)
        e.size_bytes = int(max(0, e.size_bytes))
        e.hash_hex = str(e.hash_hex or "").lower()
        e.hash_algorithm = str(e.hash_algorithm or DEFAULT_HASH_ALGORITHM).lower()
        e.mime_type = str(e.mime_type or "application/octet-stream")
        e.role = str(e.role or "artifact")
        e.metadata = dict(e.metadata or {})
        return e

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self.normalized()))


@dataclass
class StorageManifest:
    manifest_id: str
    created_at: str
    root_public_path: str
    entries: List[StorageManifestEntry] = field(default_factory=list)
    schema: str = STORAGE_MANIFEST_SCHEMA
    storage_version: str = TB_STORAGE_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)
    manifest_hash: str = ""

    def payload(self, *, include_hash: bool = False) -> Dict[str, Any]:
        out = {
            "schema": self.schema,
            "storage_version": self.storage_version,
            "manifest_id": str(self.manifest_id),
            "created_at": str(self.created_at),
            "root_public_path": str(self.root_public_path),
            "entries": [e.to_dict() for e in self.entries],
            "metadata": _json_safe(self.metadata),
        }
        if include_hash:
            out["manifest_hash"] = str(self.manifest_hash)
        return out

    def compute_hash(self) -> str:
        return _hash_object(self.payload(include_hash=False), domain="tb.storage.manifest.v1")

    def finalize(self) -> "StorageManifest":
        m = dataclasses.replace(self)
        m.entries = [e.normalized() for e in self.entries]
        m.manifest_hash = m.compute_hash()
        return m

    def to_dict(self) -> Dict[str, Any]:
        return self.finalize().payload(include_hash=True)


@dataclass
class ArtifactRecord:
    artifact_id: str
    artifact_kind: str
    public_path: str
    size_bytes: int
    hash_hex: str
    hash_algorithm: str = DEFAULT_HASH_ALGORITHM
    created_at: str = field(default_factory=_now_iso_utc)
    manifest_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema: str = ARTIFACT_RECORD_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactRecord":
        p = dict(payload or {})
        return cls(
            artifact_id=str(p.get("artifact_id", uuid.uuid4().hex)),
            artifact_kind=str(p.get("artifact_kind", ArtifactKind.BYTES.value)),
            public_path=str(p.get("public_path", "")),
            size_bytes=int(p.get("size_bytes", 0)),
            hash_hex=str(p.get("hash_hex", "")),
            hash_algorithm=str(p.get("hash_algorithm", DEFAULT_HASH_ALGORITHM)),
            created_at=str(p.get("created_at", _now_iso_utc())),
            manifest_path=str(p.get("manifest_path", "")),
            metadata=dict(p.get("metadata", {})),
            schema=str(p.get("schema", ARTIFACT_RECORD_SCHEMA)),
        )


@dataclass
class StoreIndex:
    store_id: str
    root_public_path: str
    created_at: str
    updated_at: str
    records: List[ArtifactRecord] = field(default_factory=list)
    schema: str = STORE_INDEX_SCHEMA
    storage_version: str = TB_STORAGE_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)
    index_hash: str = ""

    def payload(self, *, include_hash: bool = False) -> Dict[str, Any]:
        out = {
            "schema": self.schema,
            "storage_version": self.storage_version,
            "store_id": self.store_id,
            "root_public_path": self.root_public_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "records": [r.to_dict() for r in self.records],
            "metadata": _json_safe(self.metadata),
        }
        if include_hash:
            out["index_hash"] = self.index_hash
        return out

    def compute_hash(self) -> str:
        return _hash_object(self.payload(include_hash=False), domain="tb.storage.index.v1")

    def finalize(self) -> "StoreIndex":
        idx = dataclasses.replace(self)
        idx.index_hash = idx.compute_hash()
        return idx

    def to_dict(self) -> Dict[str, Any]:
        return self.finalize().payload(include_hash=True)


# =============================================================================
# Path policy
# =============================================================================

class SafePathPolicy:
    """Restricts all storage operations to a root directory and optional child roots."""

    def __init__(
        self,
        root: Union[str, os.PathLike[str]],
        *,
        allowed_roots: Optional[Sequence[Union[str, os.PathLike[str]]]] = None,
        allow_overwrite: bool = True,
        max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
        redact_absolute_paths: bool = True,
    ):
        self.root = Path(root).expanduser().resolve()
        self.allowed_roots = [self.root]
        for r in list(allowed_roots or []):
            rp = Path(r).expanduser().resolve()
            if rp not in self.allowed_roots:
                self.allowed_roots.append(rp)
        self.allow_overwrite = bool(allow_overwrite)
        self.max_artifact_bytes = int(max(1024, max_artifact_bytes))
        self.redact_absolute_paths = bool(redact_absolute_paths)

    @classmethod
    def from_config_policy(
        cls,
        root: Union[str, os.PathLike[str]],
        policy: Any,
        *,
        allow_overwrite: bool = True,
    ) -> "SafePathPolicy":
        allowed_roots: List[str] = []
        max_bytes = DEFAULT_MAX_ARTIFACT_BYTES
        redact = True
        try:
            p = policy.normalized() if hasattr(policy, "normalized") else policy
            allowed_roots = list(getattr(p, "allowed_roots", []) or [])
            max_bytes = int(getattr(p, "max_local_file_bytes", DEFAULT_MAX_ARTIFACT_BYTES))
            redact = bool(getattr(p, "redact_absolute_paths", True))
        except Exception:
            pass
        return cls(root, allowed_roots=allowed_roots, allow_overwrite=allow_overwrite, max_artifact_bytes=max_bytes, redact_absolute_paths=redact)

    def ensure_root(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def _is_under_allowed_root(self, path: Path) -> bool:
        try:
            rp = path.expanduser().resolve()
        except Exception:
            return False
        for root in self.allowed_roots:
            try:
                rp.relative_to(root)
                return True
            except Exception:
                continue
        return False

    def resolve(self, relative_path: Union[str, os.PathLike[str]], *, must_exist: bool = False, for_write: bool = False) -> Path:
        s = str(relative_path or "")
        if not s:
            raise PathPolicyError("Path cannot be empty.")
        p = Path(s)
        if p.is_absolute():
            candidate = p.expanduser().resolve()
        else:
            # Prevent traversal before resolution for clearer diagnostics.
            if any(part == ".." for part in p.parts):
                raise PathPolicyError(f"Unsafe relative path traversal: {s!r}")
            candidate = (self.root / p).expanduser().resolve()
        if not self._is_under_allowed_root(candidate):
            raise PathPolicyError(f"Path is outside allowed roots: {s!r}")
        if must_exist and not candidate.exists():
            raise FileNotFoundError(str(candidate))
        if for_write:
            if candidate.exists() and not self.allow_overwrite:
                raise FileExistsError(str(candidate))
            candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def public_path(self, path: Union[str, os.PathLike[str]]) -> str:
        p = Path(path).expanduser()
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if self.redact_absolute_paths:
            for root in self.allowed_roots:
                try:
                    return str(rp.relative_to(root)).replace("\\", "/")
                except Exception:
                    pass
            try:
                return str(rp.relative_to(self.root)).replace("\\", "/")
            except Exception:
                pass
            if rp.is_absolute():
                return f"<redacted>/{rp.name}"
        return str(path).replace("\\", "/")

    def validate_size(self, path: Union[str, os.PathLike[str]]) -> None:
        p = Path(path)
        if p.exists() and p.is_file():
            size = int(p.stat().st_size)
            if size > self.max_artifact_bytes:
                raise PathPolicyError(f"Artifact exceeds max_artifact_bytes: {size} > {self.max_artifact_bytes}")


# =============================================================================
# Atomic file operations
# =============================================================================

def atomic_write_bytes(
    path: Union[str, os.PathLike[str]],
    data: Union[bytes, bytearray, memoryview],
    *,
    mode: int = 0o644,
    fsync_dir: bool = True,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix or ".bin"
    tmp_name = f".{p.name}.{uuid.uuid4().hex}.tmp{suffix}"
    tmp = p.parent / tmp_name
    try:
        with tmp.open("wb") as f:
            f.write(bytes(data))
            f.flush()
            os.fsync(f.fileno())
        try:
            os.chmod(tmp, int(mode))
        except Exception:
            pass
        os.replace(tmp, p)
        if fsync_dir:
            _atomic_fsync_dir(p.parent)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def atomic_write_text(path: Union[str, os.PathLike[str]], text: str, *, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, str(text).encode(encoding, errors="ignore"))


def atomic_write_json(
    path: Union[str, os.PathLike[str]],
    payload: Any,
    *,
    pretty: bool = True,
    include_newline: bool = True,
) -> None:
    text = _canonical_dumps(payload, pretty=pretty)
    if include_newline and not text.endswith("\n"):
        text += "\n"
    atomic_write_text(path, text)


def load_json(path: Union[str, os.PathLike[str]]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def maybe_load_json(path: Union[str, os.PathLike[str]], default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return load_json(p)
    except Exception:
        return default


def copy_file_atomic(src: Union[str, os.PathLike[str]], dst: Union[str, os.PathLike[str]], *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> None:
    s = Path(src)
    d = Path(dst)
    d.parent.mkdir(parents=True, exist_ok=True)
    tmp = d.parent / f".{d.name}.{uuid.uuid4().hex}.tmp"
    try:
        with s.open("rb") as fin, tmp.open("wb") as fout:
            while True:
                chunk = fin.read(int(chunk_bytes))
                if not chunk:
                    break
                fout.write(chunk)
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(tmp, d)
        _atomic_fsync_dir(d.parent)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


# =============================================================================
# Store base class
# =============================================================================

class ArtifactStore:
    """Path-safe artifact store with manifest sidecars and a store index."""

    def __init__(
        self,
        root: Union[str, os.PathLike[str]],
        *,
        namespace: str = "artifacts",
        store_id: Optional[str] = None,
        path_policy: Optional[SafePathPolicy] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        self.root = Path(root).expanduser().resolve()
        self.namespace = _safe_name(namespace, default="artifacts")
        self.path_policy = path_policy or SafePathPolicy(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store_root = self.path_policy.resolve(self.namespace, for_write=True)
        self.store_root.mkdir(parents=True, exist_ok=True)
        self.store_id = str(store_id or f"{self.namespace}-{uuid.uuid4().hex[:12]}")
        self.metadata = dict(metadata or {})
        self.index_path = self.store_root / "_store_index.json"
        self._index: Optional[StoreIndex] = None

    # ------------------------------------------------------------------
    # Path utilities
    # ------------------------------------------------------------------
    def resolve(self, relative_path: Union[str, os.PathLike[str]], *, for_write: bool = False, must_exist: bool = False) -> Path:
        rel = Path(relative_path)
        # Records store paths public-relative to the project root. If the caller
        # passes such a path, do not prefix the namespace twice.
        if rel.parts and rel.parts[0] == self.namespace:
            return self.path_policy.resolve(rel, for_write=for_write, must_exist=must_exist)
        return self.path_policy.resolve(Path(self.namespace) / rel, for_write=for_write, must_exist=must_exist)

    def public_path(self, path: Union[str, os.PathLike[str]]) -> str:
        return self.path_policy.public_path(path)

    def _artifact_path(self, artifact_id: str, suffix: str, *, subdir: str = "") -> Path:
        aid = _safe_name(artifact_id, default="artifact")
        suff = suffix if str(suffix).startswith(".") else f".{suffix}"
        rel = Path(subdir) / f"{aid}{suff}" if subdir else Path(f"{aid}{suff}")
        return self.resolve(rel, for_write=True)

    def _manifest_sidecar_path(self, artifact_path: Path) -> Path:
        return artifact_path.with_suffix(artifact_path.suffix + ".manifest.json")

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------
    def load_index(self) -> StoreIndex:
        if self._index is not None:
            return self._index
        if self.index_path.exists():
            try:
                payload = load_json(self.index_path)
                records = [ArtifactRecord.from_dict(x) for x in payload.get("records", [])]
                self._index = StoreIndex(
                    store_id=str(payload.get("store_id", self.store_id)),
                    root_public_path=str(payload.get("root_public_path", self.public_path(self.store_root))),
                    created_at=str(payload.get("created_at", _now_iso_utc())),
                    updated_at=str(payload.get("updated_at", _now_iso_utc())),
                    records=records,
                    metadata=dict(payload.get("metadata", self.metadata)),
                    index_hash=str(payload.get("index_hash", "")),
                )
                expected = self._index.compute_hash()
                if self._index.index_hash and self._index.index_hash != expected:
                    raise ArtifactIntegrityError("Store index hash mismatch.")
                return self._index
            except Exception as exc:
                raise StorageError(f"Failed to load store index: {exc}") from exc
        self._index = StoreIndex(
            store_id=self.store_id,
            root_public_path=self.public_path(self.store_root),
            created_at=_now_iso_utc(),
            updated_at=_now_iso_utc(),
            records=[],
            metadata=dict(self.metadata),
        ).finalize()
        self.save_index()
        return self._index

    def save_index(self) -> None:
        idx = self._index or StoreIndex(
            store_id=self.store_id,
            root_public_path=self.public_path(self.store_root),
            created_at=_now_iso_utc(),
            updated_at=_now_iso_utc(),
            records=[],
            metadata=dict(self.metadata),
        )
        idx.updated_at = _now_iso_utc()
        idx = idx.finalize()
        self._index = idx
        atomic_write_json(self.index_path, idx.to_dict(), pretty=True)

    def add_record(self, record: ArtifactRecord) -> ArtifactRecord:
        idx = self.load_index()
        # Replace same artifact id, preserve latest.
        idx.records = [r for r in idx.records if r.artifact_id != record.artifact_id]
        idx.records.append(record)
        self._index = idx
        self.save_index()
        return record

    def records(self) -> List[ArtifactRecord]:
        return list(self.load_index().records)

    def find_record(self, artifact_id: str) -> Optional[ArtifactRecord]:
        aid = str(artifact_id)
        for r in self.records():
            if r.artifact_id == aid:
                return r
        return None

    # ------------------------------------------------------------------
    # Manifest / verification
    # ------------------------------------------------------------------
    def build_manifest(
        self,
        artifact_path: Union[str, os.PathLike[str]],
        *,
        artifact_kind: Union[str, ArtifactKind],
        artifact_id: str,
        role: str = "artifact",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> StorageManifest:
        p = Path(artifact_path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(str(p))
        self.path_policy.validate_size(p)
        h = _hash_file(p)
        entry = StorageManifestEntry(
            path=self.public_path(p),
            artifact_kind=str(artifact_kind.value if isinstance(artifact_kind, ArtifactKind) else artifact_kind),
            size_bytes=int(p.stat().st_size),
            hash_hex=h,
            role=role,
            metadata=dict(metadata or {}),
        )
        return StorageManifest(
            manifest_id=f"manifest-{_safe_name(artifact_id)}-{uuid.uuid4().hex[:12]}",
            created_at=_now_iso_utc(),
            root_public_path=self.public_path(self.store_root),
            entries=[entry],
            metadata={"artifact_id": str(artifact_id), **dict(metadata or {})},
        ).finalize()

    def write_manifest_sidecar(self, artifact_path: Union[str, os.PathLike[str]], manifest: StorageManifest) -> Path:
        mp = self._manifest_sidecar_path(Path(artifact_path))
        atomic_write_json(mp, manifest.to_dict(), pretty=True)
        return mp

    def verify_manifest(self, manifest_path: Union[str, os.PathLike[str]]) -> Dict[str, Any]:
        mp = Path(manifest_path)
        payload = load_json(mp)
        entries = [StorageManifestEntry(**dict(e)).normalized() for e in payload.get("entries", [])]
        manifest = StorageManifest(
            manifest_id=str(payload.get("manifest_id", "")),
            created_at=str(payload.get("created_at", "")),
            root_public_path=str(payload.get("root_public_path", "")),
            entries=entries,
            metadata=dict(payload.get("metadata", {})),
            manifest_hash=str(payload.get("manifest_hash", "")),
        )
        expected_manifest_hash = manifest.compute_hash()
        ok_manifest = bool(payload.get("manifest_hash") == expected_manifest_hash)
        file_results = []
        ok_files = True
        for e in entries:
            try:
                p = self.resolve(e.path, must_exist=True)
                actual = _hash_file(p, algorithm=e.hash_algorithm)
                ok = bool(actual == e.hash_hex and int(p.stat().st_size) == int(e.size_bytes))
                file_results.append({"path": e.path, "ok": ok, "expected_hash": e.hash_hex, "actual_hash": actual})
                ok_files = ok_files and ok
            except Exception as exc:
                file_results.append({"path": e.path, "ok": False, "error": repr(exc)})
                ok_files = False
        return {
            "ok": bool(ok_manifest and ok_files),
            "manifest_hash_ok": bool(ok_manifest),
            "file_hashes_ok": bool(ok_files),
            "expected_manifest_hash": expected_manifest_hash,
            "stored_manifest_hash": payload.get("manifest_hash", ""),
            "files": file_results,
        }

    def verify_record(self, record: ArtifactRecord) -> Dict[str, Any]:
        try:
            path = self.resolve(record.public_path, must_exist=True)
        except Exception:
            return {"ok": False, "error": "artifact_missing", "record": record.to_dict()}
        actual = _hash_file(path, algorithm=record.hash_algorithm)
        ok_hash = actual == record.hash_hex
        ok_size = int(path.stat().st_size) == int(record.size_bytes)
        manifest_result = None
        if record.manifest_path:
            try:
                mp = self.resolve(record.manifest_path, must_exist=True)
                manifest_result = self.verify_manifest(mp)
            except Exception:
                manifest_result = {"ok": False, "error": "manifest_missing"}
        return {
            "ok": bool(ok_hash and ok_size and (manifest_result is None or manifest_result.get("ok", False))),
            "hash_ok": bool(ok_hash),
            "size_ok": bool(ok_size),
            "expected_hash": record.hash_hex,
            "actual_hash": actual,
            "manifest": manifest_result,
        }

    # ------------------------------------------------------------------
    # Write/read artifacts
    # ------------------------------------------------------------------
    def _record_for_path(
        self,
        *,
        artifact_id: str,
        artifact_kind: Union[str, ArtifactKind],
        path: Path,
        manifest_path: Optional[Path],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ArtifactRecord:
        record = ArtifactRecord(
            artifact_id=str(artifact_id),
            artifact_kind=str(artifact_kind.value if isinstance(artifact_kind, ArtifactKind) else artifact_kind),
            public_path=self.public_path(path),
            size_bytes=int(path.stat().st_size),
            hash_hex=_hash_file(path),
            manifest_path=self.public_path(manifest_path) if manifest_path is not None else "",
            metadata=dict(metadata or {}),
        )
        return self.add_record(record)

    def write_bytes_artifact(
        self,
        artifact_id: str,
        data: Union[bytes, bytearray, memoryview],
        *,
        suffix: str = ".bin",
        artifact_kind: Union[str, ArtifactKind] = ArtifactKind.BYTES,
        subdir: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
        write_manifest: bool = True,
    ) -> ArtifactRecord:
        p = self._artifact_path(artifact_id, suffix, subdir=subdir)
        if len(bytes(data)) > self.path_policy.max_artifact_bytes:
            raise PathPolicyError("Data exceeds max_artifact_bytes.")
        atomic_write_bytes(p, data)
        manifest_path = None
        if write_manifest:
            manifest = self.build_manifest(p, artifact_kind=artifact_kind, artifact_id=artifact_id, metadata=metadata)
            manifest_path = self.write_manifest_sidecar(p, manifest)
        return self._record_for_path(artifact_id=artifact_id, artifact_kind=artifact_kind, path=p, manifest_path=manifest_path, metadata=metadata)

    def write_text_artifact(
        self,
        artifact_id: str,
        text: str,
        *,
        suffix: str = ".txt",
        subdir: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
        artifact_kind: Union[str, ArtifactKind] = ArtifactKind.TEXT,
    ) -> ArtifactRecord:
        return self.write_bytes_artifact(artifact_id, str(text).encode("utf-8"), suffix=suffix, artifact_kind=artifact_kind, subdir=subdir, metadata=metadata)

    def write_json_artifact(
        self,
        artifact_id: str,
        payload: Any,
        *,
        suffix: str = ".json",
        subdir: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
        artifact_kind: Union[str, ArtifactKind] = ArtifactKind.JSON,
        pretty: bool = True,
    ) -> ArtifactRecord:
        text = _canonical_dumps(payload, pretty=pretty)
        if not text.endswith("\n"):
            text += "\n"
        return self.write_bytes_artifact(artifact_id, text.encode("utf-8"), suffix=suffix, artifact_kind=artifact_kind, subdir=subdir, metadata=metadata)

    def read_bytes_artifact(self, public_or_relative_path: Union[str, os.PathLike[str]]) -> bytes:
        p = self.resolve(public_or_relative_path, must_exist=True)
        self.path_policy.validate_size(p)
        return p.read_bytes()

    def read_text_artifact(self, public_or_relative_path: Union[str, os.PathLike[str]], *, encoding: str = "utf-8") -> str:
        return self.read_bytes_artifact(public_or_relative_path).decode(encoding)

    def read_json_artifact(self, public_or_relative_path: Union[str, os.PathLike[str]]) -> Any:
        return json.loads(self.read_text_artifact(public_or_relative_path))

    def copy_in_artifact(
        self,
        src_path: Union[str, os.PathLike[str]],
        artifact_id: str,
        *,
        suffix: Optional[str] = None,
        subdir: str = "",
        artifact_kind: Union[str, ArtifactKind] = ArtifactKind.BYTES,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ArtifactRecord:
        src = Path(src_path).expanduser().resolve()
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(str(src))
        if int(src.stat().st_size) > self.path_policy.max_artifact_bytes:
            raise PathPolicyError("Source exceeds max_artifact_bytes.")
        suff = suffix or src.suffix or ".bin"
        dst = self._artifact_path(artifact_id, suff, subdir=subdir)
        copy_file_atomic(src, dst)
        manifest = self.build_manifest(dst, artifact_kind=artifact_kind, artifact_id=artifact_id, metadata=metadata)
        mp = self.write_manifest_sidecar(dst, manifest)
        return self._record_for_path(artifact_id=artifact_id, artifact_kind=artifact_kind, path=dst, manifest_path=mp, metadata=metadata)

    def write_npz_artifact(
        self,
        artifact_id: str,
        arrays: Optional[Mapping[str, Any]] = None,
        *,
        npz_writer: Optional[Any] = None,
        subdir: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
        compressed: bool = True,
    ) -> ArtifactRecord:
        if not _NUMPY_OK and arrays is not None:
            raise StorageError("NumPy is required for array-based NPZ writes.")
        p = self._artifact_path(artifact_id, ".npz", subdir=subdir)
        tmp = p.parent / f".{p.name}.{uuid.uuid4().hex}.tmp.npz"
        try:
            if npz_writer is not None:
                # Supports objects exposing save_npz(path), such as TensorTrain.
                npz_writer(tmp)
            else:
                arrs = dict(arrays or {})
                if compressed:
                    np.savez_compressed(tmp, **arrs)  # type: ignore[union-attr]
                else:
                    np.savez(tmp, **arrs)  # type: ignore[union-attr]
            # np.savez may append .npz if the path is not string-like enough in some versions; handle it.
            actual_tmp = tmp
            if not actual_tmp.exists() and Path(str(tmp) + ".npz").exists():
                actual_tmp = Path(str(tmp) + ".npz")
            try:
                with actual_tmp.open("r+b") as f:
                    os.fsync(f.fileno())
            except Exception:
                pass
            os.replace(actual_tmp, p)
            _atomic_fsync_dir(p.parent)
        finally:
            for candidate in (tmp, Path(str(tmp) + ".npz")):
                try:
                    if candidate.exists():
                        candidate.unlink()
                except Exception:
                    pass
        manifest = self.build_manifest(p, artifact_kind=ArtifactKind.NPZ, artifact_id=artifact_id, metadata=metadata)
        mp = self.write_manifest_sidecar(p, manifest)
        return self._record_for_path(artifact_id=artifact_id, artifact_kind=ArtifactKind.NPZ, path=p, manifest_path=mp, metadata=metadata)

    def export_manifest(self, artifact_id: str = "store_manifest") -> ArtifactRecord:
        idx = self.load_index().finalize()
        return self.write_json_artifact(artifact_id, idx.to_dict(), suffix=".json", subdir="manifests", artifact_kind=ArtifactKind.MANIFEST)

    def verify_all(self) -> Dict[str, Any]:
        results = []
        ok = True
        for record in self.records():
            r = self.verify_record(record)
            results.append({"artifact_id": record.artifact_id, **r})
            ok = ok and bool(r.get("ok", False))
        return {"ok": bool(ok), "record_count": len(results), "records": results}


# =============================================================================
# Specialized stores
# =============================================================================

class LedgerStore(ArtifactStore):
    """Safe persistence for full ledgers and ledger heads."""

    def __init__(self, root: Union[str, os.PathLike[str]], **kwargs: Any):
        super().__init__(root, namespace=kwargs.pop("namespace", "ledger"), **kwargs)

    def save_ledger(self, ledger_payload: Any, *, ledger_id: str = "ledger", metadata: Optional[Mapping[str, Any]] = None) -> ArtifactRecord:
        return self.write_json_artifact(ledger_id, ledger_payload, subdir="snapshots", artifact_kind=ArtifactKind.LEDGER, metadata=metadata)

    def load_ledger(self, ledger_id_or_path: str = "ledger.json") -> Any:
        record = self.find_record(ledger_id_or_path)
        path = record.public_path if record is not None else ledger_id_or_path
        return self.read_json_artifact(path)

    def save_head(self, head_payload: Mapping[str, Any], *, metadata: Optional[Mapping[str, Any]] = None) -> ArtifactRecord:
        return self.write_json_artifact("head", dict(head_payload), subdir="", artifact_kind=ArtifactKind.LEDGER, metadata=metadata)

    def load_head(self) -> Any:
        return self.read_json_artifact("head.json")


class BlockStore(ArtifactStore):
    """Safe block persistence with index/hash-addressable filenames."""

    def __init__(self, root: Union[str, os.PathLike[str]], **kwargs: Any):
        super().__init__(root, namespace=kwargs.pop("namespace", "blocks"), **kwargs)

    def _block_artifact_id(self, block: Mapping[str, Any]) -> str:
        idx = block.get("index", block.get("height", "unknown"))
        bh = str(block.get("block_hash", block.get("hash", block.get("commitment_hash", ""))))
        if bh:
            return f"block-{int(idx) if str(idx).isdigit() else _safe_name(idx)}-{bh[:16]}"
        return f"block-{_safe_name(idx)}-{_hash_object(block, domain='tb.storage.block-id.v1')[:16]}"

    def save_block(self, block_payload: Mapping[str, Any], *, metadata: Optional[Mapping[str, Any]] = None) -> ArtifactRecord:
        aid = self._block_artifact_id(block_payload)
        return self.write_json_artifact(aid, dict(block_payload), subdir="by_height", artifact_kind=ArtifactKind.BLOCK, metadata=metadata)

    def load_block_by_record(self, artifact_id: str) -> Any:
        r = self.find_record(artifact_id)
        if r is None:
            raise KeyError(artifact_id)
        return self.read_json_artifact(r.public_path)

    def list_block_records(self) -> List[ArtifactRecord]:
        return [r for r in self.records() if r.artifact_kind == ArtifactKind.BLOCK.value]


class TensorArtifactStore(ArtifactStore):
    """Safe tensor artifact storage for NPZ TensorTrain/Tucker artifacts."""

    def __init__(self, root: Union[str, os.PathLike[str]], **kwargs: Any):
        super().__init__(root, namespace=kwargs.pop("namespace", "tensors"), **kwargs)

    def save_tensor_npz(self, tensor_obj: Any, *, artifact_id: str, metadata: Optional[Mapping[str, Any]] = None) -> ArtifactRecord:
        if not hasattr(tensor_obj, "save_npz") or not callable(tensor_obj.save_npz):
            raise TypeError("tensor_obj must expose save_npz(path)")
        def writer(path: Path) -> None:
            tensor_obj.save_npz(path)
        meta = dict(metadata or {})
        if hasattr(tensor_obj, "commitment") and callable(tensor_obj.commitment):
            try:
                commitment = tensor_obj.commitment()
                meta["tensor_commitment"] = _json_safe(commitment)
            except Exception as exc:
                meta["tensor_commitment_error"] = repr(exc)
        return self.write_npz_artifact(artifact_id, npz_writer=writer, subdir="npz", metadata=meta)

    def save_array_npz(self, arrays: Mapping[str, Any], *, artifact_id: str, metadata: Optional[Mapping[str, Any]] = None) -> ArtifactRecord:
        return self.write_npz_artifact(artifact_id, arrays=arrays, subdir="npz", metadata=metadata)


class LobeCheckpointStore(ArtifactStore):
    """Safe storage for NCP/CfC lobe checkpoints and state receipts."""

    def __init__(self, root: Union[str, os.PathLike[str]], **kwargs: Any):
        super().__init__(root, namespace=kwargs.pop("namespace", "lobes"), **kwargs)

    def save_lobe_checkpoint(self, lobe_obj: Any, *, artifact_id: str, metadata: Optional[Mapping[str, Any]] = None) -> ArtifactRecord:
        if not hasattr(lobe_obj, "save_checkpoint") or not callable(lobe_obj.save_checkpoint):
            raise TypeError("lobe_obj must expose save_checkpoint(path)")
        p = self._artifact_path(artifact_id, ".npz", subdir="checkpoints")
        tmp = p.parent / f".{p.name}.{uuid.uuid4().hex}.tmp.npz"
        try:
            lobe_obj.save_checkpoint(tmp)
            actual_tmp = tmp if tmp.exists() else Path(str(tmp) + ".npz")
            if not actual_tmp.exists():
                raise StorageError("Lobe checkpoint writer did not create an NPZ file.")
            os.replace(actual_tmp, p)
            _atomic_fsync_dir(p.parent)
        finally:
            for candidate in (tmp, Path(str(tmp) + ".npz")):
                try:
                    if candidate.exists():
                        candidate.unlink()
                except Exception:
                    pass
        meta = dict(metadata or {})
        if hasattr(lobe_obj, "lobe_commitment") and callable(lobe_obj.lobe_commitment):
            try:
                meta["lobe_commitment"] = _json_safe(lobe_obj.lobe_commitment())
            except Exception as exc:
                meta["lobe_commitment_error"] = repr(exc)
        elif hasattr(lobe_obj, "snapshot") and callable(lobe_obj.snapshot):
            try:
                meta["snapshot_hash"] = _hash_object(lobe_obj.snapshot(), domain="tb.storage.lobe-snapshot.v1")
            except Exception:
                pass
        manifest = self.build_manifest(p, artifact_kind=ArtifactKind.LOBE_CHECKPOINT, artifact_id=artifact_id, metadata=meta)
        mp = self.write_manifest_sidecar(p, manifest)
        return self._record_for_path(artifact_id=artifact_id, artifact_kind=ArtifactKind.LOBE_CHECKPOINT, path=p, manifest_path=mp, metadata=meta)

    def save_lobe_state_json(self, state_payload: Mapping[str, Any], *, artifact_id: str, metadata: Optional[Mapping[str, Any]] = None) -> ArtifactRecord:
        return self.write_json_artifact(artifact_id, dict(state_payload), subdir="states", artifact_kind=ArtifactKind.LOBE_CHECKPOINT, metadata=metadata)


class DemoArtifactStore(ArtifactStore):
    """Safe export store for demo reports and app artifacts."""

    def __init__(self, root: Union[str, os.PathLike[str]], **kwargs: Any):
        super().__init__(root, namespace=kwargs.pop("namespace", "demo"), **kwargs)

    def export_json_report(self, report_payload: Any, *, artifact_id: str = "demo_report", metadata: Optional[Mapping[str, Any]] = None) -> ArtifactRecord:
        return self.write_json_artifact(artifact_id, report_payload, subdir="reports", artifact_kind=ArtifactKind.DEMO, metadata=metadata)

    def export_markdown(self, markdown: str, *, artifact_id: str = "demo_report", metadata: Optional[Mapping[str, Any]] = None) -> ArtifactRecord:
        return self.write_text_artifact(artifact_id, markdown, suffix=".md", subdir="reports", artifact_kind=ArtifactKind.DEMO, metadata=metadata)

    def export_bundle_manifest(self, *, artifact_id: str = "demo_bundle_manifest") -> ArtifactRecord:
        return self.export_manifest(artifact_id=artifact_id)


# =============================================================================
# Convenience construction
# =============================================================================

def build_project_stores(root: Union[str, os.PathLike[str]], *, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, ArtifactStore]:
    root_path = Path(root).expanduser().resolve()
    policy = SafePathPolicy(root_path)
    return {
        "ledger": LedgerStore(root_path, path_policy=policy, metadata=metadata),
        "blocks": BlockStore(root_path, path_policy=policy, metadata=metadata),
        "tensors": TensorArtifactStore(root_path, path_policy=policy, metadata=metadata),
        "lobes": LobeCheckpointStore(root_path, path_policy=policy, metadata=metadata),
        "demo": DemoArtifactStore(root_path, path_policy=policy, metadata=metadata),
    }


def storage_status(root: Optional[Union[str, os.PathLike[str]]] = None) -> Dict[str, Any]:
    payload = {
        "ok": True,
        "version": TB_STORAGE_VERSION,
        "ts": _now_iso_utc(),
        "canonical_available": bool(_CANONICAL_OK),
        "hashing_available": bool(_HASHING_OK),
        "numpy_available": bool(_NUMPY_OK),
    }
    if root is not None:
        rp = Path(root).expanduser().resolve()
        payload["root"] = str(rp)
        payload["exists"] = rp.exists()
    return payload


# =============================================================================
# Self-test / CLI
# =============================================================================

def run_self_test() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="tb_storage_selftest_") as td:
        root = Path(td)
        stores = build_project_stores(root, metadata={"self_test": True})
        ledger = stores["ledger"]
        blocks = stores["blocks"]
        tensors = stores["tensors"]
        demo = stores["demo"]

        ledger_payload = {"chain_id": "selftest", "blocks": [], "head_hash": ZERO_HASH_256}
        ledger_record = ledger.save_ledger(ledger_payload, ledger_id="ledger_selftest")  # type: ignore[attr-defined]
        loaded_ledger = ledger.read_json_artifact(ledger_record.public_path)
        if loaded_ledger.get("chain_id") != "selftest":
            raise AssertionError("ledger JSON round-trip failed")

        block_payload = {"index": 1, "block_hash": "a" * 64, "transactions": []}
        block_record = blocks.save_block(block_payload)  # type: ignore[attr-defined]
        block_verify = blocks.verify_record(block_record)
        if not block_verify.get("ok"):
            raise AssertionError(f"block record verification failed: {block_verify}")

        npz_record = None
        if _NUMPY_OK:
            npz_record = tensors.save_array_npz({"x": np.arange(8, dtype=np.float32)}, artifact_id="tensor_probe")  # type: ignore[union-attr]
            npz_verify = tensors.verify_record(npz_record)
            if not npz_verify.get("ok"):
                raise AssertionError(f"npz record verification failed: {npz_verify}")

        demo_record = demo.export_json_report({"ok": True, "records": 3}, artifact_id="demo_report")  # type: ignore[attr-defined]
        manifest_record = demo.export_bundle_manifest()

        # Path traversal must fail.
        traversal_blocked = False
        try:
            ledger.write_text_artifact("bad", "x", subdir="../escape")
        except PathPolicyError:
            traversal_blocked = True
        if not traversal_blocked:
            raise AssertionError("path traversal was not blocked")

        verify = {
            "ledger": ledger.verify_all(),
            "blocks": blocks.verify_all(),
            "tensors": tensors.verify_all(),
            "demo": demo.verify_all(),
        }
        return {
            "ok": True,
            "version": TB_STORAGE_VERSION,
            "root": str(root),
            "records": {
                "ledger": ledger_record.to_dict(),
                "block": block_record.to_dict(),
                "npz": None if npz_record is None else npz_record.to_dict(),
                "demo": demo_record.to_dict(),
                "manifest": manifest_record.to_dict(),
            },
            "verification": verify,
            "path_traversal_blocked": traversal_blocked,
        }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tensorized Blockchain safe storage layer")
    p.add_argument("--mode", choices=["status", "self-test"], default="status")
    p.add_argument("--root", default="", help="Optional storage root for status.")
    p.add_argument("--pretty", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        if args.mode == "self-test":
            out = run_self_test()
        else:
            out = storage_status(args.root or None)
        print(json.dumps(_json_safe(out), indent=2 if args.pretty else None, ensure_ascii=False))
        return 0 if bool(out.get("ok", False)) else 2
    except Exception as exc:
        err = {"ok": False, "error": repr(exc), "version": TB_STORAGE_VERSION}
        print(json.dumps(_json_safe(err), indent=2, ensure_ascii=False))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
