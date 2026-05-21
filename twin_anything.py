#!/usr/bin/env python3
# twin_anything.py
r"""
Project Chimera / Akkurat - Twin Anything Factory
=================================================

Production canonical twin materialization layer for the Tensorized Blockchain
stack. This module creates deterministic, auditable, governed digital-twin
branches for arbitrary objects, robots, processes, plans, curricula, documents,
machines, teams, organizations, environments, software processes, and abstract
operations.

Design boundaries
-----------------
- digital_twin_kernel.py owns canonical world-state, commitments, policy, and
  governed execution.
- tn.py owns low-level tensor-network algebra.
- tensorized_blockchain_kernel.py owns transactions, blocks, consensus, ledger
  persistence, and chain validation.
- this module owns canonical twin schema materialization, source manifests,
  creation receipts, and serializable factory snapshots.

Key production properties
-------------------------
- Deterministic creation mode with stable clocks and canonical input hashing.
- TwinCreationReceipt linking input hash, schema hash, source-manifest root,
  pre/post state commitments, and diagnostics.
- TwinSchema and TwinTypeRegistry for explicit materialization behavior.
- Safe local-file hashing policy; absolute local paths are never exposed in
  public snapshots by default.
- Separate runtime API and serializable snapshot API.
- Structured diagnostics instead of silent fallbacks.

Compatibility
-------------
The historic create_twin(...) API is retained and returns live runtime objects:
    {"network": ..., "control_plane": ..., "metadata": ..., "node_map": ..., "snapshot": ...}

New production callers should prefer:
    create_twin_runtime(...)
    create_twin_snapshot(...)
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib
import json
import math
import mimetypes
import os
import platform
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np


# =============================================================================
# Version / schemas
# =============================================================================

TWIN_ANYTHING_VERSION = "twin-anything-v4.0-production-materialization"
TWIN_SCHEMA_VERSION = "ProjectChimera.TwinSchema.v1"
TWIN_CREATION_RECEIPT_SCHEMA = "ProjectChimera.TwinCreationReceipt.v1"
SOURCE_MANIFEST_SCHEMA = "ProjectChimera.SourceFileManifest.v1"

_DOMAIN_SCHEMA = b"ProjectChimera:TwinSchema:v1:"
_DOMAIN_RECEIPT = b"ProjectChimera:TwinCreationReceipt:v1:"
_DOMAIN_SOURCE_LEAF = b"ProjectChimera:SourceManifestLeaf:v1:"
_DOMAIN_SOURCE_NODE = b"ProjectChimera:SourceManifestNode:v1:"
_DOMAIN_JSON = b"ProjectChimera:TwinAnything:JSON:v1:"


# =============================================================================
# Path bootstrap / dependency imports
# =============================================================================

def _module_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()


def _add_path(path: Union[str, os.PathLike, None], *, prepend: bool = True) -> Optional[Path]:
    if path is None:
        return None
    try:
        p = Path(path).expanduser().resolve()
    except Exception:
        return None
    if not p.exists():
        return None
    s = str(p)
    if prepend:
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)
    else:
        if s not in sys.path:
            sys.path.append(s)
    return p


def configure_paths(
    *,
    project_root: Optional[Union[str, os.PathLike]] = None,
    permissive: bool = True,
) -> Dict[str, Optional[str]]:
    import tempfile
    """Locate project-local modules.

    Production package deployments should use ordinary imports and pass
    permissive=False. Permissive mode is retained for local demos and notebooks.
    """
    here = _module_dir()
    candidates: List[Union[str, os.PathLike, None]] = [project_root, here]
    if permissive:
        candidates.extend([
            os.environ.get("TENSORIZED_BLOCKCHAIN_ROOT"),
            os.environ.get("LK20_ROOT"),
            os.environ.get("AKKURAT_ROOT"),
            os.environ.get("AKKURAT_COGNITIVE_ROOT"),
            here.parent,
            here.parent / "app",
            here.parent / "Akkurat",
            here.parent / "Akkurat" / "qwen3.5_4b",
            Path.cwd(),
            Path(tempfile.gettempdir()),
        ])

    found_dtk = None
    found_tn = None
    found_self = None
    for candidate in candidates:
        p = _add_path(candidate, prepend=True)
        if p is None:
            continue
        if found_dtk is None and (p / "digital_twin_kernel.py").exists():
            found_dtk = p
        if found_tn is None and (p / "tn.py").exists():
            found_tn = p
        if found_self is None and (p / "twin_anything.py").exists():
            found_self = p

    _add_path(here, prepend=True)
    return {
        "module_dir": str(here),
        "project_root": None if project_root is None else str(Path(project_root).expanduser()),
        "digital_twin_kernel_root": None if found_dtk is None else str(found_dtk),
        "tn_root": None if found_tn is None else str(found_tn),
        "twin_anything_root": None if found_self is None else str(found_self),
    }


_PATHS = configure_paths()

try:
    dtk = importlib.import_module("digital_twin_kernel")
except Exception as exc:  # pragma: no cover - import environment failure
    raise ImportError(
        "Missing required dependency digital_twin_kernel.py. Ensure it is in PYTHONPATH "
        "or set TENSORIZED_BLOCKCHAIN_ROOT / AKKURAT_ROOT."
    ) from exc

try:
    tn = importlib.import_module("tn")
    _TN_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - optional dependency
    tn = None
    _TN_IMPORT_ERROR = repr(exc)


# =============================================================================
# Helpers
# =============================================================================

def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(_json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8", errors="ignore")


def _sha256_hex_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_hex_obj(obj: Any) -> str:
    return _sha256_hex_bytes(_DOMAIN_JSON + _canonical_json_bytes(obj))


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        arr = np.nan_to_num(np.asarray(obj), nan=0.0, posinf=0.0, neginf=0.0)
        return arr.astype(float).tolist()
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in sorted(dict(obj).items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in list(obj)]
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


def _atomic_write_text(path: Union[str, os.PathLike], text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def _write_json(path: Union[str, os.PathLike], payload: Any) -> None:
    _atomic_write_text(path, json.dumps(_json_safe(payload), indent=2, ensure_ascii=False))


def _read_json(path: Union[str, os.PathLike], default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return copy.deepcopy(default)
    return json.loads(p.read_text(encoding="utf-8"))


def _slug(value: Any, default: str = "thing", max_len: int = 96) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = default
    if s[0].isdigit():
        s = f"{default}_{s}"
    return s[: int(max_len)]


def _as_mapping(x: Any) -> Dict[str, Any]:
    return dict(x) if isinstance(x, Mapping) else {}


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return list(x)
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, set):
        return list(x)
    return [x]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _guess_mime(filename: str) -> str:
    if not filename:
        return "application/octet-stream"
    guess, _ = mimetypes.guess_type(filename)
    return guess or "application/octet-stream"


def _sha256_file(path: Union[str, os.PathLike]) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _merkle_root_hex(leaves: Sequence[str], *, leaf_domain: bytes = _DOMAIN_SOURCE_LEAF, node_domain: bytes = _DOMAIN_SOURCE_NODE) -> str:
    if not leaves:
        return _sha256_hex_bytes(node_domain + b"<empty>")
    level = [_sha256_hex_bytes(leaf_domain + str(x).encode("utf-8", errors="ignore")) for x in leaves]
    raw = [bytes.fromhex(x) for x in level]
    while len(raw) > 1:
        if len(raw) % 2 == 1:
            raw.append(raw[-1])
        nxt = []
        for i in range(0, len(raw), 2):
            nxt.append(hashlib.sha256(node_domain + raw[i] + raw[i + 1]).digest())
        raw = nxt
    return raw[0].hex()


def _diagnostic_to_dict(x: Any) -> Dict[str, Any]:
    if hasattr(x, "to_dict") and callable(x.to_dict):
        try:
            return x.to_dict()
        except Exception:
            pass
    return _json_safe(x)


# =============================================================================
# Diagnostics
# =============================================================================

@dataclass
class DiagnosticEvent:
    code: str
    severity: str
    component: str
    message: str
    recoverable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


class DiagnosticLog:
    def __init__(self) -> None:
        self.events: List[DiagnosticEvent] = []

    def add(self, code: str, severity: str, component: str, message: str, *, recoverable: bool = True, **metadata: Any) -> None:
        self.events.append(DiagnosticEvent(code=str(code), severity=str(severity), component=str(component), message=str(message), recoverable=bool(recoverable), metadata=_json_safe(metadata)))

    def extend(self, events: Sequence[DiagnosticEvent]) -> None:
        self.events.extend(list(events))

    def to_list(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.events]

    @property
    def has_error(self) -> bool:
        return any(e.severity.lower() in {"error", "critical"} for e in self.events)


# =============================================================================
# Dependency status
# =============================================================================

def digital_twin_status() -> Dict[str, Any]:
    attrs = {
        "DigitalTwinsBuilder": hasattr(dtk, "DigitalTwinsBuilder"),
        "TreeTensorNetwork": hasattr(dtk, "TreeTensorNetwork"),
        "AkkuratInterface": hasattr(dtk, "AkkuratInterface"),
        "NodeKind": hasattr(dtk, "NodeKind"),
        "FusionConfig": hasattr(dtk, "FusionConfig"),
        "FusionMode": hasattr(dtk, "FusionMode"),
        "LatentGeometry": hasattr(dtk, "LatentGeometry"),
        "StateCommitment": hasattr(dtk, "StateCommitment"),
        "ActionReceipt": hasattr(dtk, "ActionReceipt"),
        "PolicyReceipt": hasattr(dtk, "PolicyReceipt"),
        "RuntimeClock": hasattr(dtk, "RuntimeClock"),
        "Action": hasattr(dtk, "Action"),
    }
    return {
        "available": True,
        "module": getattr(dtk, "__file__", ""),
        "version": getattr(dtk, "DIGITAL_TWIN_KERNEL_VERSION", ""),
        "attrs": attrs,
        "ok": bool(all(attrs.values())),
    }


def tn_status() -> Dict[str, Any]:
    if tn is None:
        return {"available": False, "module": None, "error": _TN_IMPORT_ERROR, "attrs": {}, "ok": False}
    attrs = {
        "TensorTrain": hasattr(tn, "TensorTrain"),
        "TTConfig": hasattr(tn, "TTConfig"),
        "TensorCommitment": hasattr(tn, "TensorCommitment"),
        "TuckerFusionLayer": hasattr(tn, "TuckerFusionLayer"),
        "factorize_into_modes": hasattr(tn, "factorize_into_modes"),
        "get_factors": hasattr(tn, "get_factors"),
    }
    return {
        "available": True,
        "module": getattr(tn, "__file__", ""),
        "version": getattr(tn, "TN_VERSION", ""),
        "attrs": attrs,
        "ok": bool(attrs["TensorTrain"] and attrs["TTConfig"]),
    }


def status_report() -> Dict[str, Any]:
    dstatus = digital_twin_status()
    tstatus = tn_status()
    return {
        "ok": bool(dstatus.get("ok", False)),
        "status": "ok" if dstatus.get("ok", False) else "failed",
        "ts": _now_iso_utc(),
        "version": TWIN_ANYTHING_VERSION,
        "schema_version": TWIN_SCHEMA_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "paths": configure_paths(),
        "digital_twin_kernel": dstatus,
        "tn": tstatus,
    }


# =============================================================================
# Config dataclasses
# =============================================================================

@dataclass
class TensorNetworkRuntimeConfig:
    enable_tn_profile: bool = True
    prefer_tn_projection: bool = True
    prefer_tucker_fusion: bool = True
    tt_rank: int = 8
    tucker_rank: int = 16
    tt_num_modes: int = 2
    dtype: str = "float32"
    device: str = "cpu"
    check_finite: bool = True
    run_smoke_tests: bool = False
    include_runtime_paths: bool = False

    def normalized(self) -> "TensorNetworkRuntimeConfig":
        cfg = copy.deepcopy(self)
        cfg.enable_tn_profile = bool(cfg.enable_tn_profile)
        cfg.prefer_tn_projection = bool(cfg.prefer_tn_projection)
        cfg.prefer_tucker_fusion = bool(cfg.prefer_tucker_fusion)
        cfg.tt_rank = int(max(1, cfg.tt_rank))
        cfg.tucker_rank = int(max(1, cfg.tucker_rank))
        cfg.tt_num_modes = int(max(1, cfg.tt_num_modes))
        cfg.dtype = str(cfg.dtype or "float32").lower().strip()
        if cfg.dtype not in {"float16", "float32", "float64"}:
            cfg.dtype = "float32"
        cfg.device = str(cfg.device or "cpu").lower().strip()
        if cfg.device not in {"cpu", "gpu"}:
            cfg.device = "cpu"
        cfg.check_finite = bool(cfg.check_finite)
        cfg.run_smoke_tests = bool(cfg.run_smoke_tests)
        cfg.include_runtime_paths = bool(cfg.include_runtime_paths)
        return cfg


@dataclass
class SafeFileHashPolicy:
    allow_local_file_hashing: bool = False
    allowed_roots: List[str] = field(default_factory=list)
    expose_relative_paths: bool = True
    expose_absolute_paths: bool = False
    max_hash_bytes: int = 256 * 1024 * 1024

    def normalized(self) -> "SafeFileHashPolicy":
        p = copy.deepcopy(self)
        p.allow_local_file_hashing = bool(p.allow_local_file_hashing)
        p.allowed_roots = [str(Path(x).expanduser().resolve()) for x in list(p.allowed_roots or []) if str(x or "").strip()]
        p.expose_relative_paths = bool(p.expose_relative_paths)
        p.expose_absolute_paths = bool(p.expose_absolute_paths)
        p.max_hash_bytes = int(max(1, p.max_hash_bytes))
        return p

    def is_allowed(self, path: Union[str, os.PathLike]) -> Tuple[bool, str, Optional[str]]:
        p = Path(path).expanduser()
        try:
            rp = p.resolve()
        except Exception as exc:
            return False, f"path_resolve_failed:{exc}", None
        if not self.allow_local_file_hashing:
            return False, "local_file_hashing_disabled", None
        if not rp.exists() or not rp.is_file():
            return False, "not_a_file", None
        try:
            size = rp.stat().st_size
            if size > self.max_hash_bytes:
                return False, "file_exceeds_max_hash_bytes", None
        except Exception:
            return False, "file_stat_failed", None
        roots = [Path(x).resolve() for x in self.allowed_roots]
        if not roots:
            return False, "no_allowed_roots_configured", None
        for root in roots:
            if _is_relative_to(rp, root):
                rel = str(rp.relative_to(root)).replace("\\", "/")
                return True, "ok", rel
        return False, "outside_allowed_roots", None


@dataclass
class TwinAnythingConfig:
    vector_dim: int = 256
    sketch_dim: int = 96
    history_capacity: int = 64
    seed: int = 2027
    use_tn_projection: bool = True
    latent_geometry: str = "euclidean"
    deterministic: bool = False
    deterministic_epoch: int = 0
    consensus_safe: bool = True

    install_platform_taxonomy: bool = True
    install_success_defaults: bool = True
    install_tensor_network_profile: bool = True

    thing_root_id: str = "5.0"
    project_root: str = ""
    permissive_path_search: bool = True

    tn_runtime: TensorNetworkRuntimeConfig = field(default_factory=TensorNetworkRuntimeConfig)
    file_hash_policy: SafeFileHashPolicy = field(default_factory=SafeFileHashPolicy)

    def normalized(self) -> "TwinAnythingConfig":
        cfg = copy.deepcopy(self)
        cfg.vector_dim = int(max(8, cfg.vector_dim))
        cfg.sketch_dim = int(max(8, min(int(cfg.sketch_dim), cfg.vector_dim)))
        cfg.history_capacity = int(max(1, cfg.history_capacity))
        cfg.seed = int(cfg.seed)
        cfg.use_tn_projection = bool(cfg.use_tn_projection)
        cfg.deterministic = bool(cfg.deterministic)
        cfg.deterministic_epoch = int(cfg.deterministic_epoch)
        cfg.consensus_safe = bool(cfg.consensus_safe)
        geom = str(cfg.latent_geometry or "euclidean").lower().strip()
        if geom in {"poincare", "poincaré"}:
            geom = "hyperbolic"
        if geom not in {"euclidean", "hyperbolic"}:
            geom = "euclidean"
        cfg.latent_geometry = geom
        cfg.install_platform_taxonomy = bool(cfg.install_platform_taxonomy)
        cfg.install_success_defaults = bool(cfg.install_success_defaults)
        cfg.install_tensor_network_profile = bool(cfg.install_tensor_network_profile)
        cfg.thing_root_id = str(cfg.thing_root_id or "5.0")
        cfg.project_root = str(cfg.project_root or "")
        cfg.permissive_path_search = bool(cfg.permissive_path_search)
        cfg.tn_runtime = cfg.tn_runtime.normalized()
        cfg.file_hash_policy = cfg.file_hash_policy.normalized()
        return cfg

    def clock(self) -> Any:
        if hasattr(dtk, "RuntimeClock"):
            return dtk.RuntimeClock(deterministic=self.deterministic, fixed_epoch=self.deterministic_epoch, tick=0)
        return None


@dataclass
class TwinInput:
    thing_name: str
    thing_type: str = "generic"
    description: str = ""
    observables: Dict[str, Any] = field(default_factory=dict)
    process_steps: List[str] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    components: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    risks: List[Any] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    source_files: List[Dict[str, Any]] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TwinInput":
        p = dict(payload or {})
        return cls(
            thing_name=str(p.get("thing_name", p.get("name", "Unnamed Thing"))),
            thing_type=str(p.get("thing_type", p.get("type", "generic"))),
            description=str(p.get("description", "")),
            observables=_as_mapping(p.get("observables", {})),
            process_steps=[str(x) for x in _as_list(p.get("process_steps", []))],
            resources=_as_mapping(p.get("resources", {})),
            constraints=[str(x) for x in _as_list(p.get("constraints", []))],
            components=_as_mapping(p.get("components", {})),
            environment=_as_mapping(p.get("environment", {})),
            success_criteria=_as_mapping(p.get("success_criteria", {})),
            risks=_as_list(p.get("risks", [])),
            actions=_as_list(p.get("actions", [])),
            source_files=[_as_mapping(x) for x in _as_list(p.get("source_files", []))],
            links=[str(x) for x in _as_list(p.get("links", []))],
            tags=[str(x) for x in _as_list(p.get("tags", []))],
            metadata=_as_mapping(p.get("metadata", {})),
        ).normalized()

    def normalized(self) -> "TwinInput":
        t = copy.deepcopy(self)
        t.thing_name = str(t.thing_name or "Unnamed Thing")
        t.thing_type = _slug(t.thing_type or "generic", default="generic")
        t.description = str(t.description or "")
        t.observables = _as_mapping(t.observables)
        t.process_steps = [str(x) for x in _as_list(t.process_steps)]
        t.resources = _as_mapping(t.resources)
        t.constraints = [str(x) for x in _as_list(t.constraints)]
        t.components = _as_mapping(t.components)
        t.environment = _as_mapping(t.environment)
        t.success_criteria = _as_mapping(t.success_criteria)
        t.risks = _as_list(t.risks)
        t.actions = _as_list(t.actions)
        t.source_files = [_as_mapping(x) for x in _as_list(t.source_files)]
        t.links = [str(x) for x in _as_list(t.links)]
        t.tags = [str(x) for x in _as_list(t.tags)]
        t.metadata = _as_mapping(t.metadata)
        return t

    def input_hash(self) -> str:
        return _sha256_hex_obj(asdict(self.normalized()))


# =============================================================================
# Schema, source manifests, and receipts
# =============================================================================

@dataclass(frozen=True)
class TwinSchemaNode:
    node_id: str
    name: str
    kind: str = "generic"
    parent_id: str = "5.0"
    role: str = "universal"
    committed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class TwinSchema:
    schema_version: str = TWIN_SCHEMA_VERSION
    name: str = "universal_thing_twin"
    root_id: str = "5.0"
    nodes: List[TwinSchemaNode] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def universal(root_id: str = "5.0") -> "TwinSchema":
        specs = [
            (root_id, "Thing Twin", "generic", "0", "root"),
            ("5.1", "Identity / Purpose", "data", root_id, "identity"),
            ("5.2", "Environment / Context", "physical", root_id, "environment"),
            ("5.3", "Components / Structure", "entity", root_id, "components"),
            ("5.4", "Observables / Telemetry", "observable", root_id, "observables"),
            ("5.5", "Process / Behavior Model", "model", root_id, "process"),
            ("5.6", "Resources", "resource", root_id, "resources"),
            ("5.7", "Constraints / Governance", "governance", root_id, "constraints"),
            ("5.8", "Success Criteria", "observable", root_id, "success"),
            ("5.9", "Risks / Failure Modes", "governance", root_id, "risks"),
            ("5.10", "Actions / Actuators", "actuator", root_id, "actions"),
            ("5.11", "Memory / Logs / Snapshots", "storage", root_id, "memory"),
            ("5.12", "Tensor-Network Runtime Profile", "model", root_id, "tn_profile"),
        ]
        return TwinSchema(root_id=root_id, nodes=[TwinSchemaNode(node_id=a, name=b, kind=c, parent_id=d, role=e) for a, b, c, d, e in specs], metadata={"factory_version": TWIN_ANYTHING_VERSION})

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": self.schema_version, "name": self.name, "root_id": self.root_id, "nodes": [n.to_dict() for n in self.nodes], "metadata": _json_safe(self.metadata)}

    def hash(self) -> str:
        return _sha256_hex_bytes(_DOMAIN_SCHEMA + _canonical_json_bytes(self.to_dict()))


@dataclass
class SourceFileManifest:
    schema_version: str = SOURCE_MANIFEST_SCHEMA
    filename: str = ""
    relative_path: str = ""
    mime_type: str = ""
    sha256: str = ""
    size_bytes: Optional[int] = None
    role: str = "source"
    source_ref: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    hash_status: str = "not_requested"
    diagnostic: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, policy: SafeFileHashPolicy, diagnostics: DiagnosticLog) -> "SourceFileManifest":
        p = dict(payload or {})
        raw_path = str(p.get("file_path", p.get("path", "")) or "")
        filename = str(p.get("filename", "") or "")
        if raw_path and not filename:
            filename = Path(raw_path).name
        sha = str(p.get("sha256", p.get("file_sha256", "")) or "")
        rel = str(p.get("relative_path", "") or "")
        size_bytes = None
        status = "provided_hash" if sha else "not_requested"
        diagnostic = ""
        if raw_path and not rel:
            allowed, reason, rel_candidate = policy.is_allowed(raw_path)
            if allowed:
                rel = rel_candidate or ""
                try:
                    rp = Path(raw_path).expanduser().resolve()
                    size_bytes = int(rp.stat().st_size)
                    if not sha:
                        sha = _sha256_file(rp)
                        status = "hashed"
                except Exception as exc:
                    status = "hash_failed"
                    diagnostic = repr(exc)
                    diagnostics.add("source_hash_failed", "warning", "SourceFileManifest", f"Failed to hash source file {filename or raw_path}: {exc}", reason=reason)
            else:
                status = "not_hashed"
                diagnostic = reason
                diagnostics.add("source_hash_skipped", "info", "SourceFileManifest", f"Source file hash skipped for {filename or Path(raw_path).name}: {reason}", reason=reason)
        elif raw_path and policy.expose_relative_paths and not rel:
            rel = Path(raw_path).name
        if size_bytes is None:
            try:
                if raw_path and policy.allow_local_file_hashing and Path(raw_path).expanduser().exists():
                    size_bytes = int(Path(raw_path).expanduser().stat().st_size)
            except Exception:
                pass
        mime = str(p.get("mime_type", "") or _guess_mime(filename))
        role = _slug(p.get("role", "source"), default="source")
        source_ref = str(p.get("source_ref", "") or "")
        if not source_ref:
            source_ref = f"sha256:{sha}" if sha else (f"file:{filename}" if filename else "source:unknown")
        meta = _as_mapping(p.get("metadata", {}))
        # Never expose absolute paths by default.
        if policy.expose_absolute_paths and raw_path:
            meta["absolute_path"] = str(Path(raw_path).expanduser().resolve())
        return cls(filename=filename, relative_path=rel if policy.expose_relative_paths else "", mime_type=mime, sha256=sha, size_bytes=size_bytes, role=role, source_ref=source_ref, metadata=meta, hash_status=status, diagnostic=diagnostic)

    def canonical_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "filename": self.filename,
            "relative_path": self.relative_path,
            "mime_type": self.mime_type,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "role": self.role,
            "source_ref": self.source_ref,
            "metadata": _json_safe(self.metadata),
            "hash_status": self.hash_status,
            "diagnostic": self.diagnostic,
        }

    def manifest_hash(self) -> str:
        return _sha256_hex_obj(self.canonical_payload())

    def to_dict(self) -> Dict[str, Any]:
        out = self.canonical_payload()
        out["manifest_hash"] = self.manifest_hash()
        return out


def source_manifest_root(manifests: Sequence[SourceFileManifest]) -> str:
    leaves = [m.manifest_hash() for m in sorted(manifests, key=lambda x: (x.sha256, x.filename, x.source_ref))]
    return _merkle_root_hex(leaves)


@dataclass
class TwinCreationReceipt:
    schema_version: str
    factory_version: str
    thing_name: str
    thing_type: str
    thing_slug: str
    schema_hash: str
    input_hash: str
    source_manifest_root: str
    pre_state_commitment: Dict[str, Any]
    post_state_commitment: Dict[str, Any]
    deterministic: bool
    consensus_safe: bool
    node_count: int
    node_map_hash: str
    diagnostics_hash: str
    receipt_hash: str = ""

    @staticmethod
    def create(
        *,
        thing_input: TwinInput,
        schema_hash: str,
        source_root: str,
        pre_state_commitment: Mapping[str, Any],
        post_state_commitment: Mapping[str, Any],
        deterministic: bool,
        consensus_safe: bool,
        node_count: int,
        node_map: Mapping[str, str],
        diagnostics: Sequence[Mapping[str, Any]],
    ) -> "TwinCreationReceipt":
        obj = TwinCreationReceipt(
            schema_version=TWIN_CREATION_RECEIPT_SCHEMA,
            factory_version=TWIN_ANYTHING_VERSION,
            thing_name=thing_input.thing_name,
            thing_type=thing_input.thing_type,
            thing_slug=_slug(thing_input.thing_name),
            schema_hash=str(schema_hash),
            input_hash=thing_input.input_hash(),
            source_manifest_root=str(source_root),
            pre_state_commitment=_json_safe(pre_state_commitment),
            post_state_commitment=_json_safe(post_state_commitment),
            deterministic=bool(deterministic),
            consensus_safe=bool(consensus_safe),
            node_count=int(node_count),
            node_map_hash=_sha256_hex_obj(dict(sorted(node_map.items()))),
            diagnostics_hash=_sha256_hex_obj(list(diagnostics)),
            receipt_hash="",
        )
        obj.receipt_hash = obj.compute_hash()
        return obj

    def payload(self, *, include_receipt_hash: bool = False) -> Dict[str, Any]:
        out = _json_safe(asdict(self))
        if not include_receipt_hash:
            out.pop("receipt_hash", None)
        return out

    def compute_hash(self) -> str:
        return _sha256_hex_bytes(_DOMAIN_RECEIPT + _canonical_json_bytes(self.payload()))

    def verify(self) -> bool:
        return self.receipt_hash == self.compute_hash()

    def to_dict(self) -> Dict[str, Any]:
        return self.payload(include_receipt_hash=True)


@dataclass
class TwinBuildResult:
    ok: bool
    network: Any
    control_plane: Any
    metadata: Dict[str, Any]
    node_map: Dict[str, str]
    snapshot: Dict[str, Any]
    schema: TwinSchema
    source_manifests: List[SourceFileManifest]
    creation_receipt: TwinCreationReceipt
    diagnostics: List[DiagnosticEvent] = field(default_factory=list)

    def to_serializable(self, *, include_network_snapshot: bool = True) -> Dict[str, Any]:
        out = {
            "ok": bool(self.ok),
            "metadata": _json_safe(self.metadata),
            "node_map": _json_safe(self.node_map),
            "schema": self.schema.to_dict(),
            "schema_hash": self.schema.hash(),
            "source_manifests": [m.to_dict() for m in self.source_manifests],
            "source_manifest_root": source_manifest_root(self.source_manifests),
            "creation_receipt": self.creation_receipt.to_dict(),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "snapshot": _json_safe(self.snapshot),
        }
        if include_network_snapshot:
            try:
                out["network"] = self.network.to_dict(include_histories=False)
            except Exception as exc:
                out["network_error"] = repr(exc)
        return out

    # Backward-compatible summary method.
    def to_dict(self) -> Dict[str, Any]:
        return self.to_serializable(include_network_snapshot=False)


# =============================================================================
# Twin type registry
# =============================================================================

InstallerFn = Callable[[Any, Any, TwinInput, Mapping[str, str]], None]

@dataclass
class TwinTypeSpec:
    name: str
    aliases: Tuple[str, ...]
    installer_name: str
    description: str = ""

    def matches(self, thing_type: str) -> bool:
        t = str(thing_type or "generic").lower().strip()
        if t == self.name:
            return True
        if t in self.aliases:
            return True
        return any(alias and alias in t for alias in self.aliases if len(alias) >= 4)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


class TwinTypeRegistry:
    def __init__(self) -> None:
        self._specs: Dict[str, TwinTypeSpec] = {}

    def register(self, name: str, aliases: Sequence[str], installer_name: str, description: str = "") -> None:
        spec = TwinTypeSpec(name=_slug(name, default="generic"), aliases=tuple(_slug(a, default="alias") for a in aliases), installer_name=str(installer_name), description=str(description))
        self._specs[spec.name] = spec

    def resolve(self, thing_type: str) -> TwinTypeSpec:
        t = _slug(thing_type or "generic", default="generic")
        for spec in self._specs.values():
            if spec.matches(t):
                return spec
        return self._specs.get("generic", TwinTypeSpec("generic", tuple(), "_install_generic_nodes", "Generic twin."))

    def supported_types(self) -> List[Dict[str, Any]]:
        return [self._specs[k].to_dict() for k in sorted(self._specs.keys())]

    @staticmethod
    def default() -> "TwinTypeRegistry":
        r = TwinTypeRegistry()
        r.register("plan", ["project_plan", "migration_plan", "strategy_plan"], "_install_plan_nodes", "Project, migration, recovery, and staged-execution plans.")
        r.register("software_process", ["software", "pipeline", "runtime", "ai_runtime", "digital_twin_runtime", "hybrid_ai_runtime"], "_install_software_process_nodes", "Software processes, runtimes, pipelines, and AI services.")
        r.register("industrial_machine", ["machine", "robot", "device", "equipment", "cnc", "industrial", "quadruped"], "_install_machine_nodes", "Machines, robots, devices, and physical equipment.")
        r.register("human_org", ["person", "operator", "team", "organization", "org", "school", "municipality", "kommune", "county"], "_install_human_org_nodes", "People, teams, institutions, and organizations.")
        r.register("curriculum", ["lk20", "udir", "grep", "education", "school_curriculum", "curriculum_plan"], "_install_curriculum_nodes", "Curriculum and education-domain twins.")
        r.register("document", ["knowledge_base", "corpus", "policy_document", "knowledge", "paper", "report"], "_install_document_nodes", "Documents, corpora, policies, and knowledge bases.")
        r.register("environment", ["building", "site", "city", "lab", "workspace", "field", "facility"], "_install_environment_nodes", "Physical or operational environments.")
        r.register("generic", [], "_install_generic_nodes", "Fallback generic twin.")
        return r


# =============================================================================
# Tensor-network profile builder
# =============================================================================

class TensorNetworkProfileBuilder:
    def __init__(self, cfg: TensorNetworkRuntimeConfig, *, seed: int = 2027, vector_dim: int = 256):
        self.cfg = cfg.normalized()
        self.seed = int(seed)
        self.vector_dim = int(max(8, vector_dim))

    def build_profile(self, diagnostics: Optional[DiagnosticLog] = None) -> Dict[str, Any]:
        status = tn_status()
        profile = {
            "enabled": bool(self.cfg.enable_tn_profile),
            "available": bool(status.get("available", False)),
            "status": status if self.cfg.include_runtime_paths else self._redact_paths(status),
            "runtime_config": _json_safe(asdict(self.cfg)),
            "operator_profile": {},
            "recommended_usage": {
                "projection": "digital_twin_kernel.ProjectionEngine owns deterministic projection.",
                "fusion": "digital_twin_kernel may use tn.TuckerFusionLayer for bounded fusion.",
                "storage": "tn.TensorTrain NPZ manifests and commitments verify tensor artifacts.",
            },
        }
        if not self.cfg.enable_tn_profile or tn is None:
            return profile
        try:
            if hasattr(tn, "factorize_into_modes"):
                profile["operator_profile"]["vector_dim_modes"] = list(tn.factorize_into_modes(int(self.vector_dim), int(self.cfg.tt_num_modes)))
            elif hasattr(tn, "get_factors"):
                profile["operator_profile"]["vector_dim_modes"] = list(tn.get_factors(int(self.vector_dim)))
            if hasattr(tn, "TTConfig"):
                dt = getattr(np, self.cfg.dtype, np.float32)
                try:
                    tt_cfg = tn.TTConfig(dtype=dt, device="cpu", check_finite=bool(self.cfg.check_finite), deterministic=True, seed=self.seed)
                except TypeError:
                    tt_cfg = tn.TTConfig(dtype=dt, device="cpu", check_finite=bool(self.cfg.check_finite))
                profile["operator_profile"]["tt_config"] = _json_safe(tt_cfg)
            if self.cfg.run_smoke_tests and hasattr(tn, "TensorTrain") and hasattr(tn, "TTConfig"):
                rng = np.random.default_rng(self.seed)
                dt = getattr(np, self.cfg.dtype, np.float32)
                try:
                    tt_cfg = tn.TTConfig(dtype=dt, device="cpu", check_finite=bool(self.cfg.check_finite), deterministic=True, seed=self.seed)
                except TypeError:
                    tt_cfg = tn.TTConfig(dtype=dt, device="cpu", check_finite=bool(self.cfg.check_finite))
                tt = tn.TensorTrain(output_dims=[2, 2], input_dims=[2, 2], bond_dims=[min(2, int(self.cfg.tt_rank))], config=tt_cfg, rng=rng, init_scale=1e-3, metadata={"source": "TwinAnythingFactory.tensor_profile"})
                x = np.ones((4,), dtype=dt)
                y = tt.apply(x)
                profile["operator_profile"]["tt_smoke"] = {"ok": True, "output_shape": list(np.asarray(y).shape), "output_norm": float(np.linalg.norm(np.asarray(y))), "commitment": _json_safe(tt.commitment().to_dict()) if hasattr(tt, "commitment") else None}
        except Exception as exc:
            profile["operator_profile"]["error"] = repr(exc)
            if diagnostics is not None:
                diagnostics.add("tn_profile_error", "warning", "TensorNetworkProfileBuilder", f"Tensor-network profile failed: {exc}", traceback=traceback.format_exc())
        return profile

    @staticmethod
    def _redact_paths(status: Mapping[str, Any]) -> Dict[str, Any]:
        out = copy.deepcopy(dict(status))
        if "module" in out and out["module"]:
            out["module"] = Path(str(out["module"])).name
        return out


# =============================================================================
# Factory
# =============================================================================

class TwinAnythingFactory:
    VERSION = TWIN_ANYTHING_VERSION

    def __init__(
        self,
        *,
        vector_dim: int = 256,
        sketch_dim: int = 96,
        history_capacity: int = 64,
        seed: int = 2027,
        use_tn_projection: bool = True,
        latent_geometry: str = "euclidean",
        deterministic: bool = False,
        deterministic_epoch: int = 0,
        consensus_safe: bool = True,
        install_platform_taxonomy: bool = True,
        install_success_defaults: bool = True,
        install_tensor_network_profile: bool = True,
        project_root: Optional[Union[str, os.PathLike]] = None,
        config: Optional[TwinAnythingConfig] = None,
        registry: Optional[TwinTypeRegistry] = None,
    ):
        self.cfg = (config or TwinAnythingConfig(
            vector_dim=vector_dim,
            sketch_dim=sketch_dim,
            history_capacity=history_capacity,
            seed=seed,
            use_tn_projection=use_tn_projection,
            latent_geometry=latent_geometry,
            deterministic=deterministic,
            deterministic_epoch=deterministic_epoch,
            consensus_safe=consensus_safe,
            install_platform_taxonomy=install_platform_taxonomy,
            install_success_defaults=install_success_defaults,
            install_tensor_network_profile=install_tensor_network_profile,
            project_root="" if project_root is None else str(project_root),
        )).normalized()
        configure_paths(project_root=self.cfg.project_root or None, permissive=self.cfg.permissive_path_search)
        self.registry = registry or TwinTypeRegistry.default()
        self.schema = TwinSchema.universal(root_id=self.cfg.thing_root_id)

    # ------------------------------------------------------------------
    # Main APIs
    # ------------------------------------------------------------------
    def create_twin_runtime(
        self,
        *,
        thing_name: str,
        thing_type: str = "generic",
        description: str = "",
        observables: Optional[Mapping[str, Any]] = None,
        process_steps: Optional[Sequence[str]] = None,
        resources: Optional[Mapping[str, Any]] = None,
        constraints: Optional[Sequence[str]] = None,
        components: Optional[Mapping[str, Any]] = None,
        environment: Optional[Mapping[str, Any]] = None,
        success_criteria: Optional[Mapping[str, Any]] = None,
        risks: Optional[Sequence[Any]] = None,
        actions: Optional[Sequence[Any]] = None,
        source_files: Optional[Sequence[Mapping[str, Any]]] = None,
        links: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TwinBuildResult:
        twin_input = TwinInput(
            thing_name=str(thing_name),
            thing_type=str(thing_type or "generic"),
            description=str(description or ""),
            observables=dict(observables or {}),
            process_steps=[str(x) for x in list(process_steps or [])],
            resources=dict(resources or {}),
            constraints=[str(x) for x in list(constraints or [])],
            components=dict(components or {}),
            environment=dict(environment or {}),
            success_criteria=dict(success_criteria or {}),
            risks=list(risks or []),
            actions=list(actions or []),
            source_files=[dict(x) for x in list(source_files or [])],
            links=[str(x) for x in list(links or [])],
            tags=[str(x) for x in list(tags or [])],
            metadata=dict(metadata or {}),
        ).normalized()
        return self.create_twin_from_input(twin_input)

    def create_twin_snapshot(self, *args: Any, include_network_snapshot: bool = True, **kwargs: Any) -> Dict[str, Any]:
        result = self.create_twin_runtime(*args, **kwargs)
        return result.to_serializable(include_network_snapshot=include_network_snapshot)

    def create_twin(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Backward-compatible live runtime API."""
        result = self.create_twin_runtime(*args, **kwargs)
        return {
            "network": result.network,
            "control_plane": result.control_plane,
            "metadata": result.metadata,
            "node_map": result.node_map,
            "snapshot": result.snapshot,
            "schema": result.schema.to_dict(),
            "creation_receipt": result.creation_receipt.to_dict(),
            "source_manifests": [m.to_dict() for m in result.source_manifests],
            "diagnostics": [d.to_dict() for d in result.diagnostics],
        }

    def create_twin_from_input(self, twin_input: TwinInput) -> TwinBuildResult:
        diagnostics = DiagnosticLog()
        twin_input = twin_input.normalized()
        self.schema.root_id = self.cfg.thing_root_id
        schema_hash = self.schema.hash()

        ttn = self._build_base_network(diagnostics)
        pre_commit = ttn.state_commitment().to_dict() if hasattr(ttn, "state_commitment") else {}
        cp = dtk.AkkuratInterface(ttn, strict=False)

        source_manifests = [SourceFileManifest.from_mapping(x, policy=self.cfg.file_hash_policy, diagnostics=diagnostics) for x in twin_input.source_files]
        src_root = source_manifest_root(source_manifests)

        node_map = self._install_universal_branch(ttn, twin_input, diagnostics)
        self._seed_universal_branch(ttn, twin_input, node_map, source_manifests, src_root, diagnostics)
        self._install_type_specific_nodes(ttn, twin_input, node_map, diagnostics)
        self._install_source_file_nodes(ttn, source_manifests, diagnostics)
        self._install_success_defaults_if_available(ttn, twin_input, diagnostics)
        self._install_tensor_network_profile_if_available(ttn, twin_input, diagnostics)
        self._finalize_histories(ttn, diagnostics)

        try:
            validation = ttn.validate_topology(strict=True)
        except Exception as exc:
            validation = {"ok": False, "errors": [repr(exc)]}
            diagnostics.add("topology_validation_failed", "error", "TwinAnythingFactory", f"Topology validation failed: {exc}", recoverable=False)

        post_commit = ttn.state_commitment().to_dict() if hasattr(ttn, "state_commitment") else {}
        snapshot = self._snapshot(ttn, twin_input, source_manifests, src_root, validation, diagnostics)
        metadata = {
            "factory": "TwinAnythingFactory",
            "factory_version": self.VERSION,
            "thing_name": twin_input.thing_name,
            "thing_type": twin_input.thing_type,
            "thing_slug": _slug(twin_input.thing_name),
            "root_node": self.cfg.thing_root_id,
            "node_count": len(getattr(ttn, "nodes", {})),
            "created_at": ttn.clock.stamp() if hasattr(ttn, "clock") else (_now_iso_utc() if not self.cfg.deterministic else "1970-01-01T00:00:00Z"),
            "deterministic": bool(self.cfg.deterministic),
            "consensus_safe": bool(self.cfg.consensus_safe),
            "schema_hash": schema_hash,
            "input_hash": twin_input.input_hash(),
            "source_manifest_root": src_root,
            "digital_twin_kernel": Path(str(getattr(dtk, "__file__", ""))).name,
            "tn": self._public_tn_status(),
            "config": _json_safe(asdict(self.cfg)),
        }
        receipt = TwinCreationReceipt.create(
            thing_input=twin_input,
            schema_hash=schema_hash,
            source_root=src_root,
            pre_state_commitment=pre_commit,
            post_state_commitment=post_commit,
            deterministic=self.cfg.deterministic,
            consensus_safe=self.cfg.consensus_safe,
            node_count=len(getattr(ttn, "nodes", {})),
            node_map=node_map,
            diagnostics=diagnostics.to_list(),
        )
        # Commit receipt metadata to the root branch so future state commitments can bind creation.
        try:
            if self.cfg.thing_root_id in ttn.nodes:
                ttn.nodes[self.cfg.thing_root_id].metadata.update({
                    "twin_creation_receipt_hash": receipt.receipt_hash,
                    "twin_schema_hash": schema_hash,
                    "source_manifest_root": src_root,
                    "input_hash": twin_input.input_hash(),
                })
                ttn.bump_tick()
                post_commit = ttn.state_commitment().to_dict() if hasattr(ttn, "state_commitment") else post_commit
                receipt = TwinCreationReceipt.create(
                    thing_input=twin_input,
                    schema_hash=schema_hash,
                    source_root=src_root,
                    pre_state_commitment=pre_commit,
                    post_state_commitment=post_commit,
                    deterministic=self.cfg.deterministic,
                    consensus_safe=self.cfg.consensus_safe,
                    node_count=len(getattr(ttn, "nodes", {})),
                    node_map=node_map,
                    diagnostics=diagnostics.to_list(),
                )
                snapshot["state_commitment"] = post_commit
                metadata["state_commitment"] = post_commit
                metadata["creation_receipt_hash"] = receipt.receipt_hash
        except Exception as exc:
            diagnostics.add("receipt_metadata_commit_failed", "warning", "TwinAnythingFactory", f"Failed to write creation receipt metadata: {exc}")

        return TwinBuildResult(
            ok=bool(validation.get("ok", False)) and not diagnostics.has_error,
            network=ttn,
            control_plane=cp,
            metadata=metadata,
            node_map=node_map,
            snapshot=snapshot,
            schema=copy.deepcopy(self.schema),
            source_manifests=source_manifests,
            creation_receipt=receipt,
            diagnostics=list(diagnostics.events),
        )

    # ------------------------------------------------------------------
    # Base network / kind helpers
    # ------------------------------------------------------------------
    def _build_base_network(self, diagnostics: DiagnosticLog) -> Any:
        geometry = getattr(dtk.LatentGeometry, "HYPERBOLIC").value if self.cfg.latent_geometry == "hyperbolic" else getattr(dtk.LatentGeometry, "EUCLIDEAN").value
        use_tn_projection = bool(self.cfg.use_tn_projection and self.cfg.tn_runtime.prefer_tn_projection and tn_status().get("available", False))
        try:
            if self.cfg.install_platform_taxonomy:
                return dtk.DigitalTwinsBuilder.build_platform(
                    vector_dim=int(self.cfg.vector_dim),
                    seed=int(self.cfg.seed),
                    sketch_dim=int(self.cfg.sketch_dim),
                    history_capacity=int(self.cfg.history_capacity),
                    use_tn_projection=use_tn_projection,
                    latent_geometry=geometry,
                    tt_rank=int(self.cfg.tn_runtime.tt_rank),
                    deterministic=bool(self.cfg.deterministic),
                    consensus_safe=bool(self.cfg.consensus_safe),
                )
            return dtk.TreeTensorNetwork(
                vector_dim=int(self.cfg.vector_dim),
                seed=int(self.cfg.seed),
                root_name="Digital Twin",
                sketch_dim=int(self.cfg.sketch_dim),
                history_capacity=int(self.cfg.history_capacity),
                use_tn_projection=use_tn_projection,
                latent_geometry=geometry,
                tt_rank=int(self.cfg.tn_runtime.tt_rank),
                deterministic=bool(self.cfg.deterministic),
                consensus_safe=bool(self.cfg.consensus_safe),
                clock=self.cfg.clock(),
            )
        except TypeError as exc:
            diagnostics.add("base_network_compatibility_fallback", "warning", "TwinAnythingFactory", f"Falling back to older digital_twin_kernel constructor: {exc}")
            return dtk.DigitalTwinsBuilder.build_platform(vector_dim=int(self.cfg.vector_dim), seed=int(self.cfg.seed), sketch_dim=int(self.cfg.sketch_dim), history_capacity=int(self.cfg.history_capacity))

    def _node_kind(self, kind_name: str) -> Any:
        name = str(kind_name or "generic").lower().strip()
        try:
            mapping = {
                "root": dtk.NodeKind.ROOT,
                "physical": dtk.NodeKind.PHYSICAL,
                "virtual": dtk.NodeKind.VIRTUAL,
                "data": dtk.NodeKind.DATA,
                "entity": dtk.NodeKind.ENTITY,
                "resource": dtk.NodeKind.RESOURCE,
                "sensor": dtk.NodeKind.SENSOR,
                "actuator": dtk.NodeKind.ACTUATOR,
                "property": dtk.NodeKind.PROPERTY,
                "model": dtk.NodeKind.MODEL,
                "pipeline": dtk.NodeKind.PIPELINE,
                "storage": dtk.NodeKind.STORAGE,
                "visual": dtk.NodeKind.VISUAL,
                "cognitive": dtk.NodeKind.COGNITIVE,
                "atomtn": dtk.NodeKind.ATOMTN,
                "quantum": dtk.NodeKind.QUANTUM,
                "flow": dtk.NodeKind.FLOW,
                "governance": dtk.NodeKind.GOVERNANCE,
                "observable": dtk.NodeKind.OBSERVABLE,
                "generic": dtk.NodeKind.GENERIC,
            }
            return mapping.get(name, dtk.NodeKind.GENERIC)
        except Exception:
            return dtk.NodeKind.GENERIC

    def _safe_update(self, ttn: Any, node_id: str, payload: Any, *, note: str, diagnostics: Optional[DiagnosticLog] = None) -> None:
        try:
            ttn.update_node_data(node_id, payload, note=note)
        except Exception as exc:
            if diagnostics is not None:
                diagnostics.add("node_update_failed", "error", "TwinAnythingFactory", f"Failed updating node {node_id}: {exc}", recoverable=False, note=note)
            raise

    def _add_node(self, ttn: Any, node_id: str, name: str, *, parent_id: str, kind: str = "generic", metadata: Optional[Mapping[str, Any]] = None) -> None:
        if node_id in getattr(ttn, "nodes", {}):
            if metadata:
                ttn.nodes[node_id].metadata.update(dict(metadata))
                ttn.bump_tick()
            return
        ttn.add_node(node_id, name, parent_id=parent_id, kind=self._node_kind(kind), metadata=dict(metadata or {}))

    # ------------------------------------------------------------------
    # Universal branch
    # ------------------------------------------------------------------
    def _install_universal_branch(self, ttn: Any, twin_input: TwinInput, diagnostics: DiagnosticLog) -> Dict[str, str]:
        root = str(self.cfg.thing_root_id)
        node_map: Dict[str, str] = {"root": root}
        for node in self.schema.nodes:
            name = node.name if node.node_id != root else f"Thing Twin: {twin_input.thing_name}"
            parent = node.parent_id
            metadata = {
                "schema_version": self.schema.schema_version,
                "schema_role": node.role,
                "installed_by": self.VERSION,
                "committed": bool(node.committed),
            }
            if node.node_id == root:
                metadata.update({
                    "thing_name": twin_input.thing_name,
                    "thing_type": twin_input.thing_type,
                    "description": twin_input.description,
                    "metadata": _json_safe(twin_input.metadata),
                    "tags": list(twin_input.tags),
                    "links": list(twin_input.links),
                    "schema_hash": self.schema.hash(),
                    "input_hash": twin_input.input_hash(),
                })
            self._add_node(ttn, node.node_id, name, parent_id=parent, kind=node.kind, metadata=metadata)
            node_map[node.role] = node.node_id
        node_map.update({
            "identity": "5.1", "purpose": "5.1",
            "environment": "5.2", "context": "5.2",
            "components": "5.3", "structure": "5.3",
            "observables": "5.4", "telemetry": "5.4",
            "process": "5.5", "behavior": "5.5",
            "resources": "5.6",
            "constraints": "5.7", "governance": "5.7",
            "success": "5.8", "success_criteria": "5.8",
            "risks": "5.9", "failure_modes": "5.9",
            "actions": "5.10", "actuators": "5.10",
            "memory": "5.11", "logs": "5.11", "snapshots": "5.11",
            "tn_profile": "5.12", "tensor_network_profile": "5.12",
        })
        self._install_universal_fusion(ttn, diagnostics)
        return node_map

    def _install_universal_fusion(self, ttn: Any, diagnostics: DiagnosticLog) -> None:
        specs = {
            "5.0": dict(attention_beta=2.6, residual_mix=0.16),
            "5.5": dict(attention_beta=2.4, residual_mix=0.20),
            "5.7": dict(attention_beta=2.8, residual_mix=0.14),
            "5.8": dict(attention_beta=2.6, residual_mix=0.16),
            "5.9": dict(attention_beta=2.7, residual_mix=0.14),
            "5.12": dict(attention_beta=2.9, residual_mix=0.12),
        }
        for node_id, spec in specs.items():
            if node_id not in getattr(ttn, "nodes", {}):
                continue
            try:
                ttn.set_fusion(node_id, dtk.FusionConfig(mode=dtk.FusionMode.ATTENTION, attention_beta=float(spec["attention_beta"]), residual_mix=float(spec["residual_mix"]), auto_tune_beta=True))
            except Exception as exc:
                diagnostics.add("fusion_install_failed", "warning", "TwinAnythingFactory", f"Fusion config failed for {node_id}: {exc}")

    def _seed_universal_branch(self, ttn: Any, twin_input: TwinInput, node_map: Mapping[str, str], manifests: Sequence[SourceFileManifest], src_root: str, diagnostics: DiagnosticLog) -> None:
        self._safe_update(ttn, node_map["identity"], {
            "thing_name": twin_input.thing_name,
            "thing_type": twin_input.thing_type,
            "description": twin_input.description,
            "metadata": twin_input.metadata,
            "tags": twin_input.tags,
            "links": twin_input.links,
            "slug": _slug(twin_input.thing_name),
            "input_hash": twin_input.input_hash(),
            "schema_hash": self.schema.hash(),
        }, note="thing_identity", diagnostics=diagnostics)
        self._safe_update(ttn, node_map["environment"], twin_input.environment or self._default_environment(twin_input), note="thing_environment", diagnostics=diagnostics)
        self._safe_update(ttn, node_map["components"], twin_input.components or self._infer_components(twin_input), note="thing_components", diagnostics=diagnostics)
        self._safe_update(ttn, node_map["observables"], twin_input.observables or self._default_observables(twin_input), note="thing_observables", diagnostics=diagnostics)
        self._safe_update(ttn, node_map["process"], {"process_steps": twin_input.process_steps or self._default_process_steps(twin_input), "process_type": twin_input.thing_type, "state_model": "universal_governed_process_branch"}, note="thing_process_model", diagnostics=diagnostics)
        self._safe_update(ttn, node_map["resources"], twin_input.resources or self._default_resources(twin_input), note="thing_resources", diagnostics=diagnostics)
        self._safe_update(ttn, node_map["constraints"], {"constraints": twin_input.constraints or self._default_constraints(twin_input), "governance_policy": "Actions require governed simulation, success-estimation, and auditable receipts.", "audit_required": True, "tensor_network_policy": {"low_level_ops_owned_by": "tn.py", "world_model_owned_by": "digital_twin_kernel.py", "branch_materialization_owned_by": "twin_anything.py"}}, note="thing_constraints", diagnostics=diagnostics)
        self._safe_update(ttn, node_map["success"], twin_input.success_criteria or self._default_success_criteria(twin_input), note="thing_success_criteria", diagnostics=diagnostics)
        self._safe_update(ttn, node_map["risks"], {"risks": twin_input.risks or self._default_risks(twin_input), "risk_policy": "High-risk candidates require explicit mitigation, rollback planning, and lobe/ledger receipts."}, note="thing_risk_model", diagnostics=diagnostics)
        self._safe_update(ttn, node_map["actions"], {"actions": twin_input.actions or self._default_actions(twin_input), "action_policy": "Propose -> simulate -> estimate success -> approve -> execute -> commit receipt."}, note="thing_action_model", diagnostics=diagnostics)
        self._safe_update(ttn, node_map["memory"], {"created_by": "TwinAnythingFactory", "factory_version": self.VERSION, "created_for": twin_input.thing_name, "event_log": [], "snapshots": [], "source_file_count": len(manifests), "source_manifest_root": src_root, "source_manifests": [m.to_dict() for m in manifests]}, note="thing_memory_initialization", diagnostics=diagnostics)

    # ------------------------------------------------------------------
    # Type-specific materialization
    # ------------------------------------------------------------------
    def _install_type_specific_nodes(self, ttn: Any, twin_input: TwinInput, node_map: Mapping[str, str], diagnostics: DiagnosticLog) -> None:
        spec = self.registry.resolve(twin_input.thing_type)
        installer = getattr(self, spec.installer_name, self._install_generic_nodes)
        try:
            installer(ttn, twin_input, node_map, diagnostics)
            diagnostics.add("type_registry_resolved", "info", "TwinTypeRegistry", f"Resolved thing_type={twin_input.thing_type} as {spec.name}", spec=spec.to_dict())
        except Exception as exc:
            diagnostics.add("type_installer_failed", "error", "TwinTypeRegistry", f"Type installer {spec.installer_name} failed: {exc}", recoverable=False, traceback=traceback.format_exc())
            raise

    def _add_seeded_nodes(self, ttn: Any, additions: Sequence[Tuple[str, str, str, Any]], *, parent_fallback: str, diagnostics: DiagnosticLog) -> None:
        for node_id, name, kind_name, payload in additions:
            if node_id not in ttn.nodes:
                parent = ".".join(node_id.split(".")[:-1])
                if parent not in ttn.nodes:
                    parent = parent_fallback
                ttn.add_node(node_id, name, parent_id=parent, kind=self._node_kind(kind_name), metadata={"type_specific": True, "installed_by": self.VERSION})
            self._safe_update(ttn, node_id, payload, note=f"type_specific_seed:{_slug(name)}", diagnostics=diagnostics)

    def _install_plan_nodes(self, ttn: Any, twin_input: TwinInput, node_map: Mapping[str, str], diagnostics: DiagnosticLog) -> None:
        additions = [
            ("5.5.1", "Plan Phases", "model", {"phases": twin_input.process_steps or self._default_process_steps(twin_input)}),
            ("5.5.2", "Dependencies / Critical Path", "model", {"dependencies": twin_input.metadata.get("dependencies", [])}),
            ("5.5.3", "Rollback / Recovery Path", "governance", {"rollback_required": True, "rollback_plan": twin_input.metadata.get("rollback_plan", "")}),
            ("5.8.1", "Plan Acceptance Criteria", "observable", twin_input.success_criteria or self._default_success_criteria(twin_input)),
            ("5.9.1", "Plan Failure Modes", "governance", {"risks": twin_input.risks or self._default_risks(twin_input)}),
        ]
        self._add_seeded_nodes(ttn, additions, parent_fallback="5.5", diagnostics=diagnostics)

    def _install_software_process_nodes(self, ttn: Any, twin_input: TwinInput, node_map: Mapping[str, str], diagnostics: DiagnosticLog) -> None:
        additions = [
            ("5.5.1", "Execution Pipeline", "pipeline", {"steps": twin_input.process_steps or self._default_process_steps(twin_input)}),
            ("5.5.2", "Runtime State", "model", {"runtime": twin_input.metadata.get("runtime", "python"), "language": twin_input.metadata.get("language", "python")}),
            ("5.5.3", "Validation / Test Harness", "governance", {"tests_required": True, "validation": twin_input.metadata.get("validation", {})}),
            ("5.4.1", "Software Telemetry", "observable", twin_input.observables or self._default_observables(twin_input)),
            ("5.10.1", "Automation Hooks", "actuator", {"actions": twin_input.actions or self._default_actions(twin_input)}),
            ("5.12.1", "Tensorized Runtime Hooks", "model", {"tn_projection": bool(self.cfg.use_tn_projection), "tn_status": self._public_tn_status()}),
        ]
        self._add_seeded_nodes(ttn, additions, parent_fallback="5.5", diagnostics=diagnostics)

    def _install_machine_nodes(self, ttn: Any, twin_input: TwinInput, node_map: Mapping[str, str], diagnostics: DiagnosticLog) -> None:
        additions = [
            ("5.3.1", "Machine Components", "entity", twin_input.components or self._infer_components(twin_input)),
            ("5.4.1", "Sensor Telemetry", "sensor", twin_input.observables or self._default_observables(twin_input)),
            ("5.10.1", "Controllers / Actuators", "actuator", {"actions": twin_input.actions or self._default_actions(twin_input)}),
            ("5.9.1", "Fault Modes", "governance", {"risks": twin_input.risks or self._default_risks(twin_input)}),
            ("5.6.1", "Maintenance Resources", "resource", twin_input.resources or self._default_resources(twin_input)),
            ("5.5.1", "State Estimation Model", "model", {"model_type": "machine_state_estimation", "requires_telemetry": True, "uses_baselines": True}),
        ]
        self._add_seeded_nodes(ttn, additions, parent_fallback="5.3", diagnostics=diagnostics)

    def _install_human_org_nodes(self, ttn: Any, twin_input: TwinInput, node_map: Mapping[str, str], diagnostics: DiagnosticLog) -> None:
        additions = [
            ("5.3.1", "Roles / Stakeholders", "entity", twin_input.components or self._infer_components(twin_input)),
            ("5.6.1", "Capabilities / Resources", "resource", twin_input.resources or self._default_resources(twin_input)),
            ("5.7.1", "Norms / Governance", "governance", {"constraints": twin_input.constraints or self._default_constraints(twin_input)}),
            ("5.8.1", "Outcome Criteria", "observable", twin_input.success_criteria or self._default_success_criteria(twin_input)),
            ("5.9.1", "Organizational Risks", "governance", {"risks": twin_input.risks or self._default_risks(twin_input)}),
        ]
        self._add_seeded_nodes(ttn, additions, parent_fallback="5.3", diagnostics=diagnostics)

    def _install_curriculum_nodes(self, ttn: Any, twin_input: TwinInput, node_map: Mapping[str, str], diagnostics: DiagnosticLog) -> None:
        additions = [
            ("5.3.1", "Curriculum Structure", "entity", twin_input.components or {"layers": ["framework", "subject", "grade_or_stage", "competence_aim", "assessment", "local_overlay"]}),
            ("5.4.1", "Curriculum Observables", "observable", twin_input.observables or {"required": ["coverage", "alignment", "gaps", "version_drift", "assessment_binding"]}),
            ("5.5.1", "Curriculum Binding Process", "pipeline", {"steps": twin_input.process_steps or self._default_process_steps(twin_input)}),
            ("5.7.1", "Curriculum Governance Policy", "governance", {"canonical_source": twin_input.metadata.get("canonical_source", "Udir/Grep"), "local_overlay_policy": "Local artefacts do not mutate canonical curriculum nodes.", "constraints": twin_input.constraints or self._default_constraints(twin_input)}),
            ("5.8.1", "Curriculum Success Criteria", "observable", twin_input.success_criteria or {"minimum_coverage_ratio": 0.90, "requires_subject_binding": True, "requires_grade_binding": True, "requires_version_match": True}),
        ]
        self._add_seeded_nodes(ttn, additions, parent_fallback="5.5", diagnostics=diagnostics)

    def _install_document_nodes(self, ttn: Any, twin_input: TwinInput, node_map: Mapping[str, str], diagnostics: DiagnosticLog) -> None:
        additions = [
            ("5.3.1", "Document Structure", "entity", twin_input.components or {"sections": [], "source_file_count": len(twin_input.source_files)}),
            ("5.4.1", "Document Observables", "observable", twin_input.observables or {"required": ["source_hash", "version", "coverage", "semantic_alignment", "extraction_confidence"]}),
            ("5.5.1", "Document Processing Pipeline", "pipeline", {"steps": twin_input.process_steps or self._default_process_steps(twin_input)}),
            ("5.7.1", "Document Governance", "governance", {"constraints": twin_input.constraints or self._default_constraints(twin_input), "requires_source_hash": True}),
        ]
        self._add_seeded_nodes(ttn, additions, parent_fallback="5.5", diagnostics=diagnostics)

    def _install_environment_nodes(self, ttn: Any, twin_input: TwinInput, node_map: Mapping[str, str], diagnostics: DiagnosticLog) -> None:
        additions = [
            ("5.2.1", "Environment Geometry / Zones", "physical", twin_input.environment or self._default_environment(twin_input)),
            ("5.4.1", "Environmental Telemetry", "sensor", twin_input.observables or self._default_observables(twin_input)),
            ("5.9.1", "Environmental Hazards", "governance", {"risks": twin_input.risks or self._default_risks(twin_input)}),
            ("5.6.1", "Environment Resources", "resource", twin_input.resources or self._default_resources(twin_input)),
        ]
        self._add_seeded_nodes(ttn, additions, parent_fallback="5.2", diagnostics=diagnostics)

    def _install_generic_nodes(self, ttn: Any, twin_input: TwinInput, node_map: Mapping[str, str], diagnostics: DiagnosticLog) -> None:
        additions = [
            ("5.5.1", "Generic State Model", "model", {"description": twin_input.description}),
            ("5.4.1", "Generic Measurements", "observable", twin_input.observables or self._default_observables(twin_input)),
        ]
        self._add_seeded_nodes(ttn, additions, parent_fallback="5.5", diagnostics=diagnostics)

    # ------------------------------------------------------------------
    # Source files and tensor profile
    # ------------------------------------------------------------------
    def _install_source_file_nodes(self, ttn: Any, manifests: Sequence[SourceFileManifest], diagnostics: DiagnosticLog) -> None:
        if not manifests:
            return
        if "5.11.1" not in ttn.nodes:
            ttn.add_node("5.11.1", "Source File Registry", parent_id="5.11", kind=self._node_kind("storage"), metadata={"source_manifest_root": source_manifest_root(manifests)})
        registry_payload = {"source_file_count": len(manifests), "source_manifest_root": source_manifest_root(manifests), "files": [m.to_dict() for m in manifests]}
        self._safe_update(ttn, "5.11.1", registry_payload, note="source_file_registry", diagnostics=diagnostics)
        for idx, manifest in enumerate(manifests):
            safe_id = _slug(manifest.sha256[:16] if manifest.sha256 else f"{idx + 1}_{manifest.filename}", default=f"source_{idx+1}", max_len=32)
            node_id = f"5.11.1.{safe_id}"
            if node_id not in ttn.nodes:
                ttn.add_node(node_id, f"Source File: {manifest.filename or idx + 1}", parent_id="5.11.1", kind=self._node_kind("data"), metadata={"filename": manifest.filename, "sha256": manifest.sha256, "mime_type": manifest.mime_type, "role": manifest.role, "manifest_hash": manifest.manifest_hash()})
            self._safe_update(ttn, node_id, manifest.to_dict(), note="source_file_manifest", diagnostics=diagnostics)

    def _install_tensor_network_profile_if_available(self, ttn: Any, twin_input: TwinInput, diagnostics: DiagnosticLog) -> None:
        if not self.cfg.install_tensor_network_profile or "5.12" not in getattr(ttn, "nodes", {}):
            return
        builder = TensorNetworkProfileBuilder(self.cfg.tn_runtime, seed=int(self.cfg.seed), vector_dim=int(self.cfg.vector_dim))
        profile = builder.build_profile(diagnostics)
        profile.update({"thing_type": twin_input.thing_type, "thing_name": twin_input.thing_name, "factory_version": self.VERSION})
        self._safe_update(ttn, "5.12", profile, note="tensor_network_runtime_profile", diagnostics=diagnostics)
        try:
            ttn.nodes["5.12"].metadata.update({"tn_available": bool(profile.get("available", False)), "tt_rank": int(self.cfg.tn_runtime.tt_rank), "tucker_rank": int(self.cfg.tn_runtime.tucker_rank)})
            ttn.bump_tick()
        except Exception as exc:
            diagnostics.add("tn_profile_metadata_failed", "warning", "TwinAnythingFactory", f"Tensor profile metadata update failed: {exc}")

    def _install_success_defaults_if_available(self, ttn: Any, twin_input: TwinInput, diagnostics: DiagnosticLog) -> None:
        if not self.cfg.install_success_defaults or "5.8" not in ttn.nodes:
            return
        try:
            payload = {
                "minimum_probability": _safe_float(twin_input.success_criteria.get("minimum_probability", 0.80) if isinstance(twin_input.success_criteria, Mapping) else 0.80, 0.80),
                "requires_governance_approval": True,
                "requires_audit_trail": True,
                "requires_reversibility_or_mitigation": True,
                "default_threshold_source": "TwinAnythingFactory",
                "factory_version": self.VERSION,
            }
            ttn.nodes["5.8"].metadata["success_defaults"] = _json_safe(payload)
            ttn.bump_tick()
        except Exception as exc:
            diagnostics.add("success_defaults_failed", "warning", "TwinAnythingFactory", f"Success defaults install failed: {exc}")

    def _finalize_histories(self, ttn: Any, diagnostics: DiagnosticLog) -> None:
        try:
            ttn.push_global_histories()
            return
        except Exception as exc:
            diagnostics.add("push_global_histories_failed", "warning", "TwinAnythingFactory", f"push_global_histories failed: {exc}")
        try:
            for nid in sorted(getattr(ttn, "nodes", {}).keys()):
                if hasattr(ttn, "_push_path_histories"):
                    ttn._push_path_histories(nid)
        except Exception as exc:
            diagnostics.add("push_path_histories_failed", "warning", "TwinAnythingFactory", f"Path history fallback failed: {exc}")

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    def _default_environment(self, twin_input: TwinInput) -> Dict[str, Any]:
        return {"context": "unspecified", "assumption": "Environment should be updated from observations.", "thing_type": twin_input.thing_type}

    def _infer_components(self, twin_input: TwinInput) -> Dict[str, Any]:
        if twin_input.components:
            return dict(twin_input.components)
        if twin_input.process_steps:
            return {"process_step_count": len(twin_input.process_steps), "process_steps": list(twin_input.process_steps)}
        if twin_input.source_files:
            return {"source_file_count": len(twin_input.source_files)}
        return {"components": [], "note": "No explicit components supplied."}

    def _default_observables(self, twin_input: TwinInput) -> Dict[str, Any]:
        return {"observable_status": "initial", "required": ["state", "drift", "constraint_satisfaction", "resource_readiness", "success_probability", "tensor_profile_health", "lobe_health"]}

    def _default_process_steps(self, twin_input: TwinInput) -> List[str]:
        t = str(twin_input.thing_type).lower()
        if "plan" in t:
            return ["define objective", "verify resources", "simulate plan", "approve execution", "monitor result", "rollback if needed"]
        if any(x in t for x in ["software", "runtime", "pipeline", "ai"]):
            return ["ingest input", "process state", "validate output", "estimate success", "emit telemetry", "persist snapshot"]
        if any(x in t for x in ["machine", "device", "robot", "cnc", "industrial"]):
            return ["observe sensors", "estimate health", "compare against baseline", "propose control action", "simulate", "execute if approved"]
        if any(x in t for x in ["curriculum", "lk20", "education"]):
            return ["ingest canonical curriculum", "ingest local overlay", "bind local artefact", "score coverage and alignment", "approve or quarantine", "persist verifiable snapshot"]
        if any(x in t for x in ["document", "knowledge", "corpus"]):
            return ["register source", "extract content", "parse structure", "project into latent state", "bind to target taxonomy", "persist snapshot"]
        return ["observe", "model", "estimate", "act if approved", "learn from outcome"]

    def _default_resources(self, twin_input: TwinInput) -> Dict[str, Any]:
        return {"required": ["operator_or_agent_attention", "data", "governance_policy", "digital_twin_kernel", "tn_runtime_when_available"], "status": "unknown_until_observed"}

    def _default_constraints(self, twin_input: TwinInput) -> List[str]:
        return [
            "Actions must pass governed simulation.",
            "Success probability must meet threshold before execution.",
            "Rollback or recovery path should be known for high-risk actions.",
            "All material state changes must be auditable.",
            "Low-level tensor operations remain owned by tn.py.",
            "Canonical world-state and governance remain owned by digital_twin_kernel.py.",
            "Twin creation must produce a verifiable TwinCreationReceipt.",
        ]

    def _default_success_criteria(self, twin_input: TwinInput) -> Dict[str, Any]:
        return {"success_definition": "Stable state, satisfied constraints, ready resources, acceptable risk, and success odds above threshold.", "minimum_probability": 0.80, "requires_governance_approval": True, "requires_reversibility_or_mitigation": True, "requires_audit_trail": True}

    def _default_risks(self, twin_input: TwinInput) -> List[Dict[str, Any]]:
        return [
            {"name": "insufficient_observability", "probability": 0.20, "impact": 0.40, "mitigation": "Add telemetry and update observables."},
            {"name": "unmodeled_dependency", "probability": 0.15, "impact": 0.50, "mitigation": "Expose dependencies as components/resources."},
            {"name": "tensor_runtime_unavailable", "probability": 0.10 if tn_status().get("available", False) else 0.60, "impact": 0.30, "mitigation": "Fallback to deterministic hash projection in digital_twin_kernel."},
        ]

    def _default_actions(self, twin_input: TwinInput) -> List[Dict[str, Any]]:
        return [
            {"kind": "observe", "description": "Collect or update telemetry."},
            {"kind": "simulate", "description": "Run governed sandbox simulation."},
            {"kind": "estimate_success", "description": "Estimate success odds and constraint satisfaction."},
            {"kind": "approve", "description": "Approve only if success threshold is met."},
            {"kind": "execute", "description": "Execute bounded governed action."},
            {"kind": "snapshot", "description": "Persist verifiable state snapshot."},
        ]

    # ------------------------------------------------------------------
    # Updating existing twins
    # ------------------------------------------------------------------
    def update_observation(self, network: Any, *, observation: Mapping[str, Any], note: str = "thing_observation_update") -> Dict[str, Any]:
        try:
            network.update_node_data("5.4", {"observation": dict(observation), "source": "TwinAnythingFactory.update_observation"}, note=note)
            try:
                network.update_node_data("5.11", {"last_observation": dict(observation)}, note="thing_memory_observation")
            except Exception:
                pass
            self._finalize_histories(network, DiagnosticLog())
            commit = network.state_commitment().to_dict() if hasattr(network, "state_commitment") else {}
            return {"ok": True, "node_id": "5.4", "state_commitment": commit}
        except Exception as exc:
            return {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()}

    def append_event(self, network: Any, *, event: Mapping[str, Any], note: str = "thing_event_append") -> Dict[str, Any]:
        try:
            network.update_node_data("5.11", {"event": dict(event), "source": "TwinAnythingFactory.append_event"}, note=note)
            self._finalize_histories(network, DiagnosticLog())
            commit = network.state_commitment().to_dict() if hasattr(network, "state_commitment") else {}
            return {"ok": True, "node_id": "5.11", "state_commitment": commit}
        except Exception as exc:
            return {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()}

    def attach_source_file(self, network: Any, *, source_file: Mapping[str, Any], note: str = "thing_source_file_attach") -> Dict[str, Any]:
        diag = DiagnosticLog()
        try:
            manifest = SourceFileManifest.from_mapping(source_file, policy=self.cfg.file_hash_policy, diagnostics=diag)
            self._install_source_file_nodes(network, [manifest], diag)
            self._finalize_histories(network, diag)
            return {"ok": not diag.has_error, "manifest": manifest.to_dict(), "source_manifest_root": source_manifest_root([manifest]), "diagnostics": diag.to_list(), "state_commitment": network.state_commitment().to_dict() if hasattr(network, "state_commitment") else {}}
        except Exception as exc:
            diag.add("attach_source_file_failed", "error", "TwinAnythingFactory", f"attach_source_file failed: {exc}", recoverable=False)
            return {"ok": False, "error": repr(exc), "traceback": traceback.format_exc(), "diagnostics": diag.to_list()}

    def refresh_tensor_network_profile(self, network: Any, *, note: str = "tensor_network_profile_refresh") -> Dict[str, Any]:
        diag = DiagnosticLog()
        try:
            if "5.12" not in network.nodes:
                network.add_node("5.12", "Tensor-Network Runtime Profile", parent_id=str(self.cfg.thing_root_id), kind=self._node_kind("model"))
            profile = TensorNetworkProfileBuilder(self.cfg.tn_runtime, seed=int(self.cfg.seed), vector_dim=int(self.cfg.vector_dim)).build_profile(diag)
            network.update_node_data("5.12", profile, note=note)
            self._finalize_histories(network, diag)
            return {"ok": not diag.has_error, "profile": _json_safe(profile), "diagnostics": diag.to_list(), "state_commitment": network.state_commitment().to_dict() if hasattr(network, "state_commitment") else {}}
        except Exception as exc:
            diag.add("refresh_tensor_network_profile_failed", "error", "TwinAnythingFactory", f"refresh_tensor_network_profile failed: {exc}", recoverable=False)
            return {"ok": False, "error": repr(exc), "traceback": traceback.format_exc(), "diagnostics": diag.to_list()}

    # ------------------------------------------------------------------
    # Snapshot / status helpers
    # ------------------------------------------------------------------
    def _snapshot(self, ttn: Any, twin_input: TwinInput, manifests: Sequence[SourceFileManifest], src_root: str, validation: Mapping[str, Any], diagnostics: DiagnosticLog) -> Dict[str, Any]:
        try:
            state_commitment = ttn.state_commitment().to_dict()
        except Exception as exc:
            diagnostics.add("state_commitment_failed", "error", "TwinAnythingFactory", f"state_commitment failed: {exc}", recoverable=False)
            state_commitment = {}
        try:
            thing_measure = ttn.measure(str(self.cfg.thing_root_id))
        except Exception as exc:
            thing_measure = {"error": repr(exc)}
        try:
            tn_measure = ttn.measure("5.12") if "5.12" in getattr(ttn, "nodes", {}) else None
        except Exception as exc:
            tn_measure = {"error": repr(exc)}
        return {
            "schema_version": "ProjectChimera.TwinAnythingSnapshot.v1",
            "factory_version": self.VERSION,
            "deterministic": bool(self.cfg.deterministic),
            "node_count": len(getattr(ttn, "nodes", {})),
            "thing_root": str(self.cfg.thing_root_id),
            "thing_name": twin_input.thing_name,
            "thing_type": twin_input.thing_type,
            "schema_hash": self.schema.hash(),
            "input_hash": twin_input.input_hash(),
            "source_manifest_root": src_root,
            "source_manifest_count": len(manifests),
            "state_commitment": state_commitment,
            "thing_measure": _json_safe(thing_measure),
            "tensor_network_measure": _json_safe(tn_measure),
            "topology_validation": _json_safe(validation),
            "diagnostics": diagnostics.to_list(),
        }

    def _public_tn_status(self) -> Dict[str, Any]:
        st = tn_status()
        if not self.cfg.tn_runtime.include_runtime_paths and st.get("module"):
            st = copy.deepcopy(st)
            st["module"] = Path(str(st.get("module", ""))).name
        return st

    def supported_types(self) -> List[Dict[str, Any]]:
        return self.registry.supported_types()


# =============================================================================
# Convenience functions
# =============================================================================

def build_anything_twin(
    *,
    thing_name: str,
    thing_type: str = "generic",
    description: str = "",
    observables: Optional[Mapping[str, Any]] = None,
    process_steps: Optional[Sequence[str]] = None,
    resources: Optional[Mapping[str, Any]] = None,
    constraints: Optional[Sequence[str]] = None,
    components: Optional[Mapping[str, Any]] = None,
    environment: Optional[Mapping[str, Any]] = None,
    success_criteria: Optional[Mapping[str, Any]] = None,
    risks: Optional[Sequence[Any]] = None,
    actions: Optional[Sequence[Any]] = None,
    source_files: Optional[Sequence[Mapping[str, Any]]] = None,
    links: Optional[Sequence[str]] = None,
    tags: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    config: Optional[TwinAnythingConfig] = None,
    **factory_kwargs: Any,
) -> Dict[str, Any]:
    factory = TwinAnythingFactory(config=config, **factory_kwargs)
    return factory.create_twin(
        thing_name=thing_name,
        thing_type=thing_type,
        description=description,
        observables=observables,
        process_steps=process_steps,
        resources=resources,
        constraints=constraints,
        components=components,
        environment=environment,
        success_criteria=success_criteria,
        risks=risks,
        actions=actions,
        source_files=source_files,
        links=links,
        tags=tags,
        metadata=metadata,
    )


def create_twin(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return build_anything_twin(*args, **kwargs)


def create_twin_snapshot(*args: Any, config: Optional[TwinAnythingConfig] = None, **kwargs: Any) -> Dict[str, Any]:
    factory = TwinAnythingFactory(config=config)
    return factory.create_twin_snapshot(*args, **kwargs)


def sample_twin_input() -> Dict[str, Any]:
    return {
        "thing_name": "Example Governed Process",
        "thing_type": "plan",
        "description": "A sample governed plan twin.",
        "process_steps": ["define objective", "verify resources", "simulate", "approve", "execute", "monitor"],
        "observables": {"state": "initial", "readiness": 0.0, "success_probability": None},
        "resources": {"operator": "required", "data": "required", "runtime": "digital_twin_kernel + tn.py"},
        "constraints": ["Must pass sandbox simulation.", "Must preserve audit trail."],
        "success_criteria": {"minimum_probability": 0.80, "requires_governance_approval": True},
        "risks": [{"name": "unknown_dependency", "probability": 0.2, "impact": 0.5}],
        "actions": [{"kind": "observe"}, {"kind": "simulate"}, {"kind": "approve"}],
        "tags": ["sample", "governed", "tensor-aware"],
        "metadata": {"created_by": "TwinAnythingFactory.sample_twin_input"},
    }


# =============================================================================
# Self-test
# =============================================================================

def _self_test() -> Dict[str, Any]:
    cfg = TwinAnythingConfig(vector_dim=32, sketch_dim=16, seed=77, use_tn_projection=False, deterministic=True, deterministic_epoch=0, install_tensor_network_profile=True).normalized()
    factory = TwinAnythingFactory(config=cfg)
    result1 = factory.create_twin_runtime(
        thing_name="Self Test Robot",
        thing_type="robot",
        description="Deterministic self-test robot twin.",
        observables={"battery_pct": 91.0, "status": "nominal"},
        components={"platform": "quadruped"},
        success_criteria={"minimum_probability": 0.80},
        tags=["selftest"],
    )
    result2 = factory.create_twin_runtime(
        thing_name="Self Test Robot",
        thing_type="robot",
        description="Deterministic self-test robot twin.",
        observables={"battery_pct": 91.0, "status": "nominal"},
        components={"platform": "quadruped"},
        success_criteria={"minimum_probability": 0.80},
        tags=["selftest"],
    )
    c1 = result1.network.state_commitment().to_dict()
    c2 = result2.network.state_commitment().to_dict()
    if c1["state_root"] != c2["state_root"]:
        raise AssertionError("deterministic creation state_root mismatch")
    if not result1.creation_receipt.verify():
        raise AssertionError("creation receipt failed verification")
    if not result1.ok:
        raise AssertionError(f"twin result not ok: {result1.to_serializable(include_network_snapshot=False)}")
    return {
        "ok": True,
        "state_root": c1["state_root"],
        "receipt_hash": result1.creation_receipt.receipt_hash,
        "node_count": len(result1.network.nodes),
        "source_manifest_root": result1.creation_receipt.source_manifest_root,
        "diagnostic_count": len(result1.diagnostics),
    }


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Project Chimera / Akkurat Twin Anything Factory")
    p.add_argument("--mode", choices=["status", "types", "create", "sample-input", "self-test"], default="status")
    p.add_argument("--thing-name", default="Unnamed Thing")
    p.add_argument("--thing-type", default="generic")
    p.add_argument("--description", default="")
    p.add_argument("--input-json", default="")
    p.add_argument("--output", default="")
    p.add_argument("--project-root", default="")
    p.add_argument("--vector-dim", type=int, default=256)
    p.add_argument("--sketch-dim", type=int, default=96)
    p.add_argument("--history-capacity", type=int, default=64)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--deterministic-epoch", type=int, default=0)
    p.add_argument("--disable-consensus-safe", action="store_true")
    p.add_argument("--disable-tn-projection", action="store_true")
    p.add_argument("--disable-platform-taxonomy", action="store_true")
    p.add_argument("--disable-success-defaults", action="store_true")
    p.add_argument("--disable-tn-profile", action="store_true")
    p.add_argument("--latent-geometry", choices=["euclidean", "hyperbolic"], default="euclidean")
    p.add_argument("--tt-rank", type=int, default=8)
    p.add_argument("--tucker-rank", type=int, default=16)
    p.add_argument("--allow-local-file-hashing", action="store_true")
    p.add_argument("--allowed-root", action="append", default=[])
    p.add_argument("--include-network", action="store_true")
    return p


def _config_from_args(args: argparse.Namespace) -> TwinAnythingConfig:
    tn_cfg = TensorNetworkRuntimeConfig(tt_rank=int(args.tt_rank), tucker_rank=int(args.tucker_rank), prefer_tn_projection=not bool(args.disable_tn_projection), enable_tn_profile=not bool(args.disable_tn_profile)).normalized()
    file_policy = SafeFileHashPolicy(allow_local_file_hashing=bool(args.allow_local_file_hashing), allowed_roots=[str(x) for x in list(args.allowed_root or [])]).normalized()
    return TwinAnythingConfig(
        vector_dim=int(args.vector_dim),
        sketch_dim=int(args.sketch_dim),
        history_capacity=int(args.history_capacity),
        seed=int(args.seed),
        use_tn_projection=not bool(args.disable_tn_projection),
        latent_geometry=str(args.latent_geometry),
        deterministic=bool(args.deterministic),
        deterministic_epoch=int(args.deterministic_epoch),
        consensus_safe=not bool(args.disable_consensus_safe),
        install_platform_taxonomy=not bool(args.disable_platform_taxonomy),
        install_success_defaults=not bool(args.disable_success_defaults),
        install_tensor_network_profile=not bool(args.disable_tn_profile),
        project_root=str(args.project_root or ""),
        tn_runtime=tn_cfg,
        file_hash_policy=file_policy,
    ).normalized()


def _create_from_cli(args: argparse.Namespace) -> Dict[str, Any]:
    if args.input_json:
        payload = _read_json(args.input_json, {})
        if not isinstance(payload, Mapping):
            raise ValueError("--input-json must contain a JSON object")
        twin_input = TwinInput.from_mapping(payload)
    else:
        twin_input = TwinInput(thing_name=str(args.thing_name), thing_type=str(args.thing_type), description=str(args.description)).normalized()
    cfg = _config_from_args(args)
    factory = TwinAnythingFactory(config=cfg)
    result = factory.create_twin_from_input(twin_input)
    out = result.to_serializable(include_network_snapshot=bool(args.include_network))
    if args.output:
        _write_json(args.output, out)
        out["saved_json"] = str(Path(args.output).resolve())
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        configure_paths(project_root=args.project_root or None)
        if args.mode == "status":
            out = status_report()
        elif args.mode == "types":
            out = {"ok": True, "types": TwinTypeRegistry.default().supported_types()}
        elif args.mode == "sample-input":
            out = sample_twin_input()
            if args.output:
                _write_json(args.output, out)
                out["saved_sample_input_json"] = str(Path(args.output).resolve())
        elif args.mode == "self-test":
            out = _self_test()
        elif args.mode == "create":
            out = _create_from_cli(args)
        else:
            raise ValueError(f"unknown mode: {args.mode}")
        print(json.dumps(_json_safe(out), indent=2, ensure_ascii=False))
        return 0 if bool(out.get("ok", True)) else 2
    except Exception as exc:
        err = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc(), "status": status_report()}
        print(json.dumps(_json_safe(err), indent=2, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
