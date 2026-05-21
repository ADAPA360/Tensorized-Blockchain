#!/usr/bin/env python3
# tb_execution.py
r"""
Project Chimera / Akkurat - Tensorized Blockchain Execution Pipeline
====================================================================

Purpose
-------
Canonical transaction execution boundary for the Tensorized Blockchain stack.

This module separates deterministic state-transition execution from ledger
append logic. It consumes schema-versioned transaction envelopes, executes the
requested governed digital-twin actions, captures pre/post state commitments,
produces canonical receipts, and optionally attaches neural-lobe evidence from
NCP and CfC substrates.

Design boundaries
-----------------
- This module does not append blocks.
- This module does not own consensus.
- This module does not bypass governance.
- This module does not mutate ledger storage.
- This module may execute a TreeTensorNetwork through AkkuratInterface or a
  TreeTensorNetwork-compatible object exposing update methods.

The ledger should call this module first, then commit the resulting receipts.

Primary API
-----------
- ExecutionConfig
- ExecutionDiagnostics
- ExecutionResult
- CanonicalTransactionExecutor
- execute_transaction(...)
- execute_transaction_batch(...)
- deterministic_feature_vector(...)

Compatibility
-------------
The module is dependency-light and import-safe. It can use the production
project modules when available:

- tb_schemas.py
- tb_canonical.py
- tb_hashing.py
- tb_validation.py
- digital_twin_kernel.py
- ncp.py
- cfc.py

If a high-level optional integration is unavailable, execution still works for
schema-level and generic network objects, but diagnostics will explicitly record
which integration was absent.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np


# =============================================================================
# Optional project imports
# =============================================================================

try:  # canonical serialization
    import tb_canonical as _tbc  # type: ignore
except Exception:  # pragma: no cover - fallback path
    _tbc = None  # type: ignore

try:  # domain-separated hashes / Merkle utilities
    import tb_hashing as _tbh  # type: ignore
except Exception:  # pragma: no cover - fallback path
    _tbh = None  # type: ignore

try:  # shared schema language
    import tb_schemas as schemas  # type: ignore
except Exception:  # pragma: no cover - fallback path
    schemas = None  # type: ignore

try:  # structured validators
    import tb_validation as validation  # type: ignore
except Exception:  # pragma: no cover - fallback path
    validation = None  # type: ignore

try:  # governed digital-twin kernel
    import digital_twin_kernel as dtk  # type: ignore
except Exception:  # pragma: no cover - optional integration
    dtk = None  # type: ignore


# =============================================================================
# Constants / helpers
# =============================================================================

TB_EXECUTION_VERSION = "tb-execution-v1.0-production"
TB_EXECUTION_SCHEMA = "tb.execution_result.v1"
_ZERO64 = "0" * 64
_EPS = 1e-12


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _is_np_scalar(x: Any) -> bool:
    return isinstance(x, np.generic)


def _is_np_array(x: Any) -> bool:
    return isinstance(x, np.ndarray)


def _json_safe(obj: Any) -> Any:
    if _tbc is not None and hasattr(_tbc, "canonical_json_safe"):
        try:
            return _tbc.canonical_json_safe(obj)  # type: ignore[attr-defined]
        except Exception:
            pass

    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if _is_np_scalar(obj):
        return _json_safe(obj.item())
    if _is_np_array(obj):
        arr = np.nan_to_num(np.asarray(obj), nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "__ndarray__": True,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "data": arr.astype(float).tolist(),
        }
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return [_json_safe(v) for v in sorted(obj, key=lambda v: str(v))]
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return _json_safe(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "payload") and callable(getattr(obj, "payload")):
        try:
            return _json_safe(obj.payload())
        except Exception:
            pass
    if hasattr(obj, "describe") and callable(getattr(obj, "describe")):
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


def _canonical_dumps(obj: Any) -> str:
    if _tbc is not None and hasattr(_tbc, "canonical_dumps"):
        try:
            return _tbc.canonical_dumps(obj)  # type: ignore[attr-defined]
        except Exception:
            pass
    return json.dumps(_json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_bytes(obj: Any) -> bytes:
    if _tbc is not None and hasattr(_tbc, "canonical_bytes"):
        try:
            return _tbc.canonical_bytes(obj)  # type: ignore[attr-defined]
        except Exception:
            pass
    return _canonical_dumps(obj).encode("utf-8", errors="strict")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _object_hash(obj: Any, *, domain: str = "tb.execution.object") -> str:
    if _tbh is not None and hasattr(_tbh, "domain_hash_hex"):
        try:
            return _tbh.domain_hash_hex(domain, obj)  # type: ignore[attr-defined]
        except Exception:
            pass
    framed = domain.encode("utf-8") + b"\0" + _canonical_bytes(obj)
    return _sha256_hex(framed)


def _merkle_root_hex(items: Sequence[Any], *, domain: str = "tb.execution.receipts") -> str:
    hashes = []
    for i, item in enumerate(items):
        if isinstance(item, str) and len(item) == 64 and all(c in "0123456789abcdefABCDEF" for c in item):
            hashes.append(item.lower())
        else:
            hashes.append(_object_hash({"index": i, "item": item}, domain=f"{domain}.leaf"))
    if _tbh is not None and hasattr(_tbh, "merkle_root_hex"):
        try:
            return _tbh.merkle_root_hex(hashes, domain=domain)  # type: ignore[attr-defined]
        except Exception:
            pass
    if not hashes:
        return _object_hash({"empty": True}, domain=f"{domain}.empty")
    level = [bytes.fromhex(h) for h in hashes]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        nxt = []
        for i in range(0, len(level), 2):
            nxt.append(hashlib.sha256(domain.encode("utf-8") + b"\0" + level[i] + level[i + 1]).digest())
        level = nxt
    return level[0].hex()


def _to_mapping(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return dict(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "payload") and callable(getattr(obj, "payload")):
        try:
            return dict(obj.payload())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return dict(vars(obj))
        except Exception:
            pass
    return {"value": _json_safe(obj)}


def _schema_hash(obj: Any, *, domain: str) -> str:
    if hasattr(obj, "hash") and callable(getattr(obj, "hash")):
        try:
            return str(obj.hash(domain=domain))
        except Exception:
            pass
    if hasattr(obj, "payload_hash") and callable(getattr(obj, "payload_hash")):
        try:
            return str(obj.payload_hash())
        except Exception:
            pass
    return _object_hash(_to_mapping(obj), domain=domain)


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _coerce_vector(x: Any, dim: int, *, key: str = "feature") -> np.ndarray:
    dim = int(max(1, dim))
    try:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
    except Exception:
        arr = np.zeros((0,), dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if arr.size == dim:
        return arr
    out = np.zeros((dim,), dtype=np.float32)
    if arr.size:
        for i, v in enumerate(arr):
            h = int(hashlib.sha256(f"{key}:{i}".encode()).hexdigest()[:16], 16)
            out[h % dim] += (-1.0 if h & 1 else 1.0) * np.float32(v)
    return out


def deterministic_feature_vector(payload: Any, dim: int, *, domain: str = "tb.execution.feature") -> np.ndarray:
    """Build a deterministic numeric feature vector from an arbitrary payload."""
    dim = int(max(1, dim))
    data = _canonical_bytes({"domain": domain, "payload": _json_safe(payload)})
    out = np.zeros((dim,), dtype=np.float32)
    # Expand SHA-256 blocks deterministically.
    counter = 0
    offset = 0
    while offset < dim:
        digest = hashlib.sha256(domain.encode("utf-8") + counter.to_bytes(8, "little") + data).digest()
        vals = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        vals = (vals - 127.5) / 127.5
        take = min(vals.size, dim - offset)
        out[offset : offset + take] = vals[:take]
        offset += take
        counter += 1
    n = float(np.linalg.norm(out))
    if np.isfinite(n) and n > _EPS:
        out = out / np.float32(n)
    return out.astype(np.float32, copy=False)


def _root_from_state_commitment(commitment: Any) -> str:
    root = str(_field(commitment, "state_root", ""))
    return root if root else _schema_hash(commitment, domain="tb.execution.state_commitment")


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ExecutionConfig:
    """Runtime policy for canonical transaction execution."""

    deterministic: bool = True
    consensus_safe: bool = True
    require_pre_state_match: bool = True
    require_valid_transaction: bool = True
    verify_signature: bool = False
    allow_direct_network_fallback: bool = False
    emit_lobe_receipts: bool = True
    run_ncp_lobe: bool = True
    run_cfc_lobe: bool = True
    cfc_time_delta: float = 1.0
    feature_domain: str = "tb.execution.tx_features.v1"
    strict: bool = True
    max_actions_per_transaction: int = 1
    include_debug_metadata: bool = False

    def normalized(self) -> "ExecutionConfig":
        cfg = copy.deepcopy(self)
        cfg.deterministic = bool(cfg.deterministic)
        cfg.consensus_safe = bool(cfg.consensus_safe)
        cfg.require_pre_state_match = bool(cfg.require_pre_state_match)
        cfg.require_valid_transaction = bool(cfg.require_valid_transaction)
        cfg.verify_signature = bool(cfg.verify_signature)
        cfg.allow_direct_network_fallback = bool(cfg.allow_direct_network_fallback)
        cfg.emit_lobe_receipts = bool(cfg.emit_lobe_receipts)
        cfg.run_ncp_lobe = bool(cfg.run_ncp_lobe)
        cfg.run_cfc_lobe = bool(cfg.run_cfc_lobe)
        cfg.cfc_time_delta = _as_float(cfg.cfc_time_delta, 1.0)
        if cfg.cfc_time_delta < 0.0:
            cfg.cfc_time_delta = 0.0
        cfg.feature_domain = str(cfg.feature_domain or "tb.execution.tx_features.v1")
        cfg.strict = bool(cfg.strict)
        cfg.max_actions_per_transaction = int(max(1, cfg.max_actions_per_transaction))
        cfg.include_debug_metadata = bool(cfg.include_debug_metadata)
        return cfg


@dataclass
class ExecutionDiagnostics:
    ok: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    integrations: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def warn(self, message: str) -> None:
        self.warnings.append(str(message))

    def error(self, message: str) -> None:
        self.errors.append(str(message))
        self.ok = False

    def note(self, message: str) -> None:
        self.notes.append(str(message))

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class ExecutionResult:
    """Canonical output of executing one transaction envelope."""

    tx_id: str
    tx_hash: str
    accepted: bool
    pre_state_root: str
    post_state_root: str
    action_receipts: List[Any] = field(default_factory=list)
    policy_receipt: Optional[Any] = None
    transaction_receipt: Optional[Any] = None
    transition_receipt: Optional[Any] = None
    ncp_health_receipt: Optional[Any] = None
    cfc_transition_receipt: Optional[Any] = None
    pre_state_commitment: Optional[Any] = None
    post_state_commitment: Optional[Any] = None
    execution_hash: str = ""
    diagnostics: ExecutionDiagnostics = field(default_factory=ExecutionDiagnostics)
    raw_execution: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso_utc)
    schema_version: str = TB_EXECUTION_SCHEMA

    def receipt_hashes(self) -> Dict[str, str]:
        action_hashes = [_schema_hash(r, domain="tb.execution.action_receipt") for r in self.action_receipts]
        return {
            "action_receipt_root": _merkle_root_hex(action_hashes, domain="tb.execution.action_receipt_root"),
            "policy_receipt_hash": _schema_hash(self.policy_receipt, domain="tb.execution.policy_receipt") if self.policy_receipt is not None else "",
            "transaction_receipt_hash": _schema_hash(self.transaction_receipt, domain="tb.execution.transaction_receipt") if self.transaction_receipt is not None else "",
            "transition_receipt_hash": _schema_hash(self.transition_receipt, domain="tb.execution.transition_receipt") if self.transition_receipt is not None else "",
            "ncp_health_hash": _schema_hash(self.ncp_health_receipt, domain="tb.execution.ncp_health") if self.ncp_health_receipt is not None else "",
            "cfc_transition_hash": _schema_hash(self.cfc_transition_receipt, domain="tb.execution.cfc_transition") if self.cfc_transition_receipt is not None else "",
        }

    def payload(self, *, include_execution_hash: bool = False) -> Dict[str, Any]:
        out = {
            "schema_version": self.schema_version,
            "version": TB_EXECUTION_VERSION,
            "tx_id": self.tx_id,
            "tx_hash": self.tx_hash,
            "accepted": bool(self.accepted),
            "pre_state_root": self.pre_state_root,
            "post_state_root": self.post_state_root,
            "pre_state_commitment": _json_safe(self.pre_state_commitment),
            "post_state_commitment": _json_safe(self.post_state_commitment),
            "action_receipts": [_json_safe(r) for r in self.action_receipts],
            "policy_receipt": _json_safe(self.policy_receipt),
            "transaction_receipt": _json_safe(self.transaction_receipt),
            "transition_receipt": _json_safe(self.transition_receipt),
            "ncp_health_receipt": _json_safe(self.ncp_health_receipt),
            "cfc_transition_receipt": _json_safe(self.cfc_transition_receipt),
            "receipt_hashes": self.receipt_hashes(),
            "diagnostics": self.diagnostics.to_dict(),
            "raw_execution": _json_safe(self.raw_execution),
            "created_at": self.created_at,
        }
        if include_execution_hash:
            out["execution_hash"] = self.execution_hash
        return out

    def compute_hash(self) -> str:
        return _object_hash(self.payload(include_execution_hash=False), domain="tb.execution.result")

    def finalize(self) -> "ExecutionResult":
        self.execution_hash = self.compute_hash()
        return self

    def verify(self) -> bool:
        return bool(self.execution_hash and self.execution_hash == self.compute_hash())

    def to_dict(self) -> Dict[str, Any]:
        return self.payload(include_execution_hash=True)


@dataclass
class BatchExecutionResult:
    accepted: bool
    results: List[ExecutionResult]
    pre_state_root: str
    post_state_root: str
    batch_hash: str = ""
    diagnostics: ExecutionDiagnostics = field(default_factory=ExecutionDiagnostics)
    created_at: str = field(default_factory=_now_iso_utc)
    schema_version: str = "tb.batch_execution_result.v1"

    def payload(self, *, include_batch_hash: bool = False) -> Dict[str, Any]:
        out = {
            "schema_version": self.schema_version,
            "version": TB_EXECUTION_VERSION,
            "accepted": bool(self.accepted),
            "pre_state_root": self.pre_state_root,
            "post_state_root": self.post_state_root,
            "result_hashes": [r.execution_hash for r in self.results],
            "results": [r.to_dict() for r in self.results],
            "diagnostics": self.diagnostics.to_dict(),
            "created_at": self.created_at,
        }
        if include_batch_hash:
            out["batch_hash"] = self.batch_hash
        return out

    def compute_hash(self) -> str:
        return _object_hash(self.payload(include_batch_hash=False), domain="tb.execution.batch")

    def finalize(self) -> "BatchExecutionResult":
        self.batch_hash = self.compute_hash()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return self.payload(include_batch_hash=True)


class ExecutionError(Exception):
    """Raised when strict execution fails."""


# =============================================================================
# Schema bridges
# =============================================================================

def tx_hash(envelope: Any) -> str:
    if hasattr(envelope, "tx_hash") and callable(getattr(envelope, "tx_hash")):
        try:
            return str(envelope.tx_hash())
        except Exception:
            pass
    if hasattr(envelope, "payload_hash") and callable(getattr(envelope, "payload_hash")):
        try:
            return str(envelope.payload_hash())
        except Exception:
            pass
    return _object_hash(_to_mapping(envelope), domain="tb.execution.transaction")


def action_hash(action: Any) -> str:
    if hasattr(action, "hash") and callable(getattr(action, "hash")):
        try:
            return str(action.hash())
        except Exception:
            pass
    return _object_hash(_to_mapping(action), domain="tb.execution.action")


def make_schema_state_commitment(commitment: Any) -> Any:
    """Convert any state commitment-like object to tb_schemas.StateCommitment when possible."""
    if schemas is None or not hasattr(schemas, "StateCommitment"):
        return commitment
    if isinstance(commitment, schemas.StateCommitment):  # type: ignore[attr-defined]
        return commitment
    m = _to_mapping(commitment)
    root = str(m.get("state_root", ""))
    if not root:
        root = _object_hash(m, domain="tb.execution.state_root_fallback")
    return schemas.StateCommitment(  # type: ignore[attr-defined]
        state_root=root,
        sketch_root=str(m.get("sketch_root", "")),
        metadata_root=str(m.get("metadata_root", "")),
        topology_root=str(m.get("topology_root", "")),
        history_root=str(m.get("history_root", "")),
        projector_config_hash=str(m.get("projector_config_hash", "")),
        node_count=_as_int(m.get("node_count", 0)),
        root_id=str(m.get("root_id", "0")),
        tick=_as_int(m.get("tick", 0)),
        algorithm=str(m.get("algorithm", "sha256")),
        commitment_policy=str(m.get("commitment_policy", "state+topology+committed_metadata")),
        created_at=str(m.get("created_at", "")),
        metadata={
            "source_schema": str(m.get("schema_version", m.get("schema", ""))),
            "source_commitment_hash": str(m.get("commitment_hash", "")),
            **dict(m.get("metadata", {}) if isinstance(m.get("metadata", {}), Mapping) else {}),
        },
    )


def capture_state_commitment(network: Any) -> Any:
    if hasattr(network, "state_commitment") and callable(getattr(network, "state_commitment")):
        return make_schema_state_commitment(network.state_commitment())
    if hasattr(network, "state_root") and callable(getattr(network, "state_root")):
        root = str(network.state_root())
        return make_schema_state_commitment({"state_root": root})
    if hasattr(network, "merkle_root") and callable(getattr(network, "merkle_root")):
        try:
            h = network.merkle_root()
            if isinstance(h, int):
                root = f"{h & ((1 << 64) - 1):016x}".rjust(64, "0")
            else:
                root = str(h)
            return make_schema_state_commitment({"state_root": root, "commitment_policy": "legacy_merkle_root_fallback"})
        except Exception:
            pass
    return make_schema_state_commitment({"state_root": _object_hash(_json_safe(network), domain="tb.execution.generic_state")})


def _make_action_receipt(
    *,
    action: Any,
    pre_root: str,
    post_root: str,
    approved: bool,
    executed: bool,
    status: str,
    error: str = "",
    warnings: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Any:
    ah = action_hash(action)
    kind = str(_field(action, "kind", _to_mapping(action).get("kind", "unknown")))
    node_id = _field(action, "node_id", _to_mapping(action).get("node_id", None))
    if schemas is not None and hasattr(schemas, "ActionReceipt"):
        return schemas.ActionReceipt(  # type: ignore[attr-defined]
            action_hash=ah,
            kind=kind,
            node_id=None if node_id is None else str(node_id),
            approved=bool(approved),
            executed=bool(executed),
            status=str(status),
            pre_state_root=str(pre_root),
            post_state_root=str(post_root),
            error=str(error or ""),
            warnings=list(warnings or []),
            metadata=dict(metadata or {}),
        )
    return {
        "schema_version": "tb.action_receipt.v1",
        "action_hash": ah,
        "kind": kind,
        "node_id": node_id,
        "approved": bool(approved),
        "executed": bool(executed),
        "status": str(status),
        "pre_state_root": str(pre_root),
        "post_state_root": str(post_root),
        "error": str(error or ""),
        "warnings": list(warnings or []),
        "metadata": dict(metadata or {}),
    }


def _make_policy_receipt(
    *,
    approved: bool,
    intent: str,
    pre_root: str,
    post_root: str,
    actions: Sequence[Any],
    action_receipts: Sequence[Any],
    execution_stats: Mapping[str, Any],
    rejected_actions: Optional[Sequence[Mapping[str, Any]]] = None,
    warnings: Optional[Sequence[str]] = None,
    reason: str = "",
    raw_policy_receipt: Optional[Mapping[str, Any]] = None,
) -> Any:
    action_hashes = [action_hash(a) for a in actions]
    action_receipt_hashes = [_schema_hash(r, domain="tb.execution.policy.action_receipt") for r in action_receipts]
    sim_payload = {
        "approved": bool(approved),
        "intent": intent,
        "execution_stats": _json_safe(execution_stats),
        "raw_policy_receipt": _json_safe(raw_policy_receipt or {}),
    }
    if schemas is not None and hasattr(schemas, "PolicyReceipt"):
        return schemas.PolicyReceipt(  # type: ignore[attr-defined]
            approved=bool(approved),
            intent=str(intent),
            pre_state_root=str(pre_root),
            post_state_root=str(post_root),
            simulation_hash=_object_hash(sim_payload, domain="tb.execution.policy.simulation"),
            action_hashes=action_hashes,
            rejected_actions=[dict(x) for x in list(rejected_actions or [])],
            warnings=list(warnings or []),
            reason=str(reason or ""),
            execution_stats=dict(execution_stats),
            action_receipt_hashes=action_receipt_hashes,
            metadata={"raw_policy_receipt_hash": _object_hash(raw_policy_receipt or {}, domain="tb.execution.raw_policy")},
        )
    return {
        "schema_version": "tb.policy_receipt.v1",
        "approved": bool(approved),
        "intent": intent,
        "pre_state_root": pre_root,
        "post_state_root": post_root,
        "simulation_hash": _object_hash(sim_payload, domain="tb.execution.policy.simulation"),
        "action_hashes": action_hashes,
        "action_receipt_hashes": action_receipt_hashes,
        "rejected_actions": [dict(x) for x in list(rejected_actions or [])],
        "warnings": list(warnings or []),
        "reason": reason,
        "execution_stats": dict(execution_stats),
    }


def _make_transaction_receipt(
    *,
    tx_id: str,
    th: str,
    accepted: bool,
    status: str,
    reason: str,
    pre_root: str,
    post_root: str,
    policy_receipt: Any,
    action_receipts: Sequence[Any],
) -> Any:
    policy_hash = _schema_hash(policy_receipt, domain="tb.execution.tx_receipt.policy") if policy_receipt is not None else ""
    action_receipt_hashes = [_schema_hash(r, domain="tb.execution.tx_receipt.action") for r in action_receipts]
    if schemas is not None and hasattr(schemas, "TransactionReceipt"):
        return schemas.TransactionReceipt(  # type: ignore[attr-defined]
            tx_id=str(tx_id),
            tx_hash=str(th),
            accepted=bool(accepted),
            status=str(status),
            reason=str(reason or ""),
            pre_state_root=str(pre_root),
            post_state_root=str(post_root),
            policy_receipt_hash=policy_hash,
            action_receipt_hashes=action_receipt_hashes,
        )
    return {
        "schema_version": "tb.transaction_receipt.v1",
        "tx_id": tx_id,
        "tx_hash": th,
        "accepted": bool(accepted),
        "status": status,
        "reason": reason,
        "pre_state_root": pre_root,
        "post_state_root": post_root,
        "policy_receipt_hash": policy_hash,
        "action_receipt_hashes": action_receipt_hashes,
    }


def _make_transition_receipt(
    *,
    th: str,
    pre_root: str,
    post_root: str,
    policy_receipt: Any,
    ncp_health_receipt: Any = None,
    cfc_transition_receipt: Any = None,
    tensor_commitment: Any = None,
    accepted: bool,
    reason: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Any:
    if schemas is not None and hasattr(schemas, "TransitionReceipt"):
        return schemas.TransitionReceipt(  # type: ignore[attr-defined]
            tx_hash=str(th),
            pre_state_root=str(pre_root),
            post_state_root=str(post_root),
            policy_receipt_hash=_schema_hash(policy_receipt, domain="tb.execution.transition.policy") if policy_receipt is not None else "",
            tensor_commitment_hash=_field(tensor_commitment, "commitment_hash", "") or (_schema_hash(tensor_commitment, domain="tb.execution.transition.tensor") if tensor_commitment is not None else ""),
            ncp_health_hash=_schema_hash(ncp_health_receipt, domain="tb.execution.transition.ncp") if ncp_health_receipt is not None else "",
            cfc_transition_hash=_schema_hash(cfc_transition_receipt, domain="tb.execution.transition.cfc") if cfc_transition_receipt is not None else "",
            accepted=bool(accepted),
            reason=str(reason or ""),
            metadata=dict(metadata or {}),
        )
    return {
        "schema_version": "tb.transition_receipt.v1",
        "tx_hash": th,
        "pre_state_root": pre_root,
        "post_state_root": post_root,
        "policy_receipt_hash": _schema_hash(policy_receipt, domain="tb.execution.transition.policy") if policy_receipt is not None else "",
        "ncp_health_hash": _schema_hash(ncp_health_receipt, domain="tb.execution.transition.ncp") if ncp_health_receipt is not None else "",
        "cfc_transition_hash": _schema_hash(cfc_transition_receipt, domain="tb.execution.transition.cfc") if cfc_transition_receipt is not None else "",
        "accepted": bool(accepted),
        "reason": reason,
        "metadata": dict(metadata or {}),
    }


# =============================================================================
# Action decoding / direct execution fallback
# =============================================================================

def decode_action(action_payload: Any) -> Any:
    """Decode one transaction action into a digital_twin_kernel.Action when available."""
    if hasattr(action_payload, "kind") and hasattr(action_payload, "node_id"):
        return action_payload
    p = dict(action_payload or {}) if isinstance(action_payload, Mapping) else {"kind": str(action_payload)}
    kind = str(p.get("kind", p.get("action_kind", ""))).strip()
    node_id = p.get("node_id", None)
    payload = p.get("payload", p.get("data", None))
    note = str(p.get("note", p.get("memo", "")))
    if dtk is not None and hasattr(dtk, "Action"):
        return dtk.Action(kind=kind, node_id=node_id, payload=payload, note=note)  # type: ignore[attr-defined]
    return {"kind": kind, "node_id": node_id, "payload": payload, "note": note}


def decode_transaction_actions(envelope: Any, *, max_actions: int = 1) -> List[Any]:
    payload = _field(envelope, "action", {})
    if isinstance(payload, Mapping) and isinstance(payload.get("actions"), list):
        actions = [decode_action(x) for x in payload.get("actions", [])]
    elif isinstance(payload, list):
        actions = [decode_action(x) for x in payload]
    else:
        actions = [decode_action(payload)]
    return actions[: int(max(1, max_actions))]


def _direct_execute_action(network: Any, action: Any) -> Tuple[bool, str]:
    """Execute one action without AkkuratInterface. Only used when explicitly allowed."""
    kind = str(_field(action, "kind", _to_mapping(action).get("kind", "")))
    node_id = _field(action, "node_id", _to_mapping(action).get("node_id", None))
    payload = _field(action, "payload", _to_mapping(action).get("payload", None))
    note = str(_field(action, "note", _to_mapping(action).get("note", "direct_execution")))

    try:
        if kind == "update_node_data" and hasattr(network, "update_node_data"):
            network.update_node_data(str(node_id), payload, note=note)
            return True, ""
        if kind == "update_node_latent" and hasattr(network, "update_node_latent"):
            network.update_node_latent(str(node_id), payload, note=note)
            return True, ""
        if kind == "attach_metadata":
            if node_id is None:
                return False, "attach_metadata requires node_id"
            nodes = getattr(network, "nodes", {})
            if str(node_id) not in nodes:
                return False, f"unknown node_id: {node_id}"
            md_payload = payload if isinstance(payload, Mapping) else {"payload": payload}
            nodes[str(node_id)].metadata.update(dict(md_payload))
            if hasattr(network, "bump_tick"):
                network.bump_tick()
            return True, ""
        if kind == "raise_alert":
            # Direct fallback has no alert channel; treat as applied no-op with metadata semantics.
            return True, ""
        return False, f"unsupported direct action kind: {kind}"
    except Exception as exc:
        return False, repr(exc)


# =============================================================================
# Lobe integration
# =============================================================================

def _input_dim_for_ncp(lobe: Any) -> Optional[int]:
    for attr in ("num_inputs", "input_size"):
        try:
            v = int(getattr(lobe, attr))
            if v > 0:
                return v
        except Exception:
            pass
    return None


def _input_dim_for_cfc(lobe: Any) -> Optional[int]:
    for attr in ("input_size", "num_inputs"):
        try:
            v = int(getattr(lobe, attr))
            if v > 0:
                return v
        except Exception:
            pass
    return None


def run_ncp_lobe_receipt(ncp_lobe: Any, feature_payload: Any, diagnostics: ExecutionDiagnostics, *, config: ExecutionConfig) -> Optional[Any]:
    if ncp_lobe is None or not config.emit_lobe_receipts or not config.run_ncp_lobe:
        return None
    try:
        dim = _input_dim_for_ncp(ncp_lobe)
        if dim is None:
            diagnostics.warn("NCP lobe has no discoverable input dimension; skipped.")
            return None
        x = deterministic_feature_vector(feature_payload, dim, domain=f"{config.feature_domain}.ncp")
        if hasattr(ncp_lobe, "step"):
            try:
                ncp_lobe.step(x, return_state=True)
            except TypeError:
                ncp_lobe.step(x)
        if hasattr(ncp_lobe, "lobe_health_receipt") and callable(getattr(ncp_lobe, "lobe_health_receipt")):
            return ncp_lobe.lobe_health_receipt(include_expensive=False)
        metrics = ncp_lobe.health_metrics(include_expensive=False) if hasattr(ncp_lobe, "health_metrics") else {}
        state_hash = ncp_lobe.state_hash() if hasattr(ncp_lobe, "state_hash") else _object_hash(metrics, domain="tb.execution.ncp.state")
        if schemas is not None and hasattr(schemas, "LobeHealthReceipt"):
            return schemas.LobeHealthReceipt(  # type: ignore[attr-defined]
                lobe_id=str(getattr(ncp_lobe, "lobe_id", getattr(ncp_lobe, "policy_id", "ncp_lobe"))),
                lobe_kind="ncp_liquid_reflex",
                stable=bool(metrics.get("is_stable", True)) if isinstance(metrics, Mapping) else True,
                degraded=bool(metrics.get("degraded_mode", False)) if isinstance(metrics, Mapping) else False,
                health_score=1.0 if not isinstance(metrics, Mapping) or metrics.get("is_stable", True) else 0.5,
                state_hash=str(state_hash),
                parameter_count=int(getattr(ncp_lobe, "parameter_count", lambda: 0)()),
                dense_equivalent_parameter_count=int(getattr(ncp_lobe, "equivalent_dense_parameter_count", lambda: 0)()),
                recurrent_mode=str(metrics.get("recurrent_mode", "")) if isinstance(metrics, Mapping) else "",
                metrics=dict(metrics) if isinstance(metrics, Mapping) else {},
            )
        return {"lobe_id": getattr(ncp_lobe, "policy_id", "ncp_lobe"), "state_hash": state_hash, "metrics": metrics}
    except Exception as exc:
        diagnostics.warn(f"NCP lobe receipt failed: {exc!r}")
        if config.strict and config.consensus_safe:
            raise
        return None


def run_cfc_transition_receipt(
    cfc_lobe: Any,
    feature_payload: Any,
    *,
    txh: str,
    pre_root: str,
    post_root: str,
    diagnostics: ExecutionDiagnostics,
    config: ExecutionConfig,
) -> Optional[Any]:
    if cfc_lobe is None or not config.emit_lobe_receipts or not config.run_cfc_lobe:
        return None
    try:
        dim = _input_dim_for_cfc(cfc_lobe)
        if dim is None:
            diagnostics.warn("CfC lobe has no discoverable input dimension; skipped.")
            return None
        x = deterministic_feature_vector(feature_payload, dim, domain=f"{config.feature_domain}.cfc")
        predicted_state = None
        local_receipt = None
        if hasattr(cfc_lobe, "forecast") and callable(getattr(cfc_lobe, "forecast")):
            out = cfc_lobe.forecast(x, time_delta=config.cfc_time_delta, return_receipt=True)
            if isinstance(out, tuple) and len(out) == 2:
                predicted_state, local_receipt = out
            else:
                predicted_state = out
        elif hasattr(cfc_lobe, "step"):
            out = cfc_lobe.step(x, time_delta=config.cfc_time_delta, return_state=True)
            predicted_state = out[0] if isinstance(out, tuple) else out
        predicted_hash = ""
        control_hash = ""
        if hasattr(cfc_lobe, "state_hash"):
            try:
                predicted_hash = str(cfc_lobe.state_hash(predicted_state))
            except Exception:
                pass
        if not predicted_hash:
            predicted_hash = _object_hash(predicted_state, domain="tb.execution.cfc.predicted_state")
        if hasattr(cfc_lobe, "get_control_vector"):
            try:
                control = cfc_lobe.get_control_vector(out_dim=32, state=predicted_state)
                control_hash = _object_hash(np.asarray(control).astype(np.float32), domain="tb.execution.cfc.control")
            except Exception:
                pass
        forecast_error = None
        cosine_similarity = None
        accepted = None
        if hasattr(cfc_lobe, "compare_forecast_actual"):
            try:
                err_report = cfc_lobe.compare_forecast_actual(predicted_state, predicted_state, time_delta=config.cfc_time_delta)
                forecast_error = _field(err_report, "relative_l2_error", _field(err_report, "l2_error", None))
                cosine_similarity = _field(err_report, "cosine_similarity", None)
                accepted = _field(err_report, "accepted", None)
            except Exception:
                pass
        backbone_hashes: List[str] = []
        if hasattr(cfc_lobe, "tensor_backbone_commitments"):
            try:
                backbone_hashes = [str(x) for x in cfc_lobe.tensor_backbone_commitments()]
            except Exception:
                backbone_hashes = []
        if schemas is not None and hasattr(schemas, "CfCTransitionReceipt"):
            return schemas.CfCTransitionReceipt(  # type: ignore[attr-defined]
                lobe_id=str(getattr(cfc_lobe, "policy_id", "cfc_lobe")),
                pre_state_root=str(pre_root),
                tx_hash=str(txh),
                predicted_post_state_hash=str(predicted_hash),
                actual_post_state_hash=str(post_root),
                forecast_error=None if forecast_error is None else float(forecast_error),
                cosine_similarity=None if cosine_similarity is None else float(cosine_similarity),
                time_delta=float(config.cfc_time_delta),
                control_vector_hash=str(control_hash),
                tensor_backbone_commitment_hashes=backbone_hashes,
                accepted=accepted if accepted is None else bool(accepted),
                metadata={"local_receipt_hash": _object_hash(local_receipt, domain="tb.execution.cfc.local_receipt") if local_receipt is not None else ""},
            )
        return {
            "lobe_id": getattr(cfc_lobe, "policy_id", "cfc_lobe"),
            "pre_state_root": pre_root,
            "tx_hash": txh,
            "predicted_post_state_hash": predicted_hash,
            "actual_post_state_hash": post_root,
            "time_delta": config.cfc_time_delta,
            "tensor_backbone_commitment_hashes": backbone_hashes,
        }
    except Exception as exc:
        diagnostics.warn(f"CfC transition receipt failed: {exc!r}")
        if config.strict and config.consensus_safe:
            raise
        return None


# =============================================================================
# Core executor
# =============================================================================

class CanonicalTransactionExecutor:
    """Canonical transaction execution service."""

    def __init__(self, *, config: Optional[ExecutionConfig] = None):
        self.config = (config or ExecutionConfig()).normalized()

    def _validate_transaction(self, tx: Any, diagnostics: ExecutionDiagnostics) -> None:
        if not self.config.require_valid_transaction:
            return
        if validation is not None and hasattr(validation, "validate_transaction_envelope"):
            report = validation.validate_transaction_envelope(tx)  # type: ignore[attr-defined]
            ok = bool(getattr(report, "ok", False))
            if not ok:
                diagnostics.error(f"transaction envelope validation failed: {_json_safe(report)}")
                if self.config.strict:
                    raise ExecutionError(diagnostics.errors[-1])
        elif hasattr(tx, "validate") and callable(getattr(tx, "validate")):
            issues = tx.validate()
            if issues:
                diagnostics.error(f"transaction envelope validation failed: {_json_safe(issues)}")
                if self.config.strict:
                    raise ExecutionError(diagnostics.errors[-1])

    def _check_pre_state(self, tx: Any, pre_root: str, diagnostics: ExecutionDiagnostics) -> None:
        expected = str(_field(tx, "pre_state_root", "") or "")
        if self.config.require_pre_state_match and expected and expected != pre_root:
            msg = f"transaction pre_state_root mismatch: tx={expected} actual={pre_root}"
            diagnostics.error(msg)
            if self.config.strict:
                raise ExecutionError(msg)

    def execute(
        self,
        tx: Any,
        *,
        network: Any,
        control_plane: Optional[Any] = None,
        ncp_lobe: Optional[Any] = None,
        cfc_lobe: Optional[Any] = None,
        tensor_commitment: Optional[Any] = None,
        intent: Optional[str] = None,
    ) -> ExecutionResult:
        cfg = self.config
        diagnostics = ExecutionDiagnostics(
            integrations={
                "tb_schemas": schemas is not None,
                "tb_validation": validation is not None,
                "digital_twin_kernel": dtk is not None,
                "control_plane": control_plane is not None,
                "ncp_lobe": ncp_lobe is not None,
                "cfc_lobe": cfc_lobe is not None,
            }
        )
        tx_id = str(_field(tx, "tx_id", _object_hash(tx, domain="tb.execution.tx_id")[:32]))
        th = tx_hash(tx)
        raw_execution: Dict[str, Any] = {}

        try:
            self._validate_transaction(tx, diagnostics)
            pre_commit = capture_state_commitment(network)
            pre_root = _root_from_state_commitment(pre_commit)
            self._check_pre_state(tx, pre_root, diagnostics)
            if diagnostics.errors:
                raise ExecutionError(diagnostics.errors[-1])
            actions = decode_transaction_actions(tx, max_actions=cfg.max_actions_per_transaction)
            intent_value = str(intent or _field(tx, "metadata", {}).get("intent", "transaction_execution") if isinstance(_field(tx, "metadata", {}), Mapping) else "transaction_execution")

            # Pre-execution predictive lobe evidence uses the tx and pre-state context.
            cfc_pre_feature_payload = {
                "tx_hash": th,
                "tx": _json_safe(tx),
                "pre_state_root": pre_root,
                "actions": [_json_safe(a) for a in actions],
            }

            # Run governed action path. No compatibility bypass is performed here.
            approved = False
            reason = ""
            execution_stats: Dict[str, Any] = {"applied": 0, "skipped": 0, "errors": 0, "error_details": []}
            rejected_actions: List[Dict[str, Any]] = []
            raw_policy_receipt: Dict[str, Any] = {}

            if control_plane is not None and hasattr(control_plane, "governed_actions"):
                raw = control_plane.governed_actions(actions, intent=intent_value)
                raw_execution = _to_mapping(raw)
                approved = bool(raw_execution.get("approved", False))
                execution_stats = dict(raw_execution.get("execute", raw_execution.get("executed", execution_stats)) or execution_stats)
                rejected_actions = [dict(x) for x in raw_execution.get("rejected_actions", []) if isinstance(x, Mapping)]
                raw_policy_receipt = dict(raw_execution.get("policy_receipt", {}) or {})
                reason = "approved" if approved else str(raw_policy_receipt.get("reason", "policy_rejected") or "policy_rejected")
            elif cfg.allow_direct_network_fallback:
                diagnostics.warn("governed control plane absent; using explicitly allowed direct network fallback")
                applied = 0
                skipped = 0
                errors = 0
                details = []
                for a in actions:
                    ok, err = _direct_execute_action(network, a)
                    if ok:
                        applied += 1
                    else:
                        errors += 1
                        details.append({"action": _json_safe(a), "error": err})
                approved = errors == 0
                execution_stats = {"applied": applied, "skipped": skipped, "errors": errors, "error_details": details}
                reason = "direct_fallback_applied" if approved else "direct_fallback_errors"
            else:
                msg = "control_plane with governed_actions is required unless allow_direct_network_fallback=True"
                diagnostics.error(msg)
                if cfg.strict:
                    raise ExecutionError(msg)

            post_commit = capture_state_commitment(network)
            post_root = _root_from_state_commitment(post_commit)

            # Construct action receipts independent of the dtk internal receipts so the schema layer is canonical.
            action_receipts: List[Any] = []
            executed_ok = bool(approved and int(execution_stats.get("errors", 0) or 0) == 0)
            status = "executed" if executed_ok else ("rejected" if not approved else "error")
            error_text = "" if executed_ok else reason
            for a in actions:
                action_receipts.append(
                    _make_action_receipt(
                        action=a,
                        pre_root=pre_root,
                        post_root=post_root,
                        approved=bool(approved),
                        executed=bool(executed_ok),
                        status=status,
                        error=error_text,
                        warnings=diagnostics.warnings,
                        metadata={"tx_hash": th, "execution_stats_hash": _object_hash(execution_stats, domain="tb.execution.action.stats")},
                    )
                )

            policy_receipt = _make_policy_receipt(
                approved=approved,
                intent=intent_value,
                pre_root=pre_root,
                post_root=post_root,
                actions=actions,
                action_receipts=action_receipts,
                execution_stats=execution_stats,
                rejected_actions=rejected_actions,
                warnings=diagnostics.warnings,
                reason="" if approved else reason,
                raw_policy_receipt=raw_policy_receipt,
            )

            # Post-execution lobe receipts.
            ncp_receipt = run_ncp_lobe_receipt(
                ncp_lobe,
                {
                    "tx_hash": th,
                    "pre_state_root": pre_root,
                    "post_state_root": post_root,
                    "approved": approved,
                    "execution_stats": execution_stats,
                },
                diagnostics,
                config=cfg,
            )
            cfc_receipt = run_cfc_transition_receipt(
                cfc_lobe,
                {**cfc_pre_feature_payload, "post_state_root": post_root, "approved": approved},
                txh=th,
                pre_root=pre_root,
                post_root=post_root,
                diagnostics=diagnostics,
                config=cfg,
            )

            tx_receipt = _make_transaction_receipt(
                tx_id=tx_id,
                th=th,
                accepted=approved,
                status="accepted" if approved else "rejected",
                reason="" if approved else reason,
                pre_root=pre_root,
                post_root=post_root,
                policy_receipt=policy_receipt,
                action_receipts=action_receipts,
            )
            transition_receipt = _make_transition_receipt(
                th=th,
                pre_root=pre_root,
                post_root=post_root,
                policy_receipt=policy_receipt,
                ncp_health_receipt=ncp_receipt,
                cfc_transition_receipt=cfc_receipt,
                tensor_commitment=tensor_commitment,
                accepted=approved,
                reason="" if approved else reason,
                metadata={"tx_id": tx_id, "execution_version": TB_EXECUTION_VERSION},
            )
            result = ExecutionResult(
                tx_id=tx_id,
                tx_hash=th,
                accepted=bool(approved),
                pre_state_root=pre_root,
                post_state_root=post_root,
                action_receipts=action_receipts,
                policy_receipt=policy_receipt,
                transaction_receipt=tx_receipt,
                transition_receipt=transition_receipt,
                ncp_health_receipt=ncp_receipt,
                cfc_transition_receipt=cfc_receipt,
                pre_state_commitment=pre_commit,
                post_state_commitment=post_commit,
                diagnostics=diagnostics,
                raw_execution=raw_execution if cfg.include_debug_metadata else {"approved": approved, "execution_stats": execution_stats},
            ).finalize()

            if validation is not None:
                for obj, fn_name in [
                    (policy_receipt, "validate_policy_receipt"),
                    (tx_receipt, "validate_transaction_receipt"),
                    (transition_receipt, "validate_transition_receipt"),
                ]:
                    fn = getattr(validation, fn_name, None)
                    if callable(fn):
                        rep = fn(obj)
                        if not bool(getattr(rep, "ok", False)):
                            diagnostics.warn(f"{fn_name} reported issues: {_json_safe(rep)}")
            return result

        except Exception as exc:
            if isinstance(exc, ExecutionError):
                err = str(exc)
            else:
                err = repr(exc)
                diagnostics.error(err)
                diagnostics.metadata["traceback"] = traceback.format_exc()
            # Capture best-effort post root after failure.
            try:
                post_commit = capture_state_commitment(network)
                post_root = _root_from_state_commitment(post_commit)
            except Exception:
                post_commit = None
                post_root = ""
            try:
                pre_commit = pre_commit  # type: ignore[name-defined]
                pre_root = pre_root  # type: ignore[name-defined]
            except Exception:
                pre_commit = None
                pre_root = ""
            policy_receipt = _make_policy_receipt(
                approved=False,
                intent=str(intent or "transaction_execution"),
                pre_root=pre_root,
                post_root=post_root,
                actions=[],
                action_receipts=[],
                execution_stats={"applied": 0, "skipped": 0, "errors": 1, "error_details": [{"error": err}]},
                rejected_actions=[],
                warnings=diagnostics.warnings,
                reason=err,
            )
            tx_receipt = _make_transaction_receipt(
                tx_id=tx_id,
                th=th,
                accepted=False,
                status="error",
                reason=err,
                pre_root=pre_root,
                post_root=post_root,
                policy_receipt=policy_receipt,
                action_receipts=[],
            )
            transition_receipt = _make_transition_receipt(
                th=th,
                pre_root=pre_root,
                post_root=post_root,
                policy_receipt=policy_receipt,
                accepted=False,
                reason=err,
            )
            result = ExecutionResult(
                tx_id=tx_id,
                tx_hash=th,
                accepted=False,
                pre_state_root=pre_root,
                post_state_root=post_root,
                policy_receipt=policy_receipt,
                transaction_receipt=tx_receipt,
                transition_receipt=transition_receipt,
                pre_state_commitment=pre_commit,
                post_state_commitment=post_commit,
                diagnostics=diagnostics,
                raw_execution={"error": err},
            ).finalize()
            if cfg.strict:
                # Return structured error only when caller uses non-strict. Strict raises after result construction.
                raise ExecutionError(err) from exc
            return result

    def execute_batch(
        self,
        transactions: Sequence[Any],
        *,
        network: Any,
        control_plane: Optional[Any] = None,
        ncp_lobe: Optional[Any] = None,
        cfc_lobe: Optional[Any] = None,
        tensor_commitment: Optional[Any] = None,
        intent: str = "batch_transaction_execution",
    ) -> BatchExecutionResult:
        diagnostics = ExecutionDiagnostics()
        pre_commit = capture_state_commitment(network)
        pre_root = _root_from_state_commitment(pre_commit)
        results: List[ExecutionResult] = []
        for i, tx in enumerate(transactions):
            try:
                res = self.execute(
                    tx,
                    network=network,
                    control_plane=control_plane,
                    ncp_lobe=ncp_lobe,
                    cfc_lobe=cfc_lobe,
                    tensor_commitment=tensor_commitment,
                    intent=f"{intent}:{i}",
                )
                results.append(res)
                if not res.accepted:
                    diagnostics.warn(f"transaction {i} not accepted: {res.tx_id}")
                    if self.config.strict:
                        raise ExecutionError(f"batch transaction {i} rejected")
            except Exception as exc:
                diagnostics.error(f"transaction {i} execution failed: {exc!r}")
                if self.config.strict:
                    raise
        post_commit = capture_state_commitment(network)
        post_root = _root_from_state_commitment(post_commit)
        batch = BatchExecutionResult(
            accepted=bool(results and all(r.accepted for r in results)),
            results=results,
            pre_state_root=pre_root,
            post_state_root=post_root,
            diagnostics=diagnostics,
        ).finalize()
        return batch


# =============================================================================
# Convenience functions
# =============================================================================

def execute_transaction(
    tx: Any,
    *,
    network: Any,
    control_plane: Optional[Any] = None,
    ncp_lobe: Optional[Any] = None,
    cfc_lobe: Optional[Any] = None,
    tensor_commitment: Optional[Any] = None,
    intent: Optional[str] = None,
    config: Optional[ExecutionConfig] = None,
) -> ExecutionResult:
    return CanonicalTransactionExecutor(config=config).execute(
        tx,
        network=network,
        control_plane=control_plane,
        ncp_lobe=ncp_lobe,
        cfc_lobe=cfc_lobe,
        tensor_commitment=tensor_commitment,
        intent=intent,
    )


def execute_transaction_batch(
    transactions: Sequence[Any],
    *,
    network: Any,
    control_plane: Optional[Any] = None,
    ncp_lobe: Optional[Any] = None,
    cfc_lobe: Optional[Any] = None,
    tensor_commitment: Optional[Any] = None,
    intent: str = "batch_transaction_execution",
    config: Optional[ExecutionConfig] = None,
) -> BatchExecutionResult:
    return CanonicalTransactionExecutor(config=config).execute_batch(
        transactions,
        network=network,
        control_plane=control_plane,
        ncp_lobe=ncp_lobe,
        cfc_lobe=cfc_lobe,
        tensor_commitment=tensor_commitment,
        intent=intent,
    )


# =============================================================================
# Status / self-test / CLI
# =============================================================================

def status_report() -> Dict[str, Any]:
    return {
        "ok": True,
        "version": TB_EXECUTION_VERSION,
        "ts": _now_iso_utc(),
        "integrations": {
            "tb_canonical": _tbc is not None,
            "tb_hashing": _tbh is not None,
            "tb_schemas": schemas is not None,
            "tb_validation": validation is not None,
            "digital_twin_kernel": dtk is not None,
        },
    }


def run_self_test() -> Dict[str, Any]:
    if dtk is None:
        raise RuntimeError("digital_twin_kernel.py is required for tb_execution self-test")
    # Small deterministic twin.
    network = dtk.TreeTensorNetwork(vector_dim=32, sketch_dim=16, seed=11, deterministic=True, consensus_safe=True)
    network.add_node("5.0", "Thing", parent_id="0")
    network.add_node("5.4", "Telemetry", parent_id="5.0")
    cp = dtk.AkkuratInterface(network, strict=False, consensus_safe=True)
    pre = capture_state_commitment(network)
    pre_root = _root_from_state_commitment(pre)

    if schemas is not None and hasattr(schemas, "TransactionEnvelope"):
        tx = schemas.TransactionEnvelope.new(  # type: ignore[attr-defined]
            chain_id="tb-selftest",
            sender_id="tester",
            sender_public_key="ab" * 32,
            sender_sequence=1,
            pre_state_root=pre_root,
            action={"kind": "attach_metadata", "node_id": "5.4", "payload": {"temperature_c": 33.0}, "note": "self_test"},
        )
    else:
        tx = {
            "chain_id": "tb-selftest",
            "tx_id": "selftest-tx",
            "sender_id": "tester",
            "sender_public_key": "ab" * 32,
            "sender_sequence": 1,
            "pre_state_root": pre_root,
            "action": {"kind": "attach_metadata", "node_id": "5.4", "payload": {"temperature_c": 33.0}, "note": "self_test"},
        }

    result = execute_transaction(
        tx,
        network=network,
        control_plane=cp,
        config=ExecutionConfig(strict=True, require_valid_transaction=False, emit_lobe_receipts=False),
    )
    if not result.accepted:
        raise AssertionError(f"execution not accepted: {result.to_dict()}")
    if not result.verify():
        raise AssertionError("execution hash verification failed")
    if result.pre_state_root != pre_root:
        raise AssertionError("pre-state root mismatch")
    if result.post_state_root == pre_root:
        raise AssertionError("post-state root did not change")

    # Batch non-strict rejection path with mismatched pre-root.
    bad_tx = tx.clone(pre_state_root="ff" * 32, sender_sequence=2) if hasattr(tx, "clone") else dict(tx, pre_state_root="ff" * 32, sender_sequence=2)
    nonstrict = execute_transaction(
        bad_tx,
        network=network,
        control_plane=cp,
        config=ExecutionConfig(strict=False, require_valid_transaction=False, emit_lobe_receipts=False),
    )
    if nonstrict.accepted:
        raise AssertionError("mismatched pre-state transaction should not be accepted")

    return {
        "ok": True,
        "status": status_report(),
        "execution_hash": result.execution_hash,
        "tx_hash": result.tx_hash,
        "pre_state_root": result.pre_state_root,
        "post_state_root": result.post_state_root,
        "receipt_hashes": result.receipt_hashes(),
        "nonstrict_error_ok": not nonstrict.accepted,
    }


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(description="Tensorized Blockchain canonical execution pipeline")
    p.add_argument("--mode", choices=["status", "self-test"], default="self-test")
    p.add_argument("--pretty", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        payload = status_report() if args.mode == "status" else run_self_test()
        print(json.dumps(_json_safe(payload), indent=2 if args.pretty else None, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        err = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc(), "status": status_report()}
        print(json.dumps(_json_safe(err), indent=2, ensure_ascii=False, sort_keys=True))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
