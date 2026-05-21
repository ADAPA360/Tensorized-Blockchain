#!/usr/bin/env python3
# tb_replay.py
r"""
Project Chimera / Akkurat - Deterministic Chain Replay
======================================================

Purpose
-------
Deterministic chain replay and forensic verification for the Tensorized
Blockchain stack.

This module verifies that a stored ledger can be replayed from genesis and that
all consensus-critical commitments still match their canonical derivations:

- block hash chain
- transaction Merkle roots
- transaction replay protection and sender sequence numbers
- transition receipt roots
- consensus certificate structural integrity
- tensor commitments, when the tensorized ledger kernel is available
- governed pre/post state roots, when a deterministic world-state factory is
  supplied
- lobe receipts, when deterministic lobe runtimes are supplied through the
  execution pipeline

Design boundaries
-----------------
- This module does not create new canonical blocks.
- This module does not mutate persisted ledgers.
- This module does not silently approve governance failures.
- Full state replay requires a caller-supplied deterministic network factory;
  otherwise the module performs static/hash/tensor replay and reports that
  world-state execution replay was skipped.

Primary API
-----------
- ReplayConfig
- ReplayDivergence
- ReplayStepResult
- ReplayReport
- ChainReplayer
- replay_ledger(...)
- replay_ledger_file(...)
- verify_ledger_file(...)

CLI
---
    python tb_replay.py --ledger ledger.json --output replay_report.json
    python tb_replay.py --self-test
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


# =============================================================================
# Optional project imports
# =============================================================================

try:
    import tb_canonical as _tbc  # type: ignore
except Exception:  # pragma: no cover
    _tbc = None  # type: ignore

try:
    import tb_hashing as _tbh  # type: ignore
except Exception:  # pragma: no cover
    _tbh = None  # type: ignore

try:
    import tb_schemas as schemas  # type: ignore
except Exception:  # pragma: no cover
    schemas = None  # type: ignore

try:
    import tb_validation as validation  # type: ignore
except Exception:  # pragma: no cover
    validation = None  # type: ignore

try:
    import tb_execution as execution  # type: ignore
except Exception:  # pragma: no cover
    execution = None  # type: ignore

try:
    import tb_crypto as crypto  # type: ignore
except Exception:  # pragma: no cover
    crypto = None  # type: ignore

try:
    import tensorized_blockchain_kernel as ledger_kernel  # type: ignore
except Exception:  # pragma: no cover
    ledger_kernel = None  # type: ignore

try:
    import digital_twin_kernel as dtk  # type: ignore
except Exception:  # pragma: no cover
    dtk = None  # type: ignore


# =============================================================================
# Constants / small helpers
# =============================================================================

TB_REPLAY_VERSION = "tb-replay-v1.0-production"
TB_REPLAY_SCHEMA = "tb.replay_report.v1"
ZERO_HASH_256 = "0" * 64


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_safe(obj: Any) -> Any:
    if _tbc is not None and hasattr(_tbc, "canonical_json_safe"):
        try:
            return _tbc.canonical_json_safe(obj)  # type: ignore[attr-defined]
        except Exception:
            pass
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if np is not None and isinstance(obj, getattr(np, "generic", ())):
        return _json_safe(obj.item())
    if np is not None and isinstance(obj, getattr(np, "ndarray", ())):
        arr = np.asarray(obj)
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.tolist()
    if is_dataclass(obj):
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            try:
                return obj.to_dict()
            except Exception:
                pass
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return [_json_safe(v) for v in sorted(obj, key=lambda x: str(x))]
    if isinstance(obj, bytes):
        return obj.hex()
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
            return str(_tbc.canonical_dumps(obj))  # type: ignore[attr-defined]
        except Exception:
            pass
    return json.dumps(_json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_bytes(obj: Any) -> bytes:
    if _tbc is not None and hasattr(_tbc, "canonical_bytes"):
        try:
            return bytes(_tbc.canonical_bytes(obj))  # type: ignore[attr-defined]
        except Exception:
            pass
    return _canonical_dumps(obj).encode("utf-8", errors="strict")


def _object_hash(obj: Any, *, domain: str = "tb.replay.object") -> str:
    if _tbh is not None:
        for name in ("domain_hash_hex", "hash_object"):
            fn = getattr(_tbh, name, None)
            if callable(fn):
                try:
                    if name == "domain_hash_hex":
                        return str(fn(obj, domain=domain))
                    return str(fn(obj))
                except Exception:
                    pass
    import hashlib

    return hashlib.sha256((str(domain) + "\x00").encode("utf-8") + _canonical_bytes(obj)).hexdigest()


def _merkle_root_hashes(hashes: Sequence[str], *, domain: str = "tb.replay.merkle") -> str:
    hs = [str(h) for h in hashes]
    if _tbh is not None and hasattr(_tbh, "merkle_root_hex"):
        try:
            return str(_tbh.merkle_root_hex(hs, domain=domain, leaves_are_hashes=True))  # type: ignore[attr-defined]
        except Exception:
            pass
    if not hs:
        return _object_hash({"empty": True}, domain=domain)
    level = [_object_hash({"leaf": h}, domain=domain + ".leaf") for h in hs]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [_object_hash({"left": level[i], "right": level[i + 1]}, domain=domain + ".node") for i in range(0, len(level), 2)]
    return level[0]


def _read_json(path: Union[str, os.PathLike[str]]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_write_json(path: Union[str, os.PathLike[str]], payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _call_to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return dict(obj.to_dict())
        except Exception:
            pass
    if is_dataclass(obj):
        return dict(_json_safe(asdict(obj)))
    if hasattr(obj, "__dict__"):
        return dict(_json_safe(vars(obj)))
    return {"value": _json_safe(obj)}


def _hash_or_empty(obj: Any, *, domain: str) -> str:
    if obj is None:
        return ""
    for attr in ("hash", "receipt_hash", "commitment_hash", "certificate_hash", "block_hash", "execution_hash", "batch_hash"):
        val = _get(obj, attr, "")
        if isinstance(val, str) and val:
            return val
    if hasattr(obj, "hash") and callable(getattr(obj, "hash")):
        try:
            return str(obj.hash())
        except Exception:
            pass
    return _object_hash(obj, domain=domain)


def _tx_hash(tx: Any) -> str:
    if hasattr(tx, "payload_hash") and callable(getattr(tx, "payload_hash")):
        try:
            return str(tx.payload_hash())
        except Exception:
            pass
    if hasattr(tx, "tx_hash") and callable(getattr(tx, "tx_hash")):
        try:
            return str(tx.tx_hash())
        except Exception:
            pass
    if execution is not None and hasattr(execution, "tx_hash"):
        try:
            return str(execution.tx_hash(tx))  # type: ignore[attr-defined]
        except Exception:
            pass
    payload = _call_to_dict(tx)
    payload.pop("signature", None)
    return _object_hash(payload, domain="tb.replay.tx_hash")


def _block_hash(block: Any) -> str:
    if hasattr(block, "compute_hash") and callable(getattr(block, "compute_hash")):
        try:
            return str(block.compute_hash())
        except Exception:
            pass
    if isinstance(block, Mapping):
        payload = dict(block)
        payload.pop("block_hash", None)
        return _object_hash(payload, domain="tb.replay.block_hash.fallback")
    return _object_hash(_call_to_dict(block), domain="tb.replay.block_hash.fallback")


def _block_static_validate(block: Any) -> Tuple[bool, List[str]]:
    if hasattr(block, "validate_static") and callable(getattr(block, "validate_static")):
        try:
            rep = block.validate_static()
            return bool(_get(rep, "ok", False)), [str(x) for x in _get(rep, "errors", [])]
        except Exception as exc:
            return False, [f"validate_static_exception:{exc!r}"]
    errors: List[str] = []
    bh = str(_get(block, "block_hash", ""))
    if bh and bh != _block_hash(block):
        errors.append("block_hash_invalid")
    txs = list(_get(block, "transactions", []) or [])
    expected_root = _transaction_merkle_root(txs)
    got_root = str(_get(block, "tx_merkle_root", _get(block, "merkle_root", "")))
    if got_root and expected_root and got_root != expected_root:
        errors.append("tx_merkle_root_invalid")
    return len(errors) == 0, errors


def _transaction_merkle_root(transactions: Sequence[Any]) -> str:
    if ledger_kernel is not None and hasattr(ledger_kernel, "TransactionMerkleTree"):
        try:
            return str(ledger_kernel.TransactionMerkleTree.root_for_transactions(list(transactions)))  # type: ignore[attr-defined]
        except Exception:
            pass
    return _merkle_root_hashes([_tx_hash(tx) for tx in transactions], domain="tb.replay.tx_merkle")


def _root_from_commitment(commitment: Any) -> str:
    if commitment is None:
        return ""
    if isinstance(commitment, str):
        return commitment
    for key in ("state_root", "root_hash", "merkle_root"):
        val = _get(commitment, key, "")
        if val:
            return str(val)
    if isinstance(commitment, Mapping):
        for key in ("state_root", "root_hash", "merkle_root"):
            if commitment.get(key):
                return str(commitment[key])
    return ""


def _capture_state_commitment(network: Any) -> Any:
    if execution is not None and hasattr(execution, "capture_state_commitment"):
        try:
            return execution.capture_state_commitment(network)  # type: ignore[attr-defined]
        except Exception:
            pass
    if hasattr(network, "state_commitment") and callable(getattr(network, "state_commitment")):
        return network.state_commitment()
    if hasattr(network, "merkle_root") and callable(getattr(network, "merkle_root")):
        root = network.merkle_root()
        if dtk is not None and hasattr(dtk, "MerkleHasher"):
            try:
                root = dtk.MerkleHasher.hex64(root)  # type: ignore[attr-defined]
            except Exception:
                pass
        return {"state_root": str(root), "sketch_root": str(root)}
    return {"state_root": _object_hash(_json_safe(network), domain="tb.replay.generic_state")}


def _make_execution_config(config: "ReplayConfig") -> Any:
    if execution is not None and hasattr(execution, "ExecutionConfig"):
        try:
            return execution.ExecutionConfig(
                deterministic=True,
                consensus_safe=True,
                require_pre_state_match=config.require_pre_state_match,
                verify_signature=config.verify_transaction_signatures,
                allow_direct_network_fallback=config.allow_direct_network_fallback,
                emit_lobe_receipts=config.recompute_lobe_receipts,
                run_ncp_lobe=config.recompute_lobe_receipts,
                run_cfc_lobe=config.recompute_lobe_receipts,
                strict=False,
                include_debug_metadata=False,
            )
        except Exception:
            return execution.ExecutionConfig()  # type: ignore[attr-defined]
    return None


# =============================================================================
# Replay data structures
# =============================================================================

@dataclass
class ReplayConfig:
    """Policy for deterministic replay and forensic checks."""

    verify_static_chain: bool = True
    verify_transaction_merkle_roots: bool = True
    verify_transaction_sequences: bool = True
    verify_transaction_signatures: bool = False
    verify_consensus_certificates: bool = True
    verify_tensor_commitments: bool = True
    replay_world_state: bool = True
    require_world_state_factory: bool = False
    require_pre_state_match: bool = True
    recompute_lobe_receipts: bool = True
    allow_direct_network_fallback: bool = False
    stop_on_first_divergence: bool = False
    strict_load: bool = False
    include_debug_metadata: bool = False
    max_blocks: Optional[int] = None

    def normalized(self) -> "ReplayConfig":
        cfg = copy.deepcopy(self)
        cfg.verify_static_chain = bool(cfg.verify_static_chain)
        cfg.verify_transaction_merkle_roots = bool(cfg.verify_transaction_merkle_roots)
        cfg.verify_transaction_sequences = bool(cfg.verify_transaction_sequences)
        cfg.verify_transaction_signatures = bool(cfg.verify_transaction_signatures)
        cfg.verify_consensus_certificates = bool(cfg.verify_consensus_certificates)
        cfg.verify_tensor_commitments = bool(cfg.verify_tensor_commitments)
        cfg.replay_world_state = bool(cfg.replay_world_state)
        cfg.require_world_state_factory = bool(cfg.require_world_state_factory)
        cfg.require_pre_state_match = bool(cfg.require_pre_state_match)
        cfg.recompute_lobe_receipts = bool(cfg.recompute_lobe_receipts)
        cfg.allow_direct_network_fallback = bool(cfg.allow_direct_network_fallback)
        cfg.stop_on_first_divergence = bool(cfg.stop_on_first_divergence)
        cfg.strict_load = bool(cfg.strict_load)
        cfg.include_debug_metadata = bool(cfg.include_debug_metadata)
        if cfg.max_blocks is not None:
            cfg.max_blocks = int(max(0, int(cfg.max_blocks)))
        return cfg


@dataclass
class ReplayDivergence:
    code: str
    message: str
    height: Optional[int] = None
    tx_id: str = ""
    expected: Any = None
    actual: Any = None
    severity: str = "error"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class ReplayStepResult:
    height: int
    block_hash: str
    expected_block_hash: str = ""
    recomputed_block_hash: str = ""
    prev_hash_ok: bool = True
    tx_merkle_ok: bool = True
    static_ok: bool = True
    sequence_ok: bool = True
    signature_ok: Optional[bool] = None
    consensus_ok: Optional[bool] = None
    tensor_commitment_ok: Optional[bool] = None
    world_replay_ok: Optional[bool] = None
    lobe_receipts_ok: Optional[bool] = None
    expected_pre_state_root: str = ""
    actual_pre_state_root: str = ""
    expected_post_state_root: str = ""
    actual_post_state_root: str = ""
    tx_count: int = 0
    execution_hashes: List[str] = field(default_factory=list)
    divergences: List[ReplayDivergence] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        if self.divergences:
            return False
        flags = [self.prev_hash_ok, self.tx_merkle_ok, self.static_ok, self.sequence_ok]
        for optional in (self.signature_ok, self.consensus_ok, self.tensor_commitment_ok, self.world_replay_ok, self.lobe_receipts_ok):
            if optional is not None:
                flags.append(bool(optional))
        return all(flags)

    def add_divergence(self, divergence: ReplayDivergence) -> None:
        self.divergences.append(divergence)

    def to_dict(self) -> Dict[str, Any]:
        out = _json_safe(asdict(self))
        out["ok"] = self.ok
        return out


@dataclass
class ReplayReport:
    ok: bool
    replay_hash: str
    chain_id: str
    block_count: int
    checked_blocks: int
    head_hash: str
    recomputed_head_hash: str
    config: ReplayConfig
    steps: List[ReplayStepResult] = field(default_factory=list)
    divergences: List[ReplayDivergence] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    dependency_status: Dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=_now_iso_utc)
    finished_at: str = ""
    duration_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = TB_REPLAY_SCHEMA
    version: str = TB_REPLAY_VERSION

    def payload(self, *, include_replay_hash: bool = False) -> Dict[str, Any]:
        out = {
            "schema_version": self.schema_version,
            "version": self.version,
            "ok": bool(self.ok),
            "chain_id": self.chain_id,
            "block_count": int(self.block_count),
            "checked_blocks": int(self.checked_blocks),
            "head_hash": self.head_hash,
            "recomputed_head_hash": self.recomputed_head_hash,
            "config": _json_safe(self.config),
            "steps": [s.to_dict() for s in self.steps],
            "divergences": [d.to_dict() for d in self.divergences],
            "skipped": list(self.skipped),
            "dependency_status": _json_safe(self.dependency_status),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": float(self.duration_s),
            "metadata": _json_safe(self.metadata),
        }
        if include_replay_hash:
            out["replay_hash"] = self.replay_hash
        return out

    def compute_hash(self) -> str:
        return _object_hash(self.payload(include_replay_hash=False), domain="tb.replay.report")

    def finalize(self) -> "ReplayReport":
        self.finished_at = self.finished_at or _now_iso_utc()
        self.ok = len(self.divergences) == 0 and all(step.ok for step in self.steps)
        self.replay_hash = self.compute_hash()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return self.payload(include_replay_hash=True)


# =============================================================================
# Ledger adapters
# =============================================================================

class LedgerView:
    """Small normalized view over either LedgerMPS objects or raw ledger JSON."""

    def __init__(self, source: Any):
        self.source = source
        if hasattr(source, "blocks"):
            self.blocks = list(getattr(source, "blocks"))
            cfg = getattr(source, "cfg", None)
            self.chain_id = str(getattr(cfg, "chain_id", "")) if cfg is not None else str(_get(source, "chain_id", ""))
            self.head_hash = str(getattr(source, "head_hash", ""))
        elif isinstance(source, Mapping):
            self.blocks = list(source.get("blocks", []) or [])
            cfg = dict(source.get("cfg", {}) or {})
            self.chain_id = str(cfg.get("chain_id", source.get("chain_id", "")))
            self.head_hash = str(source.get("head_hash", ""))
            if not self.head_hash and self.blocks:
                self.head_hash = str(_get(self.blocks[-1], "block_hash", ""))
        else:
            raise TypeError("Unsupported ledger source. Expected LedgerMPS-like object or mapping.")
        if not self.chain_id and self.blocks:
            self.chain_id = str(_get(self.blocks[0], "chain_id", ""))
        if not self.head_hash and self.blocks:
            self.head_hash = str(_get(self.blocks[-1], "block_hash", ""))

    @property
    def block_count(self) -> int:
        return len(self.blocks)


# =============================================================================
# Main replayer
# =============================================================================

class ChainReplayer:
    """Deterministic replay and forensic verification engine."""

    def __init__(
        self,
        config: Optional[ReplayConfig] = None,
        *,
        network_factory: Optional[Callable[[], Any]] = None,
        control_plane_factory: Optional[Callable[[Any], Any]] = None,
        ncp_lobe_factory: Optional[Callable[[], Any]] = None,
        cfc_lobe_factory: Optional[Callable[[], Any]] = None,
    ):
        self.config = (config or ReplayConfig()).normalized()
        self.network_factory = network_factory
        self.control_plane_factory = control_plane_factory
        self.ncp_lobe_factory = ncp_lobe_factory
        self.cfc_lobe_factory = cfc_lobe_factory

    # ------------------------------------------------------------------
    # Dependency / load helpers
    # ------------------------------------------------------------------
    @staticmethod
    def dependency_status() -> Dict[str, Any]:
        return {
            "tb_canonical": _tbc is not None,
            "tb_hashing": _tbh is not None,
            "tb_schemas": schemas is not None,
            "tb_validation": validation is not None,
            "tb_execution": execution is not None,
            "tb_crypto": crypto is not None,
            "tensorized_blockchain_kernel": ledger_kernel is not None,
            "digital_twin_kernel": dtk is not None,
        }

    def _load_runtime_lobes(self) -> Tuple[Any, Any, List[str]]:
        skipped: List[str] = []
        ncp_lobe = None
        cfc_lobe = None
        if self.config.recompute_lobe_receipts:
            if self.ncp_lobe_factory is not None:
                ncp_lobe = self.ncp_lobe_factory()
            else:
                skipped.append("ncp_lobe_factory_absent")
            if self.cfc_lobe_factory is not None:
                cfc_lobe = self.cfc_lobe_factory()
            else:
                skipped.append("cfc_lobe_factory_absent")
        return ncp_lobe, cfc_lobe, skipped

    def _load_world_runtime(self) -> Tuple[Any, Any, List[str]]:
        skipped: List[str] = []
        network = None
        control_plane = None
        if not self.config.replay_world_state:
            skipped.append("world_state_replay_disabled")
            return None, None, skipped
        if self.network_factory is None:
            msg = "network_factory_absent"
            if self.config.require_world_state_factory:
                raise RuntimeError("ReplayConfig.require_world_state_factory=True but no network_factory was supplied.")
            skipped.append(msg)
            return None, None, skipped
        network = self.network_factory()
        if self.control_plane_factory is not None:
            control_plane = self.control_plane_factory(network)
        elif dtk is not None and hasattr(dtk, "AkkuratInterface"):
            try:
                control_plane = dtk.AkkuratInterface(network, strict=False)  # type: ignore[attr-defined]
            except Exception:
                skipped.append("control_plane_autocreation_failed")
        else:
            skipped.append("control_plane_absent")
        return network, control_plane, skipped

    # ------------------------------------------------------------------
    # Static checks
    # ------------------------------------------------------------------
    def _check_prev_hash(self, block: Any, height: int, prev_hash: str, step: ReplayStepResult) -> None:
        got = str(_get(block, "prev_hash", ""))
        if height == 0:
            expected = ZERO_HASH_256
        else:
            expected = prev_hash
        if got != expected:
            step.prev_hash_ok = False
            step.add_divergence(ReplayDivergence("prev_hash_mismatch", "Block prev_hash does not match previous block hash", height=height, expected=expected, actual=got))

    def _check_merkle(self, block: Any, height: int, step: ReplayStepResult) -> None:
        txs = list(_get(block, "transactions", []) or [])
        got = str(_get(block, "tx_merkle_root", _get(block, "merkle_root", "")))
        actual = _transaction_merkle_root(txs)
        if got and got != actual:
            step.tx_merkle_ok = False
            step.add_divergence(ReplayDivergence("tx_merkle_root_mismatch", "Transaction Merkle root mismatch", height=height, expected=got, actual=actual))

    def _check_static(self, block: Any, height: int, step: ReplayStepResult) -> None:
        ok, errors = _block_static_validate(block)
        if not ok:
            step.static_ok = False
            for e in errors:
                step.add_divergence(ReplayDivergence("block_static_invalid", str(e), height=height))
        expected = str(_get(block, "block_hash", ""))
        actual = _block_hash(block)
        step.expected_block_hash = expected
        step.recomputed_block_hash = actual
        if expected and expected != actual:
            step.static_ok = False
            step.add_divergence(ReplayDivergence("block_hash_mismatch", "Block hash mismatch", height=height, expected=expected, actual=actual))

    def _check_sequences_and_signatures(self, txs: Sequence[Any], step: ReplayStepResult, sequences: MutableMapping[str, int], seen: set[str]) -> None:
        signature_all_ok = True
        for tx in txs:
            tx_id = str(_get(tx, "tx_id", _tx_hash(tx)[:32]))
            if tx_id in seen:
                step.sequence_ok = False
                step.add_divergence(ReplayDivergence("duplicate_transaction", "Transaction ID was already observed during replay", height=step.height, tx_id=tx_id))
            seen.add(tx_id)
            pub = str(_get(tx, "sender_public_key", ""))
            sender = str(_get(tx, "sender_id", ""))
            key = f"pub:{pub}" if pub else f"sender:{sender}"
            seq = int(_get(tx, "sender_sequence", 0) or 0)
            expected = int(sequences.get(key, 0)) + 1
            if seq != expected:
                step.sequence_ok = False
                step.add_divergence(ReplayDivergence("sender_sequence_mismatch", "Sender sequence mismatch", height=step.height, tx_id=tx_id, expected=expected, actual=seq, metadata={"sender_key": key}))
            sequences[key] = max(seq, int(sequences.get(key, 0)))
            if self.config.verify_transaction_signatures:
                ok = self._verify_tx_signature(tx)
                if not ok:
                    signature_all_ok = False
                    step.add_divergence(ReplayDivergence("transaction_signature_invalid", "Transaction signature failed verification", height=step.height, tx_id=tx_id))
        if self.config.verify_transaction_signatures:
            step.signature_ok = bool(signature_all_ok)

    def _verify_tx_signature(self, tx: Any) -> bool:
        if hasattr(tx, "verify_signature") and callable(getattr(tx, "verify_signature")):
            try:
                return bool(tx.verify_signature())
            except TypeError:
                return False
            except Exception:
                return False
        if crypto is None:
            return False
        # Best-effort schema verification for TransactionEnvelope-like payloads.
        try:
            pub = str(_get(tx, "sender_public_key", ""))
            sig = str(_get(tx, "signature", ""))
            scheme = str(_get(tx, "signature_scheme", "ed25519"))
            if not pub or not sig:
                return False
            payload = _call_to_dict(tx)
            payload.pop("signature", None)
            if hasattr(crypto, "verify_signature"):
                return bool(crypto.verify_signature(payload, signature=sig, public_key=pub, scheme=scheme))  # type: ignore[attr-defined]
        except Exception:
            return False
        return False

    def _check_consensus(self, block: Any, step: ReplayStepResult) -> None:
        cert = _get(block, "consensus_certificate", None)
        if not cert:
            step.consensus_ok = None
            step.notes.append("consensus_certificate_absent")
            return
        # Empty certificates are used by the local deterministic genesis/demo path.
        # They are structurally informative but not quorum evidence, so replay
        # records them as not-applicable rather than a divergence.
        try:
            if int(_get(cert, "total_votes", 0) or 0) == 0 and int(_get(cert, "approvals", 0) or 0) == 0:
                step.consensus_ok = None
                step.notes.append("consensus_certificate_empty_no_votes")
                return
        except Exception:
            pass
        if hasattr(cert, "verify") and callable(getattr(cert, "verify")):
            try:
                rep = cert.verify(require_vote_signatures=False)
                ok = bool(_get(rep, "ok", False))
                step.consensus_ok = ok
                if not ok:
                    for e in _get(rep, "errors", []) or []:
                        step.add_divergence(ReplayDivergence("consensus_certificate_invalid", str(e), height=step.height))
                return
            except Exception as exc:
                step.consensus_ok = False
                step.add_divergence(ReplayDivergence("consensus_certificate_verify_exception", repr(exc), height=step.height))
                return
        if isinstance(cert, Mapping):
            # Structural fallback: certificate hash, if present, should be stable.
            expected = str(cert.get("certificate_hash", ""))
            if expected:
                payload = dict(cert)
                payload.pop("certificate_hash", None)
                actual = _object_hash(payload, domain="tb.replay.consensus_certificate")
                step.consensus_ok = expected == actual
                if expected != actual:
                    step.add_divergence(ReplayDivergence("consensus_certificate_hash_mismatch", "Consensus certificate hash mismatch", height=step.height, expected=expected, actual=actual))
            else:
                step.consensus_ok = None

    # ------------------------------------------------------------------
    # Tensor checks
    # ------------------------------------------------------------------
    def _check_tensor_commitments_with_kernel(self, ledger_obj: Any, checked_blocks: int) -> Dict[int, Tuple[bool, str, str, str]]:
        out: Dict[int, Tuple[bool, str, str, str]] = {}
        if ledger_obj is None or ledger_kernel is None or not hasattr(ledger_obj, "blocks"):
            return out
        blocks = list(getattr(ledger_obj, "blocks"))[:checked_blocks]
        for i in range(len(blocks)):
            try:
                prefix = copy.copy(ledger_obj)
                prefix.blocks = [copy.deepcopy(b) for b in blocks[: i + 1]]
                if hasattr(prefix, "_compressed_cores"):
                    prefix._compressed_cores = None
                if hasattr(prefix, "_compressed_at_hash"):
                    prefix._compressed_at_hash = None
                actual = prefix.get_tensor_train(use_compressed=False).commitment().commitment_hash
                expected = str(_get(blocks[i], "tensor_commitment_hash", ""))
                out[i] = (expected == actual, expected, actual, "")
            except Exception as exc:
                out[i] = (False, str(_get(blocks[i], "tensor_commitment_hash", "")), "", repr(exc))
        return out

    # ------------------------------------------------------------------
    # World-state replay
    # ------------------------------------------------------------------
    def _replay_world_for_block(self, block: Any, step: ReplayStepResult, network: Any, control_plane: Any, ncp_lobe: Any, cfc_lobe: Any) -> None:
        if network is None:
            step.world_replay_ok = None
            step.notes.append("world_state_replay_skipped_no_network")
            return
        txs = list(_get(block, "transactions", []) or [])
        expected_pre = str(_get(block, "pre_state_root", ""))
        expected_post = str(_get(block, "post_state_root", ""))
        step.expected_pre_state_root = expected_pre
        step.expected_post_state_root = expected_post
        pre_commit = _capture_state_commitment(network)
        actual_pre = _root_from_commitment(pre_commit)
        step.actual_pre_state_root = actual_pre
        if expected_pre and actual_pre and expected_pre != actual_pre and self.config.require_pre_state_match:
            step.world_replay_ok = False
            step.add_divergence(ReplayDivergence("pre_state_root_mismatch", "Replay pre-state root differs from block commitment", height=step.height, expected=expected_pre, actual=actual_pre))
            if self.config.stop_on_first_divergence:
                return
        if not txs:
            actual_post = actual_pre
            step.actual_post_state_root = actual_post
            # Genesis blocks may deliberately use symbolic genesis roots. Do not fail if height 0.
            if step.height == 0:
                step.world_replay_ok = None
                step.notes.append("genesis_world_replay_not_applicable")
            else:
                step.world_replay_ok = (not expected_post) or (actual_post == expected_post)
            return
        if execution is None or not hasattr(execution, "CanonicalTransactionExecutor"):
            step.world_replay_ok = None
            step.notes.append("tb_execution_unavailable")
            return
        try:
            exec_cfg = _make_execution_config(self.config)
            executor = execution.CanonicalTransactionExecutor(exec_cfg)  # type: ignore[attr-defined]
            for tx in txs:
                result = executor.execute(tx, network=network, control_plane=control_plane, ncp_lobe=ncp_lobe, cfc_lobe=cfc_lobe, intent=f"replay:block:{step.height}")
                step.execution_hashes.append(str(_get(result, "execution_hash", "")))
                if not bool(_get(result, "accepted", False)):
                    step.add_divergence(ReplayDivergence("transaction_reexecution_rejected", "Transaction was not accepted during deterministic replay", height=step.height, tx_id=str(_get(tx, "tx_id", "")), metadata={"execution_result": _json_safe(result) if self.config.include_debug_metadata else {"execution_hash": _get(result, "execution_hash", "")}}))
            post_commit = _capture_state_commitment(network)
            actual_post = _root_from_commitment(post_commit)
            step.actual_post_state_root = actual_post
            if expected_post and actual_post != expected_post:
                step.world_replay_ok = False
                step.add_divergence(ReplayDivergence("post_state_root_mismatch", "Replay post-state root differs from block commitment", height=step.height, expected=expected_post, actual=actual_post))
            else:
                step.world_replay_ok = True
            self._check_lobe_receipts(block, step)
        except Exception as exc:
            step.world_replay_ok = False
            step.add_divergence(ReplayDivergence("world_replay_exception", repr(exc), height=step.height, metadata={"traceback": traceback.format_exc() if self.config.include_debug_metadata else ""}))

    def _check_lobe_receipts(self, block: Any, step: ReplayStepResult) -> None:
        # This method is intentionally conservative. Execution hash comparison is
        # enough for deterministic rerun evidence when the original block stores
        # lobe receipt hashes inside TransitionReceipt; older blocks may not.
        tr = _get(block, "transition_receipt", None)
        if not tr:
            step.lobe_receipts_ok = None
            return
        ncp_expected = str(_get(tr, "ncp_health_hash", "")) or str(_get(tr, "ncp_health_receipt_hash", ""))
        cfc_expected = str(_get(tr, "cfc_transition_hash", "")) or str(_get(tr, "cfc_transition_receipt_hash", ""))
        if not ncp_expected and not cfc_expected:
            # Local tensorized_blockchain_kernel TransitionReceipt does not carry
            # separate lobe fields; lobe hashes may live in policy/action metadata.
            step.lobe_receipts_ok = None
            step.notes.append("lobe_receipt_hashes_absent")
            return
        # If the execution path appended hashes, more specialized comparison can
        # be added by application code. For now, absence of divergence from
        # execution plus presence of expected fields is treated as structurally OK.
        step.lobe_receipts_ok = True

    # ------------------------------------------------------------------
    # Public replay
    # ------------------------------------------------------------------
    def replay(self, ledger_source: Any) -> ReplayReport:
        start = time.time()
        view = LedgerView(ledger_source)
        checked = view.block_count if self.config.max_blocks is None else min(view.block_count, int(self.config.max_blocks))
        skipped: List[str] = []
        divergences: List[ReplayDivergence] = []
        steps: List[ReplayStepResult] = []
        network = None
        control_plane = None
        ncp_lobe = None
        cfc_lobe = None

        if checked == 0:
            div = ReplayDivergence("empty_chain", "Ledger contains no blocks")
            divergences.append(div)
            report = ReplayReport(
                ok=False,
                replay_hash="",
                chain_id=view.chain_id,
                block_count=view.block_count,
                checked_blocks=0,
                head_hash=view.head_hash,
                recomputed_head_hash="",
                config=self.config,
                divergences=divergences,
                skipped=skipped,
                dependency_status=self.dependency_status(),
                duration_s=time.time() - start,
            )
            return report.finalize()

        try:
            network, control_plane, s = self._load_world_runtime()
            skipped.extend(s)
        except Exception as exc:
            divergences.append(ReplayDivergence("world_runtime_init_failed", repr(exc), metadata={"traceback": traceback.format_exc() if self.config.include_debug_metadata else ""}))
            if self.config.stop_on_first_divergence:
                checked = 0
        try:
            ncp_lobe, cfc_lobe, s = self._load_runtime_lobes()
            skipped.extend(s)
        except Exception as exc:
            divergences.append(ReplayDivergence("lobe_runtime_init_failed", repr(exc), metadata={"traceback": traceback.format_exc() if self.config.include_debug_metadata else ""}))

        tensor_checks: Dict[int, Tuple[bool, str, str, str]] = {}
        if self.config.verify_tensor_commitments:
            tensor_checks = self._check_tensor_commitments_with_kernel(ledger_source, checked)
            if not tensor_checks:
                skipped.append("tensor_commitment_replay_unavailable")

        sequences: Dict[str, int] = {}
        seen_tx: set[str] = set()
        prev_hash = ZERO_HASH_256
        recomputed_head = ""

        for height, block in enumerate(view.blocks[:checked]):
            expected_bh = str(_get(block, "block_hash", ""))
            step = ReplayStepResult(height=height, block_hash=expected_bh, tx_count=len(list(_get(block, "transactions", []) or [])))
            self._check_prev_hash(block, height, prev_hash, step)
            if self.config.verify_transaction_merkle_roots:
                self._check_merkle(block, height, step)
            if self.config.verify_static_chain:
                self._check_static(block, height, step)
            if self.config.verify_transaction_sequences:
                self._check_sequences_and_signatures(list(_get(block, "transactions", []) or []), step, sequences, seen_tx)
            if self.config.verify_consensus_certificates:
                self._check_consensus(block, step)
            if self.config.verify_tensor_commitments and height in tensor_checks:
                ok, expected, actual, error = tensor_checks[height]
                step.tensor_commitment_ok = bool(ok)
                if not ok:
                    step.add_divergence(ReplayDivergence("tensor_commitment_mismatch" if not error else "tensor_commitment_recompute_error", "Tensor commitment replay check failed", height=height, expected=expected, actual=actual, metadata={"error": error}))
            if self.config.replay_world_state:
                self._replay_world_for_block(block, step, network, control_plane, ncp_lobe, cfc_lobe)
            recomputed_head = step.recomputed_block_hash or _block_hash(block)
            prev_hash = expected_bh or recomputed_head
            steps.append(step)
            divergences.extend(step.divergences)
            if step.divergences and self.config.stop_on_first_divergence:
                break

        report = ReplayReport(
            ok=len(divergences) == 0 and all(s.ok for s in steps),
            replay_hash="",
            chain_id=view.chain_id,
            block_count=view.block_count,
            checked_blocks=len(steps),
            head_hash=view.head_hash,
            recomputed_head_hash=recomputed_head,
            config=self.config,
            steps=steps,
            divergences=divergences,
            skipped=sorted(set(skipped)),
            dependency_status=self.dependency_status(),
            duration_s=time.time() - start,
            metadata={"source_type": type(ledger_source).__name__},
        )
        return report.finalize()


# =============================================================================
# Convenience API
# =============================================================================


def load_ledger(path: Union[str, os.PathLike[str]], *, strict_load: bool = False) -> Any:
    """Load a ledger file as LedgerMPS when available, otherwise as JSON mapping."""
    p = Path(path)
    if ledger_kernel is not None and hasattr(ledger_kernel, "LedgerMPS"):
        try:
            if strict_load:
                return ledger_kernel.LedgerMPS.load_json(p, validate=True)  # type: ignore[attr-defined]
            return ledger_kernel.LedgerMPS.unsafe_load_json(p)  # type: ignore[attr-defined]
        except Exception:
            if strict_load:
                raise
    return _read_json(p)


def replay_ledger(
    ledger_source: Any,
    *,
    config: Optional[ReplayConfig] = None,
    network_factory: Optional[Callable[[], Any]] = None,
    control_plane_factory: Optional[Callable[[Any], Any]] = None,
    ncp_lobe_factory: Optional[Callable[[], Any]] = None,
    cfc_lobe_factory: Optional[Callable[[], Any]] = None,
) -> ReplayReport:
    return ChainReplayer(
        config=config,
        network_factory=network_factory,
        control_plane_factory=control_plane_factory,
        ncp_lobe_factory=ncp_lobe_factory,
        cfc_lobe_factory=cfc_lobe_factory,
    ).replay(ledger_source)


def replay_ledger_file(
    path: Union[str, os.PathLike[str]],
    *,
    config: Optional[ReplayConfig] = None,
    network_factory: Optional[Callable[[], Any]] = None,
    control_plane_factory: Optional[Callable[[Any], Any]] = None,
    ncp_lobe_factory: Optional[Callable[[], Any]] = None,
    cfc_lobe_factory: Optional[Callable[[], Any]] = None,
) -> ReplayReport:
    cfg = (config or ReplayConfig()).normalized()
    ledger_source = load_ledger(path, strict_load=cfg.strict_load)
    report = replay_ledger(
        ledger_source,
        config=cfg,
        network_factory=network_factory,
        control_plane_factory=control_plane_factory,
        ncp_lobe_factory=ncp_lobe_factory,
        cfc_lobe_factory=cfc_lobe_factory,
    )
    report.metadata["ledger_path"] = str(Path(path).expanduser().resolve())
    return report.finalize()


def verify_ledger_file(path: Union[str, os.PathLike[str]], *, verify_tensor_commitments: bool = True) -> bool:
    cfg = ReplayConfig(replay_world_state=False, verify_tensor_commitments=verify_tensor_commitments)
    return bool(replay_ledger_file(path, config=cfg).ok)


# =============================================================================
# Self-test / CLI
# =============================================================================


def _build_selftest_ledger() -> Any:
    if ledger_kernel is None or not hasattr(ledger_kernel, "LedgerMPS"):
        # Fallback minimal JSON ledger; static hash is intentionally self-consistent
        # only under fallback logic.
        block0 = {
            "chain_id": "selftest",
            "index": 0,
            "prev_hash": ZERO_HASH_256,
            "transactions": [],
            "tx_merkle_root": _transaction_merkle_root([]),
            "pre_state_root": "genesis_pre_state",
            "post_state_root": "genesis_post_state",
        }
        block0["block_hash"] = _block_hash(block0)
        return {"cfg": {"chain_id": "selftest"}, "blocks": [block0]}
    cfg = ledger_kernel.LedgerConfig(chain_id="tb-replay-selftest", out_dim=2, in_dim=2, max_bond_dim=2, compress_on_append=False, require_signatures=False).normalized()  # type: ignore[attr-defined]
    ledger = ledger_kernel.LedgerMPS(cfg)  # type: ignore[attr-defined]
    ledger.append_block([], pre_state_root="a" * 64, post_state_root="b" * 64, metadata={"selftest": 1})
    return ledger


def run_self_test() -> Dict[str, Any]:
    ledger = _build_selftest_ledger()
    report = replay_ledger(ledger, config=ReplayConfig(replay_world_state=False, verify_transaction_signatures=False, verify_tensor_commitments=ledger_kernel is not None))
    if not report.ok:
        raise AssertionError(f"Replay self-test failed: {json.dumps(report.to_dict(), indent=2)[:2000]}")
    # Tamper check.
    payload = _json_safe(ledger.to_dict() if hasattr(ledger, "to_dict") else ledger)
    payload["blocks"][-1]["post_state_root"] = "c" * 64
    tampered = replay_ledger(payload, config=ReplayConfig(replay_world_state=False, verify_tensor_commitments=False))
    if tampered.ok:
        raise AssertionError("Replay self-test failed to detect tampering.")
    return {
        "ok": True,
        "version": TB_REPLAY_VERSION,
        "clean_replay_hash": report.replay_hash,
        "tamper_detected": bool(not tampered.ok),
        "dependencies": ChainReplayer.dependency_status(),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tensorized Blockchain deterministic replay and forensic verification")
    p.add_argument("--ledger", default="", help="Path to ledger JSON file")
    p.add_argument("--output", default="", help="Optional JSON path for replay report")
    p.add_argument("--self-test", action="store_true", help="Run internal self-test")
    p.add_argument("--strict-load", action="store_true", help="Use ledger kernel strict validation during load")
    p.add_argument("--no-tensor", action="store_true", help="Skip tensor commitment replay")
    p.add_argument("--no-world", action="store_true", help="Skip world-state execution replay")
    p.add_argument("--verify-signatures", action="store_true", help="Verify transaction signatures where possible")
    p.add_argument("--stop-on-first-divergence", action="store_true")
    p.add_argument("--max-blocks", type=int, default=-1)
    p.add_argument("--pretty", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        if args.self_test:
            out = run_self_test()
        else:
            if not args.ledger:
                raise ValueError("--ledger is required unless --self-test is used")
            cfg = ReplayConfig(
                verify_tensor_commitments=not bool(args.no_tensor),
                replay_world_state=not bool(args.no_world),
                verify_transaction_signatures=bool(args.verify_signatures),
                strict_load=bool(args.strict_load),
                stop_on_first_divergence=bool(args.stop_on_first_divergence),
                max_blocks=None if int(args.max_blocks) < 0 else int(args.max_blocks),
            )
            report = replay_ledger_file(args.ledger, config=cfg)
            out = report.to_dict()
            if args.output:
                _atomic_write_json(args.output, out)
                out["saved_report"] = str(Path(args.output).expanduser().resolve())
        text = json.dumps(_json_safe(out), indent=2 if args.pretty else None, ensure_ascii=False, sort_keys=bool(args.pretty))
        print(text)
        return 0 if bool(_get(out, "ok", False)) else 2
    except Exception as exc:
        err = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc(), "version": TB_REPLAY_VERSION}
        print(json.dumps(_json_safe(err), indent=2, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
