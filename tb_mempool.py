#!/usr/bin/env python3
# tb_mempool.py
r"""
Project Chimera / Akkurat - Tensorized Blockchain Mempool
=========================================================

Purpose
-------
Production transaction queue, replay-protection, and deterministic transaction
selection boundary for the Tensorized Blockchain stack.

Responsibilities
----------------
- Accept schema-versioned signed transactions.
- Verify transaction shape and optional public-key signatures.
- Track sender sequence numbers and reject replayed transactions.
- Reject duplicate tx_ids and tx hashes.
- Hold future sender sequences without allowing them into deterministic batches
  until prior sequences are available/committed.
- Prioritize candidate transactions deterministically.
- Produce deterministic transaction batches for block proposal.
- Export/import JSON-safe snapshots for process restart and tests.

Design boundaries
-----------------
The mempool does not execute transactions, mutate digital-twin state, or create
blocks. It is a pre-ledger staging area. The ledger/proposer consumes batches;
the execution/governance layer produces receipts; consensus validates blocks.

Import policy
-------------
This module is dependency-light and will use the generated shared modules when
present:
    tb_schemas.py, tb_validation.py, tb_crypto.py, tb_canonical.py, tb_hashing.py
It also contains safe fallbacks for canonical JSON and SHA-256 so that it can be
imported early during stack bootstrapping.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import heapq
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union


# =============================================================================
# Optional stack imports
# =============================================================================

try:
    import tb_schemas as schemas  # type: ignore
except Exception:  # pragma: no cover
    schemas = None  # type: ignore

try:
    import tb_validation as validation  # type: ignore
except Exception:  # pragma: no cover
    validation = None  # type: ignore

try:
    import tb_crypto as crypto  # type: ignore
except Exception:  # pragma: no cover
    crypto = None  # type: ignore

try:
    import tb_canonical as canonical  # type: ignore
except Exception:  # pragma: no cover
    canonical = None  # type: ignore

try:
    import tb_hashing as hashing  # type: ignore
except Exception:  # pragma: no cover
    hashing = None  # type: ignore


# =============================================================================
# Constants
# =============================================================================

TB_MEMPOOL_VERSION = "tb-mempool-v1.0-production"
ZERO_HASH_256 = "0" * 64
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


# =============================================================================
# Canonical fallback helpers
# =============================================================================

def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_safe(obj: Any) -> Any:
    if canonical is not None:
        for name in ("canonical_json_safe", "json_safe"):
            fn = getattr(canonical, name, None)
            if callable(fn):
                try:
                    return fn(obj)
                except Exception:
                    pass
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            try:
                return obj.to_dict()
            except Exception:
                pass
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return _json_safe(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return _json_safe(vars(obj))
        except Exception:
            pass
    return str(obj)


def _canonical_dumps(obj: Any) -> str:
    if canonical is not None:
        for name in ("canonical_dumps", "canonical_json_dumps"):
            fn = getattr(canonical, name, None)
            if callable(fn):
                try:
                    return str(fn(obj))
                except Exception:
                    pass
    return json.dumps(_json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_bytes(obj: Any) -> bytes:
    if canonical is not None:
        fn = getattr(canonical, "canonical_bytes", None)
        if callable(fn):
            try:
                return bytes(fn(obj))
            except Exception:
                pass
    return _canonical_dumps(obj).encode("utf-8", errors="strict")


def _sha256_hex(obj: Any) -> str:
    if isinstance(obj, bytes):
        b = obj
    else:
        b = _canonical_bytes(obj)
    return hashlib.sha256(b).hexdigest()


def _domain_hash(domain: str, payload: Any) -> str:
    if hashing is not None:
        for name in ("domain_hash", "domain_hash_hex", "hash_payload"):
            fn = getattr(hashing, name, None)
            if callable(fn):
                try:
                    return str(fn(domain, payload)) if name.startswith("domain") else str(fn(payload, domain=domain))
                except Exception:
                    pass
    return hashlib.sha256((str(domain) + "\x00").encode("utf-8") + _canonical_bytes(payload)).hexdigest()


def _is_hex(s: Any, *, length: Optional[int] = None) -> bool:
    t = str(s or "").strip().lower().removeprefix("0x")
    if not t or not _HEX_RE.match(t):
        return False
    return length is None or len(t) == int(length)


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


def _tx_to_mapping(tx: Any) -> Dict[str, Any]:
    if hasattr(tx, "to_dict") and callable(getattr(tx, "to_dict")):
        return dict(tx.to_dict())
    if is_dataclass(tx):
        return dict(_json_safe(asdict(tx)))
    if isinstance(tx, Mapping):
        return dict(tx)
    if hasattr(tx, "__dict__"):
        return dict(vars(tx))
    raise TypeError("transaction must be a mapping, dataclass, or object with to_dict()")


def _tx_hash(tx: Any) -> str:
    if hasattr(tx, "tx_hash") and callable(getattr(tx, "tx_hash")):
        try:
            h = str(tx.tx_hash())
            if _is_hex(h, length=64):
                return h.lower()
        except Exception:
            pass
    if hasattr(tx, "payload_hash") and callable(getattr(tx, "payload_hash")):
        try:
            h = str(tx.payload_hash())
            if _is_hex(h, length=64):
                return h.lower()
        except Exception:
            pass
    data = _tx_to_mapping(tx)
    signing = dict(data)
    signing.pop("signature", None)
    signing.pop("schema_name", None)
    return _domain_hash("TB:TX_HASH:v1", signing)


# =============================================================================
# Errors and statuses
# =============================================================================

class MempoolError(Exception):
    """Base mempool error."""


class MempoolValidationError(MempoolError):
    """Raised by strict helpers when a transaction cannot be accepted."""


class MempoolSequenceError(MempoolError):
    """Raised for sender sequence violations."""


class MempoolStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"
    PENDING = "pending"
    SELECTED = "selected"
    COMMITTED = "committed"
    DROPPED = "dropped"


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class MempoolConfig:
    chain_id: str
    require_signature: bool = True
    verify_signature: bool = True
    accept_future_sequences: bool = True
    strict_contiguous_acceptance: bool = False
    max_pending: int = 100_000
    max_per_sender: int = 1_000
    max_action_bytes: int = 256_000
    max_tx_bytes: int = 512_000
    max_batch_count: int = 1_000
    max_batch_bytes: int = 4_000_000
    min_sender_sequence: int = 0
    default_priority: int = 0
    allow_replace_by_higher_priority: bool = False
    replacement_priority_bump: int = 1
    current_height: int = 0
    signature_context: str = "transaction"
    deterministic_ordering: bool = True
    schema_version: str = "tb.mempool_config.v1"

    def normalized(self) -> "MempoolConfig":
        cfg = copy.deepcopy(self)
        cfg.chain_id = str(cfg.chain_id)
        cfg.max_pending = max(1, int(cfg.max_pending))
        cfg.max_per_sender = max(1, int(cfg.max_per_sender))
        cfg.max_action_bytes = max(1024, int(cfg.max_action_bytes))
        cfg.max_tx_bytes = max(cfg.max_action_bytes, int(cfg.max_tx_bytes))
        cfg.max_batch_count = max(1, int(cfg.max_batch_count))
        cfg.max_batch_bytes = max(1024, int(cfg.max_batch_bytes))
        cfg.min_sender_sequence = max(0, int(cfg.min_sender_sequence))
        cfg.current_height = max(0, int(cfg.current_height))
        cfg.replacement_priority_bump = max(1, int(cfg.replacement_priority_bump))
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self.normalized()))


@dataclass
class MempoolDecision:
    accepted: bool
    status: str
    tx_id: str = ""
    tx_hash: str = ""
    sender_id: str = ""
    sender_sequence: int = -1
    reason: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso_utc)
    schema_version: str = "tb.mempool_decision.v1"

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class MempoolEntry:
    tx: Dict[str, Any]
    tx_id: str
    tx_hash: str
    sender_id: str
    sender_public_key: str
    sender_sequence: int
    priority: int = 0
    received_order: int = 0
    received_at: str = field(default_factory=_now_iso_utc)
    valid_from_height: int = 0
    valid_until_height: Optional[int] = None
    action_bytes: int = 0
    tx_bytes: int = 0
    signature_verified: bool = False
    status: str = MempoolStatus.PENDING.value
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "tb.mempool_entry.v1"

    def sort_key(self) -> Tuple[int, int, int, str, str, int]:
        # Deterministic ordering: priority descending, earlier validity, received order, hash.
        return (-int(self.priority), int(self.valid_from_height), int(self.received_order), self.tx_hash, self.sender_id, int(self.sender_sequence))

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MempoolEntry":
        p = dict(payload)
        return cls(
            tx=dict(p.get("tx", {})),
            tx_id=str(p.get("tx_id", "")),
            tx_hash=str(p.get("tx_hash", "")),
            sender_id=str(p.get("sender_id", "")),
            sender_public_key=str(p.get("sender_public_key", "")),
            sender_sequence=int(p.get("sender_sequence", -1)),
            priority=int(p.get("priority", 0)),
            received_order=int(p.get("received_order", 0)),
            received_at=str(p.get("received_at", _now_iso_utc())),
            valid_from_height=int(p.get("valid_from_height", 0)),
            valid_until_height=(None if p.get("valid_until_height", None) is None else int(p.get("valid_until_height"))),
            action_bytes=int(p.get("action_bytes", 0)),
            tx_bytes=int(p.get("tx_bytes", 0)),
            signature_verified=bool(p.get("signature_verified", False)),
            status=str(p.get("status", MempoolStatus.PENDING.value)),
            reason=str(p.get("reason", "")),
            metadata=dict(p.get("metadata", {})),
            schema_version=str(p.get("schema_version", "tb.mempool_entry.v1")),
        )


@dataclass
class MempoolBatch:
    chain_id: str
    height: int
    transactions: List[Dict[str, Any]]
    tx_hashes: List[str]
    tx_ids: List[str]
    batch_hash: str
    byte_count: int
    sender_sequences: Dict[str, List[int]]
    created_at: str = field(default_factory=_now_iso_utc)
    schema_version: str = "tb.mempool_batch.v1"

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class MempoolSnapshot:
    version: str
    config: Dict[str, Any]
    entries: List[Dict[str, Any]]
    committed_next_sequences: Dict[str, int]
    rejected_tx_hashes: List[str]
    seen_tx_hashes: List[str]
    order_counter: int
    created_at: str = field(default_factory=_now_iso_utc)
    schema_version: str = "tb.mempool_snapshot.v1"

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


# =============================================================================
# Transaction normalization and signature verification
# =============================================================================

def normalize_transaction_envelope(tx: Any) -> Any:
    """Return a tb_schemas.TransactionEnvelope when available, else mapping."""
    data = _tx_to_mapping(tx)
    if schemas is not None and hasattr(schemas, "TransactionEnvelope"):
        cls = getattr(schemas, "TransactionEnvelope")
        if isinstance(tx, cls):
            return tx
        try:
            return cls.from_dict(data)
        except Exception:
            return data
    return data


def transaction_signing_payload(tx: Any) -> Dict[str, Any]:
    obj = normalize_transaction_envelope(tx)
    if hasattr(obj, "signing_payload") and callable(getattr(obj, "signing_payload")):
        return dict(obj.signing_payload())
    data = _tx_to_mapping(obj)
    data.pop("signature", None)
    data.pop("schema_name", None)
    return data


def transaction_to_dict(tx: Any) -> Dict[str, Any]:
    obj = normalize_transaction_envelope(tx)
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return dict(obj.to_dict())
    return _tx_to_mapping(obj)


def sign_transaction(tx: Any, private_key_hex: Union[str, bytes], *, context: str = "transaction") -> Any:
    """Return a signed TransactionEnvelope/dict using tb_crypto if available.

    The signature field stores a canonical JSON SignatureEnvelope. This is more
    expressive than a raw signature and is compatible with direct verification.
    """
    if crypto is None or not hasattr(crypto, "sign_transaction_payload"):
        raise MempoolValidationError("tb_crypto.sign_transaction_payload is unavailable")
    obj = normalize_transaction_envelope(tx)
    payload = transaction_signing_payload(obj)
    public_key = str(payload.get("sender_public_key", ""))
    env = crypto.sign_transaction_payload(payload, private_key_hex, context=context, public_key_hex=public_key or None)
    env_dict = env.to_dict() if hasattr(env, "to_dict") else dict(env)
    signature_json = _canonical_dumps(env_dict)
    if hasattr(obj, "with_signature") and callable(getattr(obj, "with_signature")):
        return obj.with_signature(signature_json, scheme="ed25519")
    data = transaction_to_dict(obj)
    data["signature"] = signature_json
    data["signature_scheme"] = "ed25519"
    return data


def _signature_as_envelope(signature: str) -> Optional[Dict[str, Any]]:
    sig = str(signature or "").strip()
    if not sig:
        return None
    if sig.startswith("{"):
        try:
            obj = json.loads(sig)
            if isinstance(obj, Mapping):
                return dict(obj)
        except Exception:
            return None
    return None


def verify_transaction_signature(tx: Any, *, context: str = "transaction", strict: bool = False) -> bool:
    data = transaction_to_dict(tx)
    sig = str(data.get("signature", "") or "")
    pub = str(data.get("sender_public_key", "") or "")
    if not sig or not pub:
        if strict:
            raise MempoolValidationError("transaction signature and sender_public_key are required")
        return False
    if crypto is None:
        if strict:
            raise MempoolValidationError("tb_crypto is unavailable")
        return False
    payload = transaction_signing_payload(data)
    env = _signature_as_envelope(sig)
    try:
        if env is not None and hasattr(crypto, "verify_transaction_signature"):
            return bool(crypto.verify_transaction_signature(payload, env, context=context, strict=strict))
        # Raw hex fallback for compatibility with code that stores only signature_hex.
        raw = sig.lower().removeprefix("0x")
        if _is_hex(raw) and len(raw) == 128:
            ph = crypto.payload_hash_for_signature(payload, domain=crypto.SIG_DOMAIN_TRANSACTION)
            pre = crypto.signature_preimage(payload_hash=ph, domain=crypto.SIG_DOMAIN_TRANSACTION, context=context)
            return bool(crypto.verify_bytes_signature(pre, raw, pub, strict=strict))
    except Exception:
        if strict:
            raise
        return False
    return False


# =============================================================================
# Main mempool
# =============================================================================

class TransactionMempool:
    """Deterministic transaction queue with replay protection."""

    def __init__(self, config: Union[MempoolConfig, Mapping[str, Any], str]):
        if isinstance(config, MempoolConfig):
            self.config = config.normalized()
        elif isinstance(config, Mapping):
            self.config = MempoolConfig(**dict(config)).normalized()
        else:
            self.config = MempoolConfig(chain_id=str(config)).normalized()

        self._entries_by_hash: Dict[str, MempoolEntry] = {}
        self._txid_to_hash: Dict[str, str] = {}
        self._sender_index: Dict[str, Dict[int, str]] = {}
        self._committed_next_sequences: Dict[str, int] = {}
        self._seen_tx_hashes: set[str] = set()
        self._rejected_tx_hashes: set[str] = set()
        self._order_counter = 0

    # ------------------------------------------------------------------
    # Sequence configuration
    # ------------------------------------------------------------------
    def set_sender_next_sequence(self, sender_id: str, next_sequence: int) -> None:
        self._committed_next_sequences[str(sender_id)] = max(self.config.min_sender_sequence, int(next_sequence))

    def get_sender_next_sequence(self, sender_id: str) -> int:
        return int(self._committed_next_sequences.get(str(sender_id), self.config.min_sender_sequence))

    def sender_pending_sequences(self, sender_id: str) -> List[int]:
        idx = self._sender_index.get(str(sender_id), {})
        return sorted(int(x) for x in idx.keys())

    # ------------------------------------------------------------------
    # Acceptance
    # ------------------------------------------------------------------
    def accept_transaction(self, tx: Any, *, strict: bool = False) -> MempoolDecision:
        errors: List[str] = []
        warnings: List[str] = []
        try:
            obj = normalize_transaction_envelope(tx)
            data = transaction_to_dict(obj)
        except Exception as exc:
            decision = MempoolDecision(False, MempoolStatus.REJECTED.value, reason="normalization_failed", errors=[repr(exc)])
            if strict:
                raise MempoolValidationError(decision.reason) from exc
            return decision

        tx_id = str(data.get("tx_id", ""))
        sender_id = str(data.get("sender_id", ""))
        sender_public_key = str(data.get("sender_public_key", ""))
        sender_sequence = _as_int(data.get("sender_sequence", -1), -1)
        chain_id = str(data.get("chain_id", ""))
        txh = _tx_hash(obj)
        action = data.get("action", {})
        action_bytes = len(_canonical_bytes(action))
        tx_bytes = len(_canonical_bytes(data))

        base_kwargs = dict(tx_id=tx_id, tx_hash=txh, sender_id=sender_id, sender_sequence=sender_sequence)

        if not tx_id:
            errors.append("tx_id_empty")
        if chain_id != self.config.chain_id:
            errors.append("chain_id_mismatch")
        if sender_sequence < self.config.min_sender_sequence:
            errors.append("sender_sequence_below_minimum")
        if action_bytes > self.config.max_action_bytes:
            errors.append("action_too_large")
        if tx_bytes > self.config.max_tx_bytes:
            errors.append("transaction_too_large")
        if len(self._entries_by_hash) >= self.config.max_pending:
            errors.append("mempool_full")
        if txh in self._entries_by_hash or tx_id in self._txid_to_hash or txh in self._seen_tx_hashes:
            return MempoolDecision(False, MempoolStatus.DUPLICATE.value, reason="duplicate_transaction", errors=["duplicate_transaction"], **base_kwargs)

        per_sender = self._sender_index.get(sender_id, {})
        if len(per_sender) >= self.config.max_per_sender:
            errors.append("sender_pending_limit_exceeded")
        if sender_sequence in per_sender:
            errors.append("sender_sequence_already_pending")

        next_expected = self.get_sender_next_sequence(sender_id)
        if sender_sequence < next_expected:
            errors.append("sender_sequence_replay")
        elif sender_sequence > next_expected:
            if self.config.strict_contiguous_acceptance:
                errors.append("sender_sequence_gap")
            elif not self.config.accept_future_sequences:
                errors.append("future_sequence_rejected")
            else:
                warnings.append("future_sequence_queued")

        if validation is not None and hasattr(validation, "validate_transaction_envelope"):
            try:
                # Only shape/height here. Signature is verified below to avoid signature API drift.
                rep = validation.validate_transaction_envelope(
                    obj,
                    chain_id=self.config.chain_id,
                    verify_signature=False,
                    require_signature=self.config.require_signature,
                    current_height=self.config.current_height,
                )
                if hasattr(rep, "ok") and not bool(rep.ok):
                    for issue in getattr(rep, "issues", []):
                        code = getattr(issue, "code", None) or (issue.get("code") if isinstance(issue, Mapping) else str(issue))
                        errors.append(str(code))
                elif hasattr(rep, "to_dict"):
                    rd = rep.to_dict()
                    if not bool(rd.get("ok", True)):
                        errors.extend(str(i.get("code", "validation_error")) for i in rd.get("issues", []))
            except Exception as exc:
                errors.append(f"validation_exception:{exc!r}")

        sig_ok = False
        if self.config.require_signature or self.config.verify_signature:
            sig = str(data.get("signature", "") or "")
            if not sig:
                if self.config.require_signature:
                    errors.append("signature_missing")
            elif self.config.verify_signature:
                try:
                    sig_ok = verify_transaction_signature(obj, context=self.config.signature_context, strict=False)
                except Exception as exc:
                    errors.append(f"signature_exception:{exc!r}")
                if not sig_ok:
                    errors.append("signature_invalid")
            else:
                sig_ok = bool(sig)

        if errors:
            self._rejected_tx_hashes.add(txh)
            decision = MempoolDecision(False, MempoolStatus.REJECTED.value, reason=";".join(errors), errors=errors, warnings=warnings, metadata={"tx_bytes": tx_bytes, "action_bytes": action_bytes}, **base_kwargs)
            if strict:
                raise MempoolValidationError(decision.reason)
            return decision

        self._order_counter += 1
        priority = self._extract_priority(data)
        entry = MempoolEntry(
            tx=data,
            tx_id=tx_id,
            tx_hash=txh,
            sender_id=sender_id,
            sender_public_key=sender_public_key,
            sender_sequence=sender_sequence,
            priority=priority,
            received_order=self._order_counter,
            valid_from_height=_as_int(data.get("valid_from_height", 0), 0),
            valid_until_height=(None if data.get("valid_until_height", None) is None else _as_int(data.get("valid_until_height"), 0)),
            action_bytes=action_bytes,
            tx_bytes=tx_bytes,
            signature_verified=bool(sig_ok),
            status=MempoolStatus.PENDING.value,
            metadata={"warnings": list(warnings)},
        )
        self._entries_by_hash[txh] = entry
        self._txid_to_hash[tx_id] = txh
        self._sender_index.setdefault(sender_id, {})[sender_sequence] = txh
        self._seen_tx_hashes.add(txh)
        return MempoolDecision(True, MempoolStatus.ACCEPTED.value, reason="accepted", warnings=warnings, metadata={"priority": priority, "tx_bytes": tx_bytes}, **base_kwargs)

    def _extract_priority(self, data: Mapping[str, Any]) -> int:
        md = data.get("metadata", {}) if isinstance(data.get("metadata", {}), Mapping) else {}
        for key in ("priority", "fee", "gas_price", "urgency"):
            if key in data:
                return _as_int(data.get(key), self.config.default_priority)
            if key in md:
                return _as_int(md.get(key), self.config.default_priority)
        return int(self.config.default_priority)

    # ------------------------------------------------------------------
    # Deterministic batch selection
    # ------------------------------------------------------------------
    def produce_batch(
        self,
        *,
        max_count: Optional[int] = None,
        max_bytes: Optional[int] = None,
        height: Optional[int] = None,
        remove: bool = False,
    ) -> MempoolBatch:
        max_count = int(max_count if max_count is not None else self.config.max_batch_count)
        max_bytes = int(max_bytes if max_bytes is not None else self.config.max_batch_bytes)
        height = int(height if height is not None else self.config.current_height)
        max_count = max(1, min(max_count, self.config.max_batch_count))
        max_bytes = max(1, min(max_bytes, self.config.max_batch_bytes))

        selected: List[MempoolEntry] = []
        selected_hashes: set[str] = set()
        byte_count = 0
        expected_by_sender: Dict[str, int] = dict(self._committed_next_sequences)
        for sender in self._sender_index.keys():
            expected_by_sender.setdefault(sender, self.config.min_sender_sequence)

        # Greedy deterministic pass. Repeat so that if seq N is selected, seq N+1
        # can become eligible in the same batch.
        while len(selected) < max_count:
            candidates = [e for h, e in self._entries_by_hash.items() if h not in selected_hashes and self._is_entry_live(e, height=height)]
            candidates.sort(key=lambda e: e.sort_key())
            progress = False
            for e in candidates:
                if len(selected) >= max_count:
                    break
                expected = int(expected_by_sender.get(e.sender_id, self.config.min_sender_sequence))
                if int(e.sender_sequence) != expected:
                    continue
                if byte_count + int(e.tx_bytes) > max_bytes:
                    continue
                selected.append(e)
                selected_hashes.add(e.tx_hash)
                byte_count += int(e.tx_bytes)
                expected_by_sender[e.sender_id] = expected + 1
                progress = True
            if not progress:
                break

        txs = [copy.deepcopy(e.tx) for e in selected]
        tx_hashes = [e.tx_hash for e in selected]
        tx_ids = [e.tx_id for e in selected]
        sender_sequences: Dict[str, List[int]] = {}
        for e in selected:
            sender_sequences.setdefault(e.sender_id, []).append(int(e.sender_sequence))
        batch_hash = _domain_hash(
            "TB:MEMPOOL_BATCH:v1",
            {"chain_id": self.config.chain_id, "height": height, "tx_hashes": tx_hashes, "byte_count": byte_count},
        )
        if remove:
            for e in selected:
                self._remove_entry(e.tx_hash, status=MempoolStatus.SELECTED.value)
        return MempoolBatch(
            chain_id=self.config.chain_id,
            height=height,
            transactions=txs,
            tx_hashes=tx_hashes,
            tx_ids=tx_ids,
            batch_hash=batch_hash,
            byte_count=byte_count,
            sender_sequences=sender_sequences,
        )

    def _is_entry_live(self, e: MempoolEntry, *, height: int) -> bool:
        if e.valid_from_height > height:
            return False
        if e.valid_until_height is not None and e.valid_until_height < height:
            return False
        return e.status == MempoolStatus.PENDING.value

    # ------------------------------------------------------------------
    # Commit/drop/update operations
    # ------------------------------------------------------------------
    def mark_committed(self, txs_or_hashes: Sequence[Any]) -> int:
        count = 0
        for item in txs_or_hashes:
            txh = self._resolve_tx_hash(item)
            if not txh or txh not in self._entries_by_hash:
                continue
            e = self._entries_by_hash[txh]
            next_seq = self.get_sender_next_sequence(e.sender_id)
            if e.sender_sequence >= next_seq:
                self._committed_next_sequences[e.sender_id] = int(e.sender_sequence) + 1
            self._remove_entry(txh, status=MempoolStatus.COMMITTED.value)
            count += 1
        return count

    def drop_transaction(self, tx_or_hash: Any, *, reason: str = "dropped") -> bool:
        txh = self._resolve_tx_hash(tx_or_hash)
        if not txh or txh not in self._entries_by_hash:
            return False
        self._remove_entry(txh, status=MempoolStatus.DROPPED.value)
        return True

    def expire(self, *, height: Optional[int] = None) -> int:
        height = int(height if height is not None else self.config.current_height)
        expired = [h for h, e in self._entries_by_hash.items() if e.valid_until_height is not None and int(e.valid_until_height) < height]
        for h in expired:
            self._remove_entry(h, status=MempoolStatus.DROPPED.value)
        return len(expired)

    def _resolve_tx_hash(self, item: Any) -> str:
        if isinstance(item, str):
            s = item.strip().lower().removeprefix("0x")
            if s in self._entries_by_hash:
                return s
            if s in self._txid_to_hash:
                return self._txid_to_hash[s]
            if _is_hex(s, length=64):
                return s
            return ""
        try:
            return _tx_hash(item)
        except Exception:
            return ""

    def _remove_entry(self, tx_hash: str, *, status: str) -> Optional[MempoolEntry]:
        h = str(tx_hash).lower()
        e = self._entries_by_hash.pop(h, None)
        if e is None:
            return None
        self._txid_to_hash.pop(e.tx_id, None)
        sender = self._sender_index.get(e.sender_id)
        if sender is not None:
            sender.pop(int(e.sender_sequence), None)
            if not sender:
                self._sender_index.pop(e.sender_id, None)
        e.status = str(status)
        return e

    # ------------------------------------------------------------------
    # Introspection and persistence
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._entries_by_hash)

    def pending_count(self) -> int:
        return len(self._entries_by_hash)

    def get_transaction(self, tx_id_or_hash: str) -> Optional[Dict[str, Any]]:
        txh = self._resolve_tx_hash(tx_id_or_hash)
        e = self._entries_by_hash.get(txh)
        return copy.deepcopy(e.tx) if e is not None else None

    def pending_entries(self) -> List[MempoolEntry]:
        return sorted((copy.deepcopy(e) for e in self._entries_by_hash.values()), key=lambda e: e.sort_key())

    def snapshot(self) -> MempoolSnapshot:
        return MempoolSnapshot(
            version=TB_MEMPOOL_VERSION,
            config=self.config.to_dict(),
            entries=[e.to_dict() for e in sorted(self._entries_by_hash.values(), key=lambda x: x.sort_key())],
            committed_next_sequences={str(k): int(v) for k, v in sorted(self._committed_next_sequences.items())},
            rejected_tx_hashes=sorted(self._rejected_tx_hashes),
            seen_tx_hashes=sorted(self._seen_tx_hashes),
            order_counter=int(self._order_counter),
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.snapshot().to_dict()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransactionMempool":
        p = dict(payload)
        cfg = MempoolConfig(**dict(p.get("config", {}))).normalized()
        obj = cls(cfg)
        obj._committed_next_sequences = {str(k): int(v) for k, v in dict(p.get("committed_next_sequences", {})).items()}
        obj._rejected_tx_hashes = set(str(x) for x in list(p.get("rejected_tx_hashes", [])))
        obj._seen_tx_hashes = set(str(x) for x in list(p.get("seen_tx_hashes", [])))
        obj._order_counter = int(p.get("order_counter", 0))
        for ep in list(p.get("entries", [])):
            e = MempoolEntry.from_dict(ep)
            obj._entries_by_hash[e.tx_hash] = e
            obj._txid_to_hash[e.tx_id] = e.tx_hash
            obj._sender_index.setdefault(e.sender_id, {})[int(e.sender_sequence)] = e.tx_hash
            obj._seen_tx_hashes.add(e.tx_hash)
            obj._order_counter = max(obj._order_counter, int(e.received_order))
        return obj

    def save_json(self, path: Union[str, os.PathLike[str]]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)
        with tmp.open("w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)

    @classmethod
    def load_json(cls, path: Union[str, os.PathLike[str]]) -> "TransactionMempool":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def summary(self) -> Dict[str, Any]:
        per_sender = {s: sorted(seq.keys()) for s, seq in sorted(self._sender_index.items())}
        return {
            "version": TB_MEMPOOL_VERSION,
            "chain_id": self.config.chain_id,
            "pending_count": len(self._entries_by_hash),
            "sender_count": len(self._sender_index),
            "committed_next_sequences": dict(sorted(self._committed_next_sequences.items())),
            "pending_sequences": per_sender,
            "rejected_count": len(self._rejected_tx_hashes),
            "seen_count": len(self._seen_tx_hashes),
        }


# Backward-friendly aliases
Mempool = TransactionMempool


# =============================================================================
# Convenience functions
# =============================================================================

def create_demo_transaction(
    *,
    chain_id: str,
    sender_id: str,
    sender_public_key: str,
    sender_sequence: int,
    kind: str = "update_node_data",
    node_id: str = "5.4",
    payload: Optional[Mapping[str, Any]] = None,
    priority: int = 0,
) -> Any:
    action = {"kind": str(kind), "node_id": str(node_id), "payload": dict(payload or {"value": sender_sequence}), "note": "demo_tx"}
    metadata = {"priority": int(priority)}
    if schemas is not None and hasattr(schemas, "TransactionEnvelope"):
        return schemas.TransactionEnvelope.new(
            chain_id=chain_id,
            sender_id=sender_id,
            sender_public_key=sender_public_key,
            sender_sequence=int(sender_sequence),
            action=action,
            metadata=metadata,
        )
    tx_id = _domain_hash("TB:TX_ID:v1", {"chain_id": chain_id, "sender_id": sender_id, "seq": sender_sequence, "action": action})[:32]
    return {
        "chain_id": chain_id,
        "tx_id": tx_id,
        "sender_id": sender_id,
        "sender_public_key": sender_public_key,
        "sender_sequence": int(sender_sequence),
        "action": action,
        "nonce": _domain_hash("TB:TX_NONCE:v1", {"tx_id": tx_id})[:32],
        "timestamp": _now_iso_utc(),
        "signature_scheme": "ed25519",
        "signature": "",
        "metadata": metadata,
    }


# =============================================================================
# Self-test / CLI
# =============================================================================

def run_self_test() -> Dict[str, Any]:
    chain_id = "tb-demo-chain"
    # Use real Ed25519 signatures when crypto backend is available; otherwise run
    # shape/sequence tests with signature verification disabled.
    signed = False
    if crypto is not None and hasattr(crypto, "generate_ed25519_keypair"):
        try:
            kp = crypto.generate_ed25519_keypair(label="mempool_selftest")
            pub = kp.public_key_hex
            priv = kp.private_key_hex
            require_sig = True
            signed = True
        except Exception:
            pub = "ab" * 32
            priv = ""
            require_sig = False
    else:
        pub = "ab" * 32
        priv = ""
        require_sig = False

    mp = TransactionMempool(MempoolConfig(chain_id=chain_id, require_signature=require_sig, verify_signature=require_sig, strict_contiguous_acceptance=False))
    mp.set_sender_next_sequence("alice", 0)

    # Submit seq 1 before seq 0; it should be accepted but not batch-selected first.
    tx1 = create_demo_transaction(chain_id=chain_id, sender_id="alice", sender_public_key=pub, sender_sequence=1, priority=100)
    tx0 = create_demo_transaction(chain_id=chain_id, sender_id="alice", sender_public_key=pub, sender_sequence=0, priority=1)
    tx2 = create_demo_transaction(chain_id=chain_id, sender_id="alice", sender_public_key=pub, sender_sequence=2, priority=50)
    if signed:
        tx1 = sign_transaction(tx1, priv)
        tx0 = sign_transaction(tx0, priv)
        tx2 = sign_transaction(tx2, priv)

    d1 = mp.accept_transaction(tx1)
    d0 = mp.accept_transaction(tx0)
    ddup = mp.accept_transaction(tx0)
    d2 = mp.accept_transaction(tx2)
    if not d1.accepted or not d0.accepted or not d2.accepted:
        raise AssertionError({"d1": d1.to_dict(), "d0": d0.to_dict(), "d2": d2.to_dict()})
    if ddup.status != MempoolStatus.DUPLICATE.value:
        raise AssertionError("duplicate transaction was not rejected")

    batch = mp.produce_batch(max_count=3, remove=False)
    seqs = batch.sender_sequences.get("alice", [])
    if seqs != [0, 1, 2]:
        raise AssertionError(f"expected contiguous sequence [0,1,2], got {seqs}")
    committed = mp.mark_committed(batch.tx_hashes)
    if committed != 3 or mp.pending_count() != 0 or mp.get_sender_next_sequence("alice") != 3:
        raise AssertionError("commit bookkeeping failed")

    # Snapshot/load round trip.
    mp2 = TransactionMempool.from_dict(mp.to_dict())
    if mp2.get_sender_next_sequence("alice") != 3:
        raise AssertionError("snapshot round-trip failed")

    return {
        "ok": True,
        "version": TB_MEMPOOL_VERSION,
        "signed": bool(signed),
        "decisions": [d1.to_dict(), d0.to_dict(), ddup.to_dict(), d2.to_dict()],
        "batch": batch.to_dict(),
        "summary": mp.summary(),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tensorized Blockchain mempool utility")
    p.add_argument("--mode", choices=["self-test"], default="self-test")
    p.add_argument("--pretty", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        out = run_self_test()
        print(json.dumps(_json_safe(out), indent=2 if args.pretty else None, ensure_ascii=False))
        return 0
    except Exception as exc:
        err = {"ok": False, "error": repr(exc), "version": TB_MEMPOOL_VERSION}
        print(json.dumps(err, indent=2, ensure_ascii=False))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
