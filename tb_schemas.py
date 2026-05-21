#!/usr/bin/env python3
# tb_schemas.py
r"""
Project Chimera / Akkurat - Tensorized Blockchain Shared Schemas
=================================================================

Canonical schema-versioned dataclasses shared by the Tensorized Blockchain
ledger, governed digital twin, tensor-network runtime, cognitive lobes, API,
and demo layers.

Design goals
------------
- Dependency-light: stdlib only, with optional use of project canonical/hash
  utilities when available.
- Canonical payloads: every schema can produce deterministic dictionaries,
  canonical bytes, stable hashes, and validation diagnostics.
- Compatibility-preserving: field names align with existing production modules
  while giving the stack one common language.
- Explicit boundaries: schema objects do not execute ledger, twin, neural, or
  cryptographic logic. They describe receipts, commitments, votes, and manifests.

Core exported schemas
---------------------
- ChainIdentity
- TransactionEnvelope
- TransactionReceipt
- PolicyReceipt
- ActionReceipt
- StateCommitment
- TensorCommitment
- CompressionReport
- ConsensusVote
- ConsensusCertificate
- LobeScaleProfile
- LobeHealthReceipt
- CfCTransitionReceipt
- TwinCreationReceipt

This module is safe to import from all other scripts in the stack.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field as dc_field, fields, is_dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

try:  # Optional. The schemas remain usable without NumPy.
    import numpy as np  # type: ignore
    _NP_OK = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _NP_OK = False


# =============================================================================
# Version constants
# =============================================================================

TB_SCHEMAS_VERSION = "tb-schemas-v1.0-production"
DEFAULT_SCHEMA_VERSION = "tensorized_blockchain_schema_v1"
DEFAULT_HASH_ALGORITHM = "sha256"
ZERO_HASH_256 = "0" * 64


# =============================================================================
# Optional shared utility integration
# =============================================================================

try:  # Prefer the canonical module generated for this stack.
    from tb_canonical import canonical_json_safe as _tb_canonical_json_safe  # type: ignore
    from tb_canonical import canonical_dumps as _tb_canonical_dumps  # type: ignore
    from tb_canonical import canonical_bytes as _tb_canonical_bytes  # type: ignore
except Exception:  # pragma: no cover
    _tb_canonical_json_safe = None
    _tb_canonical_dumps = None
    _tb_canonical_bytes = None

try:  # Prefer the domain-separated hash module when present.
    from tb_hashing import domain_hash_hex as _tb_domain_hash_hex  # type: ignore
    from tb_hashing import hash_object as _tb_hash_object  # type: ignore
    from tb_hashing import merkle_root_hex as _tb_merkle_root_hex  # type: ignore
except Exception:  # pragma: no cover
    _tb_domain_hash_hex = None
    _tb_hash_object = None
    _tb_merkle_root_hex = None


# =============================================================================
# Errors and helper types
# =============================================================================

class SchemaError(ValueError):
    """Raised for invalid schema construction, normalization, or validation."""


class ValidationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


T = TypeVar("T", bound="SchemaMixin")


# =============================================================================
# Canonical helpers
# =============================================================================

def now_iso_utc() -> str:
    """Return compact second-resolution UTC timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _is_np_scalar(obj: Any) -> bool:
    return bool(_NP_OK and isinstance(obj, getattr(np, "generic", ())))


def _is_np_array(obj: Any) -> bool:
    return bool(_NP_OK and isinstance(obj, getattr(np, "ndarray", ())))


def json_safe(obj: Any) -> Any:
    """Project-compatible JSON-safe conversion."""
    if _tb_canonical_json_safe is not None:
        try:
            return _tb_canonical_json_safe(obj)
        except Exception:
            pass

    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if _is_np_scalar(obj):
        return json_safe(obj.item())
    if _is_np_array(obj):
        arr = np.asarray(obj)  # type: ignore[union-attr]
        if np.issubdtype(arr.dtype, np.floating):  # type: ignore[union-attr]
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.tolist()
    if isinstance(obj, bytes):
        return obj.hex()
    if is_dataclass(obj):
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            try:
                return obj.to_dict()  # type: ignore[no-any-return]
            except Exception:
                pass
        return json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return json_safe(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "describe") and callable(getattr(obj, "describe")):
        try:
            return json_safe(obj.describe())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return json_safe(vars(obj))
        except Exception:
            pass
    return str(obj)


def canonical_dumps(obj: Any) -> str:
    if _tb_canonical_dumps is not None:
        try:
            return str(_tb_canonical_dumps(obj))
        except Exception:
            pass
    return json.dumps(json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_bytes(obj: Any) -> bytes:
    if _tb_canonical_bytes is not None:
        try:
            return bytes(_tb_canonical_bytes(obj))
        except Exception:
            pass
    return canonical_dumps(obj).encode("utf-8", errors="strict")


def sha256_hex(data: Union[bytes, str, Any]) -> str:
    if isinstance(data, bytes):
        b = data
    elif isinstance(data, str):
        b = data.encode("utf-8", errors="strict")
    else:
        b = canonical_bytes(data)
    return hashlib.sha256(b).hexdigest()


def domain_hash(domain: str, payload: Any, *, algorithm: str = DEFAULT_HASH_ALGORITHM) -> str:
    """Domain-separated hash with fallback independent of tb_hashing.py."""
    algorithm = str(algorithm or DEFAULT_HASH_ALGORITHM).lower().strip()
    if _tb_domain_hash_hex is not None:
        try:
            return str(_tb_domain_hash_hex(domain, payload, algorithm=algorithm))
        except TypeError:
            try:
                return str(_tb_domain_hash_hex(domain, payload))
            except Exception:
                pass
        except Exception:
            pass
    if algorithm != "sha256":
        # Fallback intentionally supports SHA-256 only. tb_hashing.py can extend.
        algorithm = "sha256"
    framed = b"TBSCHEMA-DOMAIN\x00" + str(domain).encode("utf-8") + b"\x00" + canonical_bytes(payload)
    return hashlib.sha256(framed).hexdigest()


def object_hash(payload: Any, *, domain: str = "TB:OBJECT", algorithm: str = DEFAULT_HASH_ALGORITHM) -> str:
    if _tb_hash_object is not None:
        try:
            return str(_tb_hash_object(payload, domain=domain, algorithm=algorithm))
        except TypeError:
            try:
                return str(_tb_hash_object(payload))
            except Exception:
                pass
        except Exception:
            pass
    return domain_hash(domain, payload, algorithm=algorithm)


def merkle_root_hex(leaves: Sequence[Any], *, domain: str = "TB:MERKLE", algorithm: str = DEFAULT_HASH_ALGORITHM) -> str:
    """Canonical Merkle root over leaves, with domain-separated fallback."""
    arr = [json_safe(x) for x in leaves]
    if _tb_merkle_root_hex is not None:
        try:
            return str(_tb_merkle_root_hex(arr, domain=domain, algorithm=algorithm))
        except TypeError:
            try:
                return str(_tb_merkle_root_hex(arr))
            except Exception:
                pass
        except Exception:
            pass
    if not arr:
        return domain_hash(f"{domain}:EMPTY", [])
    level = [bytes.fromhex(domain_hash(f"{domain}:LEAF", {"i": i, "leaf": leaf})) for i, leaf in enumerate(arr)]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        nxt: List[bytes] = []
        for j in range(0, len(level), 2):
            nxt.append(bytes.fromhex(domain_hash(f"{domain}:NODE", {"left": level[j].hex(), "right": level[j + 1].hex()})))
        level = nxt
    return level[0].hex()


def _copy_dict(x: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(x or {})


def _str_list(xs: Optional[Sequence[Any]]) -> List[str]:
    return [str(x) for x in list(xs or [])]


def _dict_list(xs: Optional[Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    return [dict(x) for x in list(xs or [])]


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _validate_hex(value: str, *, length: Optional[int] = 64, allow_empty: bool = False) -> bool:
    s = str(value or "")
    if not s:
        return bool(allow_empty)
    if length is not None and len(s) != int(length):
        return False
    try:
        int(s, 16)
        return True
    except Exception:
        return False


def _nonempty(s: Any) -> bool:
    return bool(str(s or "").strip())


# =============================================================================
# Base dataclass mixin
# =============================================================================

@dataclass
class ValidationIssue:
    code: str
    message: str
    severity: str = ValidationSeverity.ERROR.value
    field: str = ""
    metadata: Dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


class SchemaMixin:
    """Mixin for deterministic schema dataclasses."""

    SCHEMA_NAME: ClassVar[str] = "SchemaMixin"
    HASH_DOMAIN: ClassVar[str] = "TB:SCHEMA"

    schema_version: str

    def to_dict(self, *, include_none: bool = True, include_empty: bool = True) -> Dict[str, Any]:
        if not is_dataclass(self):
            raise SchemaError("SchemaMixin requires dataclass subclasses.")
        out: Dict[str, Any] = {}
        for f in fields(self):
            name = f.name
            value = getattr(self, name)
            if value is None and not include_none:
                continue
            if not include_empty:
                if value == {} or value == [] or value == "":
                    continue
            out[name] = json_safe(value)
        out.setdefault("schema_name", self.SCHEMA_NAME)
        return out

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def canonical_json(self) -> str:
        return canonical_dumps(self.to_dict())

    def hash(self, *, domain: Optional[str] = None, algorithm: str = DEFAULT_HASH_ALGORITHM) -> str:
        return domain_hash(domain or self.HASH_DOMAIN, self.to_dict(), algorithm=algorithm)

    def payload_hash(self) -> str:
        return self.hash()

    def clone(self: T, **updates: Any) -> T:
        data = self.to_dict()
        data.pop("schema_name", None)
        data.update(updates)
        return self.__class__.from_dict(data)  # type: ignore[return-value]

    def validate(self) -> List[ValidationIssue]:
        return []

    def is_valid(self) -> bool:
        return not any(i.severity == ValidationSeverity.ERROR.value for i in self.validate())

    @classmethod
    def _field_names(cls) -> set:
        return {f.name for f in fields(cls)}

    @classmethod
    def from_dict(cls: Type[T], payload: Mapping[str, Any]) -> T:
        if not isinstance(payload, Mapping):
            raise TypeError(f"{cls.__name__}.from_dict expects a mapping")
        data = {k: v for k, v in dict(payload).items() if k in cls._field_names()}
        return cls(**data)  # type: ignore[arg-type, return-value]


# =============================================================================
# Chain and identity schemas
# =============================================================================

@dataclass
class ChainIdentity(SchemaMixin):
    SCHEMA_NAME: ClassVar[str] = "ChainIdentity"
    HASH_DOMAIN: ClassVar[str] = "TB:CHAIN_IDENTITY:v1"

    chain_id: str
    network: str = "local"
    genesis_hash: str = ZERO_HASH_256
    project: str = "Project Chimera / Akkurat Tensorized Blockchain"
    version: str = "1.0"
    created_at: str = dc_field(default_factory=now_iso_utc)
    metadata: Dict[str, Any] = dc_field(default_factory=dict)
    schema_version: str = "tb.chain_identity.v1"

    @staticmethod
    def new(*, network: str = "local", project: str = "Project Chimera / Akkurat Tensorized Blockchain", metadata: Optional[Mapping[str, Any]] = None) -> "ChainIdentity":
        seed = {"network": network, "project": project, "metadata": dict(metadata or {}), "ts": now_iso_utc(), "nonce": uuid.uuid4().hex}
        cid = "tb-" + domain_hash("TB:CHAIN_ID:GENERATE", seed)[:24]
        return ChainIdentity(chain_id=cid, network=network, project=project, metadata=dict(metadata or {}))

    def validate(self) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if not _nonempty(self.chain_id):
            issues.append(ValidationIssue("chain_id_empty", "chain_id must be non-empty", field="chain_id"))
        if self.genesis_hash and not _validate_hex(self.genesis_hash, length=64):
            issues.append(ValidationIssue("genesis_hash_invalid", "genesis_hash must be 64 hex chars", field="genesis_hash"))
        return issues


# =============================================================================
# State/tensor/compression commitments
# =============================================================================

@dataclass
class StateCommitment(SchemaMixin):
    SCHEMA_NAME: ClassVar[str] = "StateCommitment"
    HASH_DOMAIN: ClassVar[str] = "TB:STATE_COMMITMENT:v1"

    state_root: str
    sketch_root: str = ""
    metadata_root: str = ""
    topology_root: str = ""
    history_root: str = ""
    projector_config_hash: str = ""
    node_count: int = 0
    root_id: str = "0"
    tick: int = 0
    algorithm: str = DEFAULT_HASH_ALGORITHM
    commitment_policy: str = "state+topology+committed_metadata"
    created_at: str = ""
    metadata: Dict[str, Any] = dc_field(default_factory=dict)
    schema_version: str = "tb.state_commitment.v1"

    @staticmethod
    def from_roots(
        *,
        state_root: str,
        sketch_root: str = "",
        metadata_root: str = "",
        topology_root: str = "",
        history_root: str = "",
        projector_config_hash: str = "",
        node_count: int = 0,
        root_id: str = "0",
        tick: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "StateCommitment":
        return StateCommitment(
            state_root=str(state_root),
            sketch_root=str(sketch_root),
            metadata_root=str(metadata_root),
            topology_root=str(topology_root),
            history_root=str(history_root),
            projector_config_hash=str(projector_config_hash),
            node_count=int(node_count),
            root_id=str(root_id),
            tick=int(tick),
            metadata=dict(metadata or {}),
        )

    @property
    def commitment_hash(self) -> str:
        return self.hash()

    def validate(self) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if not _validate_hex(self.state_root, length=64):
            issues.append(ValidationIssue("state_root_invalid", "state_root must be 64 hex chars", field="state_root"))
        for name in ("sketch_root", "metadata_root", "topology_root", "history_root", "projector_config_hash"):
            v = getattr(self, name)
            if v and not _validate_hex(str(v), length=64):
                issues.append(ValidationIssue(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name))
        if int(self.node_count) < 0:
            issues.append(ValidationIssue("node_count_negative", "node_count cannot be negative", field="node_count"))
        return issues


@dataclass
class TensorCommitment(SchemaMixin):
    SCHEMA_NAME: ClassVar[str] = "TensorCommitment"
    HASH_DOMAIN: ClassVar[str] = "TB:TENSOR_COMMITMENT:v1"

    scheme: str = "tensor_train_mpo"
    version: str = "1"
    output_dims: List[int] = dc_field(default_factory=list)
    input_dims: List[int] = dc_field(default_factory=list)
    bond_ranks: List[int] = dc_field(default_factory=list)
    dtype: str = "float32"
    device_independent: bool = True
    core_hashes: List[str] = dc_field(default_factory=list)
    metadata_hash: str = ""
    commitment_hash: str = ""
    algorithm: str = DEFAULT_HASH_ALGORITHM
    mode_layout: List[int] = dc_field(default_factory=list)
    compression_parent_hash: str = ""
    metadata: Dict[str, Any] = dc_field(default_factory=dict)
    schema_version: str = "tb.tensor_commitment.v1"

    @classmethod
    def from_tensor_train(cls, tt: Any, *, metadata_hash: str = "", commitment_hash: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None) -> "TensorCommitment":
        if hasattr(tt, "commitment") and callable(getattr(tt, "commitment")):
            c = tt.commitment()
            if hasattr(c, "to_dict"):
                return cls.from_dict(c.to_dict())
            if isinstance(c, Mapping):
                return cls.from_dict(c)
        cores = list(getattr(tt, "cores_data", []) or [])
        core_hashes = []
        for i, core in enumerate(cores):
            if _NP_OK:
                arr = np.asarray(core)  # type: ignore[union-attr]
                core_hashes.append(domain_hash("TB:TENSOR_CORE:v1", {"i": i, "shape": list(arr.shape), "dtype": str(arr.dtype), "bytes_sha256": hashlib.sha256(arr.tobytes()).hexdigest()}))
            else:
                core_hashes.append(domain_hash("TB:TENSOR_CORE:v1", {"i": i, "repr": repr(core)}))
        obj = cls(
            output_dims=[int(x) for x in list(getattr(tt, "output_dims", []) or [])],
            input_dims=[int(x) for x in list(getattr(tt, "input_dims", []) or [])],
            bond_ranks=[int(x) for x in (tt.bond_ranks() if hasattr(tt, "bond_ranks") else list(getattr(tt, "bond_dims", []) or []))],
            dtype=str(getattr(tt, "dtype", "float32")),
            core_hashes=core_hashes,
            metadata_hash=metadata_hash or domain_hash("TB:TENSOR_METADATA:v1", dict(getattr(tt, "metadata", {}) or {})),
            metadata=dict(metadata or {}),
        )
        obj.commitment_hash = commitment_hash or obj.hash_without_self_reference()
        return obj

    def hash_without_self_reference(self) -> str:
        payload = self.to_dict()
        payload["commitment_hash"] = ""
        return domain_hash(self.HASH_DOMAIN, payload, algorithm=self.algorithm)

    def normalized(self) -> "TensorCommitment":
        obj = copy.deepcopy(self)
        obj.output_dims = [int(x) for x in obj.output_dims]
        obj.input_dims = [int(x) for x in obj.input_dims]
        obj.bond_ranks = [int(x) for x in obj.bond_ranks]
        obj.core_hashes = [str(x) for x in obj.core_hashes]
        if not obj.commitment_hash:
            obj.commitment_hash = obj.hash_without_self_reference()
        return obj

    def verify_self_hash(self) -> bool:
        if not self.commitment_hash:
            return False
        return self.commitment_hash == self.hash_without_self_reference()

    def validate(self) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if not self.output_dims or not self.input_dims:
            issues.append(ValidationIssue("tensor_dims_empty", "input_dims and output_dims must be non-empty"))
        if len(self.output_dims) != len(self.input_dims):
            issues.append(ValidationIssue("tensor_dims_rank_mismatch", "input_dims and output_dims must have same length"))
        if any(int(x) <= 0 for x in self.output_dims + self.input_dims):
            issues.append(ValidationIssue("tensor_dim_nonpositive", "all tensor dimensions must be positive"))
        for i, h in enumerate(self.core_hashes):
            if not _validate_hex(h, length=64):
                issues.append(ValidationIssue("core_hash_invalid", "core_hashes must be 64 hex chars", field=f"core_hashes[{i}]"))
        if self.metadata_hash and not _validate_hex(self.metadata_hash, length=64):
            issues.append(ValidationIssue("metadata_hash_invalid", "metadata_hash must be empty or 64 hex chars", field="metadata_hash"))
        if self.commitment_hash and not _validate_hex(self.commitment_hash, length=64):
            issues.append(ValidationIssue("commitment_hash_invalid", "commitment_hash must be empty or 64 hex chars", field="commitment_hash"))
        if self.commitment_hash and not self.verify_self_hash():
            issues.append(ValidationIssue("commitment_hash_mismatch", "commitment_hash does not match payload", field="commitment_hash"))
        return issues


@dataclass
class CompressionReport(SchemaMixin):
    SCHEMA_NAME: ClassVar[str] = "CompressionReport"
    HASH_DOMAIN: ClassVar[str] = "TB:COMPRESSION_REPORT:v1"

    algorithm: str = "tt_round"
    old_parameters: int = 0
    new_parameters: int = 0
    compression_ratio: float = 0.0
    space_saved_percent: float = 0.0
    old_bond_ranks: List[int] = dc_field(default_factory=list)
    new_bond_ranks: List[int] = dc_field(default_factory=list)
    max_rank: Optional[int] = None
    energy_tol: Optional[float] = None
    relative_error: Optional[float] = None
    fidelity: Optional[float] = None
    operator_norm_before: Optional[float] = None
    operator_norm_after: Optional[float] = None
    pre_tensor_commitment_hash: str = ""
    post_tensor_commitment_hash: str = ""
    deterministic: bool = True
    consensus_safe: bool = True
    report_hash: str = ""
    metadata: Dict[str, Any] = dc_field(default_factory=dict)
    schema_version: str = "tb.compression_report.v1"

    def hash_without_self_reference(self) -> str:
        payload = self.to_dict()
        payload["report_hash"] = ""
        return domain_hash(self.HASH_DOMAIN, payload)

    def normalized(self) -> "CompressionReport":
        obj = copy.deepcopy(self)
        obj.old_parameters = int(max(0, obj.old_parameters))
        obj.new_parameters = int(max(0, obj.new_parameters))
        if obj.old_parameters > 0:
            obj.compression_ratio = 1.0 - (float(obj.new_parameters) / float(obj.old_parameters))
            obj.space_saved_percent = 100.0 * obj.compression_ratio
        if not obj.report_hash:
            obj.report_hash = obj.hash_without_self_reference()
        return obj

    def validate(self) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if self.old_parameters < 0 or self.new_parameters < 0:
            issues.append(ValidationIssue("parameter_count_negative", "parameter counts cannot be negative"))
        for name in ("pre_tensor_commitment_hash", "post_tensor_commitment_hash", "report_hash"):
            v = getattr(self, name)
            if v and not _validate_hex(str(v), length=64):
                issues.append(ValidationIssue(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name))
        return issues


# =============================================================================
# Transaction, action, and policy receipts
# =============================================================================

@dataclass
class TransactionEnvelope(SchemaMixin):
    SCHEMA_NAME: ClassVar[str] = "TransactionEnvelope"
    HASH_DOMAIN: ClassVar[str] = "TB:TRANSACTION_ENVELOPE:v1"

    chain_id: str
    tx_id: str
    sender_id: str
    sender_public_key: str
    sender_sequence: int
    action: Dict[str, Any]
    pre_state_root: str = ""
    nonce: str = dc_field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = dc_field(default_factory=now_iso_utc)
    valid_from_height: int = 0
    valid_until_height: Optional[int] = None
    signature_scheme: str = "ed25519"
    signature: str = ""
    metadata: Dict[str, Any] = dc_field(default_factory=dict)
    schema_version: str = "tb.transaction_envelope.v1"

    @staticmethod
    def new(
        *,
        chain_id: str,
        sender_id: str,
        sender_public_key: str,
        sender_sequence: int,
        action: Mapping[str, Any],
        pre_state_root: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "TransactionEnvelope":
        seed = {"chain_id": chain_id, "sender_id": sender_id, "seq": sender_sequence, "action": dict(action), "nonce": uuid.uuid4().hex}
        tx_id = domain_hash("TB:TX_ID:v1", seed)[:32]
        return TransactionEnvelope(
            chain_id=str(chain_id),
            tx_id=tx_id,
            sender_id=str(sender_id),
            sender_public_key=str(sender_public_key),
            sender_sequence=int(sender_sequence),
            action=dict(action),
            pre_state_root=str(pre_state_root),
            metadata=dict(metadata or {}),
        )

    def signing_payload(self) -> Dict[str, Any]:
        payload = self.to_dict()
        payload.pop("signature", None)
        payload.pop("schema_name", None)
        return payload

    def tx_hash(self) -> str:
        return domain_hash("TB:TX_HASH:v1", self.signing_payload())

    def signature_hash(self) -> str:
        return domain_hash("TB:TX_SIGNATURE_HASH:v1", {"tx_hash": self.tx_hash(), "signature": self.signature, "scheme": self.signature_scheme})

    def with_signature(self, signature: str, *, scheme: Optional[str] = None) -> "TransactionEnvelope":
        return self.clone(signature=str(signature), signature_scheme=str(scheme or self.signature_scheme))

    def validate(self) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        for name in ("chain_id", "tx_id", "sender_id", "sender_public_key"):
            if not _nonempty(getattr(self, name)):
                issues.append(ValidationIssue(f"{name}_empty", f"{name} must be non-empty", field=name))
        if int(self.sender_sequence) < 0:
            issues.append(ValidationIssue("sender_sequence_negative", "sender_sequence cannot be negative", field="sender_sequence"))
        if self.pre_state_root and not _validate_hex(self.pre_state_root, length=64):
            issues.append(ValidationIssue("pre_state_root_invalid", "pre_state_root must be empty or 64 hex chars", field="pre_state_root"))
        if not isinstance(self.action, Mapping) or not self.action:
            issues.append(ValidationIssue("action_empty", "action must be a non-empty mapping", field="action"))
        if self.valid_until_height is not None and int(self.valid_until_height) < int(self.valid_from_height):
            issues.append(ValidationIssue("valid_height_range_invalid", "valid_until_height must be >= valid_from_height"))
        return issues


@dataclass
class ActionReceipt(SchemaMixin):
    SCHEMA_NAME: ClassVar[str] = "ActionReceipt"
    HASH_DOMAIN: ClassVar[str] = "TB:ACTION_RECEIPT:v1"

    action_hash: str
    kind: str
    node_id: Optional[str] = None
    approved: bool = False
    executed: bool = False
    status: str = "pending"
    pre_state_root: str = ""
    post_state_root: str = ""
    delta_norm: Optional[float] = None
    error: str = ""
    warnings: List[str] = dc_field(default_factory=list)
    timestamp: str = dc_field(default_factory=now_iso_utc)
    metadata: Dict[str, Any] = dc_field(default_factory=dict)
    schema_version: str = "tb.action_receipt.v1"

    @staticmethod
    def from_action(action: Mapping[str, Any], **kwargs: Any) -> "ActionReceipt":
        action_hash = domain_hash("TB:ACTION_HASH:v1", dict(action))
        return ActionReceipt(action_hash=action_hash, kind=str(action.get("kind", "")), node_id=action.get("node_id"), **kwargs)

    def validate(self) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if not _validate_hex(self.action_hash, length=64):
            issues.append(ValidationIssue("action_hash_invalid", "action_hash must be 64 hex chars", field="action_hash"))
        if not _nonempty(self.kind):
            issues.append(ValidationIssue("kind_empty", "kind must be non-empty", field="kind"))
        for name in ("pre_state_root", "post_state_root"):
            v = getattr(self, name)
            if v and not _validate_hex(v, length=64):
                issues.append(ValidationIssue(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name))
        return issues


@dataclass
class PolicyReceipt(SchemaMixin):
    SCHEMA_NAME: ClassVar[str] = "PolicyReceipt"
    HASH_DOMAIN: ClassVar[str] = "TB:POLICY_RECEIPT:v1"

    approved: bool
    intent: str = "direct_actions"
    policy_version: str = "policy.v1"
    pre_state_root: str = ""
    post_state_root: str = ""
    simulation_hash: str = ""
    action_hashes: List[str] = dc_field(default_factory=list)
    rejected_actions: List[Dict[str, Any]] = dc_field(default_factory=list)
    warnings: List[str] = dc_field(default_factory=list)
    reason: str = ""
    execution_stats: Dict[str, Any] = dc_field(default_factory=dict)
    action_receipt_hashes: List[str] = dc_field(default_factory=list)
    created_at: str = dc_field(default_factory=now_iso_utc)
    metadata: Dict[str, Any] = dc_field(default_factory=dict)
    schema_version: str = "tb.policy_receipt.v1"

    def receipt_hash(self) -> str:
        return self.hash()

    def validate(self) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        for name in ("pre_state_root", "post_state_root", "simulation_hash"):
            v = getattr(self, name)
            if v and not _validate_hex(v, length=64):
                issues.append(ValidationIssue(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name))
        for i, h in enumerate(self.action_hashes + self.action_receipt_hashes):
            if h and not _validate_hex(h, length=64):
                issues.append(ValidationIssue("hash_list_invalid", "action hashes must be 64 hex chars", field=f"hash[{i}]"))
        return issues


@dataclass
class TransactionReceipt(SchemaMixin):
    SCHEMA_NAME: ClassVar[str] = "TransactionReceipt"
    HASH_DOMAIN: ClassVar[str] = "TB:TRANSACTION_RECEIPT:v1"

    tx_id: str
    tx_hash: str
    accepted: bool
    status: str = "pending"
    reason: str = ""
    pre_state_root: str = ""
    post_state_root: str = ""
    policy_receipt_hash: str = ""
    action_receipt_hashes: List[str] = dc_field(default_factory=list)
    block_hash: str = ""
    block_index: Optional[int] = None
    timestamp: str = dc_field(default_factory=now_iso_utc)
    metadata: Dict[str, Any] = dc_field(default_factory=dict)
    schema_version: str = "tb.transaction_receipt.v1"

    def receipt_hash(self) -> str:
        return self.hash()

    def validate(self) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if not _nonempty(self.tx_id):
            issues.append(ValidationIssue("tx_id_empty", "tx_id must be non-empty", field="tx_id"))
        if not _validate_hex(self.tx_hash, length=64):
            issues.append(ValidationIssue("tx_hash_invalid", "tx_hash must be 64 hex chars", field="tx_hash"))
        for name in ("pre_state_root", "post_state_root", "policy_receipt_hash", "block_hash"):
            v = getattr(self, name)
            if v and not _validate_hex(v, length=64):
                issues.append(ValidationIssue(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name))
        return issues


# =============================================================================
# Consensus schemas
# =============================================================================

@dataclass
class ConsensusVote(SchemaMixin):
    SCHEMA_NAME: ClassVar[str] = "ConsensusVote"
    HASH_DOMAIN: ClassVar[str] = "TB:CONSENSUS_VOTE:v1"

    chain_id: str
    height: int
    round: int
    validator_id: str
    validator_public_key: str
    block_hash: str
    tx_merkle_root: str
    pre_state_root: str
    post_state_root: str
    tensor_commitment_hash: str = ""
    policy_receipt_root: str = ""
    approved: bool = False
    reason: str = ""
    timestamp: str = dc_field(default_factory=now_iso_utc)
    signature_scheme: str = "ed25519"
    signature: str = ""
    metadata: Dict[str, Any] = dc_field(default_factory=dict)
    schema_version: str = "tb.consensus_vote.v1"

    def signing_payload(self) -> Dict[str, Any]:
        payload = self.to_dict()
        payload.pop("signature", None)
        payload.pop("schema_name", None)
        return payload

    def vote_hash(self) -> str:
        return domain_hash("TB:CONSENSUS_VOTE_HASH:v1", self.signing_payload())

    def with_signature(self, signature: str, *, scheme: Optional[str] = None) -> "ConsensusVote":
        return self.clone(signature=str(signature), signature_scheme=str(scheme or self.signature_scheme))

    def validate(self) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        for name in ("chain_id", "validator_id", "validator_public_key"):
            if not _nonempty(getattr(self, name)):
                issues.append(ValidationIssue(f"{name}_empty", f"{name} must be non-empty", field=name))
        if self.height < 0 or self.round < 0:
            issues.append(ValidationIssue("height_or_round_negative", "height and round cannot be negative"))
        for name in ("block_hash", "tx_merkle_root", "pre_state_root", "post_state_root", "tensor_commitment_hash", "policy_receipt_root"):
            v = getattr(self, name)
            if v and not _validate_hex(v, length=64):
                issues.append(ValidationIssue(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name))
        return issues


@dataclass
class ConsensusCertificate(SchemaMixin):
    SCHEMA_NAME: ClassVar[str] = "ConsensusCertificate"
    HASH_DOMAIN: ClassVar[str] = "TB:CONSENSUS_CERTIFICATE:v1"

    chain_id: str
    height: int
    round: int
    block_hash: str
    tx_merkle_root: str
    pre_state_root: str
    post_state_root: str
    votes: List[ConsensusVote] = dc_field(default_factory=list)
    quorum_required: int = 1
    quorum_ratio: float = 0.66
    validator_set_hash: str = ""
    threshold: float = 0.0
    certificate_hash: str = ""
    created_at: str = dc_field(default_factory=now_iso_utc)
    metadata: Dict[str, Any] = dc_field(default_factory=dict)
    schema_version: str = "tb.consensus_certificate.v1"

    @classmethod
    def from_votes(
        cls,
        *,
        chain_id: str,
        height: int,
        round: int,
        block_hash: str,
        tx_merkle_root: str,
        pre_state_root: str,
        post_state_root: str,
        votes: Sequence[ConsensusVote],
        quorum_ratio: float = 0.66,
        threshold: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "ConsensusCertificate":
        approvals = [v for v in votes if bool(v.approved)]
        quorum_required = max(1, int(math.ceil(len(votes) * float(quorum_ratio)))) if votes else 1
        validator_set_hash = merkle_root_hex([v.validator_public_key for v in votes], domain="TB:VALIDATOR_SET:v1")
        cert = cls(
            chain_id=chain_id,
            height=int(height),
            round=int(round),
            block_hash=block_hash,
            tx_merkle_root=tx_merkle_root,
            pre_state_root=pre_state_root,
            post_state_root=post_state_root,
            votes=list(votes),
            quorum_required=quorum_required,
            quorum_ratio=float(quorum_ratio),
            validator_set_hash=validator_set_hash,
            threshold=float(threshold),
            metadata={"approvals": len(approvals), **dict(metadata or {})},
        )
        cert.certificate_hash = cert.hash_without_self_reference()
        return cert

    def hash_without_self_reference(self) -> str:
        payload = self.to_dict()
        payload["certificate_hash"] = ""
        return domain_hash(self.HASH_DOMAIN, payload)

    def approvals(self) -> int:
        return int(sum(1 for v in self.votes if bool(v.approved)))

    def has_quorum(self) -> bool:
        return self.approvals() >= int(max(1, self.quorum_required))

    def validate(self) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if self.height < 0 or self.round < 0:
            issues.append(ValidationIssue("height_or_round_negative", "height and round cannot be negative"))
        for name in ("block_hash", "tx_merkle_root", "pre_state_root", "post_state_root", "validator_set_hash", "certificate_hash"):
            v = getattr(self, name)
            if v and not _validate_hex(v, length=64):
                issues.append(ValidationIssue(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name))
        for i, vote in enumerate(self.votes):
            for issue in vote.validate():
                issues.append(ValidationIssue(f"vote_{i}_{issue.code}", issue.message, issue.severity, field=f"votes[{i}].{issue.field}"))
        if self.certificate_hash and self.certificate_hash != self.hash_without_self_reference():
            issues.append(ValidationIssue("certificate_hash_mismatch", "certificate_hash does not match certificate payload", field="certificate_hash"))
        return issues


# =============================================================================
# Cognitive lobe schemas
# =============================================================================

@dataclass
class LobeScaleProfile(SchemaMixin):
    SCHEMA_NAME: ClassVar[str] = "LobeScaleProfile"
    HASH_DOMAIN: ClassVar[str] = "TB:LOBE_SCALE_PROFILE:v1"

    lobe_id: str
    lobe_kind: str  # ncp | cfc | executive | custom
    active: bool = True
    parameter_budget: int = 0
    dense_equivalent_budget: int = 0
    active_parameter_count: int = 0
    hidden_size: int = 0
    num_cells: Optional[int] = None
    tt_bond_rank: int = 0
    precision: str = "float32"
    update_frequency_hz: float = 1.0
    checkpoint_policy: str = "on_scale_change"
    consensus_safe: bool = False
    device: str = "cpu"
    scale_mode: str = "operational"
    reason: str = ""
    metadata: Dict[str, Any] = dc_field(default_factory=dict)
    schema_version: str = "tb.lobe_scale_profile.v1"

    def profile_hash(self) -> str:
        return self.hash()

    def validate(self) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if not _nonempty(self.lobe_id):
            issues.append(ValidationIssue("lobe_id_empty", "lobe_id must be non-empty", field="lobe_id"))
        if int(self.parameter_budget) < 0 or int(self.dense_equivalent_budget) < 0 or int(self.active_parameter_count) < 0:
            issues.append(ValidationIssue("parameter_budget_negative", "parameter budgets cannot be negative"))
        if self.consensus_safe and str(self.device).lower() != "cpu":
            issues.append(ValidationIssue("consensus_safe_non_cpu", "consensus_safe lobe profiles must use CPU device", severity=ValidationSeverity.WARNING.value, field="device"))
        return issues


@dataclass
class LobeHealthReceipt(SchemaMixin):
    SCHEMA_NAME: ClassVar[str] = "LobeHealthReceipt"
    HASH_DOMAIN: ClassVar[str] = "TB:LOBE_HEALTH_RECEIPT:v1"

    lobe_id: str
    lobe_kind: str
    stable: bool
    degraded: bool = False
    health_score: float = 1.0
    state_hash: str = ""
    checkpoint_hash: str = ""
    scale_profile_hash: str = ""
    parameter_count: int = 0
    dense_equivalent_parameter_count: int = 0
    recurrent_mode: str = ""
    metrics: Dict[str, Any] = dc_field(default_factory=dict)
    warnings: List[str] = dc_field(default_factory=list)
    created_at: str = dc_field(default_factory=now_iso_utc)
    metadata: Dict[str, Any] = dc_field(default_factory=dict)
    schema_version: str = "tb.lobe_health_receipt.v1"

    def health_hash(self) -> str:
        return self.hash()

    def validate(self) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if not _nonempty(self.lobe_id):
            issues.append(ValidationIssue("lobe_id_empty", "lobe_id must be non-empty", field="lobe_id"))
        for name in ("state_hash", "checkpoint_hash", "scale_profile_hash"):
            v = getattr(self, name)
            if v and not _validate_hex(v, length=64):
                issues.append(ValidationIssue(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name))
        if not (0.0 <= float(self.health_score) <= 1.0):
            issues.append(ValidationIssue("health_score_out_of_range", "health_score must be between 0 and 1", field="health_score"))
        return issues


@dataclass
class CfCTransitionReceipt(SchemaMixin):
    SCHEMA_NAME: ClassVar[str] = "CfCTransitionReceipt"
    HASH_DOMAIN: ClassVar[str] = "TB:CFC_TRANSITION_RECEIPT:v1"

    lobe_id: str
    pre_state_root: str
    tx_hash: str
    predicted_post_state_hash: str
    actual_post_state_hash: str = ""
    forecast_error: Optional[float] = None
    cosine_similarity: Optional[float] = None
    time_delta: float = 1.0
    control_vector_hash: str = ""
    lobe_health_hash: str = ""
    scale_profile_hash: str = ""
    tensor_backbone_commitment_hashes: List[str] = dc_field(default_factory=list)
    accepted: Optional[bool] = None
    created_at: str = dc_field(default_factory=now_iso_utc)
    metadata: Dict[str, Any] = dc_field(default_factory=dict)
    schema_version: str = "tb.cfc_transition_receipt.v1"

    def transition_hash(self) -> str:
        return self.hash()

    def validate(self) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        for name in ("pre_state_root", "tx_hash", "predicted_post_state_hash", "actual_post_state_hash", "control_vector_hash", "lobe_health_hash", "scale_profile_hash"):
            v = getattr(self, name)
            if v and not _validate_hex(v, length=64):
                issues.append(ValidationIssue(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name))
        for i, h in enumerate(self.tensor_backbone_commitment_hashes):
            if not _validate_hex(h, length=64):
                issues.append(ValidationIssue("tensor_backbone_hash_invalid", "tensor backbone commitment hashes must be 64 hex chars", field=f"tensor_backbone_commitment_hashes[{i}]"))
        if float(self.time_delta) < 0:
            issues.append(ValidationIssue("time_delta_negative", "time_delta cannot be negative", field="time_delta"))
        return issues


# =============================================================================
# Twin materialization schemas
# =============================================================================

@dataclass
class TwinCreationReceipt(SchemaMixin):
    SCHEMA_NAME: ClassVar[str] = "TwinCreationReceipt"
    HASH_DOMAIN: ClassVar[str] = "TB:TWIN_CREATION_RECEIPT:v1"

    twin_id: str
    thing_name: str
    thing_type: str
    schema_hash: str
    input_hash: str
    initial_state_root: str
    initial_sketch_root: str = ""
    source_manifest_root: str = ""
    node_count: int = 0
    deterministic: bool = False
    factory_version: str = ""
    created_at: str = dc_field(default_factory=now_iso_utc)
    diagnostics: List[Dict[str, Any]] = dc_field(default_factory=list)
    metadata: Dict[str, Any] = dc_field(default_factory=dict)
    schema_version: str = "tb.twin_creation_receipt.v1"

    @staticmethod
    def make_twin_id(*, thing_name: str, thing_type: str, input_hash: str) -> str:
        return "twin-" + domain_hash("TB:TWIN_ID:v1", {"thing_name": thing_name, "thing_type": thing_type, "input_hash": input_hash})[:32]

    def receipt_hash(self) -> str:
        return self.hash()

    def validate(self) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        for name in ("twin_id", "thing_name", "thing_type"):
            if not _nonempty(getattr(self, name)):
                issues.append(ValidationIssue(f"{name}_empty", f"{name} must be non-empty", field=name))
        for name in ("schema_hash", "input_hash", "initial_state_root", "initial_sketch_root", "source_manifest_root"):
            v = getattr(self, name)
            if v and not _validate_hex(v, length=64):
                issues.append(ValidationIssue(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name))
        if int(self.node_count) < 0:
            issues.append(ValidationIssue("node_count_negative", "node_count cannot be negative", field="node_count"))
        return issues


# =============================================================================
# Additional optional envelope schemas useful to the stack
# =============================================================================

@dataclass
class TransitionReceipt(SchemaMixin):
    SCHEMA_NAME: ClassVar[str] = "TransitionReceipt"
    HASH_DOMAIN: ClassVar[str] = "TB:TRANSITION_RECEIPT:v1"

    tx_hash: str
    pre_state_root: str
    post_state_root: str
    policy_receipt_hash: str = ""
    tensor_commitment_hash: str = ""
    ncp_health_hash: str = ""
    cfc_transition_hash: str = ""
    accepted: bool = False
    reason: str = ""
    created_at: str = dc_field(default_factory=now_iso_utc)
    metadata: Dict[str, Any] = dc_field(default_factory=dict)
    schema_version: str = "tb.transition_receipt.v1"

    def transition_hash(self) -> str:
        return self.hash()


@dataclass
class BlockCommitment(SchemaMixin):
    SCHEMA_NAME: ClassVar[str] = "BlockCommitment"
    HASH_DOMAIN: ClassVar[str] = "TB:BLOCK_COMMITMENT:v1"

    chain_id: str
    height: int
    prev_hash: str
    tx_merkle_root: str
    pre_state_root: str
    post_state_root: str
    policy_receipt_root: str = ""
    tensor_commitment_hash: str = ""
    compression_report_hash: str = ""
    consensus_certificate_hash: str = ""
    timestamp: str = dc_field(default_factory=now_iso_utc)
    metadata_hash: str = ""
    schema_version: str = "tb.block_commitment.v1"

    def block_hash(self) -> str:
        return self.hash()


# =============================================================================
# Registry and convenience functions
# =============================================================================

_SCHEMA_REGISTRY: Dict[str, Type[SchemaMixin]] = {
    cls.__name__: cls
    for cls in [
        ChainIdentity,
        StateCommitment,
        TensorCommitment,
        CompressionReport,
        TransactionEnvelope,
        ActionReceipt,
        PolicyReceipt,
        TransactionReceipt,
        ConsensusVote,
        ConsensusCertificate,
        LobeScaleProfile,
        LobeHealthReceipt,
        CfCTransitionReceipt,
        TwinCreationReceipt,
        TransitionReceipt,
        BlockCommitment,
    ]
}

# Also map schema names for payloads with explicit schema_name.
for _cls in list(_SCHEMA_REGISTRY.values()):
    _SCHEMA_REGISTRY[getattr(_cls, "SCHEMA_NAME", _cls.__name__)] = _cls


def schema_registry() -> Dict[str, Type[SchemaMixin]]:
    return dict(_SCHEMA_REGISTRY)


def schema_from_dict(payload: Mapping[str, Any], *, schema_name: Optional[str] = None) -> SchemaMixin:
    if not isinstance(payload, Mapping):
        raise TypeError("schema_from_dict expects a mapping")
    name = schema_name or str(payload.get("schema_name", ""))
    if not name:
        # Try to infer from schema_version prefix.
        sv = str(payload.get("schema_version", ""))
        for cls in _SCHEMA_REGISTRY.values():
            try:
                # Avoid constructing; inspect default if possible.
                default_sv = ""
                for f in fields(cls):
                    if f.name == "schema_version":
                        default_sv = str(f.default) if f.default is not dataclasses.MISSING else ""
                if default_sv and default_sv == sv:
                    name = cls.__name__
                    break
            except Exception:
                continue
    cls = _SCHEMA_REGISTRY.get(name)
    if cls is None:
        raise SchemaError(f"Unknown schema_name: {name!r}")
    return cls.from_dict(payload)


def receipt_root(receipts: Sequence[Union[SchemaMixin, Mapping[str, Any], str]], *, domain: str = "TB:RECEIPT_ROOT:v1") -> str:
    leaves: List[Any] = []
    for r in receipts:
        if isinstance(r, SchemaMixin):
            leaves.append(r.hash())
        elif isinstance(r, Mapping):
            leaves.append(domain_hash("TB:RECEIPT_LEAF:v1", dict(r)))
        else:
            leaves.append(str(r))
    return merkle_root_hex(leaves, domain=domain)


def validate_schema(obj: Union[SchemaMixin, Mapping[str, Any]], *, schema_name: Optional[str] = None) -> List[ValidationIssue]:
    if isinstance(obj, SchemaMixin):
        return obj.validate()
    return schema_from_dict(obj, schema_name=schema_name).validate()


def schema_status() -> Dict[str, Any]:
    return {
        "ok": True,
        "version": TB_SCHEMAS_VERSION,
        "schema_count": len({cls.__name__ for cls in _SCHEMA_REGISTRY.values()}),
        "schemas": sorted({cls.__name__ for cls in _SCHEMA_REGISTRY.values()}),
        "canonical_module": bool(_tb_canonical_json_safe is not None),
        "hashing_module": bool(_tb_domain_hash_hex is not None),
        "numpy": bool(_NP_OK),
    }


__all__ = [
    "TB_SCHEMAS_VERSION",
    "DEFAULT_SCHEMA_VERSION",
    "DEFAULT_HASH_ALGORITHM",
    "ZERO_HASH_256",
    "SchemaError",
    "ValidationSeverity",
    "ValidationIssue",
    "SchemaMixin",
    "now_iso_utc",
    "json_safe",
    "canonical_dumps",
    "canonical_bytes",
    "sha256_hex",
    "domain_hash",
    "object_hash",
    "merkle_root_hex",
    "ChainIdentity",
    "TransactionEnvelope",
    "TransactionReceipt",
    "PolicyReceipt",
    "ActionReceipt",
    "StateCommitment",
    "TensorCommitment",
    "CompressionReport",
    "ConsensusVote",
    "ConsensusCertificate",
    "LobeScaleProfile",
    "LobeHealthReceipt",
    "CfCTransitionReceipt",
    "TwinCreationReceipt",
    "TransitionReceipt",
    "BlockCommitment",
    "schema_registry",
    "schema_from_dict",
    "receipt_root",
    "validate_schema",
    "schema_status",
]


# =============================================================================
# Self-test / CLI
# =============================================================================

def _self_test() -> Dict[str, Any]:
    chain = ChainIdentity.new(network="selftest", metadata={"purpose": "schemas"})
    state0 = StateCommitment(state_root=domain_hash("test", {"s": 0}), sketch_root=domain_hash("sketch", [0]), node_count=1)
    state1 = StateCommitment(state_root=domain_hash("test", {"s": 1}), sketch_root=domain_hash("sketch", [1]), node_count=1)

    action = {"kind": "update_node_data", "node_id": "5.4", "payload": {"temperature_c": 42}}
    tx = TransactionEnvelope.new(
        chain_id=chain.chain_id,
        sender_id="selftest_sender",
        sender_public_key="a" * 64,
        sender_sequence=0,
        action=action,
        pre_state_root=state0.state_root,
    )
    txh = tx.tx_hash()
    ar = ActionReceipt.from_action(action, approved=True, executed=True, status="executed", pre_state_root=state0.state_root, post_state_root=state1.state_root)
    pr = PolicyReceipt(approved=True, intent="selftest", pre_state_root=state0.state_root, post_state_root=state1.state_root, action_hashes=[ar.action_hash], action_receipt_hashes=[ar.hash()])
    tr = TransactionReceipt(tx_id=tx.tx_id, tx_hash=txh, accepted=True, status="accepted", pre_state_root=state0.state_root, post_state_root=state1.state_root, policy_receipt_hash=pr.hash(), action_receipt_hashes=[ar.hash()])

    tensor = TensorCommitment(
        output_dims=[2, 2],
        input_dims=[2, 2],
        bond_ranks=[2],
        core_hashes=[domain_hash("core", {"i": 0}), domain_hash("core", {"i": 1})],
        metadata_hash=domain_hash("meta", {}),
    ).normalized()
    comp = CompressionReport(old_parameters=100, new_parameters=40, old_bond_ranks=[4], new_bond_ranks=[2], pre_tensor_commitment_hash=tensor.commitment_hash, post_tensor_commitment_hash=tensor.commitment_hash).normalized()

    vote = ConsensusVote(
        chain_id=chain.chain_id,
        height=1,
        round=0,
        validator_id="v1",
        validator_public_key="b" * 64,
        block_hash=domain_hash("block", {"h": 1}),
        tx_merkle_root=merkle_root_hex([txh], domain="TB:TX_ROOT:test"),
        pre_state_root=state0.state_root,
        post_state_root=state1.state_root,
        tensor_commitment_hash=tensor.commitment_hash,
        approved=True,
    )
    cert = ConsensusCertificate.from_votes(
        chain_id=chain.chain_id,
        height=1,
        round=0,
        block_hash=vote.block_hash,
        tx_merkle_root=vote.tx_merkle_root,
        pre_state_root=state0.state_root,
        post_state_root=state1.state_root,
        votes=[vote],
    )

    scale = LobeScaleProfile(lobe_id="ncp_reflex", lobe_kind="ncp", parameter_budget=2_500_000_000, active_parameter_count=1024, hidden_size=64, tt_bond_rank=4)
    health = LobeHealthReceipt(lobe_id="ncp_reflex", lobe_kind="ncp", stable=True, state_hash=domain_hash("lobe_state", [0]), scale_profile_hash=scale.hash())
    cfc = CfCTransitionReceipt(lobe_id="cfc_predictor", pre_state_root=state0.state_root, tx_hash=txh, predicted_post_state_hash=state1.state_root, actual_post_state_hash=state1.state_root, forecast_error=0.0, lobe_health_hash=health.hash(), scale_profile_hash=scale.hash())
    twin = TwinCreationReceipt(
        twin_id=TwinCreationReceipt.make_twin_id(thing_name="SelfTest Twin", thing_type="test", input_hash=domain_hash("input", {})),
        thing_name="SelfTest Twin",
        thing_type="test",
        schema_hash=domain_hash("schema", {}),
        input_hash=domain_hash("input", {}),
        initial_state_root=state0.state_root,
        initial_sketch_root=state0.sketch_root,
        node_count=1,
        deterministic=True,
    )

    objects: List[SchemaMixin] = [chain, state0, state1, tx, ar, pr, tr, tensor, comp, vote, cert, scale, health, cfc, twin]
    invalid: Dict[str, List[Dict[str, Any]]] = {}
    for obj in objects:
        issues = obj.validate()
        errors = [i for i in issues if i.severity == ValidationSeverity.ERROR.value]
        if errors:
            invalid[obj.__class__.__name__] = [e.to_dict() for e in errors]
        # Round-trip through dict registry when schema_name is present.
        rt = schema_from_dict(obj.to_dict())
        if rt.hash() != obj.hash():
            invalid[obj.__class__.__name__ + "_roundtrip"] = [{"error": "hash mismatch"}]

    if invalid:
        raise AssertionError(f"Schema self-test failed: {invalid}")

    return {
        "ok": True,
        "version": TB_SCHEMAS_VERSION,
        "chain_id": chain.chain_id,
        "tx_hash": txh,
        "policy_receipt_hash": pr.hash(),
        "consensus_certificate_hash": cert.certificate_hash,
        "tensor_commitment_hash": tensor.commitment_hash,
        "compression_report_hash": comp.report_hash,
        "receipt_root": receipt_root([ar, pr, tr, health, cfc, twin]),
        "status": schema_status(),
    }


def _build_arg_parser() -> Any:
    import argparse
    p = argparse.ArgumentParser(description="Tensorized Blockchain shared schema utilities")
    p.add_argument("--mode", choices=["status", "self-test"], default="self-test")
    p.add_argument("--pretty", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = _build_arg_parser()
    args = p.parse_args(argv)
    try:
        out = schema_status() if args.mode == "status" else _self_test()
        print(json.dumps(json_safe(out), indent=2 if args.pretty else None, ensure_ascii=False))
        return 0
    except Exception as exc:
        err = {"ok": False, "error": repr(exc), "status": schema_status()}
        print(json.dumps(json_safe(err), indent=2, ensure_ascii=False))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
