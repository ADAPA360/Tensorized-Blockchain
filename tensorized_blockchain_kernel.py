#!/usr/bin/env python3
# tensorized_blockchain_kernel.py
r"""
Project Chimera / Akkurat - Tensorized Blockchain Kernel
========================================================

Production CPU-first cryptographic ledger and tensorized ledger-commitment
layer for governed digital twins.

Role
----
This module owns canonical ledger transactions, sequencing, replay protection,
domain-separated Merkle roots/proofs, transition receipts, consensus
certificates, block commitments, deterministic tensorized ledger compilation,
compression reports, ledger persistence, validation, and CLI demos.

Design boundaries
-----------------
- tn.py owns low-level deterministic TensorTrain / MPO / Tucker operations.
- digital_twin_kernel.py owns governed world-state topology, state commitments,
  policy simulation, and action receipts.
- twin_anything.py owns canonical object/process twin materialization.
- ncp.py / cfc.py own neural-lobe substrates and should only provide auditable
  receipts that can be attached to transitions.

Consensus model
---------------
The cryptographic ledger is canonical. TensorTrain/MPO cores are deterministic
compiled artifacts derived from committed block transition context. Tensor cores
are never the sole source of truth; they are hash-committed, reproducible,
compressible representations of ledger/twin evolution.

Public API
----------
- LedgerConfig
- SignatureKeyPair / SignatureEngine
- TensorTransaction
- MerkleProof / TransactionMerkleTree
- ValidatorVote / ConsensusCertificate
- TransitionReceipt
- TensorBlock
- LedgerMPS
- GovernedLedgerExecutor
- run_self_test / run_wiser_demo
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import hmac
import json
import math
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

# =============================================================================
# Version / domains
# =============================================================================

TENSORIZED_BLOCKCHAIN_KERNEL_VERSION = "tensorized-blockchain-kernel-v2.0-production"
LEDGER_SCHEMA = "ProjectChimera.TensorizedBlockchainLedger.v2"
TRANSACTION_SCHEMA = "ProjectChimera.TensorTransaction.v2"
BLOCK_SCHEMA = "ProjectChimera.TensorBlock.v2"
TRANSITION_RECEIPT_SCHEMA = "ProjectChimera.TransitionReceipt.v1"
CONSENSUS_CERTIFICATE_SCHEMA = "ProjectChimera.ConsensusCertificate.v1"
MERKLE_PROOF_SCHEMA = "ProjectChimera.TransactionMerkleProof.v1"

_DOMAIN_TX = b"CHIMERA_TB_TX_V2\x00"
_DOMAIN_BLOCK = b"CHIMERA_TB_BLOCK_V2\x00"
_DOMAIN_TX_LEAF = b"CHIMERA_TB_TX_LEAF_V1\x00"
_DOMAIN_TX_NODE = b"CHIMERA_TB_TX_NODE_V1\x00"
_DOMAIN_RECEIPT = b"CHIMERA_TB_TRANSITION_RECEIPT_V1\x00"
_DOMAIN_VOTE = b"CHIMERA_TB_VALIDATOR_VOTE_V1\x00"
_DOMAIN_CERT = b"CHIMERA_TB_CONSENSUS_CERTIFICATE_V1\x00"
_DOMAIN_TENSOR_CONTEXT = b"CHIMERA_TB_TENSOR_CONTEXT_V1\x00"
_DOMAIN_LEDGER_FILE = b"CHIMERA_TB_LEDGER_FILE_V2\x00"
_DOMAIN_JSON = b"CHIMERA_TB_JSON_V1\x00"

_EPS = 1e-12


# =============================================================================
# Path bootstrap and local imports
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


def configure_paths(*, project_root: Optional[Union[str, os.PathLike]] = None, permissive: bool = True) -> Dict[str, Optional[str]]:
    import tempfile
    """Locate project-local modules.

    Production package deployments should rely on normal imports and pass
    permissive=False. The permissive search is retained for local notebooks and
    demo folders.
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
    found: Dict[str, Optional[str]] = {
        "module_dir": str(here),
        "project_root": None if project_root is None else str(Path(project_root).expanduser()),
        "tn_root": None,
        "digital_twin_kernel_root": None,
        "twin_anything_root": None,
    }
    for candidate in candidates:
        p = _add_path(candidate, prepend=True)
        if p is None:
            continue
        if found["tn_root"] is None and (p / "tn.py").exists():
            found["tn_root"] = str(p)
        if found["digital_twin_kernel_root"] is None and (p / "digital_twin_kernel.py").exists():
            found["digital_twin_kernel_root"] = str(p)
        if found["twin_anything_root"] is None and (p / "twin_anything.py").exists():
            found["twin_anything_root"] = str(p)
    _add_path(here, prepend=True)
    return found


_PATHS = configure_paths()

try:
    import tn  # type: ignore
    import digital_twin_kernel as dtk  # type: ignore
except Exception as exc:  # pragma: no cover - environment error
    raise ImportError(
        f"Missing Tensorized Blockchain stack dependency: {exc}. Ensure tn.py and "
        "digital_twin_kernel.py are importable or set TENSORIZED_BLOCKCHAIN_ROOT."
    ) from exc

try:  # optional convenience for demos
    import twin_anything as ta  # type: ignore
except Exception:  # pragma: no cover - optional
    ta = None  # type: ignore

# Optional public-key signing backend. The ledger can run unsigned demos, but
# production signature validation requires Ed25519 support.
_ED25519_OK = False
_ED25519_ERR = ""
try:  # pragma: no cover - availability is environment dependent
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    from cryptography.hazmat.primitives import serialization
    _ED25519_OK = True
except Exception as exc:  # pragma: no cover
    Ed25519PrivateKey = None  # type: ignore
    Ed25519PublicKey = None  # type: ignore
    serialization = None  # type: ignore
    _ED25519_ERR = repr(exc)


# =============================================================================
# Helpers
# =============================================================================

def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(_json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8", errors="ignore")


def _sha256_hex_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_hex_obj(obj: Any, *, domain: bytes = _DOMAIN_JSON) -> str:
    return _sha256_hex_bytes(domain + _canonical_json_bytes(obj))


def _hex_to_bytes(value: str) -> bytes:
    s = str(value or "").strip().lower()
    try:
        return bytes.fromhex(s)
    except Exception:
        return hashlib.sha256(s.encode("utf-8", errors="ignore")).digest()


def _stable_hash_u32(s: str) -> int:
    h = 2166136261
    for b in (s or "").encode("utf-8", errors="ignore"):
        h ^= int(b)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


def _atomic_write_text(path: Union[str, os.PathLike], text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def _array_digest(arr: Any) -> str:
    x = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return _sha256_hex_bytes(x.astype(np.float32, copy=False).tobytes())


def _normalize_vec(x: Any, dim: int, *, key: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if arr.size != int(dim):
        out = np.zeros((int(dim),), dtype=np.float32)
        for i, v in enumerate(arr):
            h = _stable_hash_u32(f"{key}:{i}")
            out[h % int(dim)] += (-1.0 if (h & 1) else 1.0) * np.float32(v)
        arr = out
    n = float(np.linalg.norm(arr))
    if not np.isfinite(n) or n <= _EPS:
        return np.zeros((int(dim),), dtype=np.float32)
    return (arr / n).astype(np.float32, copy=False)


def _deterministic_vector_from_payload(payload: Mapping[str, Any], dim: int) -> np.ndarray:
    """Generate a deterministic normalized vector from canonical payload bytes."""
    base = _DOMAIN_TENSOR_CONTEXT + _canonical_json_bytes(payload)
    vals: List[float] = []
    counter = 0
    while len(vals) < int(dim):
        digest = hashlib.sha256(base + int(counter).to_bytes(8, "little", signed=False)).digest()
        counter += 1
        for i in range(0, len(digest), 4):
            u = int.from_bytes(digest[i:i + 4], "little", signed=False)
            vals.append((float(u) / float(2**32 - 1)) * 2.0 - 1.0)
            if len(vals) >= int(dim):
                break
    return _normalize_vec(np.asarray(vals, dtype=np.float32), int(dim), key="deterministic_vector")


def _merkle_root_hex_from_hashes(hashes: Sequence[str], *, leaf_domain: bytes, node_domain: bytes) -> str:
    if not hashes:
        return _sha256_hex_bytes(node_domain + b"<empty>")
    level = [hashlib.sha256(leaf_domain + _hex_to_bytes(h)).digest() for h in hashes]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        nxt = []
        for i in range(0, len(level), 2):
            nxt.append(hashlib.sha256(node_domain + level[i] + level[i + 1]).digest())
        level = nxt
    return level[0].hex()


def _dataclass_from_dict(cls: Any, payload: Mapping[str, Any]) -> Any:
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in dict(payload).items() if k in names})


# =============================================================================
# Signature support
# =============================================================================

@dataclass(frozen=True)
class SignatureKeyPair:
    scheme: str
    public_key: str
    private_key: str


class SignatureEngine:
    """Public-key signature adapter.

    Production mode uses Ed25519 when the optional cryptography package is
    installed. A legacy hmac-demo scheme is retained only for local demos and is
    rejected by LedgerConfig.require_public_key_signatures=True.
    """

    @staticmethod
    def available_schemes() -> Dict[str, Any]:
        return {
            "ed25519": {"available": bool(_ED25519_OK), "error": _ED25519_ERR},
            "hmac-demo": {"available": True, "production_safe": False},
            "none": {"available": True, "production_safe": False},
        }

    @staticmethod
    def generate_ed25519_keypair() -> SignatureKeyPair:
        if not _ED25519_OK:
            raise RuntimeError(f"Ed25519 backend unavailable: {_ED25519_ERR}")
        priv = Ed25519PrivateKey.generate()  # type: ignore[union-attr]
        pub = priv.public_key()
        private_raw = priv.private_bytes(
            encoding=serialization.Encoding.Raw,  # type: ignore[union-attr]
            format=serialization.PrivateFormat.Raw,  # type: ignore[union-attr]
            encryption_algorithm=serialization.NoEncryption(),  # type: ignore[union-attr]
        )
        public_raw = pub.public_bytes(
            encoding=serialization.Encoding.Raw,  # type: ignore[union-attr]
            format=serialization.PublicFormat.Raw,  # type: ignore[union-attr]
        )
        return SignatureKeyPair("ed25519", public_raw.hex(), private_raw.hex())

    @staticmethod
    def _load_private_ed25519(private_key: Union[str, bytes]) -> Any:
        if not _ED25519_OK:
            raise RuntimeError(f"Ed25519 backend unavailable: {_ED25519_ERR}")
        if isinstance(private_key, bytes):
            raw = private_key
        else:
            s = str(private_key).strip()
            if "BEGIN" in s:
                return serialization.load_pem_private_key(s.encode("utf-8"), password=None)  # type: ignore[union-attr]
            raw = bytes.fromhex(s)
        if len(raw) != 32:
            raise ValueError("Ed25519 raw private key must be 32 bytes.")
        return Ed25519PrivateKey.from_private_bytes(raw)  # type: ignore[union-attr]

    @staticmethod
    def _load_public_ed25519(public_key: Union[str, bytes]) -> Any:
        if not _ED25519_OK:
            raise RuntimeError(f"Ed25519 backend unavailable: {_ED25519_ERR}")
        if isinstance(public_key, bytes):
            raw = public_key
        else:
            s = str(public_key).strip()
            if "BEGIN" in s:
                return serialization.load_pem_public_key(s.encode("utf-8"))  # type: ignore[union-attr]
            raw = bytes.fromhex(s)
        if len(raw) != 32:
            raise ValueError("Ed25519 raw public key must be 32 bytes.")
        return Ed25519PublicKey.from_public_bytes(raw)  # type: ignore[union-attr]

    @staticmethod
    def public_from_private(private_key: Union[str, bytes], *, scheme: str = "ed25519") -> str:
        scheme = str(scheme).lower().strip()
        if scheme != "ed25519":
            raise ValueError("public_from_private currently supports ed25519 only.")
        priv = SignatureEngine._load_private_ed25519(private_key)
        pub = priv.public_key()
        raw = pub.public_bytes(
            encoding=serialization.Encoding.Raw,  # type: ignore[union-attr]
            format=serialization.PublicFormat.Raw,  # type: ignore[union-attr]
        )
        return raw.hex()

    @staticmethod
    def sign(payload: bytes, private_key: Union[str, bytes], *, scheme: str = "ed25519") -> str:
        scheme = str(scheme).lower().strip()
        if scheme == "ed25519":
            priv = SignatureEngine._load_private_ed25519(private_key)
            return priv.sign(payload).hex()
        if scheme == "hmac-demo":
            key = private_key if isinstance(private_key, bytes) else str(private_key).encode("utf-8")
            return hmac.new(key, payload, hashlib.sha256).hexdigest()
        if scheme in {"", "none", "unsigned"}:
            return ""
        raise ValueError(f"Unsupported signature scheme: {scheme}")

    @staticmethod
    def verify(payload: bytes, signature: str, public_key: Union[str, bytes], *, scheme: str = "ed25519") -> bool:
        scheme = str(scheme).lower().strip()
        sig = str(signature or "")
        if scheme == "ed25519":
            try:
                pub = SignatureEngine._load_public_ed25519(public_key)
                pub.verify(bytes.fromhex(sig), payload)
                return True
            except Exception:
                return False
        if scheme == "hmac-demo":
            key = public_key if isinstance(public_key, bytes) else str(public_key).encode("utf-8")
            expected = hmac.new(key, payload, hashlib.sha256).hexdigest()
            return hmac.compare_digest(expected, sig)
        if scheme in {"", "none", "unsigned"}:
            return sig == ""
        return False


# =============================================================================
# Configuration and receipts
# =============================================================================

@dataclass
class LedgerConfig:
    chain_id: str = "project-chimera-local"
    vector_dim: int = 256
    out_dim: int = 16
    in_dim: int = 16
    max_bond_dim: int = 8
    energy_tol: float = 0.999
    seed: int = 2027
    tt_rank: int = 8
    consensus_threshold: float = 0.85
    quorum_ratio: float = 0.66
    require_signatures: bool = False
    require_public_key_signatures: bool = True
    allowed_signature_schemes: Tuple[str, ...] = ("ed25519",)
    deterministic_blocks: bool = True
    genesis_timestamp: float = 0.0
    block_time_step_s: float = 1.0
    dense_max_entries: int = 1_000_000
    compile_dtype: str = "float32"
    consensus_safe_tensors: bool = True
    compress_on_append: bool = True

    def normalized(self) -> "LedgerConfig":
        cfg = copy.deepcopy(self)
        cfg.chain_id = str(cfg.chain_id or "project-chimera-local")
        cfg.out_dim = int(max(1, cfg.out_dim))
        cfg.in_dim = int(max(1, cfg.in_dim))
        cfg.vector_dim = int(cfg.out_dim * cfg.in_dim)
        cfg.max_bond_dim = int(max(1, cfg.max_bond_dim))
        cfg.energy_tol = float(cfg.energy_tol)
        if not (0.0 < cfg.energy_tol <= 1.0):
            cfg.energy_tol = 0.999
        cfg.seed = int(cfg.seed)
        cfg.tt_rank = int(max(1, cfg.tt_rank))
        cfg.consensus_threshold = _clip01(cfg.consensus_threshold)
        cfg.quorum_ratio = _clip01(cfg.quorum_ratio) or 0.66
        cfg.require_signatures = bool(cfg.require_signatures)
        cfg.require_public_key_signatures = bool(cfg.require_public_key_signatures)
        cfg.allowed_signature_schemes = tuple(str(x).lower().strip() for x in cfg.allowed_signature_schemes if str(x).strip()) or ("ed25519",)
        cfg.deterministic_blocks = bool(cfg.deterministic_blocks)
        cfg.genesis_timestamp = float(cfg.genesis_timestamp)
        cfg.block_time_step_s = float(cfg.block_time_step_s)
        cfg.dense_max_entries = int(max(1, cfg.dense_max_entries))
        cfg.compile_dtype = str(cfg.compile_dtype or "float32").lower().strip()
        if cfg.compile_dtype not in {"float32", "float64"}:
            cfg.compile_dtype = "float32"
        cfg.consensus_safe_tensors = bool(cfg.consensus_safe_tensors)
        cfg.compress_on_append = bool(cfg.compress_on_append)
        return cfg


@dataclass
class TransitionReceipt:
    schema_version: str
    chain_id: str
    tx_merkle_root: str
    pre_state_root: str
    post_state_root: str
    policy_receipt_hashes: List[str]
    action_receipt_hashes: List[str]
    tensor_commitment_hash: str
    compression_report_hash: str
    metadata_hash: str
    receipt_hash: str = ""

    @staticmethod
    def create(
        *,
        chain_id: str,
        tx_merkle_root: str,
        pre_state_root: str,
        post_state_root: str,
        policy_receipts: Sequence[Any] = (),
        action_receipts: Sequence[Any] = (),
        tensor_commitment_hash: str = "",
        compression_report_hash: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "TransitionReceipt":
        policy_hashes = []
        for r in list(policy_receipts or []):
            if isinstance(r, str):
                policy_hashes.append(r)
            elif isinstance(r, Mapping):
                policy_hashes.append(str(r.get("receipt_hash") or _sha256_hex_obj(r)))
            elif hasattr(r, "receipt_hash"):
                policy_hashes.append(str(r.receipt_hash))
            else:
                policy_hashes.append(_sha256_hex_obj(r))
        action_hashes = []
        for r in list(action_receipts or []):
            if isinstance(r, str):
                action_hashes.append(r)
            elif isinstance(r, Mapping):
                action_hashes.append(str(r.get("receipt_hash") or _sha256_hex_obj(r)))
            elif hasattr(r, "receipt_hash"):
                action_hashes.append(str(r.receipt_hash))
            else:
                action_hashes.append(_sha256_hex_obj(r))
        obj = TransitionReceipt(
            schema_version=TRANSITION_RECEIPT_SCHEMA,
            chain_id=str(chain_id),
            tx_merkle_root=str(tx_merkle_root),
            pre_state_root=str(pre_state_root),
            post_state_root=str(post_state_root),
            policy_receipt_hashes=sorted(policy_hashes),
            action_receipt_hashes=sorted(action_hashes),
            tensor_commitment_hash=str(tensor_commitment_hash),
            compression_report_hash=str(compression_report_hash),
            metadata_hash=_sha256_hex_obj(dict(metadata or {})),
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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransitionReceipt":
        p = dict(payload or {})
        return cls(
            schema_version=str(p.get("schema_version", TRANSITION_RECEIPT_SCHEMA)),
            chain_id=str(p.get("chain_id", "")),
            tx_merkle_root=str(p.get("tx_merkle_root", "")),
            pre_state_root=str(p.get("pre_state_root", "")),
            post_state_root=str(p.get("post_state_root", "")),
            policy_receipt_hashes=[str(x) for x in p.get("policy_receipt_hashes", [])],
            action_receipt_hashes=[str(x) for x in p.get("action_receipt_hashes", [])],
            tensor_commitment_hash=str(p.get("tensor_commitment_hash", "")),
            compression_report_hash=str(p.get("compression_report_hash", "")),
            metadata_hash=str(p.get("metadata_hash", "")),
            receipt_hash=str(p.get("receipt_hash", "")),
        )


# =============================================================================
# Transactions and Merkle proofs
# =============================================================================

@dataclass
class TensorTransaction:
    schema_version: str
    tx_id: str
    chain_id: str
    sender_id: str
    sender_public_key: str
    sender_sequence: int
    twin_node_id: str
    action: dtk.Action
    timestamp: float
    nonce: str = ""
    signature_scheme: str = "ed25519"
    signature: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new(
        *,
        sender_id: str,
        twin_node_id: str,
        action: dtk.Action,
        chain_id: str,
        sender_sequence: int,
        sender_public_key: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
        tx_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        nonce: Optional[str] = None,
        signature_scheme: str = "ed25519",
    ) -> "TensorTransaction":
        tx = TensorTransaction(
            schema_version=TRANSACTION_SCHEMA,
            tx_id=tx_id or uuid.uuid4().hex,
            chain_id=str(chain_id),
            sender_id=str(sender_id),
            sender_public_key=str(sender_public_key or ""),
            sender_sequence=int(sender_sequence),
            twin_node_id=str(twin_node_id),
            action=action,
            timestamp=float(time.time() if timestamp is None else timestamp),
            nonce=nonce or uuid.uuid4().hex,
            signature_scheme=str(signature_scheme or "ed25519").lower().strip(),
            metadata=dict(metadata or {}),
        )
        return tx

    def action_payload(self) -> Dict[str, Any]:
        if hasattr(self.action, "to_dict"):
            try:
                return self.action.to_dict()
            except Exception:
                pass
        return {
            "kind": str(getattr(self.action, "kind", "")),
            "node_id": getattr(self.action, "node_id", None),
            "payload": _json_safe(getattr(self.action, "payload", None)),
            "note": str(getattr(self.action, "note", "")),
        }

    def payload(self, *, include_signature: bool = False) -> Dict[str, Any]:
        out = {
            "schema_version": self.schema_version,
            "tx_id": str(self.tx_id),
            "chain_id": str(self.chain_id),
            "sender_id": str(self.sender_id),
            "sender_public_key": str(self.sender_public_key),
            "sender_sequence": int(self.sender_sequence),
            "twin_node_id": str(self.twin_node_id),
            "action": self.action_payload(),
            "timestamp": float(self.timestamp),
            "nonce": str(self.nonce),
            "signature_scheme": str(self.signature_scheme),
            "metadata": _json_safe(self.metadata),
        }
        if include_signature:
            out["signature"] = str(self.signature)
        return out

    def signing_bytes(self) -> bytes:
        return _DOMAIN_TX + _canonical_json_bytes(self.payload(include_signature=False))

    def payload_hash(self) -> str:
        return _sha256_hex_bytes(self.signing_bytes())

    def sign(self, private_key: Union[str, bytes], *, scheme: Optional[str] = None) -> str:
        sc = str(scheme or self.signature_scheme or "ed25519").lower().strip()
        self.signature_scheme = sc
        if sc == "ed25519" and not self.sender_public_key:
            self.sender_public_key = SignatureEngine.public_from_private(private_key, scheme=sc)
        self.signature = SignatureEngine.sign(self.signing_bytes(), private_key, scheme=sc)
        return self.signature

    def verify_signature(self) -> bool:
        sc = str(self.signature_scheme or "").lower().strip()
        if sc in {"", "none", "unsigned"}:
            return not bool(self.signature)
        if not self.signature or not self.sender_public_key:
            return False
        return SignatureEngine.verify(self.signing_bytes(), self.signature, self.sender_public_key, scheme=sc)

    def to_dict(self) -> Dict[str, Any]:
        return self.payload(include_signature=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TensorTransaction":
        p = dict(payload or {})
        a = p.get("action", {})
        if isinstance(a, dtk.Action):
            action = a
        else:
            action = dtk.Action(
                kind=str(dict(a or {}).get("kind", "")),
                node_id=dict(a or {}).get("node_id"),
                payload=dict(a or {}).get("payload"),
                note=str(dict(a or {}).get("note", "")),
            )
        return cls(
            schema_version=str(p.get("schema_version", TRANSACTION_SCHEMA)),
            tx_id=str(p.get("tx_id", uuid.uuid4().hex)),
            chain_id=str(p.get("chain_id", "")),
            sender_id=str(p.get("sender_id", "unknown")),
            sender_public_key=str(p.get("sender_public_key", "")),
            sender_sequence=int(p.get("sender_sequence", 0)),
            twin_node_id=str(p.get("twin_node_id", p.get("node_id", ""))),
            action=action,
            timestamp=float(p.get("timestamp", 0.0)),
            nonce=str(p.get("nonce", "")),
            signature_scheme=str(p.get("signature_scheme", "ed25519")),
            signature=str(p.get("signature", "")),
            metadata=dict(p.get("metadata", {})),
        )


@dataclass
class MerkleProof:
    schema_version: str
    tx_hash: str
    leaf_hash: str
    index: int
    siblings: List[Dict[str, str]]
    root: str

    def verify(self) -> bool:
        try:
            cur = hashlib.sha256(_DOMAIN_TX_LEAF + _hex_to_bytes(self.tx_hash)).digest()
            if cur.hex() != self.leaf_hash:
                return False
            for sib in self.siblings:
                side = str(sib.get("side", "")).lower()
                h = _hex_to_bytes(str(sib.get("hash", "")))
                if side == "left":
                    cur = hashlib.sha256(_DOMAIN_TX_NODE + h + cur).digest()
                elif side == "right":
                    cur = hashlib.sha256(_DOMAIN_TX_NODE + cur + h).digest()
                else:
                    return False
            return cur.hex() == self.root
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


class TransactionMerkleTree:
    def __init__(self, tx_hashes: Sequence[str]):
        self.tx_hashes = [str(x) for x in list(tx_hashes)]
        self.levels: List[List[bytes]] = []
        if not self.tx_hashes:
            self.root = _sha256_hex_bytes(_DOMAIN_TX_NODE + b"<empty>")
            return
        level = [hashlib.sha256(_DOMAIN_TX_LEAF + _hex_to_bytes(h)).digest() for h in self.tx_hashes]
        self.levels.append(level)
        while len(level) > 1:
            work = list(level)
            if len(work) % 2 == 1:
                work.append(work[-1])
            nxt = []
            for i in range(0, len(work), 2):
                nxt.append(hashlib.sha256(_DOMAIN_TX_NODE + work[i] + work[i + 1]).digest())
            level = nxt
            self.levels.append(level)
        self.root = self.levels[-1][0].hex()

    @staticmethod
    def root_for_transactions(txs: Sequence[TensorTransaction]) -> str:
        return TransactionMerkleTree([tx.payload_hash() for tx in txs]).root

    def proof(self, index: int) -> MerkleProof:
        if not self.tx_hashes:
            raise ValueError("Cannot build proof for empty transaction tree.")
        idx = int(index)
        if idx < 0 or idx >= len(self.tx_hashes):
            raise IndexError("transaction index out of range")
        cur_idx = idx
        siblings: List[Dict[str, str]] = []
        for level in self.levels[:-1]:
            work = list(level)
            if len(work) % 2 == 1:
                work.append(work[-1])
            if cur_idx % 2 == 0:
                sib_idx = cur_idx + 1
                siblings.append({"side": "right", "hash": work[sib_idx].hex()})
            else:
                sib_idx = cur_idx - 1
                siblings.append({"side": "left", "hash": work[sib_idx].hex()})
            cur_idx //= 2
        leaf_hash = hashlib.sha256(_DOMAIN_TX_LEAF + _hex_to_bytes(self.tx_hashes[idx])).hexdigest()
        return MerkleProof(MERKLE_PROOF_SCHEMA, self.tx_hashes[idx], leaf_hash, idx, siblings, self.root)


# =============================================================================
# Consensus
# =============================================================================

@dataclass
class ValidatorVote:
    schema_version: str
    validator_id: str
    chain_id: str
    tx_merkle_root: str
    pre_state_root: str
    post_state_root: str
    transition_receipt_hash: str
    tensor_commitment_hash: str
    policy_receipt_hash: str
    approved: bool
    timestamp: float
    signature_scheme: str = "ed25519"
    validator_public_key: str = ""
    signature: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new(
        *,
        validator_id: str,
        chain_id: str,
        tx_merkle_root: str,
        pre_state_root: str,
        post_state_root: str,
        transition_receipt_hash: str,
        tensor_commitment_hash: str,
        policy_receipt_hash: str = "",
        approved: bool = True,
        timestamp: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "ValidatorVote":
        return ValidatorVote(
            schema_version="ProjectChimera.ValidatorVote.v1",
            validator_id=str(validator_id),
            chain_id=str(chain_id),
            tx_merkle_root=str(tx_merkle_root),
            pre_state_root=str(pre_state_root),
            post_state_root=str(post_state_root),
            transition_receipt_hash=str(transition_receipt_hash),
            tensor_commitment_hash=str(tensor_commitment_hash),
            policy_receipt_hash=str(policy_receipt_hash),
            approved=bool(approved),
            timestamp=float(time.time() if timestamp is None else timestamp),
            metadata=dict(metadata or {}),
        )

    def payload(self, *, include_signature: bool = False) -> Dict[str, Any]:
        out = _json_safe(asdict(self))
        if not include_signature:
            out.pop("signature", None)
        return out

    def signing_bytes(self) -> bytes:
        return _DOMAIN_VOTE + _canonical_json_bytes(self.payload(include_signature=False))

    def vote_hash(self) -> str:
        return _sha256_hex_bytes(self.signing_bytes())

    def sign(self, private_key: Union[str, bytes], *, scheme: str = "ed25519") -> str:
        self.signature_scheme = str(scheme).lower().strip()
        if self.signature_scheme == "ed25519" and not self.validator_public_key:
            self.validator_public_key = SignatureEngine.public_from_private(private_key, scheme="ed25519")
        self.signature = SignatureEngine.sign(self.signing_bytes(), private_key, scheme=self.signature_scheme)
        return self.signature

    def verify_signature(self) -> bool:
        if not self.signature:
            return False
        return SignatureEngine.verify(self.signing_bytes(), self.signature, self.validator_public_key, scheme=self.signature_scheme)

    def to_dict(self) -> Dict[str, Any]:
        return self.payload(include_signature=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidatorVote":
        p = dict(payload or {})
        return cls(
            schema_version=str(p.get("schema_version", "ProjectChimera.ValidatorVote.v1")),
            validator_id=str(p.get("validator_id", "")),
            chain_id=str(p.get("chain_id", "")),
            tx_merkle_root=str(p.get("tx_merkle_root", "")),
            pre_state_root=str(p.get("pre_state_root", "")),
            post_state_root=str(p.get("post_state_root", "")),
            transition_receipt_hash=str(p.get("transition_receipt_hash", "")),
            tensor_commitment_hash=str(p.get("tensor_commitment_hash", "")),
            policy_receipt_hash=str(p.get("policy_receipt_hash", "")),
            approved=bool(p.get("approved", False)),
            timestamp=float(p.get("timestamp", 0.0)),
            signature_scheme=str(p.get("signature_scheme", "ed25519")),
            validator_public_key=str(p.get("validator_public_key", "")),
            signature=str(p.get("signature", "")),
            metadata=dict(p.get("metadata", {})),
        )


@dataclass
class ConsensusCertificate:
    schema_version: str
    chain_id: str
    tx_merkle_root: str
    pre_state_root: str
    post_state_root: str
    transition_receipt_hash: str
    tensor_commitment_hash: str
    approvals: int
    total_votes: int
    quorum_required: int
    threshold: float
    votes: List[ValidatorVote] = field(default_factory=list)
    certificate_hash: str = ""

    @staticmethod
    def create(
        *,
        chain_id: str,
        tx_merkle_root: str,
        pre_state_root: str,
        post_state_root: str,
        transition_receipt_hash: str,
        tensor_commitment_hash: str,
        votes: Sequence[ValidatorVote],
        quorum_required: int,
        threshold: float,
    ) -> "ConsensusCertificate":
        filtered = [v for v in list(votes or []) if bool(v.approved)]
        obj = ConsensusCertificate(
            schema_version=CONSENSUS_CERTIFICATE_SCHEMA,
            chain_id=str(chain_id),
            tx_merkle_root=str(tx_merkle_root),
            pre_state_root=str(pre_state_root),
            post_state_root=str(post_state_root),
            transition_receipt_hash=str(transition_receipt_hash),
            tensor_commitment_hash=str(tensor_commitment_hash),
            approvals=len(filtered),
            total_votes=len(list(votes or [])),
            quorum_required=int(max(1, quorum_required)),
            threshold=float(threshold),
            votes=list(votes or []),
        )
        obj.certificate_hash = obj.compute_hash()
        return obj

    def payload(self, *, include_certificate_hash: bool = False) -> Dict[str, Any]:
        out = _json_safe(asdict(self))
        out["votes"] = [v.to_dict() for v in self.votes]
        if not include_certificate_hash:
            out.pop("certificate_hash", None)
        return out

    def compute_hash(self) -> str:
        return _sha256_hex_bytes(_DOMAIN_CERT + _canonical_json_bytes(self.payload()))

    def verify(self, *, require_vote_signatures: bool = False) -> Dict[str, Any]:
        errors: List[str] = []
        if self.certificate_hash != self.compute_hash():
            errors.append("certificate_hash_invalid")
        approvals = 0
        seen_validators = set()
        for i, v in enumerate(self.votes):
            if v.validator_id in seen_validators:
                errors.append(f"duplicate_validator:{v.validator_id}")
            seen_validators.add(v.validator_id)
            if v.chain_id != self.chain_id:
                errors.append(f"vote_chain_id_mismatch:{i}")
            if v.tx_merkle_root != self.tx_merkle_root:
                errors.append(f"vote_tx_merkle_mismatch:{i}")
            if v.pre_state_root != self.pre_state_root or v.post_state_root != self.post_state_root:
                errors.append(f"vote_state_root_mismatch:{i}")
            if v.transition_receipt_hash != self.transition_receipt_hash:
                errors.append(f"vote_transition_receipt_mismatch:{i}")
            if v.tensor_commitment_hash != self.tensor_commitment_hash:
                errors.append(f"vote_tensor_commitment_mismatch:{i}")
            if require_vote_signatures and not v.verify_signature():
                errors.append(f"vote_signature_invalid:{i}")
            if v.approved:
                approvals += 1
        if approvals != self.approvals:
            errors.append("approval_count_mismatch")
        if approvals < int(self.quorum_required):
            errors.append("quorum_not_met")
        return {"ok": len(errors) == 0, "errors": errors, "approvals": approvals, "quorum_required": int(self.quorum_required)}

    def to_dict(self) -> Dict[str, Any]:
        return self.payload(include_certificate_hash=True)

    @classmethod
    def empty(cls, *, chain_id: str, tx_merkle_root: str, pre_state_root: str, post_state_root: str, transition_receipt_hash: str, tensor_commitment_hash: str) -> "ConsensusCertificate":
        return cls.create(chain_id=chain_id, tx_merkle_root=tx_merkle_root, pre_state_root=pre_state_root, post_state_root=post_state_root, transition_receipt_hash=transition_receipt_hash, tensor_commitment_hash=tensor_commitment_hash, votes=[], quorum_required=1, threshold=0.0)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConsensusCertificate":
        p = dict(payload or {})
        votes = [ValidatorVote.from_dict(v) for v in list(p.get("votes", []))]
        return cls(
            schema_version=str(p.get("schema_version", CONSENSUS_CERTIFICATE_SCHEMA)),
            chain_id=str(p.get("chain_id", "")),
            tx_merkle_root=str(p.get("tx_merkle_root", "")),
            pre_state_root=str(p.get("pre_state_root", "")),
            post_state_root=str(p.get("post_state_root", "")),
            transition_receipt_hash=str(p.get("transition_receipt_hash", "")),
            tensor_commitment_hash=str(p.get("tensor_commitment_hash", "")),
            approvals=int(p.get("approvals", 0)),
            total_votes=int(p.get("total_votes", len(votes))),
            quorum_required=int(p.get("quorum_required", 1)),
            threshold=float(p.get("threshold", 0.0)),
            votes=votes,
            certificate_hash=str(p.get("certificate_hash", "")),
        )


# =============================================================================
# Blocks
# =============================================================================

@dataclass
class TensorBlock:
    schema_version: str
    chain_id: str
    index: int
    timestamp: float
    transactions: List[TensorTransaction]
    prev_hash: str
    tx_merkle_root: str
    pre_state_root: str
    post_state_root: str
    tensor_context_hash: str
    tensor_commitment_hash: str
    compression_report_hash: str
    transition_receipt: TransitionReceipt
    consensus_certificate: ConsensusCertificate
    block_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(
        *,
        chain_id: str,
        index: int,
        transactions: Sequence[TensorTransaction],
        prev_hash: str,
        pre_state_root: str,
        post_state_root: str,
        transition_receipt: TransitionReceipt,
        consensus_certificate: ConsensusCertificate,
        timestamp: float,
        tensor_commitment_hash: str = "",
        compression_report_hash: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "TensorBlock":
        txs = list(transactions or [])
        tx_root = TransactionMerkleTree.root_for_transactions(txs)
        block = TensorBlock(
            schema_version=BLOCK_SCHEMA,
            chain_id=str(chain_id),
            index=int(index),
            timestamp=float(timestamp),
            transactions=txs,
            prev_hash=str(prev_hash),
            tx_merkle_root=tx_root,
            pre_state_root=str(pre_state_root),
            post_state_root=str(post_state_root),
            tensor_context_hash="",
            tensor_commitment_hash=str(tensor_commitment_hash),
            compression_report_hash=str(compression_report_hash),
            transition_receipt=transition_receipt,
            consensus_certificate=consensus_certificate,
            metadata=dict(metadata or {}),
        )
        block.tensor_context_hash = block.compute_tensor_context_hash()
        block.block_hash = block.compute_hash()
        return block

    def tensor_context_payload(self) -> Dict[str, Any]:
        """Payload used for deterministic tensor core derivation.

        This deliberately excludes tensor_commitment_hash, compression_report_hash,
        transition_receipt_hash, consensus_certificate_hash, and block_hash to
        avoid circular definitions.
        """
        return {
            "schema_version": self.schema_version,
            "chain_id": self.chain_id,
            "index": int(self.index),
            "prev_hash": self.prev_hash,
            "tx_merkle_root": self.tx_merkle_root,
            "pre_state_root": self.pre_state_root,
            "post_state_root": self.post_state_root,
            "tx_hashes": [tx.payload_hash() for tx in self.transactions],
            "metadata_hash": _sha256_hex_obj(self.metadata),
        }

    def compute_tensor_context_hash(self) -> str:
        return _sha256_hex_bytes(_DOMAIN_TENSOR_CONTEXT + _canonical_json_bytes(self.tensor_context_payload()))

    def commitment_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "chain_id": self.chain_id,
            "index": int(self.index),
            "timestamp": float(self.timestamp),
            "tx_hashes": [tx.payload_hash() for tx in self.transactions],
            "prev_hash": self.prev_hash,
            "tx_merkle_root": self.tx_merkle_root,
            "pre_state_root": self.pre_state_root,
            "post_state_root": self.post_state_root,
            "tensor_context_hash": self.tensor_context_hash,
            "tensor_commitment_hash": self.tensor_commitment_hash,
            "compression_report_hash": self.compression_report_hash,
            "transition_receipt_hash": self.transition_receipt.receipt_hash,
            "consensus_certificate_hash": self.consensus_certificate.certificate_hash,
            "metadata": _json_safe(self.metadata),
        }

    def compute_hash(self) -> str:
        return _sha256_hex_bytes(_DOMAIN_BLOCK + _canonical_json_bytes(self.commitment_payload()))

    def validate_hash(self) -> bool:
        return self.block_hash == self.compute_hash()

    def validate_static(self) -> Dict[str, Any]:
        errors: List[str] = []
        if self.tx_merkle_root != TransactionMerkleTree.root_for_transactions(self.transactions):
            errors.append("tx_merkle_root_invalid")
        if self.tensor_context_hash != self.compute_tensor_context_hash():
            errors.append("tensor_context_hash_invalid")
        if not self.transition_receipt.verify():
            errors.append("transition_receipt_invalid")
        if self.transition_receipt.tx_merkle_root != self.tx_merkle_root:
            errors.append("transition_tx_root_mismatch")
        if self.transition_receipt.pre_state_root != self.pre_state_root or self.transition_receipt.post_state_root != self.post_state_root:
            errors.append("transition_state_root_mismatch")
        cert_check = self.consensus_certificate.verify(require_vote_signatures=False)
        if not cert_check["ok"] and self.consensus_certificate.total_votes > 0:
            errors.extend([f"consensus:{x}" for x in cert_check["errors"]])
        if not self.validate_hash():
            errors.append("block_hash_invalid")
        return {"ok": len(errors) == 0, "errors": errors}

    def transaction_proof(self, tx_index: int) -> MerkleProof:
        return TransactionMerkleTree([tx.payload_hash() for tx in self.transactions]).proof(tx_index)

    def summary(self) -> Dict[str, Any]:
        return {
            "index": int(self.index),
            "hash": self.block_hash[:16] + "...",
            "prev_hash": self.prev_hash[:16] + "..." if self.prev_hash else "",
            "tx_count": len(self.transactions),
            "tx_merkle_root": self.tx_merkle_root[:16] + "...",
            "pre_state_root": self.pre_state_root[:16] + "...",
            "post_state_root": self.post_state_root[:16] + "...",
            "tensor_commitment_hash": self.tensor_commitment_hash[:16] + "...",
            "compression_report_hash": self.compression_report_hash[:16] + "...",
            "valid_hash": self.validate_hash(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "chain_id": self.chain_id,
            "index": int(self.index),
            "timestamp": float(self.timestamp),
            "transactions": [tx.to_dict() for tx in self.transactions],
            "prev_hash": self.prev_hash,
            "tx_merkle_root": self.tx_merkle_root,
            "pre_state_root": self.pre_state_root,
            "post_state_root": self.post_state_root,
            "tensor_context_hash": self.tensor_context_hash,
            "tensor_commitment_hash": self.tensor_commitment_hash,
            "compression_report_hash": self.compression_report_hash,
            "transition_receipt": self.transition_receipt.to_dict(),
            "consensus_certificate": self.consensus_certificate.to_dict(),
            "block_hash": self.block_hash,
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TensorBlock":
        p = dict(payload or {})
        tr = TransitionReceipt.from_dict(p.get("transition_receipt", {}))
        cert = ConsensusCertificate.from_dict(p.get("consensus_certificate", {}))
        return cls(
            schema_version=str(p.get("schema_version", BLOCK_SCHEMA)),
            chain_id=str(p.get("chain_id", "")),
            index=int(p.get("index", 0)),
            timestamp=float(p.get("timestamp", 0.0)),
            transactions=[TensorTransaction.from_dict(x) for x in list(p.get("transactions", []))],
            prev_hash=str(p.get("prev_hash", "")),
            tx_merkle_root=str(p.get("tx_merkle_root", "")),
            pre_state_root=str(p.get("pre_state_root", "")),
            post_state_root=str(p.get("post_state_root", "")),
            tensor_context_hash=str(p.get("tensor_context_hash", "")),
            tensor_commitment_hash=str(p.get("tensor_commitment_hash", "")),
            compression_report_hash=str(p.get("compression_report_hash", "")),
            transition_receipt=tr,
            consensus_certificate=cert,
            block_hash=str(p.get("block_hash", "")),
            metadata=dict(p.get("metadata", {})),
        )


# =============================================================================
# Governed execution adapter
# =============================================================================

class GovernedLedgerExecutor:
    """Adapter for policy-gated transaction execution.

    No silent governance bypass is performed. Compatibility errors are surfaced
    to the caller because a production ledger must not silently approve actions.
    """

    def __init__(self, control_plane: dtk.AkkuratInterface):
        self.control_plane = control_plane

    def execute_transactions(self, txs: Sequence[TensorTransaction], *, intent: str = "ledger_transaction_batch") -> Dict[str, Any]:
        network = getattr(self.control_plane, "network", None)
        pre_commitment = network.state_commitment() if network is not None and hasattr(network, "state_commitment") else None
        actions = [tx.action for tx in list(txs or [])]
        report = self.control_plane.governed_actions(actions, intent=intent)
        post_commitment = network.state_commitment() if network is not None and hasattr(network, "state_commitment") else None
        return {
            "intent": intent,
            "approved": bool(report.get("approved", False)),
            "pre_state_commitment": pre_commitment.to_dict() if pre_commitment is not None else None,
            "post_state_commitment": post_commitment.to_dict() if post_commitment is not None else None,
            "governance_report": _json_safe(report),
            "policy_receipt": report.get("policy_receipt"),
            "action_receipts": report.get("execute", {}).get("action_receipts", []) if isinstance(report.get("execute"), Mapping) else [],
        }


# =============================================================================
# Ledger
# =============================================================================

class LedgerMPS:
    """Cryptographic block ledger with deterministic TensorTrain compilation."""

    def __init__(self, cfg: Optional[LedgerConfig] = None):
        self.cfg = (cfg or LedgerConfig()).normalized()
        self.blocks: List[TensorBlock] = []
        self.account_sequences: Dict[str, int] = {}
        self.seen_tx_ids: set[str] = set()
        self._compressed_cores: Optional[List[np.ndarray]] = None
        self._compressed_at_hash: Optional[str] = None
        self._last_compression_report: Dict[str, Any] = {}
        self._create_genesis_block()

    # ------------------------------------------------------------------
    # Genesis / time
    # ------------------------------------------------------------------
    def _block_timestamp(self, index: int, timestamp: Optional[float] = None) -> float:
        if timestamp is not None:
            return float(timestamp)
        if self.cfg.deterministic_blocks:
            return float(self.cfg.genesis_timestamp + float(index) * float(self.cfg.block_time_step_s))
        return float(time.time())

    def _create_genesis_block(self) -> None:
        if self.blocks:
            return
        txs: List[TensorTransaction] = []
        tx_root = TransactionMerkleTree.root_for_transactions(txs)
        receipt = TransitionReceipt.create(
            chain_id=self.cfg.chain_id,
            tx_merkle_root=tx_root,
            pre_state_root="genesis_pre_state",
            post_state_root="genesis_post_state",
            metadata={"type": "genesis"},
        )
        cert = ConsensusCertificate.empty(
            chain_id=self.cfg.chain_id,
            tx_merkle_root=tx_root,
            pre_state_root="genesis_pre_state",
            post_state_root="genesis_post_state",
            transition_receipt_hash=receipt.receipt_hash,
            tensor_commitment_hash="",
        )
        genesis = TensorBlock.create(
            chain_id=self.cfg.chain_id,
            index=0,
            transactions=txs,
            prev_hash="0" * 64,
            pre_state_root="genesis_pre_state",
            post_state_root="genesis_post_state",
            transition_receipt=receipt,
            consensus_certificate=cert,
            timestamp=self._block_timestamp(0),
            metadata={"version": TENSORIZED_BLOCKCHAIN_KERNEL_VERSION, "type": "genesis", "created_at": "deterministic" if self.cfg.deterministic_blocks else _now_iso_utc()},
        )
        # Compute tensor commitment with genesis included, then seal block.
        self.blocks.append(genesis)
        tc = self.get_tensor_train(use_compressed=False).commitment()
        genesis.tensor_commitment_hash = tc.commitment_hash
        genesis.transition_receipt.tensor_commitment_hash = tc.commitment_hash
        genesis.transition_receipt.receipt_hash = genesis.transition_receipt.compute_hash()
        genesis.consensus_certificate.tensor_commitment_hash = tc.commitment_hash
        genesis.consensus_certificate.transition_receipt_hash = genesis.transition_receipt.receipt_hash
        genesis.consensus_certificate.certificate_hash = genesis.consensus_certificate.compute_hash()
        genesis.block_hash = genesis.compute_hash()
        self._rebuild_sequence_index()

    @property
    def head_hash(self) -> str:
        return self.blocks[-1].block_hash if self.blocks else ""

    # ------------------------------------------------------------------
    # Transaction validation / replay protection
    # ------------------------------------------------------------------
    def _transaction_identity_key(self, tx: TensorTransaction) -> str:
        if tx.sender_public_key:
            return f"pub:{tx.sender_public_key}"
        return f"sender:{tx.sender_id}"

    def _validate_transactions_for_append(self, txs: Sequence[TensorTransaction]) -> List[str]:
        errors: List[str] = []
        local_seq = dict(self.account_sequences)
        local_seen = set(self.seen_tx_ids)
        for tx in list(txs or []):
            if tx.chain_id != self.cfg.chain_id:
                errors.append(f"tx_chain_id_mismatch:{tx.tx_id}")
            if tx.tx_id in local_seen:
                errors.append(f"tx_replay_duplicate_id:{tx.tx_id}")
            local_seen.add(tx.tx_id)
            key = self._transaction_identity_key(tx)
            expected = int(local_seq.get(key, 0)) + 1
            if int(tx.sender_sequence) != expected:
                errors.append(f"tx_sequence_invalid:{tx.tx_id}:expected_{expected}_got_{tx.sender_sequence}")
            local_seq[key] = int(tx.sender_sequence)
            scheme = str(tx.signature_scheme or "").lower().strip()
            if self.cfg.require_signatures:
                if scheme not in self.cfg.allowed_signature_schemes:
                    errors.append(f"tx_signature_scheme_not_allowed:{tx.tx_id}:{scheme}")
                if self.cfg.require_public_key_signatures and scheme in {"hmac-demo", "none", "unsigned", ""}:
                    errors.append(f"tx_public_key_signature_required:{tx.tx_id}")
                if not tx.verify_signature():
                    errors.append(f"tx_signature_invalid:{tx.tx_id}")
        return errors

    def _commit_transactions_to_index(self, txs: Sequence[TensorTransaction]) -> None:
        for tx in list(txs or []):
            self.seen_tx_ids.add(tx.tx_id)
            key = self._transaction_identity_key(tx)
            self.account_sequences[key] = max(int(self.account_sequences.get(key, 0)), int(tx.sender_sequence))

    def _rebuild_sequence_index(self) -> None:
        self.account_sequences = {}
        self.seen_tx_ids = set()
        for block in self.blocks:
            for tx in block.transactions:
                self._commit_transactions_to_index([tx])

    # ------------------------------------------------------------------
    # Tensor compilation
    # ------------------------------------------------------------------
    def _bond_weight(self, context_hash: str, left_rank: int, right_rank: int) -> np.ndarray:
        W = np.zeros((int(left_rank), int(right_rank)), dtype=np.float32)
        scale = 1.0 / math.sqrt(float(max(1, min(int(left_rank), int(right_rank)))))
        seed = _stable_hash_u32(f"bond::{self.cfg.seed}::{context_hash}")
        for l in range(int(left_rank)):
            for r in range(int(right_rank)):
                if left_rank == 1 or right_rank == 1 or l == r:
                    h = _stable_hash_u32(f"{seed}:{l}:{r}")
                    W[l, r] = scale * (1.0 if (h & 1) == 0 else -1.0)
        if not np.any(W):
            W[0, 0] = 1.0
        return W

    def _derive_core(self, block: TensorBlock, *, index: int, nblocks: int) -> np.ndarray:
        left_rank = 1 if index == 0 else self.cfg.max_bond_dim
        right_rank = 1 if index == nblocks - 1 else self.cfg.max_bond_dim
        payload = block.tensor_context_payload()
        payload["tensor_context_hash"] = block.compute_tensor_context_hash()
        latent = _deterministic_vector_from_payload(payload, self.cfg.vector_dim)
        mat = latent.reshape(self.cfg.out_dim, self.cfg.in_dim)
        weights = self._bond_weight(block.tensor_context_hash or block.compute_tensor_context_hash(), left_rank, right_rank)
        core = weights[:, None, None, :] * mat[None, :, :, None]
        return core.astype(np.float32, copy=False)

    def _invalidate_compression_cache(self) -> None:
        self._compressed_cores = None
        self._compressed_at_hash = None
        self._last_compression_report = {}

    def compile_cores(self, *, use_compressed: bool = True) -> List[np.ndarray]:
        if use_compressed and self._compressed_cores is not None and self._compressed_at_hash == self.head_hash and len(self._compressed_cores) == len(self.blocks):
            return [np.asarray(c, dtype=np.float32).copy() for c in self._compressed_cores]
        n = len(self.blocks)
        return [self._derive_core(block, index=i, nblocks=n) for i, block in enumerate(self.blocks)]

    def get_tensor_train(self, *, use_compressed: bool = True) -> tn.TensorTrain:
        cores = self.compile_cores(use_compressed=use_compressed)
        dt = np.float32 if self.cfg.compile_dtype == "float32" else np.float64
        tt_cfg = tn.TTConfig(dtype=dt, device="cpu", check_finite=True, seed=self.cfg.seed, deterministic=True, max_dense_entries=self.cfg.dense_max_entries, consensus_safe=self.cfg.consensus_safe_tensors)
        return tn.TensorTrain(
            output_dims=[self.cfg.out_dim for _ in cores],
            input_dims=[self.cfg.in_dim for _ in cores],
            cores_data=cores,
            config=tt_cfg,
            metadata={"source": "LedgerMPS.get_tensor_train", "version": TENSORIZED_BLOCKCHAIN_KERNEL_VERSION},
            commit_metadata={"chain_id": self.cfg.chain_id, "block_count": len(self.blocks), "schema": LEDGER_SCHEMA},
        )

    def tensor_commitment_hash(self) -> str:
        return self.get_tensor_train(use_compressed=False).commitment().commitment_hash

    def compress_ledger(self) -> Dict[str, Any]:
        if len(self.blocks) < 2:
            report = {"status": "skipped", "reason": "chain_too_short", "block_count": len(self.blocks), "report_hash": _sha256_hex_obj({"status": "skipped", "reason": "chain_too_short"})}
            self._last_compression_report = report
            return report
        tt = self.get_tensor_train(use_compressed=False)
        try:
            rounded, report_obj = tt.round_with_report(max_rank=self.cfg.max_bond_dim, energy_tol=self.cfg.energy_tol, seed=self.cfg.seed, exact_error_max_entries=self.cfg.dense_max_entries)
            report = report_obj.to_dict() if hasattr(report_obj, "to_dict") else _json_safe(report_obj)
            self._compressed_cores = [np.asarray(c, dtype=np.float32).copy() for c in rounded.cores_data]
            self._compressed_at_hash = self.head_hash
            report["status"] = report.get("status", "rounded")
            report["block_count"] = len(self.blocks)
            report["report_hash"] = _sha256_hex_obj(report)
            self._last_compression_report = report
            return report
        except Exception as exc:
            report = {"status": "failed", "error": repr(exc), "block_count": len(self.blocks)}
            report["report_hash"] = _sha256_hex_obj(report)
            self._last_compression_report = report
            return report

    # ------------------------------------------------------------------
    # Append blocks
    # ------------------------------------------------------------------
    def _extract_receipts_from_execution(self, execution_report: Optional[Mapping[str, Any]]) -> Tuple[List[Any], List[Any], str, str]:
        if not execution_report:
            return [], [], "", ""
        gov = execution_report.get("governance_report", {}) if isinstance(execution_report, Mapping) else {}
        policy = execution_report.get("policy_receipt") if isinstance(execution_report, Mapping) else None
        action_receipts = execution_report.get("action_receipts", []) if isinstance(execution_report, Mapping) else []
        pre_root = ""
        post_root = ""
        if isinstance(execution_report.get("pre_state_commitment"), Mapping):
            pre_root = str(execution_report["pre_state_commitment"].get("state_root", ""))
        if isinstance(execution_report.get("post_state_commitment"), Mapping):
            post_root = str(execution_report["post_state_commitment"].get("state_root", ""))
        if not pre_root and isinstance(gov, Mapping):
            pre_root = str(gov.get("pre_state_root", ""))
        if not post_root and isinstance(gov, Mapping):
            post_root = str(gov.get("post_state_root", ""))
        policies = [policy] if policy else []
        return policies, list(action_receipts or []), pre_root, post_root

    def append_block(
        self,
        transactions: Sequence[TensorTransaction],
        *,
        pre_state_root: str,
        post_state_root: str,
        policy_receipts: Sequence[Any] = (),
        action_receipts: Sequence[Any] = (),
        consensus_certificate: Optional[ConsensusCertificate] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> TensorBlock:
        txs = list(transactions or [])
        errors = self._validate_transactions_for_append(txs)
        if errors:
            raise ValueError(f"Transaction validation failed: {errors}")
        index = len(self.blocks)
        tx_root = TransactionMerkleTree.root_for_transactions(txs)
        transition = TransitionReceipt.create(
            chain_id=self.cfg.chain_id,
            tx_merkle_root=tx_root,
            pre_state_root=str(pre_state_root),
            post_state_root=str(post_state_root),
            policy_receipts=policy_receipts,
            action_receipts=action_receipts,
            metadata=metadata,
        )
        cert = consensus_certificate or ConsensusCertificate.empty(
            chain_id=self.cfg.chain_id,
            tx_merkle_root=tx_root,
            pre_state_root=str(pre_state_root),
            post_state_root=str(post_state_root),
            transition_receipt_hash=transition.receipt_hash,
            tensor_commitment_hash="",
        )
        block = TensorBlock.create(
            chain_id=self.cfg.chain_id,
            index=index,
            transactions=txs,
            prev_hash=self.head_hash,
            pre_state_root=str(pre_state_root),
            post_state_root=str(post_state_root),
            transition_receipt=transition,
            consensus_certificate=cert,
            timestamp=self._block_timestamp(index, timestamp),
            metadata={"created_at": "deterministic" if self.cfg.deterministic_blocks else _now_iso_utc(), **dict(metadata or {})},
        )
        self.blocks.append(block)
        self._invalidate_compression_cache()
        tensor_hash = self.tensor_commitment_hash()
        compression_report = self.compress_ledger() if self.cfg.compress_on_append else {"status": "not_requested"}
        compression_hash = str(compression_report.get("report_hash") or _sha256_hex_obj(compression_report))
        block.tensor_commitment_hash = tensor_hash
        block.compression_report_hash = compression_hash
        block.transition_receipt.tensor_commitment_hash = tensor_hash
        block.transition_receipt.compression_report_hash = compression_hash
        block.transition_receipt.receipt_hash = block.transition_receipt.compute_hash()
        block.consensus_certificate.transition_receipt_hash = block.transition_receipt.receipt_hash
        block.consensus_certificate.tensor_commitment_hash = tensor_hash
        block.consensus_certificate.certificate_hash = block.consensus_certificate.compute_hash()
        block.block_hash = block.compute_hash()
        self._commit_transactions_to_index(txs)
        return block

    def append_executed_transition(
        self,
        transactions: Sequence[TensorTransaction],
        execution_report: Mapping[str, Any],
        *,
        consensus_certificate: Optional[ConsensusCertificate] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TensorBlock:
        policies, actions, pre, post = self._extract_receipts_from_execution(execution_report)
        if not pre or not post:
            raise ValueError("execution_report must include pre/post state commitments or roots.")
        if not bool(execution_report.get("approved", False)):
            raise ValueError("Cannot append rejected governed transition as an approved block.")
        return self.append_block(transactions, pre_state_root=pre, post_state_root=post, policy_receipts=policies, action_receipts=actions, consensus_certificate=consensus_certificate, metadata=metadata)

    # ------------------------------------------------------------------
    # Validation / persistence
    # ------------------------------------------------------------------
    def validate_chain(self, *, verify_tensor_commitments: bool = True, verify_signatures: Optional[bool] = None) -> Dict[str, Any]:
        errors: List[str] = []
        if not self.blocks:
            return {"ok": False, "errors": ["empty_chain"], "block_count": 0, "head_hash": ""}
        expected_sequences: Dict[str, int] = {}
        seen_tx: set[str] = set()
        for i, block in enumerate(self.blocks):
            if block.index != i:
                errors.append(f"index_mismatch:{i}:{block.index}")
            if block.chain_id != self.cfg.chain_id:
                errors.append(f"chain_id_mismatch:{i}")
            if i == 0:
                if block.prev_hash != "0" * 64:
                    errors.append("genesis_prev_hash_invalid")
            elif block.prev_hash != self.blocks[i - 1].block_hash:
                errors.append(f"prev_hash_invalid:{i}")
            st = block.validate_static()
            if not st["ok"]:
                errors.extend([f"block_{i}:{e}" for e in st["errors"]])
            for tx in block.transactions:
                if tx.tx_id in seen_tx:
                    errors.append(f"tx_replay_duplicate_id:{tx.tx_id}")
                seen_tx.add(tx.tx_id)
                key = self._transaction_identity_key(tx)
                expected = int(expected_sequences.get(key, 0)) + 1
                if int(tx.sender_sequence) != expected:
                    errors.append(f"tx_sequence_invalid:{tx.tx_id}:expected_{expected}_got_{tx.sender_sequence}")
                expected_sequences[key] = int(tx.sender_sequence)
                if tx.chain_id != self.cfg.chain_id:
                    errors.append(f"tx_chain_mismatch:{tx.tx_id}")
                need_sig = self.cfg.require_signatures if verify_signatures is None else bool(verify_signatures)
                if need_sig and not tx.verify_signature():
                    errors.append(f"tx_signature_invalid:{tx.tx_id}")
        if verify_tensor_commitments:
            for i, block in enumerate(self.blocks):
                # Recompute commitment at each prefix by compiling a shallow prefix ledger.
                prefix = copy.copy(self)
                prefix.blocks = [copy.deepcopy(b) for b in self.blocks[: i + 1]]
                prefix._compressed_cores = None
                prefix._compressed_at_hash = None
                try:
                    actual = prefix.get_tensor_train(use_compressed=False).commitment().commitment_hash
                    if actual != block.tensor_commitment_hash:
                        errors.append(f"tensor_commitment_invalid:{i}")
                except Exception as exc:
                    errors.append(f"tensor_commitment_check_failed:{i}:{exc!r}")
        return {"ok": len(errors) == 0, "errors": errors, "block_count": len(self.blocks), "head_hash": self.head_hash}

    def to_dict(self, *, include_compressed_cores: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": LEDGER_SCHEMA,
            "version": TENSORIZED_BLOCKCHAIN_KERNEL_VERSION,
            "cfg": _json_safe(asdict(self.cfg)),
            "blocks": [b.to_dict() for b in self.blocks],
            "account_sequences": dict(self.account_sequences),
            "seen_tx_ids": sorted(self.seen_tx_ids),
            "compressed_at_hash": self._compressed_at_hash,
            "last_compression_report": _json_safe(self._last_compression_report),
        }
        if include_compressed_cores and self._compressed_cores is not None:
            payload["compressed_cores"] = [_json_safe(c) for c in self._compressed_cores]
        payload["ledger_file_hash"] = _sha256_hex_bytes(_DOMAIN_LEDGER_FILE + _canonical_json_bytes({k: v for k, v in payload.items() if k != "ledger_file_hash"}))
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, validate: bool = True) -> "LedgerMPS":
        p = dict(payload or {})
        cfg_payload = dict(p.get("cfg", {}))
        cfg = LedgerConfig(**{k: v for k, v in cfg_payload.items() if k in LedgerConfig.__dataclass_fields__}).normalized()
        obj = cls(cfg)
        blocks = [TensorBlock.from_dict(x) for x in list(p.get("blocks", []))]
        if blocks:
            obj.blocks = blocks
        compressed = p.get("compressed_cores")
        if isinstance(compressed, list) and compressed:
            obj._compressed_cores = [np.asarray(c, dtype=np.float32) for c in compressed]
            obj._compressed_at_hash = str(p.get("compressed_at_hash", obj.head_hash))
        obj._last_compression_report = dict(p.get("last_compression_report", {}))
        obj._rebuild_sequence_index()
        if validate:
            check = obj.validate_chain(verify_tensor_commitments=True, verify_signatures=cfg.require_signatures)
            if not check["ok"]:
                raise ValueError(f"Ledger validation failed on load: {check['errors']}")
        return obj

    def save_json(self, path: Union[str, os.PathLike], *, include_compressed_cores: bool = False) -> None:
        _atomic_write_text(path, json.dumps(self.to_dict(include_compressed_cores=include_compressed_cores), indent=2, ensure_ascii=False))

    @classmethod
    def load_json(cls, path: Union[str, os.PathLike], *, validate: bool = True) -> "LedgerMPS":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), validate=validate)

    @classmethod
    def unsafe_load_json(cls, path: Union[str, os.PathLike]) -> "LedgerMPS":
        return cls.load_json(path, validate=False)

    def summary(self) -> Dict[str, Any]:
        validation = self.validate_chain(verify_tensor_commitments=False, verify_signatures=False)
        return {
            "version": TENSORIZED_BLOCKCHAIN_KERNEL_VERSION,
            "schema_version": LEDGER_SCHEMA,
            "chain_id": self.cfg.chain_id,
            "block_count": len(self.blocks),
            "head_hash": self.head_hash,
            "valid_static": bool(validation["ok"]),
            "cfg": _json_safe(asdict(self.cfg)),
            "compressed": bool(self._compressed_cores is not None and self._compressed_at_hash == self.head_hash),
            "last_compression_report": _json_safe(self._last_compression_report),
            "blocks": [b.summary() for b in self.blocks[-5:]],
        }


# =============================================================================
# Demo / self-test
# =============================================================================

def dependency_status() -> Dict[str, Any]:
    return {
        "version": TENSORIZED_BLOCKCHAIN_KERNEL_VERSION,
        "ts": _now_iso_utc(),
        "paths": configure_paths(),
        "signature_schemes": SignatureEngine.available_schemes(),
        "tn": {
            "module": getattr(tn, "__file__", ""),
            "version": getattr(tn, "TN_VERSION", ""),
            "TensorTrain": hasattr(tn, "TensorTrain"),
            "TensorCommitment": hasattr(tn, "TensorCommitment"),
            "round_with_report": hasattr(getattr(tn, "TensorTrain", object), "round_with_report"),
        },
        "digital_twin_kernel": {
            "module": getattr(dtk, "__file__", ""),
            "version": getattr(dtk, "DIGITAL_TWIN_KERNEL_VERSION", ""),
            "TreeTensorNetwork": hasattr(dtk, "TreeTensorNetwork"),
            "AkkuratInterface": hasattr(dtk, "AkkuratInterface"),
            "StateCommitment": hasattr(dtk, "StateCommitment"),
            "PolicyReceipt": hasattr(dtk, "PolicyReceipt"),
        },
        "twin_anything": {
            "available": ta is not None,
            "module": getattr(ta, "__file__", "") if ta is not None else None,
        },
    }


def _build_demo_twin(cfg: LedgerConfig) -> Dict[str, Any]:
    if ta is not None and hasattr(ta, "TwinAnythingFactory"):
        factory = ta.TwinAnythingFactory(vector_dim=cfg.vector_dim, sketch_dim=min(96, cfg.vector_dim), history_capacity=32, seed=cfg.seed, use_tn_projection=False, deterministic=True)
        result = factory.create_twin_runtime(
            thing_name="Hazard Robot Alpha",
            thing_type="industrial_machine",
            description="Alicante search-and-rescue robot digital product passport.",
            observables={"temperature_c": 22.0, "gas_ppm": 0.0, "battery_pct": 91.0, "status": "nominal"},
            components={"platform": "quadruped", "payloads": ["thermal_camera", "gas_sensor", "lidar"]},
            success_criteria={"minimum_probability": 0.80, "requires_governance_approval": True},
            tags=["wiser", "alicante", "edge_robotics", "digital_product_passport"],
        )
        return {
            "network": result.network,
            "control_plane": result.control_plane,
            "metadata": result.metadata,
            "snapshot": result.snapshot,
            "creation_receipt": result.creation_receipt.to_dict() if hasattr(result.creation_receipt, "to_dict") else None,
        }
    # Fallback when twin_anything.py is not available.
    ttn = dtk.DigitalTwinsBuilder.build_platform(vector_dim=cfg.vector_dim, seed=cfg.seed, sketch_dim=min(96, cfg.vector_dim), history_capacity=32, use_tn_projection=False, deterministic=True, consensus_safe=True)
    cp = dtk.AkkuratInterface(ttn, strict=False)
    return {"network": ttn, "control_plane": cp, "metadata": {"fallback": "digital_twin_kernel"}, "snapshot": {}}


def run_self_test() -> Dict[str, Any]:
    cfg = LedgerConfig(chain_id="self-test-chain", out_dim=4, in_dim=4, max_bond_dim=2, seed=7, require_signatures=True, compress_on_append=True).normalized()
    keys = SignatureEngine.generate_ed25519_keypair()
    twin_data = _build_demo_twin(cfg)
    ttn = twin_data["network"]
    cp = twin_data["control_plane"]
    ledger = LedgerMPS(cfg)

    tx1 = TensorTransaction.new(
        sender_id="self_test_robot",
        sender_public_key=keys.public_key,
        sender_sequence=1,
        chain_id=cfg.chain_id,
        twin_node_id="5.4",
        action=dtk.Action("update_node_data", node_id="5.4", payload={"temperature_c": 44.0, "status": "warm"}, note="self_test_telemetry"),
        timestamp=1.0,
        nonce="self-test-1",
    )
    tx1.sign(keys.private_key)
    executor = GovernedLedgerExecutor(cp)
    exec1 = executor.execute_transactions([tx1], intent="self_test_tx1")
    if not exec1["approved"]:
        raise AssertionError(f"governance rejected tx1: {exec1}")
    b1 = ledger.append_executed_transition([tx1], exec1, metadata={"intent": "self_test_tx1"})

    tx2 = TensorTransaction.new(
        sender_id="self_test_robot",
        sender_public_key=keys.public_key,
        sender_sequence=2,
        chain_id=cfg.chain_id,
        twin_node_id="5.4",
        action=dtk.Action("update_node_data", node_id="5.4", payload={"temperature_c": 51.0, "status": "hot"}, note="self_test_telemetry_2"),
        timestamp=2.0,
        nonce="self-test-2",
    )
    tx2.sign(keys.private_key)
    exec2 = executor.execute_transactions([tx2], intent="self_test_tx2")
    if not exec2["approved"]:
        raise AssertionError(f"governance rejected tx2: {exec2}")
    b2 = ledger.append_executed_transition([tx2], exec2, metadata={"intent": "self_test_tx2"})

    validation = ledger.validate_chain(verify_tensor_commitments=True, verify_signatures=True)
    if not validation["ok"]:
        raise AssertionError(f"chain validation failed: {validation}")
    tt = ledger.get_tensor_train()
    y = tt.apply(np.ones((tt.input_dim,), dtype=np.float32))
    if np.asarray(y).shape != (tt.output_dim,):
        raise AssertionError("TensorTrain output shape mismatch")
    proof = b2.transaction_proof(0)
    if not proof.verify():
        raise AssertionError("Merkle proof failed")

    # Persistence round-trip with strict validation.
    tmp = Path(os.getenv("TB_SELFTEST_LEDGER", "./_tb_selftest_ledger.json"))
    ledger.save_json(tmp)
    loaded = LedgerMPS.load_json(tmp, validate=True)
    try:
        tmp.unlink()
    except Exception:
        pass
    if loaded.head_hash != ledger.head_hash:
        raise AssertionError("ledger round-trip head hash mismatch")

    return {
        "ok": True,
        "blocks": [b1.summary(), b2.summary()],
        "validation": validation,
        "tensor_train": tt.describe() if hasattr(tt, "describe") else {},
        "last_compression_report": ledger._last_compression_report,
        "merkle_proof_verified": proof.verify(),
        "head_hash": ledger.head_hash,
        "dependency_status": dependency_status(),
    }


def run_wiser_demo(*, output: Optional[Union[str, os.PathLike]] = None) -> Dict[str, Any]:
    cfg = LedgerConfig(chain_id="alicante-wiser-demo", out_dim=8, in_dim=8, max_bond_dim=4, seed=2027, require_signatures=True, compress_on_append=True).normalized()
    keys = SignatureEngine.generate_ed25519_keypair()
    twin_data = _build_demo_twin(cfg)
    cp = twin_data["control_plane"]
    ttn = twin_data["network"]
    ledger = LedgerMPS(cfg)
    executor = GovernedLedgerExecutor(cp)

    txs: List[TensorTransaction] = []
    scenarios = [
        (1, {"temperature_c": 450.0, "gas_ppm": 80.0, "status": "fire_zone"}, "fire_zone_entry"),
        (2, {"temperature_c": 390.0, "gas_ppm": 65.0, "status": "egress"}, "fire_zone_egress"),
    ]
    blocks: List[TensorBlock] = []
    executions: List[Dict[str, Any]] = []
    for seq, payload, note in scenarios:
        tx = TensorTransaction.new(
            sender_id="Alicante_Robot_01",
            sender_public_key=keys.public_key,
            sender_sequence=seq,
            chain_id=cfg.chain_id,
            twin_node_id="5.4",
            action=dtk.Action("update_node_data", node_id="5.4", payload=payload, note=note),
            timestamp=float(seq),
            nonce=f"demo-{seq}",
            metadata={"scenario": "alicante_fire_dept", "passport_node": "5.0"},
        )
        tx.sign(keys.private_key)
        exec_report = executor.execute_transactions([tx], intent=note)
        executions.append(exec_report)
        block = ledger.append_executed_transition([tx], exec_report, metadata={"scenario": note})
        txs.append(tx)
        blocks.append(block)

    validation = ledger.validate_chain(verify_tensor_commitments=True, verify_signatures=True)
    report: Dict[str, Any] = {
        "ok": bool(validation["ok"]),
        "demo": "ALICANTE_FIRE_DEPT_WISER_TENSORIZED_BLOCKCHAIN_V2",
        "ledger": ledger.summary(),
        "blocks": [b.summary() for b in blocks],
        "execution": executions,
        "validation": validation,
        "current_state_commitment": ttn.state_commitment().to_dict() if hasattr(ttn, "state_commitment") else None,
        "dependency_status": dependency_status(),
    }
    if output:
        ledger.save_json(output, include_compressed_cores=False)
        report["saved_ledger"] = str(Path(output).expanduser().resolve())
    return report


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Project Chimera / Akkurat Tensorized Blockchain Kernel")
    p.add_argument("--mode", choices=["status", "self-test", "demo", "keygen"], default="demo")
    p.add_argument("--output", default="", help="Optional output JSON path for demo ledger.")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        if args.mode == "status":
            payload = dependency_status()
        elif args.mode == "self-test":
            payload = run_self_test()
        elif args.mode == "keygen":
            payload = {"ok": True, "keypair": _json_safe(SignatureEngine.generate_ed25519_keypair())}
        else:
            payload = run_wiser_demo(output=args.output or None)
        print(json.dumps(_json_safe(payload), indent=2 if args.pretty else None, ensure_ascii=False))
        return 0 if bool(payload.get("ok", True)) else 2
    except Exception as exc:
        err = {"ok": False, "error": repr(exc), "mode": args.mode, "dependency_status": dependency_status()}
        print(json.dumps(_json_safe(err), indent=2, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
