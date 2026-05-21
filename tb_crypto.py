#!/usr/bin/env python3
# tb_crypto.py
r"""
Project Chimera / Akkurat - Tensorized Blockchain Crypto Layer
==============================================================

Purpose
-------
Public-key identity, signing, and verification utilities for the Tensorized
Blockchain stack.

This module is intentionally narrow:
- Ed25519 is the production signing scheme.
- HMAC is deliberately not implemented as a transaction-signing fallback.
- If the cryptographic backend is unavailable, signing and verification fail
  explicitly with CryptoUnavailableError.
- secp256k1 is reserved for a future adapter and is not silently emulated.

Responsibilities
----------------
- Ed25519 key generation, loading, saving, signing, and verification.
- Deterministic demo/test key generation from 32-byte seeds.
- Public-key fingerprints and identity envelopes.
- Transaction payload signatures.
- Validator vote signatures.
- Canonical payload hashing through tb_canonical/tb_hashing when available.
- Dependency-light operation with stdlib fallbacks for canonical JSON and SHA-256.

Security model
--------------
Ed25519 signs a domain-separated canonical preimage:

    SHA256(canonical_payload) is computed first.
    The signed message is a canonical signature preimage containing:
        - signature schema
        - domain
        - payload hash
        - optional context

This prevents the same signature from being replayed across domains such as
transactions, validator votes, lobe receipts, and checkpoint manifests.

The module stores private keys only when callers explicitly request it. Demo JSON
and PEM export helpers are provided for local development; production deployments
should use OS/HSM/secret-manager backed key storage.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import dataclasses
import hashlib
import json
import os
import secrets
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union


# =============================================================================
# Optional canonical/hash helpers from the shared stack
# =============================================================================

try:  # Prefer the shared serializer if present.
    from tb_canonical import canonical_bytes as _stack_canonical_bytes  # type: ignore
    from tb_canonical import canonical_json_dumps as _stack_canonical_json_dumps  # type: ignore
    from tb_canonical import json_safe as _stack_json_safe  # type: ignore
except Exception:  # pragma: no cover - fallback path
    _stack_canonical_bytes = None
    _stack_canonical_json_dumps = None
    _stack_json_safe = None

try:  # Prefer shared domain-separated hashing if present.
    from tb_hashing import hash_bytes as _stack_hash_bytes  # type: ignore
    from tb_hashing import hash_payload as _stack_hash_payload  # type: ignore
except Exception:  # pragma: no cover - fallback path
    _stack_hash_bytes = None
    _stack_hash_payload = None


# =============================================================================
# Optional cryptography backend
# =============================================================================

_CRYPTOGRAPHY_OK = False
_CRYPTOGRAPHY_ERROR = ""

try:
    from cryptography.exceptions import InvalidSignature  # type: ignore
    from cryptography.hazmat.primitives import serialization  # type: ignore
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # type: ignore
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    _CRYPTOGRAPHY_OK = True
except Exception as exc:  # pragma: no cover - depends on runtime env
    InvalidSignature = Exception  # type: ignore
    serialization = None  # type: ignore
    Ed25519PrivateKey = None  # type: ignore
    Ed25519PublicKey = None  # type: ignore
    _CRYPTOGRAPHY_OK = False
    _CRYPTOGRAPHY_ERROR = repr(exc)


# =============================================================================
# Constants
# =============================================================================

TB_CRYPTO_VERSION = "tb-crypto-v1.0-production"
DEFAULT_SCHEME = "ed25519"

SIG_DOMAIN_TRANSACTION = "tb.transaction.v1"
SIG_DOMAIN_VALIDATOR_VOTE = "tb.validator_vote.v1"
SIG_DOMAIN_CONSENSUS_CERTIFICATE = "tb.consensus_certificate.v1"
SIG_DOMAIN_LOBE_RECEIPT = "tb.lobe_receipt.v1"
SIG_DOMAIN_CHECKPOINT = "tb.checkpoint.v1"
SIG_DOMAIN_GENERIC = "tb.generic.v1"

_SIGNATURE_PREIMAGE_SCHEMA = "tb.signature_preimage.v1"
_SIGNATURE_ENVELOPE_SCHEMA = "tb.signature_envelope.v1"
_PUBLIC_IDENTITY_SCHEMA = "tb.public_identity.v1"
_VALIDATOR_VOTE_SCHEMA = "tb.validator_vote_payload.v1"
_KEYPAIR_JSON_SCHEMA = "tb.ed25519_keypair_json.v1"

_RAW_ED25519_PRIVATE_BYTES = 32
_RAW_ED25519_PUBLIC_BYTES = 32


# =============================================================================
# Errors
# =============================================================================

class CryptoError(Exception):
    """Base error for tb_crypto."""


class CryptoUnavailableError(CryptoError):
    """Raised when required public-key crypto backend is unavailable."""


class KeyFormatError(CryptoError):
    """Raised for malformed key material."""


class SignatureVerificationError(CryptoError):
    """Raised when explicit strict verification fails."""


class UnsupportedSignatureScheme(CryptoError):
    """Raised when a requested signature scheme is unsupported."""


# =============================================================================
# Canonical serialization and hashing fallbacks
# =============================================================================

def _json_safe(obj: Any) -> Any:
    if _stack_json_safe is not None:
        return _stack_json_safe(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else str(obj)
    if isinstance(obj, bytes):
        return {"__bytes_hex__": obj.hex()}
    if is_dataclass(obj):
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


def canonical_json_dumps(obj: Any) -> str:
    if _stack_canonical_json_dumps is not None:
        return _stack_canonical_json_dumps(obj)
    return json.dumps(_json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_bytes(obj: Any) -> bytes:
    if _stack_canonical_bytes is not None:
        return _stack_canonical_bytes(obj)
    return canonical_json_dumps(obj).encode("utf-8", errors="strict")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def hash_bytes(data: bytes, *, domain: str = "tb.crypto.bytes.v1") -> str:
    if _stack_hash_bytes is not None:
        try:
            return str(_stack_hash_bytes(data, domain=domain))
        except TypeError:
            return str(_stack_hash_bytes(data))
    prefix = f"{domain}\x00".encode("utf-8", errors="strict")
    return sha256_hex(prefix + bytes(data))


def hash_payload(payload: Any, *, domain: str = "tb.crypto.payload.v1") -> str:
    if _stack_hash_payload is not None:
        try:
            return str(_stack_hash_payload(payload, domain=domain))
        except TypeError:
            return str(_stack_hash_payload(payload))
    return hash_bytes(canonical_bytes(payload), domain=domain)


# =============================================================================
# Byte/encoding helpers
# =============================================================================

def _normalize_scheme(scheme: str) -> str:
    s = str(scheme or DEFAULT_SCHEME).lower().replace("-", "").replace("_", "").strip()
    if s in {"ed25519", "eddsa25519"}:
        return DEFAULT_SCHEME
    if s in {"secp256k1", "ecdsa_secp256k1", "ecdsa"}:
        raise UnsupportedSignatureScheme("secp256k1 adapter is reserved for a future release; use ed25519 now.")
    raise UnsupportedSignatureScheme(f"Unsupported signature scheme: {scheme!r}")


def _strip_hex_prefix(value: str) -> str:
    s = str(value or "").strip()
    if s.startswith("0x") or s.startswith("0X"):
        s = s[2:]
    return s.lower()


def _bytes_from_hex(value: Union[str, bytes], *, expected_len: Optional[int] = None, name: str = "hex") -> bytes:
    if isinstance(value, bytes):
        raw = bytes(value)
    else:
        s = _strip_hex_prefix(str(value))
        try:
            raw = bytes.fromhex(s)
        except ValueError as exc:
            raise KeyFormatError(f"{name} is not valid hexadecimal.") from exc
    if expected_len is not None and len(raw) != int(expected_len):
        raise KeyFormatError(f"{name} must be {expected_len} bytes, got {len(raw)} bytes.")
    return raw


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(bytes(data)).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    s = str(value).strip()
    pad = "=" * ((4 - len(s) % 4) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii"))


def _require_crypto_backend() -> None:
    if not _CRYPTOGRAPHY_OK:
        raise CryptoUnavailableError(
            "Ed25519 signing requires the 'cryptography' package. "
            f"Import error: {_CRYPTOGRAPHY_ERROR or 'unknown'}"
        )


def _atomic_write_bytes(path: Union[str, os.PathLike[str]], data: bytes, *, overwrite: bool = False) -> None:
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(bytes(data))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def _atomic_write_text(path: Union[str, os.PathLike[str]], text: str, *, overwrite: bool = False) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"), overwrite=overwrite)


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass(frozen=True)
class PublicIdentity:
    """Public signing identity suitable for ledgers, validators, and transactions."""

    scheme: str
    public_key_hex: str
    fingerprint: str
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema: str = _PUBLIC_IDENTITY_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "scheme": _normalize_scheme(self.scheme),
            "public_key_hex": _strip_hex_prefix(self.public_key_hex),
            "fingerprint": str(self.fingerprint),
            "label": str(self.label),
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_public_key(cls, public_key_hex: str, *, label: str = "", metadata: Optional[Mapping[str, Any]] = None) -> "PublicIdentity":
        pub = normalize_public_key_hex(public_key_hex)
        return cls(
            scheme=DEFAULT_SCHEME,
            public_key_hex=pub,
            fingerprint=key_fingerprint(pub),
            label=str(label),
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class SignatureEnvelope:
    """Domain-separated signature envelope for a canonical payload hash."""

    scheme: str
    signer_public_key_hex: str
    key_fingerprint: str
    domain: str
    payload_hash: str
    signature_hex: str
    context_hash: str = ""
    schema: str = _SIGNATURE_ENVELOPE_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "scheme": _normalize_scheme(self.scheme),
            "signer_public_key_hex": normalize_public_key_hex(self.signer_public_key_hex),
            "key_fingerprint": str(self.key_fingerprint),
            "domain": str(self.domain),
            "payload_hash": _strip_hex_prefix(self.payload_hash),
            "context_hash": _strip_hex_prefix(self.context_hash),
            "signature_hex": _strip_hex_prefix(self.signature_hex),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SignatureEnvelope":
        p = dict(payload)
        return cls(
            scheme=str(p.get("scheme", DEFAULT_SCHEME)),
            signer_public_key_hex=str(p.get("signer_public_key_hex", p.get("public_key_hex", ""))),
            key_fingerprint=str(p.get("key_fingerprint", p.get("fingerprint", ""))),
            domain=str(p.get("domain", SIG_DOMAIN_GENERIC)),
            payload_hash=str(p.get("payload_hash", "")),
            context_hash=str(p.get("context_hash", "")),
            signature_hex=str(p.get("signature_hex", p.get("signature", ""))),
            schema=str(p.get("schema", _SIGNATURE_ENVELOPE_SCHEMA)),
        )


@dataclass(frozen=True)
class Ed25519KeyPair:
    """Raw Ed25519 keypair represented as hex strings.

    private_key_hex contains the 32-byte Ed25519 seed/private scalar material
    accepted by cryptography.Ed25519PrivateKey.from_private_bytes(...).
    Treat this field as secret.
    """

    private_key_hex: str
    public_key_hex: str
    fingerprint: str
    scheme: str = DEFAULT_SCHEME
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _normalize_scheme(self.scheme)
        priv = normalize_private_key_hex(self.private_key_hex)
        pub = normalize_public_key_hex(self.public_key_hex)
        derived = derive_public_key_hex(priv)
        if derived != pub:
            raise KeyFormatError("private_key_hex does not derive the supplied public_key_hex.")
        fp = key_fingerprint(pub)
        if self.fingerprint and self.fingerprint != fp:
            raise KeyFormatError("fingerprint does not match public_key_hex.")
        object.__setattr__(self, "private_key_hex", priv)
        object.__setattr__(self, "public_key_hex", pub)
        object.__setattr__(self, "fingerprint", fp)
        object.__setattr__(self, "scheme", DEFAULT_SCHEME)

    def public_identity(self) -> PublicIdentity:
        return PublicIdentity(
            scheme=DEFAULT_SCHEME,
            public_key_hex=self.public_key_hex,
            fingerprint=self.fingerprint,
            label=self.label,
            metadata=dict(self.metadata),
        )

    def sign_payload(
        self,
        payload: Any,
        *,
        domain: str = SIG_DOMAIN_GENERIC,
        context: Optional[Any] = None,
    ) -> SignatureEnvelope:
        return sign_payload(payload, self.private_key_hex, domain=domain, context=context, public_key_hex=self.public_key_hex)

    def to_private_json(self, *, include_private: bool = True) -> Dict[str, Any]:
        out = {
            "schema": _KEYPAIR_JSON_SCHEMA,
            "scheme": DEFAULT_SCHEME,
            "public_key_hex": self.public_key_hex,
            "fingerprint": self.fingerprint,
            "label": self.label,
            "metadata": _json_safe(self.metadata),
        }
        if include_private:
            out["private_key_hex"] = self.private_key_hex
        return out

    @classmethod
    def from_private_json(cls, payload: Mapping[str, Any]) -> "Ed25519KeyPair":
        p = dict(payload)
        priv = str(p.get("private_key_hex", ""))
        pub = str(p.get("public_key_hex", "")) or derive_public_key_hex(priv)
        return cls(
            private_key_hex=priv,
            public_key_hex=pub,
            fingerprint=str(p.get("fingerprint", "")) or key_fingerprint(pub),
            label=str(p.get("label", "")),
            metadata=dict(p.get("metadata", {})),
        )


@dataclass(frozen=True)
class ValidatorVote:
    """Serializable validator vote payload. Signature is handled separately."""

    validator_id: str
    block_hash: str
    decision: str
    pre_state_root: str = ""
    post_state_root: str = ""
    tx_merkle_root: str = ""
    tensor_commitment_hash: str = ""
    policy_receipt_hash: str = ""
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema: str = _VALIDATOR_VOTE_SCHEMA

    def payload(self) -> Dict[str, Any]:
        decision = str(self.decision).lower().strip()
        if decision not in {"approve", "reject", "abstain"}:
            decision = "abstain"
        return {
            "schema": self.schema,
            "validator_id": str(self.validator_id),
            "block_hash": _strip_hex_prefix(self.block_hash),
            "decision": decision,
            "pre_state_root": _strip_hex_prefix(self.pre_state_root),
            "post_state_root": _strip_hex_prefix(self.post_state_root),
            "tx_merkle_root": _strip_hex_prefix(self.tx_merkle_root),
            "tensor_commitment_hash": _strip_hex_prefix(self.tensor_commitment_hash),
            "policy_receipt_hash": _strip_hex_prefix(self.policy_receipt_hash),
            "reason": str(self.reason),
            "metadata": _json_safe(self.metadata),
        }

    def sign(self, private_key_hex: str, *, public_key_hex: Optional[str] = None) -> Dict[str, Any]:
        sig = sign_payload(self.payload(), private_key_hex, domain=SIG_DOMAIN_VALIDATOR_VOTE, public_key_hex=public_key_hex)
        return {"vote": self.payload(), "signature": sig.to_dict()}

    @staticmethod
    def verify(signed_vote: Mapping[str, Any], *, strict: bool = False) -> bool:
        p = dict(signed_vote)
        ok = verify_payload_signature(p.get("vote", {}), p.get("signature", {}), domain=SIG_DOMAIN_VALIDATOR_VOTE, strict=strict)
        return bool(ok)


# =============================================================================
# Key normalization and generation
# =============================================================================

def normalize_private_key_hex(private_key_hex: Union[str, bytes]) -> str:
    raw = _bytes_from_hex(private_key_hex, expected_len=_RAW_ED25519_PRIVATE_BYTES, name="ed25519 private key")
    return raw.hex()


def normalize_public_key_hex(public_key_hex: Union[str, bytes]) -> str:
    raw = _bytes_from_hex(public_key_hex, expected_len=_RAW_ED25519_PUBLIC_BYTES, name="ed25519 public key")
    return raw.hex()


def private_key_from_hex(private_key_hex: Union[str, bytes]) -> Any:
    _require_crypto_backend()
    raw = _bytes_from_hex(private_key_hex, expected_len=_RAW_ED25519_PRIVATE_BYTES, name="ed25519 private key")
    return Ed25519PrivateKey.from_private_bytes(raw)  # type: ignore[union-attr]


def public_key_from_hex(public_key_hex: Union[str, bytes]) -> Any:
    _require_crypto_backend()
    raw = _bytes_from_hex(public_key_hex, expected_len=_RAW_ED25519_PUBLIC_BYTES, name="ed25519 public key")
    return Ed25519PublicKey.from_public_bytes(raw)  # type: ignore[union-attr]


def derive_public_key_hex(private_key_hex: Union[str, bytes]) -> str:
    key = private_key_from_hex(private_key_hex)
    pub = key.public_key().public_bytes(  # type: ignore[union-attr]
        encoding=serialization.Encoding.Raw,  # type: ignore[union-attr]
        format=serialization.PublicFormat.Raw,  # type: ignore[union-attr]
    )
    return pub.hex()


def key_fingerprint(public_key_hex: Union[str, bytes], *, prefix: str = DEFAULT_SCHEME, length: int = 32) -> str:
    pub = _bytes_from_hex(public_key_hex, expected_len=_RAW_ED25519_PUBLIC_BYTES, name="ed25519 public key")
    h = hash_bytes(pub, domain="tb.crypto.ed25519.public_key.v1")
    n = int(max(8, min(64, length)))
    return f"{prefix}:{h[:n]}"


def generate_ed25519_keypair(
    *,
    seed: Optional[Union[str, bytes, int]] = None,
    label: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Ed25519KeyPair:
    """Generate an Ed25519 keypair.

    If seed is provided, generation is deterministic and intended for tests/demo.
    Accepted seed forms:
    - 32-byte bytes
    - 64-hex-character string
    - arbitrary string, hashed to 32 bytes
    - integer, converted through canonical string hashing
    """
    _require_crypto_backend()
    if seed is None:
        raw = secrets.token_bytes(_RAW_ED25519_PRIVATE_BYTES)
    elif isinstance(seed, bytes):
        raw = seed if len(seed) == _RAW_ED25519_PRIVATE_BYTES else hashlib.sha256(seed).digest()
    elif isinstance(seed, int):
        raw = hashlib.sha256(f"tb-demo-seed:{seed}".encode("utf-8")).digest()
    else:
        s = str(seed).strip()
        if len(_strip_hex_prefix(s)) == 64:
            try:
                raw = bytes.fromhex(_strip_hex_prefix(s))
            except Exception:
                raw = hashlib.sha256(s.encode("utf-8")).digest()
        else:
            raw = hashlib.sha256(s.encode("utf-8")).digest()

    private_hex = raw.hex()
    public_hex = derive_public_key_hex(private_hex)
    return Ed25519KeyPair(
        private_key_hex=private_hex,
        public_key_hex=public_hex,
        fingerprint=key_fingerprint(public_hex),
        label=str(label),
        metadata=dict(metadata or {}),
    )


# =============================================================================
# Signature preimages/signing/verification
# =============================================================================

def payload_hash_for_signature(payload: Any, *, domain: str) -> str:
    return hash_payload(payload, domain=f"{domain}.payload")


def context_hash(context: Optional[Any]) -> str:
    if context is None:
        return ""
    return hash_payload(context, domain="tb.crypto.signature_context.v1")


def signature_preimage(
    *,
    payload_hash: str,
    domain: str,
    context: Optional[Any] = None,
) -> bytes:
    ch = context_hash(context)
    preimage = {
        "schema": _SIGNATURE_PREIMAGE_SCHEMA,
        "domain": str(domain),
        "payload_hash": _strip_hex_prefix(payload_hash),
        "context_hash": _strip_hex_prefix(ch),
    }
    return canonical_bytes(preimage)


def sign_bytes(message: bytes, private_key_hex: Union[str, bytes]) -> str:
    key = private_key_from_hex(private_key_hex)
    sig = key.sign(bytes(message))  # type: ignore[union-attr]
    return sig.hex()


def verify_bytes_signature(message: bytes, signature_hex: Union[str, bytes], public_key_hex: Union[str, bytes], *, strict: bool = False) -> bool:
    try:
        pub = public_key_from_hex(public_key_hex)
        sig = _bytes_from_hex(signature_hex, expected_len=64, name="ed25519 signature")
        pub.verify(sig, bytes(message))  # type: ignore[union-attr]
        return True
    except Exception as exc:
        if strict:
            raise SignatureVerificationError(f"Signature verification failed: {exc}") from exc
        return False


def sign_payload(
    payload: Any,
    private_key_hex: Union[str, bytes],
    *,
    domain: str = SIG_DOMAIN_GENERIC,
    context: Optional[Any] = None,
    public_key_hex: Optional[str] = None,
) -> SignatureEnvelope:
    scheme = DEFAULT_SCHEME
    priv = normalize_private_key_hex(private_key_hex)
    pub = normalize_public_key_hex(public_key_hex or derive_public_key_hex(priv))
    derived = derive_public_key_hex(priv)
    if pub != derived:
        raise KeyFormatError("public_key_hex does not match private_key_hex.")
    ph = payload_hash_for_signature(payload, domain=domain)
    pre = signature_preimage(payload_hash=ph, domain=domain, context=context)
    sig = sign_bytes(pre, priv)
    return SignatureEnvelope(
        scheme=scheme,
        signer_public_key_hex=pub,
        key_fingerprint=key_fingerprint(pub),
        domain=str(domain),
        payload_hash=ph,
        context_hash=context_hash(context),
        signature_hex=sig,
    )


def verify_payload_signature(
    payload: Any,
    signature: Union[SignatureEnvelope, Mapping[str, Any]],
    *,
    domain: Optional[str] = None,
    context: Optional[Any] = None,
    strict: bool = False,
) -> bool:
    try:
        env = signature if isinstance(signature, SignatureEnvelope) else SignatureEnvelope.from_mapping(signature)
        _normalize_scheme(env.scheme)
        if domain is not None and str(env.domain) != str(domain):
            raise SignatureVerificationError(f"signature domain mismatch: {env.domain!r} != {domain!r}")
        expected_ph = payload_hash_for_signature(payload, domain=env.domain)
        if _strip_hex_prefix(expected_ph) != _strip_hex_prefix(env.payload_hash):
            raise SignatureVerificationError("payload hash mismatch")
        expected_ch = context_hash(context)
        if _strip_hex_prefix(expected_ch) != _strip_hex_prefix(env.context_hash):
            raise SignatureVerificationError("context hash mismatch")
        pub = normalize_public_key_hex(env.signer_public_key_hex)
        fp = key_fingerprint(pub)
        if env.key_fingerprint and env.key_fingerprint != fp:
            raise SignatureVerificationError("key fingerprint mismatch")
        pre = signature_preimage(payload_hash=env.payload_hash, domain=env.domain, context=context)
        return verify_bytes_signature(pre, env.signature_hex, pub, strict=True)
    except Exception as exc:
        if strict:
            if isinstance(exc, SignatureVerificationError):
                raise
            raise SignatureVerificationError(f"payload signature verification failed: {exc}") from exc
        return False


# =============================================================================
# Transaction/vote convenience functions
# =============================================================================

def sign_transaction_payload(
    transaction_payload: Any,
    private_key_hex: Union[str, bytes],
    *,
    context: Optional[Any] = None,
    public_key_hex: Optional[str] = None,
) -> SignatureEnvelope:
    return sign_payload(
        transaction_payload,
        private_key_hex,
        domain=SIG_DOMAIN_TRANSACTION,
        context=context,
        public_key_hex=public_key_hex,
    )


def verify_transaction_signature(
    transaction_payload: Any,
    signature: Union[SignatureEnvelope, Mapping[str, Any]],
    *,
    context: Optional[Any] = None,
    strict: bool = False,
) -> bool:
    return verify_payload_signature(
        transaction_payload,
        signature,
        domain=SIG_DOMAIN_TRANSACTION,
        context=context,
        strict=strict,
    )


def sign_validator_vote(vote_payload: Any, private_key_hex: Union[str, bytes], *, public_key_hex: Optional[str] = None) -> SignatureEnvelope:
    return sign_payload(vote_payload, private_key_hex, domain=SIG_DOMAIN_VALIDATOR_VOTE, public_key_hex=public_key_hex)


def verify_validator_vote_signature(vote_payload: Any, signature: Union[SignatureEnvelope, Mapping[str, Any]], *, strict: bool = False) -> bool:
    return verify_payload_signature(vote_payload, signature, domain=SIG_DOMAIN_VALIDATOR_VOTE, strict=strict)


# =============================================================================
# PEM/JSON key IO
# =============================================================================

def private_key_to_pem(private_key_hex: Union[str, bytes], *, passphrase: Optional[Union[str, bytes]] = None) -> bytes:
    _require_crypto_backend()
    key = private_key_from_hex(private_key_hex)
    if passphrase is None:
        enc = serialization.NoEncryption()  # type: ignore[union-attr]
    else:
        pw = passphrase.encode("utf-8") if isinstance(passphrase, str) else bytes(passphrase)
        enc = serialization.BestAvailableEncryption(pw)  # type: ignore[union-attr]
    return key.private_bytes(  # type: ignore[union-attr]
        encoding=serialization.Encoding.PEM,  # type: ignore[union-attr]
        format=serialization.PrivateFormat.PKCS8,  # type: ignore[union-attr]
        encryption_algorithm=enc,
    )


def public_key_to_pem(public_key_hex: Union[str, bytes]) -> bytes:
    _require_crypto_backend()
    pub = public_key_from_hex(public_key_hex)
    return pub.public_bytes(  # type: ignore[union-attr]
        encoding=serialization.Encoding.PEM,  # type: ignore[union-attr]
        format=serialization.PublicFormat.SubjectPublicKeyInfo,  # type: ignore[union-attr]
    )


def private_key_hex_from_pem(pem_data: Union[str, bytes], *, passphrase: Optional[Union[str, bytes]] = None) -> str:
    _require_crypto_backend()
    data = pem_data.encode("utf-8") if isinstance(pem_data, str) else bytes(pem_data)
    pw = passphrase.encode("utf-8") if isinstance(passphrase, str) else passphrase
    key = serialization.load_pem_private_key(data, password=pw)  # type: ignore[union-attr]
    if not isinstance(key, Ed25519PrivateKey):  # type: ignore[arg-type]
        raise KeyFormatError("PEM private key is not Ed25519.")
    raw = key.private_bytes(  # type: ignore[union-attr]
        encoding=serialization.Encoding.Raw,  # type: ignore[union-attr]
        format=serialization.PrivateFormat.Raw,  # type: ignore[union-attr]
        encryption_algorithm=serialization.NoEncryption(),  # type: ignore[union-attr]
    )
    return raw.hex()


def public_key_hex_from_pem(pem_data: Union[str, bytes]) -> str:
    _require_crypto_backend()
    data = pem_data.encode("utf-8") if isinstance(pem_data, str) else bytes(pem_data)
    key = serialization.load_pem_public_key(data)  # type: ignore[union-attr]
    if not isinstance(key, Ed25519PublicKey):  # type: ignore[arg-type]
        raise KeyFormatError("PEM public key is not Ed25519.")
    raw = key.public_bytes(  # type: ignore[union-attr]
        encoding=serialization.Encoding.Raw,  # type: ignore[union-attr]
        format=serialization.PublicFormat.Raw,  # type: ignore[union-attr]
    )
    return raw.hex()


def save_private_key_pem(
    path: Union[str, os.PathLike[str]],
    private_key_hex: Union[str, bytes],
    *,
    passphrase: Optional[Union[str, bytes]] = None,
    overwrite: bool = False,
) -> None:
    _atomic_write_bytes(path, private_key_to_pem(private_key_hex, passphrase=passphrase), overwrite=overwrite)


def load_private_key_pem(path: Union[str, os.PathLike[str]], *, passphrase: Optional[Union[str, bytes]] = None) -> str:
    return private_key_hex_from_pem(Path(path).read_bytes(), passphrase=passphrase)


def save_public_key_pem(path: Union[str, os.PathLike[str]], public_key_hex: Union[str, bytes], *, overwrite: bool = False) -> None:
    _atomic_write_bytes(path, public_key_to_pem(public_key_hex), overwrite=overwrite)


def load_public_key_pem(path: Union[str, os.PathLike[str]]) -> str:
    return public_key_hex_from_pem(Path(path).read_bytes())


def save_keypair_json(
    path: Union[str, os.PathLike[str]],
    keypair: Ed25519KeyPair,
    *,
    include_private: bool = True,
    overwrite: bool = False,
) -> None:
    _atomic_write_text(path, json.dumps(keypair.to_private_json(include_private=include_private), indent=2, ensure_ascii=False), overwrite=overwrite)


def load_keypair_json(path: Union[str, os.PathLike[str]]) -> Ed25519KeyPair:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Ed25519KeyPair.from_private_json(data)


# =============================================================================
# Status and self-test
# =============================================================================

def crypto_status() -> Dict[str, Any]:
    return {
        "ok": bool(_CRYPTOGRAPHY_OK),
        "version": TB_CRYPTO_VERSION,
        "default_scheme": DEFAULT_SCHEME,
        "cryptography_available": bool(_CRYPTOGRAPHY_OK),
        "cryptography_error": _CRYPTOGRAPHY_ERROR,
        "domains": {
            "transaction": SIG_DOMAIN_TRANSACTION,
            "validator_vote": SIG_DOMAIN_VALIDATOR_VOTE,
            "consensus_certificate": SIG_DOMAIN_CONSENSUS_CERTIFICATE,
            "lobe_receipt": SIG_DOMAIN_LOBE_RECEIPT,
            "checkpoint": SIG_DOMAIN_CHECKPOINT,
        },
    }


def run_self_test() -> Dict[str, Any]:
    _require_crypto_backend()

    kp1 = generate_ed25519_keypair(seed="deterministic-validator-1", label="validator_1")
    kp2 = generate_ed25519_keypair(seed="deterministic-validator-1", label="validator_1_again")
    if kp1.private_key_hex != kp2.private_key_hex or kp1.public_key_hex != kp2.public_key_hex:
        raise AssertionError("deterministic key generation failed")

    tx = {
        "chain_id": "tb-demo-chain",
        "sender_sequence": 1,
        "action": {"kind": "update_node_data", "node_id": "5.4", "payload": {"temperature_c": 42.0}},
    }
    sig = sign_transaction_payload(tx, kp1.private_key_hex)
    if not verify_transaction_signature(tx, sig):
        raise AssertionError("transaction signature verification failed")
    tampered = dict(tx)
    tampered["sender_sequence"] = 2
    if verify_transaction_signature(tampered, sig):
        raise AssertionError("tampered transaction verified unexpectedly")

    vote = ValidatorVote(
        validator_id="validator_1",
        block_hash="a" * 64,
        decision="approve",
        pre_state_root="b" * 64,
        post_state_root="c" * 64,
        tx_merkle_root="d" * 64,
        tensor_commitment_hash="e" * 64,
    )
    signed_vote = vote.sign(kp1.private_key_hex)
    if not ValidatorVote.verify(signed_vote):
        raise AssertionError("validator vote verification failed")

    # PEM round-trip.
    priv_pem = private_key_to_pem(kp1.private_key_hex)
    pub_pem = public_key_to_pem(kp1.public_key_hex)
    if private_key_hex_from_pem(priv_pem) != kp1.private_key_hex:
        raise AssertionError("private PEM round-trip failed")
    if public_key_hex_from_pem(pub_pem) != kp1.public_key_hex:
        raise AssertionError("public PEM round-trip failed")

    return {
        "ok": True,
        "status": crypto_status(),
        "public_identity": kp1.public_identity().to_dict(),
        "transaction_signature": sig.to_dict(),
        "signed_vote_ok": True,
    }


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tensorized Blockchain public-key crypto utilities")
    p.add_argument("--mode", choices=["status", "self-test", "generate-demo-key", "fingerprint"], default="status")
    p.add_argument("--seed", default="", help="Optional deterministic demo seed.")
    p.add_argument("--label", default="", help="Optional label for generated demo key.")
    p.add_argument("--public-key-hex", default="", help="Public key for fingerprint mode.")
    p.add_argument("--output-json", default="", help="Optional keypair JSON output path for generate-demo-key.")
    p.add_argument("--public-pem", default="", help="Optional public PEM output path for generate-demo-key.")
    p.add_argument("--private-pem", default="", help="Optional private PEM output path for generate-demo-key.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output files.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        if args.mode == "status":
            out = crypto_status()
        elif args.mode == "self-test":
            out = run_self_test()
        elif args.mode == "fingerprint":
            if not args.public_key_hex:
                raise ValueError("--public-key-hex is required for fingerprint mode")
            out = {
                "ok": True,
                "public_key_hex": normalize_public_key_hex(args.public_key_hex),
                "fingerprint": key_fingerprint(args.public_key_hex),
            }
        elif args.mode == "generate-demo-key":
            seed: Optional[Union[str, bytes, int]] = args.seed or None
            kp = generate_ed25519_keypair(seed=seed, label=args.label, metadata={"generated_by": TB_CRYPTO_VERSION})
            out = kp.to_private_json(include_private=True)
            out["warning"] = "Demo/private key material. Do not commit production secrets."
            if args.output_json:
                save_keypair_json(args.output_json, kp, include_private=True, overwrite=bool(args.overwrite))
                out["saved_keypair_json"] = str(Path(args.output_json).resolve())
            if args.public_pem:
                save_public_key_pem(args.public_pem, kp.public_key_hex, overwrite=bool(args.overwrite))
                out["saved_public_pem"] = str(Path(args.public_pem).resolve())
            if args.private_pem:
                save_private_key_pem(args.private_pem, kp.private_key_hex, overwrite=bool(args.overwrite))
                out["saved_private_pem"] = str(Path(args.private_pem).resolve())
        else:  # pragma: no cover
            raise ValueError(f"unknown mode: {args.mode}")
        print(json.dumps(_json_safe(out), indent=2, ensure_ascii=False))
        return 0 if bool(out.get("ok", True)) else 2
    except Exception as exc:
        err = {"ok": False, "error": repr(exc), "status": crypto_status()}
        print(json.dumps(_json_safe(err), indent=2, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
