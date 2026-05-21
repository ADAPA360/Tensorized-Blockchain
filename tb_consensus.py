#!/usr/bin/env python3
# tb_consensus.py
r"""
Project Chimera / Akkurat - Tensorized Blockchain Consensus Layer
=================================================================

Production transition-based consensus and validator-certificate logic for the
Tensorized Blockchain stack.

Role
----
This module upgrades the older snapshot-only Proof-of-Coherence pattern into a
transition-based validator workflow. Validators vote over exact transition
facts:

- chain id
- block height and round
- candidate block hash
- transaction Merkle root
- pre-state root
- post-state root
- policy receipt root/hash
- tensor commitment hash

Design boundaries
-----------------
- This file does not execute digital-twin actions.
- This file does not build blocks or mutate ledgers.
- This file does not perform neural-lobe inference.
- This file only validates transition material, signs validator votes, checks
  quorum, and builds/verifies consensus certificates.

Dependencies
------------
Required project modules when running inside the full stack:

- tb_canonical.py
- tb_hashing.py
- tb_crypto.py
- tb_schemas.py
- tb_validation.py

The module is dependency-light and CPU/deterministic by design.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


# =============================================================================
# Version
# =============================================================================

TB_CONSENSUS_VERSION = "tb-consensus-v1.0-production"
CONSENSUS_VOTE_SIGNING_DOMAIN = "tb.validator_vote.v1"
CONSENSUS_CERTIFICATE_DOMAIN = "tb.consensus_certificate.v1"
VALIDATOR_SET_DOMAIN = "tb.validator_set.v1"
TRANSITION_PROPOSAL_DOMAIN = "tb.transition_proposal.v1"


# =============================================================================
# Shared imports with safe local fallbacks
# =============================================================================

try:  # pragma: no cover - preferred project path
    from tb_canonical import canonical_bytes, canonical_dumps, canonical_loads, canonicalize, json_dumps_pretty
except Exception:  # pragma: no cover - minimal fallback
    def _fallback_safe(obj: Any) -> Any:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if is_dataclass(obj):
            return _fallback_safe(asdict(obj))
        if isinstance(obj, Mapping):
            return {str(k): _fallback_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_fallback_safe(v) for v in obj]
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return _fallback_safe(obj.to_dict())
        return str(obj)

    def canonicalize(obj: Any, *args: Any, **kwargs: Any) -> Any:
        return _fallback_safe(obj)

    def canonical_dumps(obj: Any, *args: Any, **kwargs: Any) -> str:
        return json.dumps(_fallback_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def canonical_bytes(obj: Any, *args: Any, **kwargs: Any) -> bytes:
        return canonical_dumps(obj).encode("utf-8")

    def canonical_loads(text: str, *args: Any, **kwargs: Any) -> Any:
        return json.loads(text)

    def json_dumps_pretty(obj: Any, *args: Any, **kwargs: Any) -> str:
        return json.dumps(_fallback_safe(obj), sort_keys=True, indent=2, ensure_ascii=False)

try:  # pragma: no cover - preferred project path
    from tb_hashing import domain_hash, merkle_root, HashDigest
except Exception:  # pragma: no cover - minimal fallback
    import hashlib

    @dataclass
    class HashDigest:  # type: ignore[no-redef]
        algorithm: str
        digest_hex: str
        domain: str = ""
        tag: str = ""

    def domain_hash(payload: Any, *, domain: str, tag: str = "", algorithm: Optional[str] = None, **_: Any) -> HashDigest:
        pre = (str(domain) + "\x00" + str(tag) + "\x00").encode("utf-8") + canonical_bytes(payload)
        return HashDigest("sha256", hashlib.sha256(pre).hexdigest(), str(domain), str(tag))

    def merkle_root(leaves: Sequence[Any], *, domain: str = "tb.merkle.v1", leaves_are_hashes: bool = False, **_: Any) -> HashDigest:
        if not leaves:
            return domain_hash("<empty>", domain=domain, tag="empty")
        level = []
        for leaf in leaves:
            if leaves_are_hashes:
                level.append(bytes.fromhex(str(leaf)))
            else:
                level.append(bytes.fromhex(domain_hash(leaf, domain=domain, tag="leaf").digest_hex))
        while len(level) > 1:
            if len(level) % 2:
                level.append(level[-1])
            nxt = []
            for i in range(0, len(level), 2):
                nxt.append(bytes.fromhex(domain_hash(level[i].hex() + level[i + 1].hex(), domain=domain, tag="node").digest_hex))
            level = nxt
        return HashDigest("sha256", level[0].hex(), str(domain), "root")

try:  # pragma: no cover - preferred project path
    from tb_schemas import (
        ConsensusCertificate,
        ConsensusVote,
        PolicyReceipt,
        StateCommitment,
        TensorCommitment,
        TransitionReceipt,
    )
except Exception as exc:  # pragma: no cover - production stack should provide schemas
    raise ImportError("tb_consensus.py requires tb_schemas.py in the production stack.") from exc

try:  # pragma: no cover - preferred project path
    import tb_crypto
except Exception:  # pragma: no cover - signature verification will report unavailable
    tb_crypto = None  # type: ignore

try:  # pragma: no cover - preferred project path
    from tb_validation import ValidationIssue, ValidationReport, validate_consensus_certificate, validate_consensus_vote
except Exception:  # pragma: no cover - minimal fallback
    @dataclass
    class ValidationIssue:  # type: ignore[no-redef]
        code: str
        message: str
        severity: str = "error"
        field: str = ""
        path: str = ""
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class ValidationReport:  # type: ignore[no-redef]
        ok: bool = True
        subject: str = ""
        version: str = TB_CONSENSUS_VERSION
        issues: List[ValidationIssue] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)

        def add(self, code: str, message: str, *, severity: str = "error", field: str = "", **metadata: Any) -> None:
            self.issues.append(ValidationIssue(code=code, message=message, severity=severity, field=field, metadata=metadata))
            if severity == "error":
                self.ok = False

        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)

    def validate_consensus_vote(vote: Any, **kwargs: Any) -> ValidationReport:  # type: ignore[no-redef]
        return ValidationReport(ok=True, subject="consensus_vote")

    def validate_consensus_certificate(cert: Any, **kwargs: Any) -> ValidationReport:  # type: ignore[no-redef]
        return ValidationReport(ok=True, subject="consensus_certificate")


# =============================================================================
# Helpers
# =============================================================================

_HEX64 = set("0123456789abcdefABCDEF")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _hash_hex(payload: Any, *, domain: str, tag: str = "") -> str:
    h = domain_hash(payload, domain=domain, tag=tag)
    return str(getattr(h, "digest_hex", h))


def _is_hex_hash(value: Any, *, lengths: Sequence[int] = (64,)) -> bool:
    s = str(value or "")
    return len(s) in set(int(x) for x in lengths) and all(c in _HEX64 for c in s)


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return dict(obj.to_dict())
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {}


def _without_fields(obj: Any, fields_to_remove: Iterable[str]) -> Dict[str, Any]:
    data = _as_dict(obj)
    data.pop("schema_name", None)
    for f in fields_to_remove:
        data.pop(str(f), None)
    return data


def _vote_signing_payload(vote: Union[ConsensusVote, Mapping[str, Any]]) -> Dict[str, Any]:
    return _without_fields(vote, {"signature"})


def _certificate_hash_payload(cert: Union[ConsensusCertificate, Mapping[str, Any]]) -> Dict[str, Any]:
    return _without_fields(cert, {"certificate_hash"})


def vote_hash(vote: Union[ConsensusVote, Mapping[str, Any]]) -> str:
    """Hash a consensus vote excluding the signature field."""
    return _hash_hex(_vote_signing_payload(vote), domain=CONSENSUS_VOTE_SIGNING_DOMAIN, tag="vote")


def certificate_hash(cert: Union[ConsensusCertificate, Mapping[str, Any]]) -> str:
    """Hash a consensus certificate excluding certificate_hash itself.

    Prefer the shared schema implementation when present so this module remains
    byte-compatible with tb_schemas.py and tb_validation.py.
    """
    try:
        obj = _coerce_certificate(cert)
        fn = getattr(obj, "hash_without_self_reference", None)
        if callable(fn):
            return str(fn())
    except Exception:
        pass
    return _hash_hex(_certificate_hash_payload(cert), domain=CONSENSUS_CERTIFICATE_DOMAIN, tag="certificate")


def validator_set_hash(validators: Sequence[Union["ValidatorIdentity", Mapping[str, Any]]]) -> str:
    payload = []
    for v in validators:
        d = v.to_dict(include_private=False) if isinstance(v, ValidatorIdentity) else dict(v)
        d.pop("private_key_hex", None)
        payload.append(d)
    payload.sort(key=lambda x: (str(x.get("validator_id", "")), str(x.get("public_key_hex", ""))))
    return _hash_hex(payload, domain=VALIDATOR_SET_DOMAIN, tag="validator_set")


def _coerce_vote(vote: Union[ConsensusVote, Mapping[str, Any]]) -> ConsensusVote:
    if isinstance(vote, ConsensusVote):
        return vote
    return ConsensusVote.from_dict(dict(vote))


def _coerce_certificate(cert: Union[ConsensusCertificate, Mapping[str, Any]]) -> ConsensusCertificate:
    if isinstance(cert, ConsensusCertificate):
        return cert
    return ConsensusCertificate.from_dict(dict(cert))


def _root_from_obj(obj: Any) -> str:
    """Extract a root/hash from a schema object or raw string."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    d = _as_dict(obj)
    for k in ("state_root", "commitment_hash", "report_hash", "payload_hash", "hash", "root"):
        v = d.get(k)
        if v:
            return str(v)
    if hasattr(obj, "payload_hash") and callable(obj.payload_hash):
        try:
            return str(obj.payload_hash())
        except Exception:
            pass
    return _hash_hex(d, domain="tb.consensus.object_root.v1")


def _metadata_set(obj: Any, key: str, value: Any) -> None:
    try:
        md = getattr(obj, "metadata", None)
        if isinstance(md, dict):
            md[key] = value
    except Exception:
        pass


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class ConsensusConfig:
    """Runtime configuration for transition consensus."""

    chain_id: str
    quorum_ratio: float = 2.0 / 3.0
    min_approvals: int = 1
    round: int = 0
    require_signatures: bool = True
    require_known_validators: bool = True
    require_policy_root_match: bool = True
    require_tensor_commitment_match: bool = False
    require_transition_receipt: bool = False
    min_transition_score: float = 1.0
    allow_abstain_votes: bool = False
    max_vote_age_s: Optional[int] = None
    deterministic: bool = True
    consensus_safe: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "ConsensusConfig":
        c = copy.deepcopy(self)
        c.chain_id = str(c.chain_id)
        c.quorum_ratio = float(max(0.0, min(1.0, c.quorum_ratio)))
        if c.quorum_ratio <= 0.0:
            c.quorum_ratio = 2.0 / 3.0
        c.min_approvals = int(max(1, c.min_approvals))
        c.round = int(max(0, c.round))
        c.require_signatures = bool(c.require_signatures)
        c.require_known_validators = bool(c.require_known_validators)
        c.require_policy_root_match = bool(c.require_policy_root_match)
        c.require_tensor_commitment_match = bool(c.require_tensor_commitment_match)
        c.require_transition_receipt = bool(c.require_transition_receipt)
        c.min_transition_score = float(max(0.0, min(1.0, c.min_transition_score)))
        c.allow_abstain_votes = bool(c.allow_abstain_votes)
        if c.max_vote_age_s is not None:
            c.max_vote_age_s = int(max(1, c.max_vote_age_s))
        c.deterministic = bool(c.deterministic)
        c.consensus_safe = bool(c.consensus_safe)
        c.metadata = dict(c.metadata or {})
        return c

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self.normalized())


@dataclass
class ValidatorIdentity:
    """Public validator identity used for vote verification and quorum weighting."""

    validator_id: str
    public_key_hex: str
    weight: int = 1
    active: bool = True
    label: str = ""
    private_key_hex: str = ""  # optional for local/demo signing only; omitted from public hash.
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "tb.validator_identity.v1"

    def __post_init__(self) -> None:
        self.validator_id = str(self.validator_id)
        self.public_key_hex = str(self.public_key_hex)
        self.private_key_hex = str(self.private_key_hex or "")
        self.weight = int(max(0, self.weight))
        self.active = bool(self.active)
        self.label = str(self.label or "")
        self.metadata = dict(self.metadata or {})

    def to_dict(self, *, include_private: bool = False) -> Dict[str, Any]:
        data = asdict(self)
        if not include_private:
            data.pop("private_key_hex", None)
        return canonicalize(data)

    @property
    def fingerprint(self) -> str:
        if tb_crypto is not None and hasattr(tb_crypto, "key_fingerprint"):
            try:
                return str(tb_crypto.key_fingerprint(self.public_key_hex))
            except Exception:
                pass
        return _hash_hex(self.public_key_hex, domain="tb.validator_key_fingerprint.v1")[:32]


@dataclass
class ValidatorSet:
    """Deterministic weighted validator set."""

    validators: List[ValidatorIdentity] = field(default_factory=list)
    set_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "tb.validator_set.v1"

    def __post_init__(self) -> None:
        self.validators = [v if isinstance(v, ValidatorIdentity) else ValidatorIdentity(**dict(v)) for v in self.validators]
        self.validators.sort(key=lambda v: (v.validator_id, v.public_key_hex))
        self.metadata = dict(self.metadata or {})
        if not self.set_id:
            self.set_id = self.hash()

    def active_validators(self) -> List[ValidatorIdentity]:
        return [v for v in self.validators if v.active and v.weight > 0]

    def by_id(self) -> Dict[str, ValidatorIdentity]:
        return {v.validator_id: v for v in self.active_validators()}

    def by_public_key(self) -> Dict[str, ValidatorIdentity]:
        return {v.public_key_hex.lower(): v for v in self.active_validators()}

    def total_weight(self) -> int:
        return int(sum(v.weight for v in self.active_validators()))

    def quorum_weight(self, quorum_ratio: float, *, min_approvals: int = 1) -> int:
        total = self.total_weight()
        q = int(math.ceil(float(total) * float(quorum_ratio))) if total > 0 else int(min_approvals)
        return int(max(int(min_approvals), q))

    def hash(self) -> str:
        return validator_set_hash(self.validators)

    def to_dict(self, *, include_private: bool = False) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "set_id": self.set_id or self.hash(),
            "validator_set_hash": self.hash(),
            "total_weight": self.total_weight(),
            "validators": [v.to_dict(include_private=include_private) for v in self.validators],
            "metadata": canonicalize(self.metadata),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ValidatorSet":
        return cls(
            validators=[ValidatorIdentity(**dict(x)) for x in list(payload.get("validators", []))],
            set_id=str(payload.get("set_id", "")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class TransitionProposal:
    """Canonical proposal over which validators vote."""

    chain_id: str
    height: int
    round: int
    block_hash: str
    tx_merkle_root: str
    pre_state_root: str
    post_state_root: str
    policy_receipt_root: str = ""
    tensor_commitment_hash: str = ""
    transition_receipt_hash: str = ""
    proposer_id: str = ""
    created_at: str = field(default_factory=_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "tb.transition_proposal.v1"

    def __post_init__(self) -> None:
        self.chain_id = str(self.chain_id)
        self.height = int(self.height)
        self.round = int(max(0, self.round))
        self.block_hash = str(self.block_hash)
        self.tx_merkle_root = str(self.tx_merkle_root)
        self.pre_state_root = str(self.pre_state_root)
        self.post_state_root = str(self.post_state_root)
        self.policy_receipt_root = str(self.policy_receipt_root or "")
        self.tensor_commitment_hash = str(self.tensor_commitment_hash or "")
        self.transition_receipt_hash = str(self.transition_receipt_hash or "")
        self.proposer_id = str(self.proposer_id or "")
        self.created_at = str(self.created_at or _now_iso())
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> Dict[str, Any]:
        return canonicalize(asdict(self))

    def hash(self) -> str:
        return _hash_hex(self.to_dict(), domain=TRANSITION_PROPOSAL_DOMAIN, tag="proposal")

    @classmethod
    def from_transition_receipt(
        cls,
        *,
        chain_id: str,
        height: int,
        round: int,
        block_hash: str,
        tx_merkle_root: str,
        receipt: TransitionReceipt,
        policy_receipt_root: str = "",
        tensor_commitment_hash: str = "",
        proposer_id: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "TransitionProposal":
        return cls(
            chain_id=chain_id,
            height=height,
            round=round,
            block_hash=block_hash,
            tx_merkle_root=tx_merkle_root,
            pre_state_root=str(receipt.pre_state_root),
            post_state_root=str(receipt.post_state_root),
            policy_receipt_root=policy_receipt_root or str(receipt.policy_receipt_hash or ""),
            tensor_commitment_hash=tensor_commitment_hash or str(receipt.tensor_commitment_hash or ""),
            transition_receipt_hash=receipt.payload_hash(),
            proposer_id=proposer_id,
            metadata=dict(metadata or {}),
        )


@dataclass
class TransitionValidationReport:
    ok: bool
    score: float
    proposal_hash: str
    checks: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "tb.transition_validation_report.v1"

    def to_dict(self) -> Dict[str, Any]:
        return canonicalize(asdict(self))

    def hash(self) -> str:
        return _hash_hex(self.to_dict(), domain="tb.transition_validation_report.v1")


@dataclass
class VoteVerificationReport:
    ok: bool
    vote_hash: str
    validator_id: str
    approved: bool
    known_validator: bool = False
    signature_ok: bool = False
    target_ok: bool = False
    duplicate: bool = False
    weight: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return canonicalize(asdict(self))


@dataclass
class CertificateVerificationReport:
    ok: bool
    certificate_hash: str
    approvals: int
    approval_weight: int
    quorum_required: int
    quorum_ratio: float
    total_weight: int
    vote_count: int
    valid_vote_count: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    vote_reports: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return canonicalize(asdict(self))


# =============================================================================
# Signature helpers
# =============================================================================

class ConsensusError(Exception):
    """Base exception for consensus failures."""


class QuorumError(ConsensusError):
    """Raised when a certificate cannot satisfy quorum."""


class VoteSignatureError(ConsensusError):
    """Raised when vote signing or verification fails."""


def sign_vote(vote: ConsensusVote, private_key_hex: str) -> ConsensusVote:
    """Return a copy of *vote* with an Ed25519 signature over the vote payload."""
    if tb_crypto is None:
        raise VoteSignatureError("tb_crypto is unavailable; cannot sign consensus vote.")
    payload = _vote_signing_payload(vote)
    sig = tb_crypto.sign_bytes(canonical_bytes(payload), private_key_hex)
    out = vote.clone(signature=str(sig), signature_scheme="ed25519")
    return out


def verify_vote_signature(vote: Union[ConsensusVote, Mapping[str, Any]], *, strict: bool = False) -> bool:
    """Verify a vote signature over the vote payload."""
    v = _coerce_vote(vote)
    if not v.signature:
        if strict:
            raise VoteSignatureError("vote has no signature")
        return False
    if tb_crypto is None:
        if strict:
            raise VoteSignatureError("tb_crypto is unavailable; cannot verify vote signature")
        return False
    try:
        return bool(
            tb_crypto.verify_bytes_signature(
                canonical_bytes(_vote_signing_payload(v)),
                v.signature,
                v.validator_public_key,
                strict=True,
            )
        )
    except Exception as exc:
        if strict:
            raise VoteSignatureError(f"vote signature verification failed: {exc}") from exc
        return False


# =============================================================================
# Transition validation
# =============================================================================

class TransitionCoherenceValidator:
    """
    Deterministic validator for transition material.

    This class verifies that proposal roots agree with optional receipts and
    commitments. It intentionally checks exact hashes/roots, not fuzzy neural
    similarity. Neural/tensor similarity signals should be committed separately
    as lobe receipts or tensor reports.
    """

    def __init__(self, config: ConsensusConfig):
        self.config = config.normalized()

    def validate(
        self,
        proposal: TransitionProposal,
        *,
        transition_receipt: Optional[TransitionReceipt] = None,
        policy_receipt: Optional[PolicyReceipt] = None,
        pre_state: Optional[Union[StateCommitment, str]] = None,
        post_state: Optional[Union[StateCommitment, str]] = None,
        tensor_commitment: Optional[Union[TensorCommitment, str]] = None,
    ) -> TransitionValidationReport:
        checks: Dict[str, bool] = {}
        errors: List[str] = []
        warnings: List[str] = []

        checks["chain_id"] = bool(proposal.chain_id == self.config.chain_id)
        if not checks["chain_id"]:
            errors.append("chain_id_mismatch")

        for name in ("block_hash", "tx_merkle_root", "pre_state_root", "post_state_root"):
            value = getattr(proposal, name)
            checks[f"{name}_hex"] = _is_hex_hash(value)
            if not checks[f"{name}_hex"]:
                errors.append(f"{name}_invalid")

        if transition_receipt is not None:
            checks["transition_receipt_accepted"] = bool(transition_receipt.accepted)
            checks["transition_receipt_pre_root"] = str(transition_receipt.pre_state_root) == str(proposal.pre_state_root)
            checks["transition_receipt_post_root"] = str(transition_receipt.post_state_root) == str(proposal.post_state_root)
            if proposal.transition_receipt_hash:
                checks["transition_receipt_hash"] = str(transition_receipt.payload_hash()) == str(proposal.transition_receipt_hash)
            else:
                checks["transition_receipt_hash"] = True
            for key in ("transition_receipt_accepted", "transition_receipt_pre_root", "transition_receipt_post_root", "transition_receipt_hash"):
                if not checks[key]:
                    errors.append(key)
        elif self.config.require_transition_receipt:
            checks["transition_receipt_present"] = False
            errors.append("transition_receipt_missing")

        if policy_receipt is not None:
            checks["policy_approved"] = bool(policy_receipt.approved)
            if proposal.policy_receipt_root:
                checks["policy_receipt_root"] = str(policy_receipt.payload_hash()) == str(proposal.policy_receipt_root)
            else:
                checks["policy_receipt_root"] = True
            if policy_receipt.pre_state_root:
                checks["policy_pre_root"] = str(policy_receipt.pre_state_root) == str(proposal.pre_state_root)
            else:
                checks["policy_pre_root"] = True
            if policy_receipt.post_state_root:
                checks["policy_post_root"] = str(policy_receipt.post_state_root) == str(proposal.post_state_root)
            else:
                checks["policy_post_root"] = True
            for key in ("policy_approved", "policy_receipt_root", "policy_pre_root", "policy_post_root"):
                if not checks[key]:
                    errors.append(key)
        elif self.config.require_policy_root_match and proposal.policy_receipt_root:
            warnings.append("policy_receipt_not_provided_for_local_validation")

        if pre_state is not None:
            checks["pre_state_root_match"] = _root_from_obj(pre_state) == str(proposal.pre_state_root)
            if not checks["pre_state_root_match"]:
                errors.append("pre_state_root_mismatch")

        if post_state is not None:
            checks["post_state_root_match"] = _root_from_obj(post_state) == str(proposal.post_state_root)
            if not checks["post_state_root_match"]:
                errors.append("post_state_root_mismatch")

        if tensor_commitment is not None:
            checks["tensor_commitment_match"] = _root_from_obj(tensor_commitment) == str(proposal.tensor_commitment_hash)
            if self.config.require_tensor_commitment_match and not checks["tensor_commitment_match"]:
                errors.append("tensor_commitment_mismatch")
        elif self.config.require_tensor_commitment_match:
            checks["tensor_commitment_present"] = False
            errors.append("tensor_commitment_missing")

        if not checks:
            score = 0.0
        else:
            score = float(sum(1 for x in checks.values() if bool(x)) / max(1, len(checks)))

        ok = bool(len(errors) == 0 and score >= self.config.min_transition_score)
        return TransitionValidationReport(
            ok=ok,
            score=score,
            proposal_hash=proposal.hash(),
            checks=checks,
            errors=errors,
            warnings=warnings,
            metadata={"config": self.config.to_dict()},
        )


# =============================================================================
# Consensus engine
# =============================================================================

class TransitionConsensusEngine:
    """Transition-based quorum certificate engine."""

    def __init__(self, config: ConsensusConfig, validator_set: Optional[ValidatorSet] = None):
        self.config = config.normalized()
        self.validator_set = validator_set or ValidatorSet([])
        self.transition_validator = TransitionCoherenceValidator(self.config)

    @property
    def validator_set_hash(self) -> str:
        return self.validator_set.hash() if self.validator_set.validators else ""

    def make_vote(
        self,
        proposal: TransitionProposal,
        validator: ValidatorIdentity,
        *,
        approved: bool,
        reason: str = "",
        private_key_hex: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ConsensusVote:
        """Create and optionally sign a validator vote for a transition proposal."""
        vote = ConsensusVote(
            chain_id=proposal.chain_id,
            height=int(proposal.height),
            round=int(proposal.round),
            validator_id=validator.validator_id,
            validator_public_key=validator.public_key_hex,
            block_hash=proposal.block_hash,
            tx_merkle_root=proposal.tx_merkle_root,
            pre_state_root=proposal.pre_state_root,
            post_state_root=proposal.post_state_root,
            tensor_commitment_hash=proposal.tensor_commitment_hash,
            policy_receipt_root=proposal.policy_receipt_root,
            approved=bool(approved),
            reason=str(reason or ("approved" if approved else "rejected")),
            timestamp=_now_iso(),
            signature_scheme="ed25519",
            signature="",
            metadata={
                "proposal_hash": proposal.hash(),
                "validator_fingerprint": validator.fingerprint,
                "validator_set_hash": self.validator_set_hash,
                **dict(metadata or {}),
            },
        )
        key = private_key_hex or validator.private_key_hex
        if key:
            vote = sign_vote(vote, key)
        elif self.config.require_signatures:
            raise VoteSignatureError(f"private key missing for validator {validator.validator_id!r}")
        return vote

    def verify_vote(
        self,
        vote: Union[ConsensusVote, Mapping[str, Any]],
        proposal: TransitionProposal,
        *,
        seen_validators: Optional[set] = None,
        require_signature: Optional[bool] = None,
    ) -> VoteVerificationReport:
        v = _coerce_vote(vote)
        errors: List[str] = []
        warnings: List[str] = []

        schema_report = validate_consensus_vote(v, verify_signature=False, require_signature=False)
        if not getattr(schema_report, "ok", True):
            for issue in getattr(schema_report, "issues", []):
                if getattr(issue, "severity", "error") == "error":
                    errors.append(f"schema:{getattr(issue, 'code', 'issue')}")
                else:
                    warnings.append(f"schema:{getattr(issue, 'code', 'issue')}")

        target_fields = (
            "chain_id",
            "height",
            "round",
            "block_hash",
            "tx_merkle_root",
            "pre_state_root",
            "post_state_root",
            "tensor_commitment_hash",
            "policy_receipt_root",
        )
        target_ok = True
        for f in target_fields:
            if str(getattr(v, f)) != str(getattr(proposal, f)):
                target_ok = False
                errors.append(f"target_mismatch:{f}")

        known_validator = False
        weight = 0
        if self.validator_set.validators:
            by_id = self.validator_set.by_id()
            ident = by_id.get(v.validator_id)
            if ident is None:
                errors.append("unknown_validator") if self.config.require_known_validators else warnings.append("unknown_validator")
            else:
                known_validator = True
                weight = int(ident.weight)
                if str(ident.public_key_hex).lower() != str(v.validator_public_key).lower():
                    errors.append("validator_public_key_mismatch")
                    known_validator = False
                    weight = 0
        elif self.config.require_known_validators:
            errors.append("validator_set_empty")
        else:
            weight = 1

        duplicate = False
        if seen_validators is not None:
            if v.validator_id in seen_validators:
                duplicate = True
                errors.append("duplicate_validator_vote")
            else:
                seen_validators.add(v.validator_id)

        need_sig = self.config.require_signatures if require_signature is None else bool(require_signature)
        signature_ok = False
        if v.signature:
            signature_ok = verify_vote_signature(v, strict=False)
            if not signature_ok:
                errors.append("signature_invalid")
        elif need_sig:
            errors.append("signature_missing")
        else:
            warnings.append("signature_absent")

        if not bool(v.approved) and not self.config.allow_abstain_votes:
            # Rejection votes can be valid but do not count for approval quorum.
            pass

        ok = bool(len(errors) == 0)
        return VoteVerificationReport(
            ok=ok,
            vote_hash=vote_hash(v),
            validator_id=v.validator_id,
            approved=bool(v.approved),
            known_validator=known_validator,
            signature_ok=signature_ok,
            target_ok=target_ok,
            duplicate=duplicate,
            weight=weight,
            errors=errors,
            warnings=warnings,
            metadata={"schema_report": schema_report.to_dict() if hasattr(schema_report, "to_dict") else _as_dict(schema_report)},
        )

    def build_certificate(
        self,
        proposal: TransitionProposal,
        votes: Sequence[Union[ConsensusVote, Mapping[str, Any]]],
        *,
        require_quorum: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ConsensusCertificate:
        """Build a consensus certificate from votes. Raises if quorum is required but absent."""
        coerced = [_coerce_vote(v) for v in votes]
        verified: List[ConsensusVote] = []
        reports: List[VoteVerificationReport] = []
        seen: set = set()
        approval_weight = 0
        approvals = 0

        for v in coerced:
            rep = self.verify_vote(v, proposal, seen_validators=seen)
            reports.append(rep)
            if rep.ok:
                verified.append(v)
                if v.approved:
                    approvals += 1
                    approval_weight += int(rep.weight)

        total_weight = self.validator_set.total_weight() if self.validator_set.validators else len(verified)
        quorum_required = self.validator_set.quorum_weight(self.config.quorum_ratio, min_approvals=self.config.min_approvals) if self.validator_set.validators else int(max(self.config.min_approvals, math.ceil(len(coerced) * self.config.quorum_ratio)))
        has_quorum = bool(approval_weight >= quorum_required and approvals >= self.config.min_approvals)

        if require_quorum and not has_quorum:
            raise QuorumError(
                f"quorum not satisfied: approval_weight={approval_weight}, approvals={approvals}, required={quorum_required}"
            )

        # Deterministic vote order: validator id then vote hash.
        verified.sort(key=lambda v: (str(v.validator_id), vote_hash(v)))

        cert = ConsensusCertificate(
            chain_id=proposal.chain_id,
            height=int(proposal.height),
            round=int(proposal.round),
            block_hash=proposal.block_hash,
            tx_merkle_root=proposal.tx_merkle_root,
            pre_state_root=proposal.pre_state_root,
            post_state_root=proposal.post_state_root,
            votes=verified,
            quorum_required=int(quorum_required),
            quorum_ratio=float(self.config.quorum_ratio),
            validator_set_hash=self.validator_set_hash,
            threshold=float(self.config.min_transition_score),
            certificate_hash="",
            created_at=_now_iso(),
            metadata={
                "ok": bool(has_quorum),
                "approvals": int(approvals),
                "approval_weight": int(approval_weight),
                "total_weight": int(total_weight),
                "valid_vote_count": int(len(verified)),
                "vote_merkle_root": merkle_root([vote_hash(v) for v in verified], domain="tb.consensus.votes.v1", leaves_are_hashes=True).digest_hex,
                "proposal_hash": proposal.hash(),
                "reports": [r.to_dict() for r in reports],
                **dict(metadata or {}),
            },
        )
        cert.certificate_hash = certificate_hash(cert)
        return cert

    def verify_certificate(
        self,
        certificate: Union[ConsensusCertificate, Mapping[str, Any]],
        proposal: Optional[TransitionProposal] = None,
        *,
        require_signatures: Optional[bool] = None,
    ) -> CertificateVerificationReport:
        cert = _coerce_certificate(certificate)
        errors: List[str] = []
        warnings: List[str] = []
        seen: set = set()
        vote_reports: List[VoteVerificationReport] = []
        approvals = 0
        approval_weight = 0
        valid_votes = 0

        expected_hash = certificate_hash(cert)
        if cert.certificate_hash and cert.certificate_hash != expected_hash:
            errors.append("certificate_hash_invalid")
        elif not cert.certificate_hash:
            warnings.append("certificate_hash_missing")

        if self.validator_set_hash and cert.validator_set_hash and cert.validator_set_hash != self.validator_set_hash:
            errors.append("validator_set_hash_mismatch")

        if proposal is None:
            proposal = TransitionProposal(
                chain_id=cert.chain_id,
                height=int(cert.height),
                round=int(cert.round),
                block_hash=cert.block_hash,
                tx_merkle_root=cert.tx_merkle_root,
                pre_state_root=cert.pre_state_root,
                post_state_root=cert.post_state_root,
                # Tensor/policy roots are checked at vote level only when present on votes.
                policy_receipt_root=str(cert.votes[0].policy_receipt_root) if cert.votes else "",
                tensor_commitment_hash=str(cert.votes[0].tensor_commitment_hash) if cert.votes else "",
            )
        else:
            for f in ("chain_id", "height", "round", "block_hash", "tx_merkle_root", "pre_state_root", "post_state_root"):
                if str(getattr(cert, f)) != str(getattr(proposal, f)):
                    errors.append(f"certificate_target_mismatch:{f}")

        for v in cert.votes:
            rep = self.verify_vote(
                v,
                proposal,
                seen_validators=seen,
                require_signature=self.config.require_signatures if require_signatures is None else bool(require_signatures),
            )
            vote_reports.append(rep)
            if rep.ok:
                valid_votes += 1
                if v.approved:
                    approvals += 1
                    approval_weight += int(rep.weight)

        total_weight = self.validator_set.total_weight() if self.validator_set.validators else valid_votes
        quorum_required = int(cert.quorum_required or 0)
        if quorum_required <= 0:
            quorum_required = self.validator_set.quorum_weight(self.config.quorum_ratio, min_approvals=self.config.min_approvals) if self.validator_set.validators else int(max(self.config.min_approvals, math.ceil(max(1, len(cert.votes)) * self.config.quorum_ratio)))

        if approval_weight < quorum_required:
            errors.append("quorum_weight_not_met")
        if approvals < self.config.min_approvals:
            errors.append("min_approvals_not_met")

        schema_report = validate_consensus_certificate(cert, verify_vote_signatures=False, require_vote_signatures=False)
        if not getattr(schema_report, "ok", True):
            for issue in getattr(schema_report, "issues", []):
                if getattr(issue, "severity", "error") == "error":
                    errors.append(f"schema:{getattr(issue, 'code', 'issue')}")
                else:
                    warnings.append(f"schema:{getattr(issue, 'code', 'issue')}")

        ok = bool(len(errors) == 0)
        return CertificateVerificationReport(
            ok=ok,
            certificate_hash=cert.certificate_hash or expected_hash,
            approvals=int(approvals),
            approval_weight=int(approval_weight),
            quorum_required=int(quorum_required),
            quorum_ratio=float(cert.quorum_ratio or self.config.quorum_ratio),
            total_weight=int(total_weight),
            vote_count=len(cert.votes),
            valid_vote_count=int(valid_votes),
            errors=errors,
            warnings=warnings,
            vote_reports=[r.to_dict() for r in vote_reports],
            metadata={"schema_report": schema_report.to_dict() if hasattr(schema_report, "to_dict") else _as_dict(schema_report)},
        )

    def validate_transition(self, proposal: TransitionProposal, **kwargs: Any) -> TransitionValidationReport:
        return self.transition_validator.validate(proposal, **kwargs)


# =============================================================================
# Convenience API
# =============================================================================


def create_consensus_vote(
    proposal: TransitionProposal,
    validator: ValidatorIdentity,
    *,
    private_key_hex: Optional[str] = None,
    approved: bool = True,
    reason: str = "approved",
    config: Optional[ConsensusConfig] = None,
    validator_set: Optional[ValidatorSet] = None,
) -> ConsensusVote:
    cfg = config or ConsensusConfig(chain_id=proposal.chain_id, require_signatures=bool(private_key_hex or validator.private_key_hex))
    return TransitionConsensusEngine(cfg, validator_set).make_vote(
        proposal,
        validator,
        approved=approved,
        reason=reason,
        private_key_hex=private_key_hex,
    )


def create_consensus_certificate(
    proposal: TransitionProposal,
    votes: Sequence[Union[ConsensusVote, Mapping[str, Any]]],
    *,
    validator_set: Optional[ValidatorSet] = None,
    config: Optional[ConsensusConfig] = None,
    require_quorum: bool = True,
) -> ConsensusCertificate:
    cfg = config or ConsensusConfig(chain_id=proposal.chain_id)
    return TransitionConsensusEngine(cfg, validator_set).build_certificate(proposal, votes, require_quorum=require_quorum)


def verify_consensus_certificate(
    certificate: Union[ConsensusCertificate, Mapping[str, Any]],
    *,
    proposal: Optional[TransitionProposal] = None,
    validator_set: Optional[ValidatorSet] = None,
    config: Optional[ConsensusConfig] = None,
) -> CertificateVerificationReport:
    cert = _coerce_certificate(certificate)
    cfg = config or ConsensusConfig(chain_id=cert.chain_id)
    return TransitionConsensusEngine(cfg, validator_set).verify_certificate(cert, proposal)


def transition_proposal_from_block_payload(block: Mapping[str, Any], *, round: int = 0, proposer_id: str = "") -> TransitionProposal:
    """Best-effort adapter from a block-like mapping into a TransitionProposal."""
    p = dict(block)
    return TransitionProposal(
        chain_id=str(p.get("chain_id", p.get("metadata", {}).get("chain_id", ""))),
        height=int(p.get("height", p.get("index", 0))),
        round=int(p.get("round", round)),
        block_hash=str(p.get("block_hash", p.get("hash", ""))),
        tx_merkle_root=str(p.get("tx_merkle_root", p.get("merkle_root", ""))),
        pre_state_root=str(p.get("pre_state_root", "")),
        post_state_root=str(p.get("post_state_root", p.get("world_root", ""))),
        policy_receipt_root=str(p.get("policy_receipt_root", "")),
        tensor_commitment_hash=str(p.get("tensor_commitment_hash", "")),
        transition_receipt_hash=str(p.get("transition_receipt_hash", "")),
        proposer_id=str(p.get("proposer_id", proposer_id)),
        metadata={"source": "transition_proposal_from_block_payload"},
    )


# =============================================================================
# Demo keys and self-test
# =============================================================================


def make_demo_validator_set(n: int = 4, *, chain_id: str = "demo-chain") -> Tuple[ValidatorSet, Dict[str, str]]:
    """Create deterministic demo validators. Returns (validator_set, private_keys_by_id)."""
    if tb_crypto is None:
        raise RuntimeError("tb_crypto unavailable; demo validator generation requires tb_crypto.py")
    validators: List[ValidatorIdentity] = []
    privs: Dict[str, str] = {}
    for i in range(int(max(1, n))):
        kp = tb_crypto.generate_ed25519_keypair(seed=f"{chain_id}:validator:{i}", label=f"validator_{i}")
        vid = f"validator_{i}"
        validators.append(
            ValidatorIdentity(
                validator_id=vid,
                public_key_hex=kp.public_key_hex,
                private_key_hex=kp.private_key_hex,
                weight=1,
                active=True,
                label=kp.label,
            )
        )
        privs[vid] = kp.private_key_hex
    return ValidatorSet(validators), privs


def run_self_test() -> Dict[str, Any]:
    chain_id = "tb-consensus-self-test"
    vset, privs = make_demo_validator_set(4, chain_id=chain_id)
    cfg = ConsensusConfig(chain_id=chain_id, quorum_ratio=0.66, min_approvals=3, require_signatures=True)
    engine = TransitionConsensusEngine(cfg, vset)

    proposal = TransitionProposal(
        chain_id=chain_id,
        height=1,
        round=0,
        block_hash=_hash_hex({"block": 1}, domain="selftest.block"),
        tx_merkle_root=merkle_root(["txa", "txb"], domain="selftest.tx").digest_hex,
        pre_state_root=_hash_hex({"state": "pre"}, domain="selftest.state"),
        post_state_root=_hash_hex({"state": "post"}, domain="selftest.state"),
        policy_receipt_root=_hash_hex({"approved": True}, domain="selftest.policy"),
        tensor_commitment_hash=_hash_hex({"tensor": "ok"}, domain="selftest.tensor"),
        proposer_id="selftest_proposer",
    )

    votes = []
    for validator in vset.active_validators()[:3]:
        votes.append(engine.make_vote(proposal, validator, approved=True, private_key_hex=privs[validator.validator_id]))

    cert = engine.build_certificate(proposal, votes)
    report = engine.verify_certificate(cert, proposal)
    if not report.ok:
        raise AssertionError(f"certificate should verify: {report.to_dict()}")

    # Tampering probe: altered vote should fail signature or target verification.
    tampered = votes[0].clone(post_state_root=_hash_hex({"state": "evil"}, domain="selftest.state"))
    tamper_report = engine.verify_vote(tampered, proposal)
    if tamper_report.ok:
        raise AssertionError("tampered vote unexpectedly verified")

    # Quorum failure probe: only two approvals when three required.
    try:
        engine.build_certificate(proposal, votes[:2], require_quorum=True)
        raise AssertionError("quorum failure was not raised")
    except QuorumError:
        pass

    transition_report = engine.validate_transition(proposal)
    if not transition_report.ok:
        raise AssertionError(f"transition should validate: {transition_report.to_dict()}")

    return {
        "ok": True,
        "version": TB_CONSENSUS_VERSION,
        "validator_set_hash": vset.hash(),
        "proposal_hash": proposal.hash(),
        "certificate_hash": cert.certificate_hash,
        "verification": report.to_dict(),
        "tamper_detected": not tamper_report.ok,
        "transition_validation": transition_report.to_dict(),
    }


# =============================================================================
# CLI
# =============================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tensorized Blockchain transition consensus utilities")
    p.add_argument("--mode", choices=["self-test", "demo-validator-set"], default="self-test")
    p.add_argument("--validators", type=int, default=4)
    p.add_argument("--chain-id", default="demo-chain")
    p.add_argument("--output", default="")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        if args.mode == "self-test":
            out = run_self_test()
        else:
            vset, privs = make_demo_validator_set(args.validators, chain_id=args.chain_id)
            out = {
                "ok": True,
                "version": TB_CONSENSUS_VERSION,
                "validator_set": vset.to_dict(include_private=False),
                "private_keys_by_id": privs,
                "warning": "Demo private keys are for local testing only.",
            }
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json_dumps_pretty(out), encoding="utf-8")
        print(json_dumps_pretty(out))
        return 0 if bool(out.get("ok", False)) else 2
    except Exception as exc:
        err = {"ok": False, "version": TB_CONSENSUS_VERSION, "error": repr(exc)}
        print(json_dumps_pretty(err))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
