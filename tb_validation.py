#!/usr/bin/env python3
# tb_validation.py
r"""
Project Chimera / Akkurat - Tensorized Blockchain Validation Layer
==================================================================

Purpose
-------
Runtime validators and invariant checks for the Tensorized Blockchain stack.

This module is intentionally dependency-light and import-safe. It is designed
for use by the ledger, twin, tensor, lobe, API, demo, and persistence layers
without causing circular imports.

Responsibilities
----------------
- Validate transaction envelopes and optional signatures.
- Validate block structures and block commitments.
- Validate state commitments, tensor commitments, compression reports, and
  consensus certificates.
- Validate lobe receipts and transition receipts.
- Validate TreeTensorNetwork-like topology integrity.
- Validate loaded ledgers: block hashes, chain linkage, duplicate transaction
  detection, and sender sequence monotonicity.
- Provide structured validation reports rather than scattered exceptions.

Design notes
------------
Validation is deliberately split into two layers:

1. Schema validation
   Calls the schema object's own validate() method when available.

2. Invariant validation
   Checks cross-object properties: no duplicate transactions, monotonic sender
   sequence numbers, block linkage, Merkle roots, quorum consistency, topology
   reachability, and commitment self-hashes.

The validators accept dataclasses, mappings, and runtime objects exposing
``to_dict()``. This makes the module compatible with generated stack scripts
while keeping it independent of their concrete implementations.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import hmac
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, field as dc_field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Type, Union


# =============================================================================
# Optional shared project imports
# =============================================================================

try:  # Prefer generated schema classes.
    import tb_schemas as schemas  # type: ignore
except Exception:  # pragma: no cover - standalone fallback
    schemas = None  # type: ignore

try:
    import tb_hashing as hashing  # type: ignore
except Exception:  # pragma: no cover - standalone fallback
    hashing = None  # type: ignore

try:
    import tb_canonical as canonical  # type: ignore
except Exception:  # pragma: no cover - standalone fallback
    canonical = None  # type: ignore

try:
    import tb_crypto as crypto  # type: ignore
except Exception:  # pragma: no cover - crypto optional at validation import time
    crypto = None  # type: ignore


# =============================================================================
# Constants and small helpers
# =============================================================================

TB_VALIDATION_VERSION = "tb-validation-v1.0-production"
ZERO_HASH_256 = "0" * 64
_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_HEX_ANY_RE = re.compile(r"^[0-9a-fA-F]+$")


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _is_mapping(x: Any) -> bool:
    return isinstance(x, Mapping)


def _json_safe(obj: Any) -> Any:
    if canonical is not None and hasattr(canonical, "canonical_json_safe"):
        try:
            return canonical.canonical_json_safe(obj)
        except Exception:
            pass
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, Enum):
        return obj.value
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


def _canonical_dumps(obj: Any) -> str:
    if canonical is not None and hasattr(canonical, "canonical_dumps"):
        try:
            return canonical.canonical_dumps(obj)
        except Exception:
            pass
    return json.dumps(_json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _domain_hash(domain: str, obj: Any) -> str:
    if hashing is not None and hasattr(hashing, "domain_hash"):
        try:
            return str(hashing.domain_hash(domain, obj))
        except Exception:
            pass
    if canonical is not None and hasattr(canonical, "domain_hash"):
        try:
            return str(canonical.domain_hash(domain, obj))
        except Exception:
            pass
    payload = (str(domain) + "\x1f" + _canonical_dumps(obj)).encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()


def _merkle_root_hex(leaves: Sequence[Any], *, domain: str = "TB:VALIDATION:MERKLE:v1") -> str:
    if hashing is not None and hasattr(hashing, "merkle_root_hex"):
        try:
            return str(hashing.merkle_root_hex(list(leaves), domain=domain))
        except TypeError:
            try:
                return str(hashing.merkle_root_hex(list(leaves)))
            except Exception:
                pass
        except Exception:
            pass
    if not leaves:
        return _domain_hash(domain + ":EMPTY", [])
    level = [bytes.fromhex(_normalize_hexish(_hash_leaf(x, domain=domain + ":LEAF"))) for x in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        nxt: List[bytes] = []
        for i in range(0, len(level), 2):
            nxt.append(hashlib.sha256((domain + ":NODE").encode("utf-8") + level[i] + level[i + 1]).digest())
        level = nxt
    return level[0].hex()


def _hash_leaf(value: Any, *, domain: str = "TB:VALIDATION:LEAF:v1") -> str:
    if isinstance(value, str) and _is_hex(value):
        return _normalize_hexish(value)
    if hasattr(value, "hash") and callable(getattr(value, "hash")):
        try:
            h = str(value.hash())
            if _is_hex(h):
                return _normalize_hexish(h)
        except Exception:
            pass
    if hasattr(value, "payload_hash") and callable(getattr(value, "payload_hash")):
        try:
            h = str(value.payload_hash())
            if _is_hex(h):
                return _normalize_hexish(h)
        except Exception:
            pass
    return _domain_hash(domain, _to_mapping(value))


def _to_mapping(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            out = obj.to_dict()
            if isinstance(out, Mapping):
                return dict(out)
        except TypeError:
            try:
                out = obj.to_dict(include_histories=False)
                if isinstance(out, Mapping):
                    return dict(out)
            except Exception:
                pass
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return dict(vars(obj))
        except Exception:
            pass
    return {"value": obj}


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _call0(obj: Any, name: str) -> Any:
    fn = getattr(obj, name, None)
    if callable(fn):
        return fn()
    return None


def _is_hex(s: Any) -> bool:
    if not isinstance(s, str):
        return False
    ss = s.strip().lower()
    if ss.startswith("0x"):
        ss = ss[2:]
    return bool(ss) and len(ss) % 2 == 0 and bool(_HEX_ANY_RE.match(ss))


def _is_hex64(s: Any, *, allow_empty: bool = False, allow_zero: bool = True) -> bool:
    if s is None:
        return bool(allow_empty)
    ss = str(s).strip()
    if ss == "":
        return bool(allow_empty)
    if not _HEX64_RE.match(ss):
        return False
    if not allow_zero and ss.lower() == ZERO_HASH_256:
        return False
    return True


def _normalize_hexish(s: Any, *, length: Optional[int] = None) -> str:
    ss = str(s or "").strip().lower()
    if ss.startswith("0x"):
        ss = ss[2:]
    if length is not None:
        ss = ss.zfill(int(length))[-int(length):]
    return ss


def _nonempty(value: Any) -> bool:
    return bool(str(value or "").strip())


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _listify(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _issue_from_schema_issue(issue: Any) -> "ValidationIssue":
    if isinstance(issue, ValidationIssue):
        return issue
    data = _to_mapping(issue)
    return ValidationIssue(
        code=str(data.get("code", "schema_issue")),
        message=str(data.get("message", data.get("msg", "schema validation issue"))),
        severity=str(data.get("severity", "error")),
        field=str(data.get("field", "")),
        metadata=dict(data.get("metadata", {}) or {}),
    )


def _schema_from_mapping(cls_name: str, payload: Any) -> Any:
    if schemas is None:
        return payload
    cls = getattr(schemas, cls_name, None)
    if cls is None:
        return payload
    if isinstance(payload, cls):
        return payload
    if isinstance(payload, Mapping):
        try:
            return cls.from_dict(payload) if hasattr(cls, "from_dict") else cls(**dict(payload))
        except Exception:
            return payload
    return payload


def _object_hash(obj: Any, *, method_names: Sequence[str] = ("hash", "payload_hash"), domain: str = "TB:VALIDATION:OBJECT:v1") -> str:
    for name in method_names:
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                h = str(fn())
                if _is_hex64(h):
                    return h.lower()
            except Exception:
                pass
    return _domain_hash(domain, _to_mapping(obj))


# =============================================================================
# Validation report structures
# =============================================================================

class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    code: str
    message: str
    severity: str = Severity.ERROR.value
    field: str = ""
    path: str = ""
    metadata: Dict[str, Any] = dc_field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": str(self.code),
            "message": str(self.message),
            "severity": str(self.severity),
            "field": str(self.field),
            "path": str(self.path),
            "metadata": _json_safe(self.metadata),
        }


@dataclass
class ValidationReport:
    ok: bool = True
    subject: str = ""
    version: str = TB_VALIDATION_VERSION
    issues: List[ValidationIssue] = dc_field(default_factory=list)
    metadata: Dict[str, Any] = dc_field(default_factory=dict)

    def add(
        self,
        code: str,
        message: str,
        *,
        severity: str = Severity.ERROR.value,
        field: str = "",
        path: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "ValidationReport":
        sev = str(severity or Severity.ERROR.value).lower()
        self.issues.append(ValidationIssue(str(code), str(message), sev, str(field), str(path), dict(metadata or {})))
        if sev == Severity.ERROR.value:
            self.ok = False
        return self

    def error(self, code: str, message: str, **kwargs: Any) -> "ValidationReport":
        return self.add(code, message, severity=Severity.ERROR.value, **kwargs)

    def warning(self, code: str, message: str, **kwargs: Any) -> "ValidationReport":
        return self.add(code, message, severity=Severity.WARNING.value, **kwargs)

    def info(self, code: str, message: str, **kwargs: Any) -> "ValidationReport":
        return self.add(code, message, severity=Severity.INFO.value, **kwargs)

    def merge(self, other: "ValidationReport", *, path_prefix: str = "") -> "ValidationReport":
        for issue in other.issues:
            copied = copy.deepcopy(issue)
            if path_prefix:
                copied.path = f"{path_prefix}.{copied.path}" if copied.path else path_prefix
            self.issues.append(copied)
            if copied.severity == Severity.ERROR.value:
                self.ok = False
        self.metadata.update({k: v for k, v in other.metadata.items() if k not in self.metadata})
        return self

    def require(self, condition: bool, code: str, message: str, **kwargs: Any) -> "ValidationReport":
        if not bool(condition):
            self.error(code, message, **kwargs)
        return self

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR.value]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING.value]

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValidationError(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "subject": str(self.subject),
            "version": str(self.version),
            "issue_count": int(len(self.issues)),
            "error_count": int(len(self.errors)),
            "warning_count": int(len(self.warnings)),
            "issues": [i.to_dict() for i in self.issues],
            "metadata": _json_safe(self.metadata),
        }

    def __bool__(self) -> bool:
        return bool(self.ok)


class ValidationError(ValueError):
    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(_canonical_dumps(report.to_dict()))


# =============================================================================
# Generic schema and signature validators
# =============================================================================

def _apply_schema_validate(obj: Any, report: ValidationReport) -> ValidationReport:
    fn = getattr(obj, "validate", None)
    if callable(fn):
        try:
            issues = fn()
            for issue in list(issues or []):
                report.issues.append(_issue_from_schema_issue(issue))
            if any(i.severity == Severity.ERROR.value for i in report.issues):
                report.ok = False
        except Exception as exc:
            report.error("schema_validate_exception", f"schema validate() raised: {exc!r}")
    return report


def validate_schema_object(obj: Any, *, subject: str = "schema_object") -> ValidationReport:
    report = ValidationReport(subject=subject)
    if obj is None:
        return report.error("object_missing", "object is required")
    _apply_schema_validate(obj, report)
    return report


def validate_signature_payload(
    payload: Mapping[str, Any],
    *,
    public_key: str,
    signature: str,
    scheme: str = "ed25519",
    context: str = "transaction",
    required: bool = True,
) -> ValidationReport:
    report = ValidationReport(subject="signature")
    if not required and not signature:
        return report.warning("signature_missing_optional", "signature is missing but optional")
    if not signature:
        return report.error("signature_missing", "signature is required", field="signature")
    if not public_key:
        return report.error("public_key_missing", "public key is required", field="public_key")
    if str(scheme).lower() != "ed25519":
        return report.error("signature_scheme_unsupported", f"unsupported signature scheme: {scheme}", field="signature_scheme")
    if crypto is None or not hasattr(crypto, "verify_payload_signature"):
        return report.warning("crypto_backend_unavailable", "crypto backend unavailable; signature shape checked only")
    try:
        ok = bool(crypto.verify_payload_signature(payload, signature, public_key, scheme=scheme, context=context))
    except TypeError:
        try:
            ok = bool(crypto.verify_payload_signature(payload, signature, public_key))
        except Exception as exc:
            return report.error("signature_verify_exception", f"signature verification raised: {exc!r}")
    except Exception as exc:
        return report.error("signature_verify_exception", f"signature verification raised: {exc!r}")
    if not ok:
        report.error("signature_invalid", "signature verification failed")
    return report


# =============================================================================
# Commitment validators
# =============================================================================

def validate_state_commitment(commitment: Any, *, allow_empty_optional_roots: bool = True) -> ValidationReport:
    obj = _schema_from_mapping("StateCommitment", commitment)
    report = ValidationReport(subject="state_commitment")
    if obj is None or obj == {}:
        return report.error("state_commitment_missing", "state commitment is required")
    _apply_schema_validate(obj, report)
    data = _to_mapping(obj)
    state_root = data.get("state_root", "")
    if not _is_hex64(state_root, allow_empty=False, allow_zero=False):
        report.error("state_root_invalid", "state_root must be a non-zero 64-character hex digest", field="state_root")
    for name in ("sketch_root", "metadata_root", "topology_root", "history_root", "projector_config_hash"):
        v = data.get(name, "")
        if v or not allow_empty_optional_roots:
            if not _is_hex64(v, allow_empty=allow_empty_optional_roots):
                report.error(f"{name}_invalid", f"{name} must be empty or a 64-character hex digest", field=name)
    if _as_int(data.get("node_count", 0)) < 0:
        report.error("node_count_negative", "node_count cannot be negative", field="node_count")
    if _as_int(data.get("tick", 0)) < 0:
        report.error("tick_negative", "tick cannot be negative", field="tick")
    return report


def validate_tensor_commitment(commitment: Any, *, verify_self_hash: bool = True, consensus_safe: bool = False) -> ValidationReport:
    obj = _schema_from_mapping("TensorCommitment", commitment)
    report = ValidationReport(subject="tensor_commitment")
    if obj is None or obj == {}:
        return report.error("tensor_commitment_missing", "tensor commitment is required")
    _apply_schema_validate(obj, report)
    data = _to_mapping(obj)

    output_dims = [int(x) for x in _listify(data.get("output_dims", [])) if str(x).strip() != ""]
    input_dims = [int(x) for x in _listify(data.get("input_dims", [])) if str(x).strip() != ""]
    if not output_dims or not input_dims:
        report.error("tensor_dims_empty", "input_dims and output_dims must be non-empty", field="input_dims/output_dims")
    if any(d <= 0 for d in output_dims + input_dims):
        report.error("tensor_dims_nonpositive", "all tensor dimensions must be positive", field="input_dims/output_dims")
    if len(output_dims) != len(input_dims):
        report.error("tensor_dim_rank_mismatch", "input_dims and output_dims must have the same number of modes")

    core_hashes = [str(x) for x in _listify(data.get("core_hashes", []))]
    if not core_hashes:
        report.error("core_hashes_empty", "tensor commitment must include at least one core hash", field="core_hashes")
    for i, h in enumerate(core_hashes):
        if not _is_hex64(h, allow_empty=False):
            report.error("core_hash_invalid", "core hash must be a 64-character hex digest", field=f"core_hashes[{i}]")

    for name in ("metadata_hash", "compression_parent_hash"):
        v = str(data.get(name, "") or "")
        if v and not _is_hex64(v):
            report.error(f"{name}_invalid", f"{name} must be empty or a 64-character hex digest", field=name)

    if consensus_safe:
        dtype = str(data.get("dtype", "")).lower()
        if dtype not in {"float32", "<class 'numpy.float32'>", "float64", "<class 'numpy.float64'>"}:
            report.error("tensor_dtype_not_consensus_safe", f"dtype {dtype!r} is not an allowed consensus dtype", field="dtype")
        if not bool(data.get("device_independent", False)):
            report.error("tensor_not_device_independent", "consensus tensor commitment must be device independent", field="device_independent")

    if verify_self_hash:
        stored = str(data.get("commitment_hash", "") or "")
        if stored:
            expected = None
            fn = getattr(obj, "hash_without_self_reference", None)
            if callable(fn):
                try:
                    expected = str(fn())
                except Exception as exc:
                    report.error("tensor_self_hash_exception", f"failed to recompute tensor commitment hash: {exc!r}")
            elif isinstance(data, Mapping):
                payload = dict(data)
                payload["commitment_hash"] = ""
                expected = _domain_hash("TB:TENSOR_COMMITMENT:v1", payload)
            if expected and not hmac.compare_digest(stored.lower(), expected.lower()):
                report.error("tensor_commitment_hash_mismatch", "stored commitment_hash does not match recomputed hash", field="commitment_hash", metadata={"expected": expected, "actual": stored})
        else:
            report.warning("tensor_commitment_hash_missing", "commitment_hash is missing; self-hash not verified", field="commitment_hash")
    return report


def validate_compression_report(report_obj: Any, *, verify_self_hash: bool = True) -> ValidationReport:
    obj = _schema_from_mapping("CompressionReport", report_obj)
    report = ValidationReport(subject="compression_report")
    if obj is None or obj == {}:
        return report.error("compression_report_missing", "compression report is required")
    _apply_schema_validate(obj, report)
    data = _to_mapping(obj)
    old_params = _as_int(data.get("old_parameters", 0))
    new_params = _as_int(data.get("new_parameters", 0))
    if old_params < 0 or new_params < 0:
        report.error("compression_params_negative", "parameter counts cannot be negative")
    if old_params and new_params > old_params:
        report.warning("compression_expands_parameters", "new_parameters is greater than old_parameters")
    ratio = _as_float(data.get("compression_ratio", 0.0))
    if ratio < -100.0 or ratio > 1.0:
        report.error("compression_ratio_invalid", "compression_ratio must be <= 1.0 and not extreme", field="compression_ratio")
    for name in ("pre_tensor_commitment_hash", "post_tensor_commitment_hash", "report_hash"):
        v = str(data.get(name, "") or "")
        if v and not _is_hex64(v):
            report.error(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name)
    if verify_self_hash and str(data.get("report_hash", "") or ""):
        fn = getattr(obj, "hash_without_self_reference", None)
        if callable(fn):
            try:
                expected = str(fn())
                actual = str(data.get("report_hash"))
                if not hmac.compare_digest(expected.lower(), actual.lower()):
                    report.error("compression_report_hash_mismatch", "report_hash does not match recomputed hash", field="report_hash", metadata={"expected": expected, "actual": actual})
            except Exception as exc:
                report.error("compression_report_hash_exception", f"failed to recompute report hash: {exc!r}")
    return report


# =============================================================================
# Transaction and receipt validators
# =============================================================================

def validate_transaction_envelope(
    tx: Any,
    *,
    chain_id: Optional[str] = None,
    verify_signature: bool = False,
    require_signature: bool = False,
    current_height: Optional[int] = None,
) -> ValidationReport:
    obj = _schema_from_mapping("TransactionEnvelope", tx)
    report = ValidationReport(subject="transaction_envelope")
    if obj is None or obj == {}:
        return report.error("transaction_missing", "transaction envelope is required")
    _apply_schema_validate(obj, report)
    data = _to_mapping(obj)

    for name in ("chain_id", "tx_id", "sender_id", "sender_public_key"):
        if not _nonempty(data.get(name, "")):
            report.error(f"{name}_empty", f"{name} must be non-empty", field=name)
    if chain_id is not None and str(data.get("chain_id", "")) != str(chain_id):
        report.error("chain_id_mismatch", "transaction chain_id does not match expected chain_id", field="chain_id", metadata={"expected": chain_id, "actual": data.get("chain_id")})
    if _as_int(data.get("sender_sequence", -1), -1) < 0:
        report.error("sender_sequence_invalid", "sender_sequence must be non-negative", field="sender_sequence")
    if data.get("pre_state_root") and not _is_hex64(data.get("pre_state_root")):
        report.error("pre_state_root_invalid", "pre_state_root must be empty or 64 hex chars", field="pre_state_root")
    if not isinstance(data.get("action"), Mapping) or not dict(data.get("action") or {}):
        report.error("action_invalid", "action must be a non-empty mapping", field="action")
    if current_height is not None:
        h = int(current_height)
        if _as_int(data.get("valid_from_height", 0)) > h:
            report.error("transaction_not_yet_valid", "transaction valid_from_height is above current height", field="valid_from_height")
        until = data.get("valid_until_height", None)
        if until is not None and _as_int(until) < h:
            report.error("transaction_expired", "transaction valid_until_height is below current height", field="valid_until_height")

    sig = str(data.get("signature", "") or "")
    if require_signature and not sig:
        report.error("signature_missing", "transaction signature is required", field="signature")
    if verify_signature and (sig or require_signature):
        signing_payload = None
        fn = getattr(obj, "signing_payload", None)
        if callable(fn):
            try:
                signing_payload = fn()
            except Exception as exc:
                report.error("signing_payload_exception", f"could not produce signing payload: {exc!r}")
        if signing_payload is None:
            signing_payload = dict(data)
            signing_payload.pop("signature", None)
            signing_payload.pop("schema_name", None)
        report.merge(validate_signature_payload(
            signing_payload,
            public_key=str(data.get("sender_public_key", "")),
            signature=sig,
            scheme=str(data.get("signature_scheme", "ed25519")),
            context="transaction",
            required=require_signature,
        ), path_prefix="signature")
    return report


def validate_action_receipt(receipt: Any) -> ValidationReport:
    obj = _schema_from_mapping("ActionReceipt", receipt)
    report = ValidationReport(subject="action_receipt")
    if obj is None or obj == {}:
        return report.error("action_receipt_missing", "action receipt is required")
    _apply_schema_validate(obj, report)
    data = _to_mapping(obj)
    if not _is_hex64(data.get("action_hash", "")):
        report.error("action_hash_invalid", "action_hash must be 64 hex chars", field="action_hash")
    if not _nonempty(data.get("kind", "")):
        report.error("action_kind_empty", "kind must be non-empty", field="kind")
    for name in ("pre_state_root", "post_state_root"):
        v = data.get(name, "")
        if v and not _is_hex64(v):
            report.error(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name)
    return report


def validate_policy_receipt(receipt: Any) -> ValidationReport:
    obj = _schema_from_mapping("PolicyReceipt", receipt)
    report = ValidationReport(subject="policy_receipt")
    if obj is None or obj == {}:
        return report.error("policy_receipt_missing", "policy receipt is required")
    _apply_schema_validate(obj, report)
    data = _to_mapping(obj)
    for name in ("pre_state_root", "post_state_root", "simulation_hash"):
        v = data.get(name, "")
        if v and not _is_hex64(v):
            report.error(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name)
    for i, h in enumerate(_listify(data.get("action_hashes", []))):
        if not _is_hex64(h):
            report.error("action_hash_invalid", "action_hashes entries must be 64 hex chars", field=f"action_hashes[{i}]")
    for i, h in enumerate(_listify(data.get("action_receipt_hashes", []))):
        if not _is_hex64(h):
            report.error("action_receipt_hash_invalid", "action_receipt_hashes entries must be 64 hex chars", field=f"action_receipt_hashes[{i}]")
    return report


def validate_transaction_receipt(receipt: Any) -> ValidationReport:
    obj = _schema_from_mapping("TransactionReceipt", receipt)
    report = ValidationReport(subject="transaction_receipt")
    if obj is None or obj == {}:
        return report.error("transaction_receipt_missing", "transaction receipt is required")
    _apply_schema_validate(obj, report)
    data = _to_mapping(obj)
    for name in ("tx_hash", "pre_state_root", "post_state_root", "policy_receipt_hash", "block_hash"):
        v = data.get(name, "")
        if v and not _is_hex64(v):
            report.error(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name)
    if _as_int(data.get("block_index", 0)) < 0:
        report.error("block_index_negative", "block_index cannot be negative", field="block_index")
    return report


# =============================================================================
# Consensus validators
# =============================================================================

def validate_consensus_vote(vote: Any, *, verify_signature: bool = False, require_signature: bool = False) -> ValidationReport:
    obj = _schema_from_mapping("ConsensusVote", vote)
    report = ValidationReport(subject="consensus_vote")
    if obj is None or obj == {}:
        return report.error("consensus_vote_missing", "consensus vote is required")
    _apply_schema_validate(obj, report)
    data = _to_mapping(obj)
    if _as_int(data.get("height", -1), -1) < 0 or _as_int(data.get("round", -1), -1) < 0:
        report.error("height_or_round_negative", "height and round must be non-negative")
    for name in ("block_hash", "tx_merkle_root", "pre_state_root", "post_state_root", "tensor_commitment_hash", "policy_receipt_root"):
        v = str(data.get(name, "") or "")
        if v and not _is_hex64(v):
            report.error(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name)
    if verify_signature or require_signature:
        signing_payload = None
        fn = getattr(obj, "signing_payload", None)
        if callable(fn):
            try:
                signing_payload = fn()
            except Exception as exc:
                report.error("vote_signing_payload_exception", f"could not produce vote signing payload: {exc!r}")
        if signing_payload is None:
            signing_payload = dict(data)
            signing_payload.pop("signature", None)
            signing_payload.pop("schema_name", None)
        report.merge(validate_signature_payload(
            signing_payload,
            public_key=str(data.get("validator_public_key", "")),
            signature=str(data.get("signature", "") or ""),
            scheme=str(data.get("signature_scheme", "ed25519")),
            context="validator_vote",
            required=require_signature,
        ), path_prefix="signature")
    return report


def validate_consensus_certificate(cert: Any, *, verify_vote_signatures: bool = False, require_vote_signatures: bool = False, verify_self_hash: bool = True) -> ValidationReport:
    obj = _schema_from_mapping("ConsensusCertificate", cert)
    report = ValidationReport(subject="consensus_certificate")
    if obj is None or obj == {}:
        return report.error("consensus_certificate_missing", "consensus certificate is required")
    _apply_schema_validate(obj, report)
    data = _to_mapping(obj)
    votes = list(getattr(obj, "votes", data.get("votes", [])) or [])

    if _as_int(data.get("height", -1), -1) < 0 or _as_int(data.get("round", -1), -1) < 0:
        report.error("height_or_round_negative", "height and round must be non-negative")
    for name in ("block_hash", "tx_merkle_root", "pre_state_root", "post_state_root", "validator_set_hash", "certificate_hash"):
        v = str(data.get(name, "") or "")
        if v and not _is_hex64(v):
            report.error(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name)

    validator_ids: Set[str] = set()
    approvals = 0
    for i, vote in enumerate(votes):
        vrep = validate_consensus_vote(vote, verify_signature=verify_vote_signatures, require_signature=require_vote_signatures)
        report.merge(vrep, path_prefix=f"votes[{i}]")
        vdata = _to_mapping(vote)
        vid = str(vdata.get("validator_id", ""))
        if vid:
            if vid in validator_ids:
                report.error("duplicate_validator_vote", "validator voted more than once", path=f"votes[{i}]", metadata={"validator_id": vid})
            validator_ids.add(vid)
        for field_name in ("chain_id", "height", "round", "block_hash", "tx_merkle_root", "pre_state_root", "post_state_root"):
            if str(vdata.get(field_name, "")) != str(data.get(field_name, "")):
                report.error("vote_certificate_field_mismatch", f"vote {field_name} does not match certificate", path=f"votes[{i}].{field_name}", metadata={"expected": data.get(field_name), "actual": vdata.get(field_name)})
        if bool(vdata.get("approved", False)):
            approvals += 1

    quorum_required = max(1, _as_int(data.get("quorum_required", 1), 1))
    if votes and quorum_required > len(votes):
        report.error("quorum_required_exceeds_votes", "quorum_required cannot exceed vote count", field="quorum_required")
    if approvals < quorum_required:
        report.error("consensus_quorum_not_met", "approved votes do not meet quorum", metadata={"approvals": approvals, "quorum_required": quorum_required})

    if verify_self_hash and str(data.get("certificate_hash", "") or ""):
        fn = getattr(obj, "hash_without_self_reference", None)
        if callable(fn):
            try:
                expected = str(fn())
                actual = str(data.get("certificate_hash"))
                if not hmac.compare_digest(expected.lower(), actual.lower()):
                    report.error("certificate_hash_mismatch", "certificate_hash does not match recomputed hash", field="certificate_hash", metadata={"expected": expected, "actual": actual})
            except Exception as exc:
                report.error("certificate_hash_exception", f"failed to recompute certificate hash: {exc!r}")
    return report


# =============================================================================
# Lobe and transition receipt validators
# =============================================================================

def validate_lobe_scale_profile(profile: Any) -> ValidationReport:
    obj = _schema_from_mapping("LobeScaleProfile", profile)
    report = ValidationReport(subject="lobe_scale_profile")
    if obj is None or obj == {}:
        return report.error("lobe_scale_profile_missing", "lobe scale profile is required")
    _apply_schema_validate(obj, report)
    data = _to_mapping(obj)
    if not _nonempty(data.get("lobe_id", "")):
        report.error("lobe_id_empty", "lobe_id must be non-empty", field="lobe_id")
    if _as_int(data.get("parameter_budget", 0)) < 0 or _as_int(data.get("dense_equivalent_budget", 0)) < 0:
        report.error("lobe_budget_negative", "parameter budgets cannot be negative")
    if _as_int(data.get("active_parameter_count", 0)) < 0:
        report.error("active_parameter_count_negative", "active_parameter_count cannot be negative")
    if _as_int(data.get("hidden_size", 0)) < 0:
        report.error("hidden_size_negative", "hidden_size cannot be negative")
    if _as_float(data.get("update_frequency_hz", 0.0)) < 0:
        report.error("update_frequency_negative", "update_frequency_hz cannot be negative")
    return report


def validate_lobe_health_receipt(receipt: Any) -> ValidationReport:
    obj = _schema_from_mapping("LobeHealthReceipt", receipt)
    report = ValidationReport(subject="lobe_health_receipt")
    if obj is None or obj == {}:
        return report.error("lobe_health_receipt_missing", "lobe health receipt is required")
    _apply_schema_validate(obj, report)
    data = _to_mapping(obj)
    if not _nonempty(data.get("lobe_id", "")):
        report.error("lobe_id_empty", "lobe_id must be non-empty", field="lobe_id")
    score = _as_float(data.get("health_score", 0.0))
    if score < 0.0 or score > 1.0:
        report.error("health_score_out_of_range", "health_score must be in [0, 1]", field="health_score")
    for name in ("state_hash", "checkpoint_hash", "scale_profile_hash"):
        v = str(data.get(name, "") or "")
        if v and not _is_hex64(v):
            report.error(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name)
    if _as_int(data.get("parameter_count", 0)) < 0 or _as_int(data.get("dense_equivalent_parameter_count", 0)) < 0:
        report.error("lobe_parameter_counts_negative", "lobe parameter counts cannot be negative")
    return report


def validate_cfc_transition_receipt(receipt: Any) -> ValidationReport:
    obj = _schema_from_mapping("CfCTransitionReceipt", receipt)
    report = ValidationReport(subject="cfc_transition_receipt")
    if obj is None or obj == {}:
        return report.error("cfc_transition_receipt_missing", "CfC transition receipt is required")
    _apply_schema_validate(obj, report)
    data = _to_mapping(obj)
    for name in ("pre_state_root", "tx_hash", "predicted_post_state_hash", "actual_post_state_hash", "control_vector_hash", "lobe_health_hash", "scale_profile_hash"):
        v = str(data.get(name, "") or "")
        if v and not _is_hex64(v):
            report.error(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name)
    if _as_float(data.get("forecast_error", 0.0)) < 0.0:
        report.error("forecast_error_negative", "forecast_error cannot be negative", field="forecast_error")
    cos = data.get("cosine_similarity", None)
    if cos is not None:
        c = _as_float(cos)
        if c < -1.000001 or c > 1.000001:
            report.error("cosine_similarity_out_of_range", "cosine_similarity must be in [-1, 1]", field="cosine_similarity")
    if _as_float(data.get("time_delta", 0.0)) < 0.0:
        report.error("time_delta_negative", "time_delta cannot be negative", field="time_delta")
    for i, h in enumerate(_listify(data.get("tensor_backbone_commitment_hashes", []))):
        if not _is_hex64(h):
            report.error("tensor_backbone_commitment_hash_invalid", "tensor backbone commitment hashes must be 64 hex chars", field=f"tensor_backbone_commitment_hashes[{i}]")
    return report


def validate_twin_creation_receipt(receipt: Any) -> ValidationReport:
    obj = _schema_from_mapping("TwinCreationReceipt", receipt)
    report = ValidationReport(subject="twin_creation_receipt")
    if obj is None or obj == {}:
        return report.error("twin_creation_receipt_missing", "twin creation receipt is required")
    _apply_schema_validate(obj, report)
    data = _to_mapping(obj)
    for name in ("schema_hash", "input_hash", "initial_state_root", "initial_sketch_root", "source_manifest_root"):
        v = str(data.get(name, "") or "")
        if v and not _is_hex64(v):
            report.error(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name)
    if _as_int(data.get("node_count", 0)) < 0:
        report.error("node_count_negative", "node_count cannot be negative", field="node_count")
    return report


def validate_transition_receipt(receipt: Any) -> ValidationReport:
    obj = _schema_from_mapping("TransitionReceipt", receipt)
    report = ValidationReport(subject="transition_receipt")
    if obj is None or obj == {}:
        return report.error("transition_receipt_missing", "transition receipt is required")
    _apply_schema_validate(obj, report)
    data = _to_mapping(obj)
    for name in ("tx_hash", "pre_state_root", "post_state_root", "policy_receipt_hash", "tensor_commitment_hash", "ncp_health_hash", "cfc_transition_hash"):
        v = str(data.get(name, "") or "")
        if v and not _is_hex64(v):
            report.error(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name)
    return report


# =============================================================================
# Topology validation
# =============================================================================

def validate_topology_integrity(network_or_payload: Any, *, root_id: Optional[str] = None, require_all_reachable: bool = True) -> ValidationReport:
    """Validate TreeTensorNetwork-like topology maps.

    Accepts a live TreeTensorNetwork object or a serialized mapping containing
    ``nodes``, ``children``, ``parent``, and optionally ``root_id``.
    """
    report = ValidationReport(subject="topology_integrity")
    data = _to_mapping(network_or_payload)
    nodes = _get(network_or_payload, "nodes", data.get("nodes", {})) or {}
    children = _get(network_or_payload, "children", data.get("children", {})) or {}
    parent = _get(network_or_payload, "parent", data.get("parent", {})) or {}
    rid = str(root_id if root_id is not None else _get(network_or_payload, "root_id", data.get("root_id", "0")))

    if not isinstance(nodes, Mapping):
        return report.error("nodes_not_mapping", "nodes must be a mapping")
    if not isinstance(children, Mapping):
        return report.error("children_not_mapping", "children must be a mapping")
    if not isinstance(parent, Mapping):
        return report.error("parent_not_mapping", "parent must be a mapping")

    node_ids = {str(k) for k in nodes.keys()}
    if rid not in node_ids:
        report.error("root_missing", "root_id is not present in nodes", field="root_id", metadata={"root_id": rid})

    # Validate child references and reciprocal parent references.
    for p, kids in children.items():
        ps = str(p)
        if ps not in node_ids:
            report.error("children_parent_missing", "children map contains parent not present in nodes", path=f"children.{ps}")
        if not isinstance(kids, (list, tuple)):
            report.error("children_not_list", "children entry must be a list", path=f"children.{ps}")
            continue
        seen_kids: Set[str] = set()
        for child in kids:
            cs = str(child)
            if cs in seen_kids:
                report.error("duplicate_child_reference", "parent has duplicate child reference", path=f"children.{ps}", metadata={"child": cs})
            seen_kids.add(cs)
            if cs not in node_ids:
                report.error("child_missing", "child reference is not present in nodes", path=f"children.{ps}", metadata={"child": cs})
            actual_parent = str(parent.get(cs, ""))
            if cs != rid and actual_parent != ps:
                report.error("parent_child_mismatch", "child parent map does not match children map", path=f"parent.{cs}", metadata={"expected": ps, "actual": actual_parent})

    for child, p in parent.items():
        cs, ps = str(child), str(p)
        if cs not in node_ids:
            report.error("parent_child_missing", "parent map contains child not present in nodes", path=f"parent.{cs}")
        if ps not in node_ids:
            report.error("parent_parent_missing", "parent map references missing parent node", path=f"parent.{cs}", metadata={"parent": ps})
        if cs == rid:
            report.error("root_has_parent", "root node must not have a parent", path=f"parent.{cs}")
        if ps in children and cs not in [str(x) for x in children.get(ps, [])]:
            report.error("parent_not_reciprocated", "parent relation is not present in children map", path=f"parent.{cs}", metadata={"parent": ps})

    # Cycle and reachability check by DFS.
    visited: Set[str] = set()
    visiting: Set[str] = set()

    def dfs(nid: str, stack: List[str]) -> None:
        if nid in visiting:
            report.error("topology_cycle", "cycle detected in topology", path="children", metadata={"cycle_stack": stack + [nid]})
            return
        if nid in visited:
            return
        visiting.add(nid)
        for k in children.get(nid, []) or []:
            dfs(str(k), stack + [nid])
        visiting.remove(nid)
        visited.add(nid)

    if rid in node_ids:
        dfs(rid, [])
    if require_all_reachable:
        unreachable = sorted(node_ids - visited)
        if unreachable:
            report.error("unreachable_nodes", "some nodes are not reachable from root", metadata={"unreachable": unreachable[:50], "count": len(unreachable)})

    # Validate node dimensions where obvious.
    for nid, node in nodes.items():
        nd = _to_mapping(node)
        vector_dim = nd.get("vector_dim", getattr(node, "vector_dim", None))
        state = nd.get("state", getattr(node, "state", None))
        if vector_dim is not None:
            vd = _as_int(vector_dim, -1)
            if vd <= 0:
                report.error("node_vector_dim_invalid", "node vector_dim must be positive", path=f"nodes.{nid}.vector_dim")
            if state is not None:
                try:
                    # Avoid importing numpy: len of flattened list-like state is enough for serialized states.
                    if isinstance(state, (list, tuple)) and len(state) != vd:
                        report.error("node_state_dim_mismatch", "node state length does not match vector_dim", path=f"nodes.{nid}.state", metadata={"expected": vd, "actual": len(state)})
                except Exception:
                    pass
    return report


# =============================================================================
# Block and ledger validators
# =============================================================================

def _block_hash(block: Any) -> str:
    for name in ("block_hash", "hash"):
        v = _get(block, name, "")
        if isinstance(v, str) and _is_hex64(v):
            return v.lower()
    for method in ("compute_hash", "block_hash", "hash"):
        fn = getattr(block, method, None)
        if callable(fn):
            try:
                h = str(fn())
                if _is_hex64(h):
                    return h.lower()
            except Exception:
                pass
    data = _to_mapping(block)
    if "block_hash" in data:
        payload = dict(data)
        payload["block_hash"] = ""
        return _domain_hash("TB:BLOCK:FALLBACK_HASH:v1", payload)
    return _domain_hash("TB:BLOCK:FALLBACK_HASH:v1", data)


def _tx_hash(tx: Any) -> str:
    for method in ("tx_hash", "payload_hash", "hash"):
        fn = getattr(tx, method, None)
        if callable(fn):
            try:
                h = str(fn())
                if _is_hex64(h):
                    return h.lower()
            except Exception:
                pass
    data = _to_mapping(tx)
    if data.get("tx_hash") and _is_hex64(data.get("tx_hash")):
        return str(data.get("tx_hash")).lower()
    if data.get("tx_id") and not data.get("signature"):
        return _domain_hash("TB:TX_HASH:FALLBACK:v1", data)
    payload = dict(data)
    payload.pop("signature", None)
    return _domain_hash("TB:TX_HASH:FALLBACK:v1", payload)


def _extract_transactions_from_block(block: Any) -> List[Any]:
    txs = _get(block, "transactions", None)
    if txs is None:
        txs = _get(block, "txs", None)
    if txs is None:
        data = _to_mapping(block)
        txs = data.get("transactions", data.get("txs", []))
    return list(txs or [])


def validate_block_structure(block: Any, *, expected_index: Optional[int] = None, expected_prev_hash: Optional[str] = None, chain_id: Optional[str] = None, verify_merkle: bool = True) -> ValidationReport:
    report = ValidationReport(subject="block_structure")
    if block is None:
        return report.error("block_missing", "block is required")
    data = _to_mapping(block)

    index = data.get("index", data.get("height", None))
    if index is None:
        report.error("block_index_missing", "block must include index or height", field="index")
    else:
        idx = _as_int(index, -1)
        if idx < 0:
            report.error("block_index_negative", "block index/height must be non-negative", field="index")
        if expected_index is not None and idx != int(expected_index):
            report.error("block_index_mismatch", "block index does not match expected index", field="index", metadata={"expected": expected_index, "actual": idx})

    if chain_id is not None and data.get("chain_id") is not None and str(data.get("chain_id")) != str(chain_id):
        report.error("block_chain_id_mismatch", "block chain_id does not match expected chain_id", field="chain_id", metadata={"expected": chain_id, "actual": data.get("chain_id")})

    prev_hash = str(data.get("prev_hash", data.get("previous_hash", "")) or "")
    if expected_prev_hash is not None and prev_hash != str(expected_prev_hash):
        report.error("prev_hash_mismatch", "block prev_hash does not match expected previous hash", field="prev_hash", metadata={"expected": expected_prev_hash, "actual": prev_hash})
    if prev_hash and not _is_hex64(prev_hash):
        report.error("prev_hash_invalid", "prev_hash must be 64 hex chars", field="prev_hash")

    for name in ("block_hash", "world_root", "pre_state_root", "post_state_root", "state_root", "tensor_commitment_hash", "compression_report_hash", "consensus_certificate_hash"):
        v = str(data.get(name, "") or "")
        if v and name != "world_root" and not _is_hex64(v):
            report.error(f"{name}_invalid", f"{name} must be empty or 64 hex chars", field=name)
        elif name == "world_root" and v and len(v) not in (16, 64):
            # Older sketch roots may be 16 hex chars; warn instead of error.
            if not _is_hex(v):
                report.warning("world_root_nonstandard", "world_root is not a standard 16/64 hex root", field="world_root")

    txs = _extract_transactions_from_block(block)
    seen_tx_hashes: Set[str] = set()
    for i, tx in enumerate(txs):
        trep = validate_transaction_envelope(tx, chain_id=chain_id) if (_get(tx, "sender_sequence", None) is not None or isinstance(tx, Mapping)) else ValidationReport(subject="legacy_tx")
        if trep.issues:
            report.merge(trep, path_prefix=f"transactions[{i}]")
        th = _tx_hash(tx)
        if th in seen_tx_hashes:
            report.error("duplicate_tx_in_block", "duplicate transaction detected inside block", path=f"transactions[{i}]", metadata={"tx_hash": th})
        seen_tx_hashes.add(th)

    if verify_merkle:
        stored = str(data.get("tx_merkle_root", data.get("merkle_root", "")) or "")
        if stored:
            if not _is_hex64(stored):
                report.error("tx_merkle_root_invalid", "transaction Merkle root must be 64 hex chars", field="tx_merkle_root/merkle_root")
            else:
                computed = _merkle_root_hex([_tx_hash(tx) for tx in txs], domain="TB:TX_MERKLE:v1")
                if not hmac.compare_digest(stored.lower(), computed.lower()):
                    # Some legacy scripts hash leaves differently. Keep as warning if block explicitly marks legacy.
                    legacy = bool(data.get("metadata", {}).get("legacy_merkle", False)) if isinstance(data.get("metadata"), Mapping) else False
                    if legacy:
                        report.warning("tx_merkle_root_legacy_mismatch", "Merkle root differs from production domain-separated root; legacy block metadata allows warning", field="merkle_root")
                    else:
                        report.error("tx_merkle_root_mismatch", "transaction Merkle root does not match transactions", field="merkle_root", metadata={"expected": computed, "actual": stored})

    stored_hash = str(data.get("block_hash", "") or "")
    if stored_hash and _is_hex64(stored_hash):
        computed = None
        fn = getattr(block, "compute_hash", None)
        if callable(fn):
            try:
                computed = str(fn())
            except Exception:
                computed = None
        if computed and _is_hex64(computed) and not hmac.compare_digest(stored_hash.lower(), computed.lower()):
            report.error("block_hash_mismatch", "block_hash does not match compute_hash()", field="block_hash", metadata={"expected": computed, "actual": stored_hash})
    return report


def validate_sender_sequences(transactions: Sequence[Any], *, expected_next: Optional[Mapping[str, int]] = None, strict_gapless: bool = True) -> ValidationReport:
    report = ValidationReport(subject="sender_sequences")
    by_sender: Dict[str, List[Tuple[int, int, Any]]] = {}
    for i, tx in enumerate(transactions):
        data = _to_mapping(tx)
        sender = str(data.get("sender_id", ""))
        if not sender:
            report.error("sender_id_missing", "transaction missing sender_id", path=f"transactions[{i}]")
            continue
        seq = _as_int(data.get("sender_sequence", -1), -1)
        if seq < 0:
            report.error("sender_sequence_missing", "transaction missing non-negative sender_sequence", path=f"transactions[{i}]")
            continue
        by_sender.setdefault(sender, []).append((seq, i, tx))

    for sender, items in by_sender.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        seen: Set[int] = set()
        for seq, idx, _ in items_sorted:
            if seq in seen:
                report.error("duplicate_sender_sequence", "duplicate sequence number for sender", path=f"transactions[{idx}]", metadata={"sender_id": sender, "sequence": seq})
            seen.add(seq)
        if expected_next and sender in expected_next:
            start = int(expected_next[sender])
            if items_sorted and items_sorted[0][0] != start:
                report.error("sender_sequence_start_mismatch", "first sequence for sender does not match expected next sequence", metadata={"sender_id": sender, "expected": start, "actual": items_sorted[0][0]})
        if strict_gapless and len(items_sorted) > 1:
            for (a, _, _), (b, idx_b, _) in zip(items_sorted, items_sorted[1:]):
                if b != a + 1:
                    report.error("sender_sequence_gap", "sender sequence numbers must be gapless within validation set", path=f"transactions[{idx_b}]", metadata={"sender_id": sender, "previous": a, "actual": b})
    report.metadata["sender_count"] = len(by_sender)
    report.metadata["transaction_count"] = len(transactions)
    return report


def validate_no_duplicate_transactions(transactions: Sequence[Any]) -> ValidationReport:
    report = ValidationReport(subject="duplicate_transactions")
    seen_hashes: Set[str] = set()
    seen_ids: Set[str] = set()
    for i, tx in enumerate(transactions):
        data = _to_mapping(tx)
        tx_id = str(data.get("tx_id", "") or "")
        th = _tx_hash(tx)
        if tx_id:
            if tx_id in seen_ids:
                report.error("duplicate_tx_id", "duplicate tx_id detected", path=f"transactions[{i}]", metadata={"tx_id": tx_id})
            seen_ids.add(tx_id)
        if th in seen_hashes:
            report.error("duplicate_tx_hash", "duplicate tx_hash detected", path=f"transactions[{i}]", metadata={"tx_hash": th})
        seen_hashes.add(th)
    report.metadata["unique_tx_hashes"] = len(seen_hashes)
    return report


def validate_loaded_ledger(
    ledger: Any,
    *,
    chain_id: Optional[str] = None,
    validate_signatures: bool = False,
    require_signatures: bool = False,
    strict_sender_sequences: bool = True,
    verify_merkle: bool = True,
) -> ValidationReport:
    report = ValidationReport(subject="loaded_ledger")
    blocks = _get(ledger, "blocks", None)
    if blocks is None:
        data = _to_mapping(ledger)
        blocks = data.get("blocks", [])
    blocks = list(blocks or [])
    if not blocks:
        return report.error("ledger_empty", "ledger must contain at least one block")

    all_txs: List[Any] = []
    prev_hash_expected: Optional[str] = None
    for i, block in enumerate(blocks):
        brep = validate_block_structure(block, expected_index=i, expected_prev_hash=prev_hash_expected, chain_id=chain_id, verify_merkle=verify_merkle)
        report.merge(brep, path_prefix=f"blocks[{i}]")
        txs = _extract_transactions_from_block(block)
        for j, tx in enumerate(txs):
            trep = validate_transaction_envelope(tx, chain_id=chain_id, verify_signature=validate_signatures, require_signature=require_signatures, current_height=i)
            report.merge(trep, path_prefix=f"blocks[{i}].transactions[{j}]")
        all_txs.extend(txs)
        bh = _block_hash(block)
        prev_hash_expected = bh

    report.merge(validate_no_duplicate_transactions(all_txs), path_prefix="transactions")
    if strict_sender_sequences:
        report.merge(validate_sender_sequences(all_txs, strict_gapless=False), path_prefix="transactions")

    # Use ledger's own validate_chain if available, but do not trust it exclusively.
    fn = getattr(ledger, "validate_chain", None)
    if callable(fn):
        try:
            out = fn()
            if isinstance(out, Mapping):
                if not bool(out.get("ok", False)):
                    report.error("ledger_validate_chain_failed", "ledger.validate_chain() reported failure", metadata=dict(out))
            elif out is False:
                report.error("ledger_validate_chain_failed", "ledger.validate_chain() returned False")
        except Exception as exc:
            report.error("ledger_validate_chain_exception", f"ledger.validate_chain() raised: {exc!r}")

    report.metadata.update({"block_count": len(blocks), "transaction_count": len(all_txs)})
    return report


# =============================================================================
# Aggregate validation dispatch
# =============================================================================

_VALIDATOR_BY_SCHEMA_NAME: Dict[str, Callable[[Any], ValidationReport]] = {
    "StateCommitment": validate_state_commitment,
    "TensorCommitment": validate_tensor_commitment,
    "CompressionReport": validate_compression_report,
    "TransactionEnvelope": validate_transaction_envelope,
    "ActionReceipt": validate_action_receipt,
    "PolicyReceipt": validate_policy_receipt,
    "TransactionReceipt": validate_transaction_receipt,
    "ConsensusVote": validate_consensus_vote,
    "ConsensusCertificate": validate_consensus_certificate,
    "LobeScaleProfile": validate_lobe_scale_profile,
    "LobeHealthReceipt": validate_lobe_health_receipt,
    "CfCTransitionReceipt": validate_cfc_transition_receipt,
    "TwinCreationReceipt": validate_twin_creation_receipt,
    "TransitionReceipt": validate_transition_receipt,
}


def validate_any(obj: Any, *, schema_name: Optional[str] = None) -> ValidationReport:
    data = _to_mapping(obj)
    name = schema_name or str(data.get("schema_name", "") or getattr(obj, "SCHEMA_NAME", ""))
    if name in _VALIDATOR_BY_SCHEMA_NAME:
        return _VALIDATOR_BY_SCHEMA_NAME[name](obj)
    # Heuristic dispatch.
    keys = set(data.keys())
    if {"state_root", "node_count"} & keys:
        return validate_state_commitment(obj)
    if {"core_hashes", "output_dims", "input_dims"} <= keys:
        return validate_tensor_commitment(obj)
    if {"chain_id", "tx_id", "sender_id", "action"} <= keys:
        return validate_transaction_envelope(obj)
    if {"votes", "quorum_required", "certificate_hash"} & keys:
        return validate_consensus_certificate(obj)
    if {"nodes", "children", "parent"} <= keys:
        return validate_topology_integrity(obj)
    return validate_schema_object(obj, subject=name or "unknown_object")


# =============================================================================
# Strict wrappers
# =============================================================================

def assert_valid_transaction_envelope(tx: Any, **kwargs: Any) -> Any:
    validate_transaction_envelope(tx, **kwargs).raise_if_invalid()
    return tx


def assert_valid_state_commitment(commitment: Any, **kwargs: Any) -> Any:
    validate_state_commitment(commitment, **kwargs).raise_if_invalid()
    return commitment


def assert_valid_tensor_commitment(commitment: Any, **kwargs: Any) -> Any:
    validate_tensor_commitment(commitment, **kwargs).raise_if_invalid()
    return commitment


def assert_valid_loaded_ledger(ledger: Any, **kwargs: Any) -> Any:
    validate_loaded_ledger(ledger, **kwargs).raise_if_invalid()
    return ledger


def assert_valid_topology(network_or_payload: Any, **kwargs: Any) -> Any:
    validate_topology_integrity(network_or_payload, **kwargs).raise_if_invalid()
    return network_or_payload


# =============================================================================
# Self-test and CLI
# =============================================================================

def run_self_test() -> Dict[str, Any]:
    reports: Dict[str, Any] = {}
    h1 = _domain_hash("TB:TEST", {"x": 1})
    h2 = _domain_hash("TB:TEST", {"x": 2})

    if schemas is not None:
        StateCommitment = getattr(schemas, "StateCommitment")
        TensorCommitment = getattr(schemas, "TensorCommitment")
        TransactionEnvelope = getattr(schemas, "TransactionEnvelope")
        ConsensusVote = getattr(schemas, "ConsensusVote")
        ConsensusCertificate = getattr(schemas, "ConsensusCertificate")

        sc = StateCommitment(state_root=h1, sketch_root=h2, node_count=2)
        reports["state_commitment"] = validate_state_commitment(sc).to_dict()

        tc = TensorCommitment(output_dims=[2, 2], input_dims=[2, 2], bond_ranks=[2], core_hashes=[h1, h2], metadata_hash=h1)
        if hasattr(tc, "hash_without_self_reference"):
            tc.commitment_hash = tc.hash_without_self_reference()
        reports["tensor_commitment"] = validate_tensor_commitment(tc).to_dict()

        tx = TransactionEnvelope(chain_id="tb-test", tx_id="tx1", sender_id="alice", sender_public_key="demo_public_key", sender_sequence=0, action={"kind": "noop"}, pre_state_root=h1)
        reports["transaction"] = validate_transaction_envelope(tx).to_dict()

        v1 = ConsensusVote(chain_id="tb-test", height=1, round=0, validator_id="v1", validator_public_key="pk1", block_hash=h1, tx_merkle_root=h2, pre_state_root=h1, post_state_root=h2, approved=True)
        v2 = ConsensusVote(chain_id="tb-test", height=1, round=0, validator_id="v2", validator_public_key="pk2", block_hash=h1, tx_merkle_root=h2, pre_state_root=h1, post_state_root=h2, approved=True)
        cert = ConsensusCertificate.from_votes(chain_id="tb-test", height=1, round=0, block_hash=h1, tx_merkle_root=h2, pre_state_root=h1, post_state_root=h2, votes=[v1, v2], quorum_ratio=0.66)
        reports["consensus_certificate"] = validate_consensus_certificate(cert).to_dict()
    else:
        reports["state_commitment"] = validate_state_commitment({"state_root": h1, "node_count": 1}).to_dict()
        reports["transaction"] = validate_transaction_envelope({"chain_id": "tb-test", "tx_id": "tx1", "sender_id": "alice", "sender_public_key": "pk", "sender_sequence": 0, "action": {"kind": "noop"}}).to_dict()

    topology = {
        "root_id": "0",
        "nodes": {"0": {"vector_dim": 2, "state": [0.0, 1.0]}, "1": {"vector_dim": 2, "state": [1.0, 0.0]}},
        "children": {"0": ["1"], "1": []},
        "parent": {"1": "0"},
    }
    reports["topology"] = validate_topology_integrity(topology).to_dict()

    ok = all(bool(r.get("ok")) for r in reports.values())
    return {"ok": ok, "version": TB_VALIDATION_VERSION, "reports": reports}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tensorized Blockchain validation utilities")
    p.add_argument("--self-test", action="store_true", help="Run module self-test")
    p.add_argument("--json", default="", help="Validate a JSON file with heuristic dispatch")
    p.add_argument("--kind", default="", help="Optional schema kind for JSON validation")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        if args.self_test or not args.json:
            out = run_self_test()
        else:
            payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
            out = validate_any(payload, schema_name=args.kind or None).to_dict()
        print(json.dumps(_json_safe(out), indent=2 if args.pretty else None, ensure_ascii=False))
        return 0 if bool(out.get("ok", False)) else 2
    except Exception as exc:
        print(json.dumps({"ok": False, "error": repr(exc)}, indent=2), file=sys.stderr)
        return 1


__all__ = [
    "TB_VALIDATION_VERSION",
    "Severity",
    "ValidationIssue",
    "ValidationReport",
    "ValidationError",
    "validate_schema_object",
    "validate_signature_payload",
    "validate_state_commitment",
    "validate_tensor_commitment",
    "validate_compression_report",
    "validate_transaction_envelope",
    "validate_action_receipt",
    "validate_policy_receipt",
    "validate_transaction_receipt",
    "validate_consensus_vote",
    "validate_consensus_certificate",
    "validate_lobe_scale_profile",
    "validate_lobe_health_receipt",
    "validate_cfc_transition_receipt",
    "validate_twin_creation_receipt",
    "validate_transition_receipt",
    "validate_topology_integrity",
    "validate_block_structure",
    "validate_sender_sequences",
    "validate_no_duplicate_transactions",
    "validate_loaded_ledger",
    "validate_any",
    "assert_valid_transaction_envelope",
    "assert_valid_state_commitment",
    "assert_valid_tensor_commitment",
    "assert_valid_loaded_ledger",
    "assert_valid_topology",
    "run_self_test",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
