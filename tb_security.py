#!/usr/bin/env python3
# tb_security.py
r"""
Project Chimera / Akkurat - Tensorized Blockchain Security Policy
==================================================================

Security policy and threat-model checks for the Tensorized Blockchain stack.

Purpose
-------
This module centralizes runtime safety policy for files, checkpoints, tensor
materialization, signatures, key handling, external AI/cortex integrations, and
unsafe load prevention. It is deliberately dependency-light and import-safe so
it can be used by storage, API, CLI, execution, lobe runtime, and ledger modules
without circular imports.

Responsibilities
----------------
- Safe file-access rules.
- Maximum memory and dense-materialization guards.
- Unsafe load prevention.
- Checkpoint verification policy.
- External cortex/LLM consensus-safety rules.
- Signature policy and key-storage warnings.
- Structured security reports for demo/production observability.

Design principles
-----------------
- Security failures are explicit and structured.
- Production and consensus-safe modes prefer denial over silent fallback.
- Demo mode may be permissive, but must report degraded safety posture.
- No private-key material is logged or returned by helper functions.
- This module never performs network access and never imports heavy runtime code.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import json
import math
import os
import platform
import re
import stat
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


# =============================================================================
# Version and constants
# =============================================================================

TB_SECURITY_VERSION = "tb-security-v1.0-production"
DEFAULT_HASH_ALGORITHM = "sha256"
ZERO_HASH_256 = "0" * 64

DEFAULT_MAX_FILE_BYTES = 512 * 1024 * 1024  # 512 MiB
DEFAULT_MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
DEFAULT_MAX_DENSE_ENTRIES = 1_000_000
DEFAULT_MAX_DENSE_BYTES = 512 * 1024 * 1024  # 512 MiB
DEFAULT_MAX_JSON_BYTES = 64 * 1024 * 1024  # 64 MiB
DEFAULT_MAX_PARAMETERS_PER_LOBE = 2_500_000_000
DEFAULT_MAX_ACTIVE_PARAMETERS_TOTAL = 5_000_000_000
DEFAULT_ALLOWED_SIGNATURE_SCHEMES = ("ed25519",)
DEFAULT_DEMO_SIGNATURE_SCHEMES = ("ed25519", "hmac_demo")

PRIVATE_KEY_NAME_PATTERNS = (
    "private",
    "privkey",
    "secret",
    "seed",
    "wallet",
    "mnemonic",
    "id_ed25519",
    "id_rsa",
    "id_ecdsa",
    "id_secp256k1",
)

SENSITIVE_EXTENSIONS = (
    ".key",
    ".pem",
    ".p12",
    ".pfx",
    ".wallet",
    ".seed",
)

SAFE_TEXT_EXTENSIONS = (
    ".json",
    ".jsonl",
    ".txt",
    ".md",
    ".csv",
    ".yaml",
    ".yml",
    ".toml",
)

SAFE_BINARY_EXTENSIONS = (
    ".npz",
    ".npy",
    ".bin",
    ".dat",
)

SAFE_SOURCE_EXTENSIONS = SAFE_TEXT_EXTENSIONS + SAFE_BINARY_EXTENSIONS + (
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
)


# =============================================================================
# Optional integration with generated stack modules
# =============================================================================

try:  # Optional; keep this module usable alone.
    from tb_canonical import canonical_json_safe as _canonical_json_safe  # type: ignore
    from tb_canonical import canonical_dumps as _canonical_dumps  # type: ignore
    from tb_canonical import canonical_bytes as _canonical_bytes  # type: ignore
except Exception:  # pragma: no cover
    _canonical_json_safe = None
    _canonical_dumps = None
    _canonical_bytes = None

try:  # Optional central config integration.
    from tb_config import build_default_config as _build_default_config  # type: ignore
except Exception:  # pragma: no cover
    _build_default_config = None


# =============================================================================
# Errors, enums, and report objects
# =============================================================================

class SecurityError(RuntimeError):
    """Raised when a requested operation violates the configured security policy."""


class SecurityPolicyError(SecurityError):
    """Raised for policy configuration errors or denied operations."""


class UnsafePathError(SecurityPolicyError):
    """Raised when a path is outside an allowed root or otherwise unsafe."""


class UnsafeLoadError(SecurityPolicyError):
    """Raised when a load/checkpoint operation is unsafe or unverifiable."""


class SecuritySeverity(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RuntimeSecurityMode(str, enum.Enum):
    DEMO = "demo"
    TEST = "test"
    PRODUCTION = "production"
    DETERMINISTIC_REPLAY = "deterministic_replay"
    CONSENSUS_SAFE = "consensus_safe"


class SecurityDecision(str, enum.Enum):
    ALLOW = "allow"
    WARN = "warn"
    DENY = "deny"


class KeyMaterialKind(str, enum.Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    UNKNOWN = "unknown"


@dataclass
class SecurityIssue:
    code: str
    severity: str = SecuritySeverity.ERROR.value
    message: str = ""
    component: str = "security"
    decision: str = SecurityDecision.DENY.value
    recoverable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class SecurityReport:
    ok: bool
    component: str = "security"
    mode: str = RuntimeSecurityMode.PRODUCTION.value
    issues: List[SecurityIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(
        self,
        code: str,
        message: str,
        *,
        severity: Union[str, SecuritySeverity] = SecuritySeverity.ERROR,
        decision: Union[str, SecurityDecision] = SecurityDecision.DENY,
        component: Optional[str] = None,
        recoverable: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        sev = severity.value if isinstance(severity, SecuritySeverity) else str(severity)
        dec = decision.value if isinstance(decision, SecurityDecision) else str(decision)
        self.issues.append(
            SecurityIssue(
                code=str(code),
                severity=sev,
                message=str(message),
                component=str(component or self.component),
                decision=dec,
                recoverable=bool(recoverable),
                metadata=dict(metadata or {}),
            )
        )
        if sev in {SecuritySeverity.ERROR.value, SecuritySeverity.CRITICAL.value} or dec == SecurityDecision.DENY.value:
            self.ok = False

    def extend(self, other: "SecurityReport") -> None:
        self.issues.extend(other.issues)
        if not other.ok:
            self.ok = False
        self.metadata.update({f"{other.component}_{k}": v for k, v in other.metadata.items()})

    def raise_if_failed(self) -> None:
        if self.ok:
            return
        messages = [f"{i.code}: {i.message}" for i in self.issues if i.decision == SecurityDecision.DENY.value]
        raise SecurityPolicyError("; ".join(messages) if messages else "Security report failed")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "component": self.component,
            "mode": self.mode,
            "issues": [i.to_dict() for i in self.issues],
            "metadata": _json_safe(self.metadata),
        }


# =============================================================================
# Canonical/hash helpers
# =============================================================================

def _json_safe(obj: Any) -> Any:
    if _canonical_json_safe is not None:
        try:
            return _canonical_json_safe(obj)
        except Exception:
            pass
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.hex()
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


def canonical_dumps(obj: Any) -> str:
    if _canonical_dumps is not None:
        try:
            return str(_canonical_dumps(obj))
        except Exception:
            pass
    return json.dumps(_json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_bytes(obj: Any) -> bytes:
    if _canonical_bytes is not None:
        try:
            return bytes(_canonical_bytes(obj))
        except Exception:
            pass
    return canonical_dumps(obj).encode("utf-8", errors="strict")


def hash_bytes(data: bytes, *, algorithm: str = DEFAULT_HASH_ALGORITHM) -> str:
    alg = str(algorithm or DEFAULT_HASH_ALGORITHM).lower().strip()
    if alg != "sha256":
        if alg == "blake3":
            try:
                import blake3  # type: ignore
                return blake3.blake3(bytes(data)).hexdigest()
            except Exception as exc:  # pragma: no cover
                raise SecurityPolicyError("BLAKE3 requested but blake3 package is unavailable") from exc
        raise SecurityPolicyError(f"Unsupported hash algorithm: {algorithm}")
    return hashlib.sha256(bytes(data)).hexdigest()


def hash_object(obj: Any, *, domain: str = "TB_SECURITY_OBJECT_V1", algorithm: str = DEFAULT_HASH_ALGORITHM) -> str:
    payload = {"domain": str(domain), "payload": _json_safe(obj)}
    return hash_bytes(canonical_bytes(payload), algorithm=algorithm)


def hash_file(path: Union[str, os.PathLike[str]], *, algorithm: str = DEFAULT_HASH_ALGORITHM, chunk_bytes: int = 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    if str(algorithm).lower().strip() != "sha256":
        # Keep streaming fallback simple and explicit.
        return hash_bytes(p.read_bytes(), algorithm=algorithm)
    with p.open("rb") as f:
        while True:
            chunk = f.read(int(chunk_bytes))
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# =============================================================================
# Policy dataclasses
# =============================================================================

@dataclass
class FileAccessPolicy:
    allowed_roots: List[str] = field(default_factory=list)
    allow_absolute_paths: bool = False
    allow_path_traversal: bool = False
    allow_symlinks: bool = False
    allow_local_file_hashing: bool = False
    allow_sensitive_key_paths: bool = False
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    allowed_extensions: List[str] = field(default_factory=lambda: list(SAFE_SOURCE_EXTENSIONS))
    denied_extensions: List[str] = field(default_factory=lambda: list(SENSITIVE_EXTENSIONS))
    redact_absolute_paths: bool = True

    def normalized(self) -> "FileAccessPolicy":
        roots: List[str] = []
        for r in self.allowed_roots:
            if not str(r).strip():
                continue
            try:
                roots.append(str(Path(r).expanduser().resolve()))
            except Exception:
                roots.append(str(r))
        allowed_ext = [_normalize_ext(x) for x in self.allowed_extensions if str(x).strip()]
        denied_ext = [_normalize_ext(x) for x in self.denied_extensions if str(x).strip()]
        return FileAccessPolicy(
            allowed_roots=roots,
            allow_absolute_paths=bool(self.allow_absolute_paths),
            allow_path_traversal=bool(self.allow_path_traversal),
            allow_symlinks=bool(self.allow_symlinks),
            allow_local_file_hashing=bool(self.allow_local_file_hashing),
            allow_sensitive_key_paths=bool(self.allow_sensitive_key_paths),
            max_file_bytes=max(1, int(self.max_file_bytes)),
            allowed_extensions=allowed_ext,
            denied_extensions=denied_ext,
            redact_absolute_paths=bool(self.redact_absolute_paths),
        )


@dataclass
class MemoryGuardPolicy:
    max_dense_entries: int = DEFAULT_MAX_DENSE_ENTRIES
    max_dense_bytes: int = DEFAULT_MAX_DENSE_BYTES
    max_json_bytes: int = DEFAULT_MAX_JSON_BYTES
    max_checkpoint_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES
    max_lobe_parameters: int = DEFAULT_MAX_PARAMETERS_PER_LOBE
    max_active_parameters_total: int = DEFAULT_MAX_ACTIVE_PARAMETERS_TOTAL
    allow_large_allocations_in_demo: bool = False

    def normalized(self) -> "MemoryGuardPolicy":
        return MemoryGuardPolicy(
            max_dense_entries=max(1, int(self.max_dense_entries)),
            max_dense_bytes=max(1, int(self.max_dense_bytes)),
            max_json_bytes=max(1, int(self.max_json_bytes)),
            max_checkpoint_bytes=max(1, int(self.max_checkpoint_bytes)),
            max_lobe_parameters=max(1, int(self.max_lobe_parameters)),
            max_active_parameters_total=max(1, int(self.max_active_parameters_total)),
            allow_large_allocations_in_demo=bool(self.allow_large_allocations_in_demo),
        )


@dataclass
class LoadSafetyPolicy:
    allow_pickle: bool = False
    require_manifest: bool = True
    require_hash_verification: bool = True
    require_checkpoint_hash: bool = True
    allow_unsafe_load_json: bool = False
    allow_missing_manifest_in_demo: bool = True

    def normalized(self) -> "LoadSafetyPolicy":
        return LoadSafetyPolicy(
            allow_pickle=bool(self.allow_pickle),
            require_manifest=bool(self.require_manifest),
            require_hash_verification=bool(self.require_hash_verification),
            require_checkpoint_hash=bool(self.require_checkpoint_hash),
            allow_unsafe_load_json=bool(self.allow_unsafe_load_json),
            allow_missing_manifest_in_demo=bool(self.allow_missing_manifest_in_demo),
        )


@dataclass
class CheckpointPolicy:
    require_integrity_hash: bool = True
    require_manifest_hash: bool = True
    require_lobe_manifest: bool = True
    require_consensus_safe_mode_for_ledger_use: bool = True
    allowed_checkpoint_extensions: List[str] = field(default_factory=lambda: [".npz", ".json"])
    max_checkpoint_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES

    def normalized(self) -> "CheckpointPolicy":
        return CheckpointPolicy(
            require_integrity_hash=bool(self.require_integrity_hash),
            require_manifest_hash=bool(self.require_manifest_hash),
            require_lobe_manifest=bool(self.require_lobe_manifest),
            require_consensus_safe_mode_for_ledger_use=bool(self.require_consensus_safe_mode_for_ledger_use),
            allowed_checkpoint_extensions=[_normalize_ext(x) for x in self.allowed_checkpoint_extensions],
            max_checkpoint_bytes=max(1, int(self.max_checkpoint_bytes)),
        )


@dataclass
class ExternalAIPolicy:
    allow_external_cortex: bool = True
    allow_external_cortex_in_consensus: bool = False
    require_committed_outputs_for_consensus: bool = True
    allow_live_network_calls: bool = False
    max_prompt_bytes: int = 256 * 1024
    max_response_bytes: int = 1024 * 1024

    def normalized(self) -> "ExternalAIPolicy":
        return ExternalAIPolicy(
            allow_external_cortex=bool(self.allow_external_cortex),
            allow_external_cortex_in_consensus=bool(self.allow_external_cortex_in_consensus),
            require_committed_outputs_for_consensus=bool(self.require_committed_outputs_for_consensus),
            allow_live_network_calls=bool(self.allow_live_network_calls),
            max_prompt_bytes=max(1, int(self.max_prompt_bytes)),
            max_response_bytes=max(1, int(self.max_response_bytes)),
        )


@dataclass
class SignaturePolicy:
    allowed_schemes: List[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_SIGNATURE_SCHEMES))
    allow_hmac_demo: bool = False
    require_public_key: bool = True
    require_key_fingerprint: bool = True
    require_validator_vote_signatures: bool = True
    require_transaction_signatures: bool = True
    min_ed25519_public_key_hex_len: int = 64

    def normalized(self) -> "SignaturePolicy":
        schemes = [str(x).lower().strip() for x in self.allowed_schemes if str(x).strip()]
        if self.allow_hmac_demo and "hmac_demo" not in schemes:
            schemes.append("hmac_demo")
        return SignaturePolicy(
            allowed_schemes=schemes or list(DEFAULT_ALLOWED_SIGNATURE_SCHEMES),
            allow_hmac_demo=bool(self.allow_hmac_demo),
            require_public_key=bool(self.require_public_key),
            require_key_fingerprint=bool(self.require_key_fingerprint),
            require_validator_vote_signatures=bool(self.require_validator_vote_signatures),
            require_transaction_signatures=bool(self.require_transaction_signatures),
            min_ed25519_public_key_hex_len=max(32, int(self.min_ed25519_public_key_hex_len)),
        )


@dataclass
class KeyStoragePolicy:
    allow_plaintext_private_keys: bool = False
    warn_on_world_readable_key_files: bool = True
    allow_private_key_export: bool = False
    require_private_key_file_permissions: bool = True
    recommended_private_key_mode_octal: str = "0o600"
    key_directory_name_hint: str = ".tb_keys"

    def normalized(self) -> "KeyStoragePolicy":
        return KeyStoragePolicy(
            allow_plaintext_private_keys=bool(self.allow_plaintext_private_keys),
            warn_on_world_readable_key_files=bool(self.warn_on_world_readable_key_files),
            allow_private_key_export=bool(self.allow_private_key_export),
            require_private_key_file_permissions=bool(self.require_private_key_file_permissions),
            recommended_private_key_mode_octal=str(self.recommended_private_key_mode_octal or "0o600"),
            key_directory_name_hint=str(self.key_directory_name_hint or ".tb_keys"),
        )


@dataclass
class RuntimeSafetyPolicy:
    mode: str = RuntimeSecurityMode.PRODUCTION.value
    consensus_safe: bool = True
    deterministic_replay: bool = False
    allow_gpu_consensus: bool = False
    allowed_consensus_dtypes: List[str] = field(default_factory=lambda: ["float32", "float64"])
    require_cpu_for_consensus: bool = True
    allow_degraded_mode_in_production: bool = False
    allow_silent_fallbacks: bool = False

    def normalized(self) -> "RuntimeSafetyPolicy":
        mode = _coerce_mode(self.mode)
        return RuntimeSafetyPolicy(
            mode=mode.value,
            consensus_safe=bool(self.consensus_safe),
            deterministic_replay=bool(self.deterministic_replay),
            allow_gpu_consensus=bool(self.allow_gpu_consensus),
            allowed_consensus_dtypes=[str(x).lower().strip() for x in self.allowed_consensus_dtypes if str(x).strip()],
            require_cpu_for_consensus=bool(self.require_cpu_for_consensus),
            allow_degraded_mode_in_production=bool(self.allow_degraded_mode_in_production),
            allow_silent_fallbacks=bool(self.allow_silent_fallbacks),
        )


@dataclass
class SecurityPolicy:
    runtime: RuntimeSafetyPolicy = field(default_factory=RuntimeSafetyPolicy)
    files: FileAccessPolicy = field(default_factory=FileAccessPolicy)
    memory: MemoryGuardPolicy = field(default_factory=MemoryGuardPolicy)
    load: LoadSafetyPolicy = field(default_factory=LoadSafetyPolicy)
    checkpoints: CheckpointPolicy = field(default_factory=CheckpointPolicy)
    external_ai: ExternalAIPolicy = field(default_factory=ExternalAIPolicy)
    signatures: SignaturePolicy = field(default_factory=SignaturePolicy)
    key_storage: KeyStoragePolicy = field(default_factory=KeyStoragePolicy)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "SecurityPolicy":
        return SecurityPolicy(
            runtime=self.runtime.normalized(),
            files=self.files.normalized(),
            memory=self.memory.normalized(),
            load=self.load.normalized(),
            checkpoints=self.checkpoints.normalized(),
            external_ai=self.external_ai.normalized(),
            signatures=self.signatures.normalized(),
            key_storage=self.key_storage.normalized(),
            metadata=dict(self.metadata or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self.normalized()))

    def policy_hash(self) -> str:
        return hash_object(self.to_dict(), domain="TB_SECURITY_POLICY_V1")


# =============================================================================
# Policy builders
# =============================================================================

def _coerce_mode(mode: Any) -> RuntimeSecurityMode:
    if isinstance(mode, RuntimeSecurityMode):
        return mode
    s = str(mode or RuntimeSecurityMode.PRODUCTION.value).lower().strip().replace("-", "_")
    aliases = {
        "prod": RuntimeSecurityMode.PRODUCTION.value,
        "production": RuntimeSecurityMode.PRODUCTION.value,
        "deterministic": RuntimeSecurityMode.DETERMINISTIC_REPLAY.value,
        "replay": RuntimeSecurityMode.DETERMINISTIC_REPLAY.value,
        "consensus": RuntimeSecurityMode.CONSENSUS_SAFE.value,
        "consensus_safe": RuntimeSecurityMode.CONSENSUS_SAFE.value,
        "dev": RuntimeSecurityMode.DEMO.value,
        "development": RuntimeSecurityMode.DEMO.value,
    }
    s = aliases.get(s, s)
    try:
        return RuntimeSecurityMode(s)
    except Exception:
        return RuntimeSecurityMode.PRODUCTION


def _normalize_ext(value: Any) -> str:
    s = str(value or "").lower().strip()
    if not s:
        return s
    return s if s.startswith(".") else f".{s}"


def build_security_policy(
    *,
    mode: Union[str, RuntimeSecurityMode] = RuntimeSecurityMode.PRODUCTION,
    allowed_roots: Optional[Sequence[Union[str, os.PathLike[str]]]] = None,
    project_root: Optional[Union[str, os.PathLike[str]]] = None,
    artifact_root: Optional[Union[str, os.PathLike[str]]] = None,
    consensus_safe: Optional[bool] = None,
    deterministic_replay: Optional[bool] = None,
) -> SecurityPolicy:
    m = _coerce_mode(mode)
    roots: List[str] = []
    for r in list(allowed_roots or []):
        try:
            roots.append(str(Path(r).expanduser().resolve()))
        except Exception:
            pass
    for r in (project_root, artifact_root):
        if r is not None:
            try:
                roots.append(str(Path(r).expanduser().resolve()))
            except Exception:
                pass
    roots = sorted(set(roots))

    is_demo = m in {RuntimeSecurityMode.DEMO, RuntimeSecurityMode.TEST}
    is_replay = m in {RuntimeSecurityMode.DETERMINISTIC_REPLAY, RuntimeSecurityMode.CONSENSUS_SAFE}
    cs = bool(consensus_safe) if consensus_safe is not None else (not is_demo or is_replay)
    dr = bool(deterministic_replay) if deterministic_replay is not None else (m == RuntimeSecurityMode.DETERMINISTIC_REPLAY)

    return SecurityPolicy(
        runtime=RuntimeSafetyPolicy(
            mode=m.value,
            consensus_safe=cs,
            deterministic_replay=dr,
            allow_gpu_consensus=False,
            require_cpu_for_consensus=True,
            allow_degraded_mode_in_production=False,
            allow_silent_fallbacks=False,
        ),
        files=FileAccessPolicy(
            allowed_roots=roots,
            allow_absolute_paths=False,
            allow_path_traversal=False,
            allow_symlinks=False,
            allow_local_file_hashing=is_demo,
            allow_sensitive_key_paths=False,
            max_file_bytes=DEFAULT_MAX_FILE_BYTES,
            redact_absolute_paths=True,
        ),
        memory=MemoryGuardPolicy(
            allow_large_allocations_in_demo=is_demo,
        ),
        load=LoadSafetyPolicy(
            allow_pickle=False,
            require_manifest=not is_demo,
            require_hash_verification=True,
            require_checkpoint_hash=not is_demo,
            allow_unsafe_load_json=False,
            allow_missing_manifest_in_demo=is_demo,
        ),
        checkpoints=CheckpointPolicy(
            require_integrity_hash=not is_demo,
            require_manifest_hash=not is_demo,
            require_lobe_manifest=not is_demo,
            require_consensus_safe_mode_for_ledger_use=True,
        ),
        external_ai=ExternalAIPolicy(
            allow_external_cortex=is_demo,
            allow_external_cortex_in_consensus=False,
            require_committed_outputs_for_consensus=True,
            allow_live_network_calls=False,
        ),
        signatures=SignaturePolicy(
            allowed_schemes=list(DEFAULT_DEMO_SIGNATURE_SCHEMES if is_demo else DEFAULT_ALLOWED_SIGNATURE_SCHEMES),
            allow_hmac_demo=is_demo,
            require_public_key=True,
            require_key_fingerprint=True,
            require_validator_vote_signatures=True,
            require_transaction_signatures=True,
        ),
        key_storage=KeyStoragePolicy(),
        metadata={
            "created_at": now_iso_utc(),
            "tb_security_version": TB_SECURITY_VERSION,
        },
    ).normalized()


def production_policy(*, allowed_roots: Optional[Sequence[Union[str, os.PathLike[str]]]] = None) -> SecurityPolicy:
    return build_security_policy(mode=RuntimeSecurityMode.PRODUCTION, allowed_roots=allowed_roots)


def demo_policy(*, allowed_roots: Optional[Sequence[Union[str, os.PathLike[str]]]] = None) -> SecurityPolicy:
    return build_security_policy(mode=RuntimeSecurityMode.DEMO, allowed_roots=allowed_roots, consensus_safe=False)


def deterministic_replay_policy(*, allowed_roots: Optional[Sequence[Union[str, os.PathLike[str]]]] = None) -> SecurityPolicy:
    return build_security_policy(mode=RuntimeSecurityMode.DETERMINISTIC_REPLAY, allowed_roots=allowed_roots, consensus_safe=True, deterministic_replay=True)


def load_security_policy(path: Union[str, os.PathLike[str]]) -> SecurityPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return security_policy_from_mapping(payload)


def save_security_policy(policy: SecurityPolicy, path: Union[str, os.PathLike[str]], *, pretty: bool = True) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(policy.normalized().to_dict(), indent=2 if pretty else None, sort_keys=True, ensure_ascii=False)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def security_policy_from_mapping(payload: Mapping[str, Any]) -> SecurityPolicy:
    p = dict(payload or {})
    return SecurityPolicy(
        runtime=RuntimeSafetyPolicy(**_filter_dataclass_kwargs(RuntimeSafetyPolicy, p.get("runtime", {}))),
        files=FileAccessPolicy(**_filter_dataclass_kwargs(FileAccessPolicy, p.get("files", {}))),
        memory=MemoryGuardPolicy(**_filter_dataclass_kwargs(MemoryGuardPolicy, p.get("memory", {}))),
        load=LoadSafetyPolicy(**_filter_dataclass_kwargs(LoadSafetyPolicy, p.get("load", {}))),
        checkpoints=CheckpointPolicy(**_filter_dataclass_kwargs(CheckpointPolicy, p.get("checkpoints", {}))),
        external_ai=ExternalAIPolicy(**_filter_dataclass_kwargs(ExternalAIPolicy, p.get("external_ai", {}))),
        signatures=SignaturePolicy(**_filter_dataclass_kwargs(SignaturePolicy, p.get("signatures", {}))),
        key_storage=KeyStoragePolicy(**_filter_dataclass_kwargs(KeyStoragePolicy, p.get("key_storage", {}))),
        metadata=dict(p.get("metadata", {})),
    ).normalized()


def _filter_dataclass_kwargs(cls: Any, values: Any) -> Dict[str, Any]:
    if not isinstance(values, Mapping):
        return {}
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in dict(values).items() if k in names}


# =============================================================================
# Path and file-access checks
# =============================================================================

def _is_relative_traversal(path_str: str) -> bool:
    parts = Path(path_str).parts
    return ".." in parts


def _looks_sensitive_path(path: Union[str, os.PathLike[str]]) -> bool:
    p = Path(path)
    lower = str(p).lower()
    if p.suffix.lower() in SENSITIVE_EXTENSIONS:
        return True
    return any(pattern in lower for pattern in PRIVATE_KEY_NAME_PATTERNS)


def resolve_safe_path(
    path: Union[str, os.PathLike[str]],
    *,
    policy: Optional[SecurityPolicy] = None,
    must_exist: bool = False,
    for_write: bool = False,
    purpose: str = "file_access",
) -> Path:
    report = check_path(path, policy=policy, must_exist=must_exist, for_write=for_write, purpose=purpose)
    report.raise_if_failed()
    resolved = report.metadata.get("resolved_path")
    if not resolved:
        raise UnsafePathError("Path resolution failed")
    return Path(str(resolved))


def check_path(
    path: Union[str, os.PathLike[str]],
    *,
    policy: Optional[SecurityPolicy] = None,
    must_exist: bool = False,
    for_write: bool = False,
    purpose: str = "file_access",
) -> SecurityReport:
    pol = (policy or production_policy()).normalized()
    fp = pol.files
    report = SecurityReport(ok=True, component="path", mode=pol.runtime.mode)
    raw = str(path or "").strip()
    if not raw:
        report.add("empty_path", "Path cannot be empty", severity=SecuritySeverity.ERROR)
        return report

    p = Path(raw).expanduser()
    if p.is_absolute() and not fp.allow_absolute_paths:
        # Absolute paths are still allowed if they are under explicitly allowed roots.
        pass
    if _is_relative_traversal(raw) and not fp.allow_path_traversal:
        report.add("path_traversal", f"Path traversal is not allowed: {raw!r}", severity=SecuritySeverity.ERROR)
        return report

    try:
        resolved = p.resolve(strict=False)
    except Exception as exc:
        report.add("path_resolve_failed", f"Could not resolve path: {exc}", severity=SecuritySeverity.ERROR)
        return report

    allowed_roots = [Path(r).expanduser().resolve() for r in fp.allowed_roots if str(r).strip()]
    if allowed_roots:
        ok_root = any(_path_is_under(resolved, root) for root in allowed_roots)
        if not ok_root:
            report.add(
                "path_outside_allowed_roots",
                "Path is outside configured allowed roots",
                severity=SecuritySeverity.ERROR,
                metadata={"path": _public_path(resolved, pol), "allowed_roots": [_public_path(r, pol) for r in allowed_roots]},
            )
    elif not fp.allow_absolute_paths and resolved.is_absolute() and pol.runtime.mode not in {RuntimeSecurityMode.DEMO.value, RuntimeSecurityMode.TEST.value}:
        report.add(
            "no_allowed_roots",
            "No allowed roots configured for absolute path access",
            severity=SecuritySeverity.ERROR,
            metadata={"path": _public_path(resolved, pol)},
        )

    if must_exist and not resolved.exists():
        report.add("path_missing", "Path does not exist", severity=SecuritySeverity.ERROR, metadata={"path": _public_path(resolved, pol)})

    if resolved.exists():
        if resolved.is_symlink() and not fp.allow_symlinks:
            report.add("symlink_denied", "Symlink access is disabled", severity=SecuritySeverity.ERROR, metadata={"path": _public_path(resolved, pol)})
        if resolved.is_file():
            check_file_size_report = check_file_size(resolved, policy=pol)
            report.extend(check_file_size_report)
    elif for_write:
        parent = resolved.parent
        if parent.exists() and not parent.is_dir():
            report.add("parent_not_directory", "Parent path is not a directory", severity=SecuritySeverity.ERROR)

    ext = resolved.suffix.lower()
    if ext and fp.denied_extensions and ext in fp.denied_extensions:
        report.add("denied_extension", f"File extension {ext!r} is denied", severity=SecuritySeverity.ERROR)
    if ext and fp.allowed_extensions and ext not in fp.allowed_extensions and purpose not in {"key_storage", "internal_artifact"}:
        report.add("extension_not_allowlisted", f"File extension {ext!r} is not allowlisted", severity=SecuritySeverity.WARNING, decision=SecurityDecision.WARN)

    if _looks_sensitive_path(resolved) and not fp.allow_sensitive_key_paths and purpose != "key_storage":
        report.add("sensitive_path_denied", "Sensitive key-like path denied outside key-storage operations", severity=SecuritySeverity.ERROR)

    report.metadata.update({
        "path": _public_path(p, pol),
        "resolved_path": str(resolved),
        "public_resolved_path": _public_path(resolved, pol),
        "purpose": purpose,
        "for_write": bool(for_write),
        "must_exist": bool(must_exist),
    })
    return report


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def _public_path(path: Union[str, os.PathLike[str]], policy: SecurityPolicy) -> str:
    p = Path(path)
    if not policy.files.redact_absolute_paths:
        return str(p)
    try:
        resolved = p.expanduser().resolve(strict=False)
        for root_s in policy.files.allowed_roots:
            root = Path(root_s).expanduser().resolve(strict=False)
            try:
                return str(resolved.relative_to(root))
            except Exception:
                continue
        if resolved.is_absolute():
            return f"<redacted>/{resolved.name}"
    except Exception:
        pass
    return str(p.name if p.name else p)


def check_file_size(path: Union[str, os.PathLike[str]], *, policy: Optional[SecurityPolicy] = None, max_bytes: Optional[int] = None) -> SecurityReport:
    pol = (policy or production_policy()).normalized()
    report = SecurityReport(ok=True, component="file_size", mode=pol.runtime.mode)
    p = Path(path)
    try:
        size = p.stat().st_size
    except Exception as exc:
        report.add("stat_failed", f"Could not stat file: {exc}", severity=SecuritySeverity.ERROR)
        return report
    limit = int(max_bytes if max_bytes is not None else pol.files.max_file_bytes)
    if size > limit:
        report.add(
            "file_too_large",
            f"File exceeds size policy: {size} > {limit}",
            severity=SecuritySeverity.ERROR,
            metadata={"size_bytes": size, "max_bytes": limit, "path": _public_path(p, pol)},
        )
    report.metadata.update({"size_bytes": int(size), "max_bytes": int(limit), "path": _public_path(p, pol)})
    return report


def hash_local_file_if_allowed(path: Union[str, os.PathLike[str]], *, policy: Optional[SecurityPolicy] = None) -> Tuple[Optional[str], SecurityReport]:
    pol = (policy or production_policy()).normalized()
    report = check_path(path, policy=pol, must_exist=True, purpose="local_file_hashing")
    if not pol.files.allow_local_file_hashing:
        report.add("local_hashing_denied", "Local file hashing is disabled by policy", severity=SecuritySeverity.ERROR)
    if not report.ok:
        return None, report
    digest = hash_file(Path(str(report.metadata["resolved_path"])))
    report.metadata["sha256"] = digest
    return digest, report


# =============================================================================
# Memory, dense materialization, and parameter-budget guards
# =============================================================================

def estimate_dense_bytes(shape: Sequence[int], *, dtype_bytes: int = 4) -> int:
    entries = 1
    for d in shape:
        entries *= max(0, int(d))
    return int(entries * max(1, int(dtype_bytes)))


def check_dense_materialization(
    shape: Sequence[int],
    *,
    dtype_bytes: int = 4,
    policy: Optional[SecurityPolicy] = None,
    purpose: str = "dense_materialization",
) -> SecurityReport:
    pol = (policy or production_policy()).normalized()
    report = SecurityReport(ok=True, component="memory", mode=pol.runtime.mode)
    dims = [int(x) for x in shape]
    entries = 1
    for d in dims:
        if d < 0:
            report.add("negative_dimension", "Dense shape contains negative dimension", severity=SecuritySeverity.ERROR, metadata={"shape": dims})
            return report
        entries *= d
    bytes_needed = int(entries * max(1, int(dtype_bytes)))
    if entries > pol.memory.max_dense_entries:
        severity = SecuritySeverity.WARNING if (pol.runtime.mode == RuntimeSecurityMode.DEMO.value and pol.memory.allow_large_allocations_in_demo) else SecuritySeverity.ERROR
        decision = SecurityDecision.WARN if severity == SecuritySeverity.WARNING else SecurityDecision.DENY
        report.add(
            "dense_entries_exceeded",
            f"Dense materialization entries exceed guard: {entries} > {pol.memory.max_dense_entries}",
            severity=severity,
            decision=decision,
            metadata={"shape": dims, "entries": entries, "max_dense_entries": pol.memory.max_dense_entries},
        )
    if bytes_needed > pol.memory.max_dense_bytes:
        severity = SecuritySeverity.WARNING if (pol.runtime.mode == RuntimeSecurityMode.DEMO.value and pol.memory.allow_large_allocations_in_demo) else SecuritySeverity.ERROR
        decision = SecurityDecision.WARN if severity == SecuritySeverity.WARNING else SecurityDecision.DENY
        report.add(
            "dense_bytes_exceeded",
            f"Dense materialization bytes exceed guard: {bytes_needed} > {pol.memory.max_dense_bytes}",
            severity=severity,
            decision=decision,
            metadata={"shape": dims, "bytes_needed": bytes_needed, "max_dense_bytes": pol.memory.max_dense_bytes},
        )
    report.metadata.update({"shape": dims, "entries": int(entries), "bytes_needed": int(bytes_needed), "purpose": purpose})
    return report


def ensure_dense_materialization_allowed(shape: Sequence[int], *, dtype_bytes: int = 4, policy: Optional[SecurityPolicy] = None) -> None:
    r = check_dense_materialization(shape, dtype_bytes=dtype_bytes, policy=policy)
    r.raise_if_failed()


def check_lobe_parameter_budget(
    *,
    lobe_id: str,
    parameter_count: int,
    dense_equivalent_parameter_count: Optional[int] = None,
    active_total_parameters: Optional[int] = None,
    policy: Optional[SecurityPolicy] = None,
) -> SecurityReport:
    pol = (policy or production_policy()).normalized()
    report = SecurityReport(ok=True, component="lobe_budget", mode=pol.runtime.mode)
    pc = int(parameter_count)
    depc = int(dense_equivalent_parameter_count) if dense_equivalent_parameter_count is not None else None
    if pc > pol.memory.max_lobe_parameters:
        report.add(
            "lobe_active_parameters_exceeded",
            f"Lobe active parameter count exceeds policy: {pc} > {pol.memory.max_lobe_parameters}",
            severity=SecuritySeverity.ERROR,
            metadata={"lobe_id": lobe_id, "parameter_count": pc},
        )
    if depc is not None and depc > pol.memory.max_lobe_parameters:
        report.add(
            "lobe_dense_equivalent_exceeded",
            f"Lobe dense-equivalent parameter count exceeds policy: {depc} > {pol.memory.max_lobe_parameters}",
            severity=SecuritySeverity.ERROR,
            metadata={"lobe_id": lobe_id, "dense_equivalent_parameter_count": depc},
        )
    if active_total_parameters is not None and int(active_total_parameters) > pol.memory.max_active_parameters_total:
        report.add(
            "total_active_parameters_exceeded",
            f"Total active parameter budget exceeded: {active_total_parameters} > {pol.memory.max_active_parameters_total}",
            severity=SecuritySeverity.ERROR,
            metadata={"active_total_parameters": int(active_total_parameters)},
        )
    report.metadata.update({
        "lobe_id": str(lobe_id),
        "parameter_count": pc,
        "dense_equivalent_parameter_count": depc,
        "active_total_parameters": active_total_parameters,
    })
    return report


# =============================================================================
# Load/checkpoint verification policy
# =============================================================================

def check_json_load_allowed(
    path: Union[str, os.PathLike[str]],
    *,
    policy: Optional[SecurityPolicy] = None,
    expected_hash: Optional[str] = None,
    max_bytes: Optional[int] = None,
) -> SecurityReport:
    pol = (policy or production_policy()).normalized()
    report = check_path(path, policy=pol, must_exist=True, purpose="json_load")
    if not report.ok:
        return report
    p = Path(str(report.metadata["resolved_path"]))
    size_report = check_file_size(p, policy=pol, max_bytes=max_bytes or pol.memory.max_json_bytes)
    report.extend(size_report)
    if p.suffix.lower() not in SAFE_TEXT_EXTENSIONS:
        report.add("json_extension_unusual", "JSON load path has unusual extension", severity=SecuritySeverity.WARNING, decision=SecurityDecision.WARN)
    if expected_hash:
        digest = hash_file(p)
        if digest.lower() != str(expected_hash).lower():
            report.add("json_hash_mismatch", "JSON file hash does not match expected hash", severity=SecuritySeverity.ERROR, metadata={"expected": expected_hash, "actual": digest})
        report.metadata["sha256"] = digest
    elif pol.load.require_hash_verification and pol.runtime.mode not in {RuntimeSecurityMode.DEMO.value, RuntimeSecurityMode.TEST.value}:
        report.add("json_expected_hash_missing", "Production JSON loads should provide expected_hash", severity=SecuritySeverity.WARNING, decision=SecurityDecision.WARN)
    return report


def check_npz_load_allowed(
    path: Union[str, os.PathLike[str]],
    *,
    policy: Optional[SecurityPolicy] = None,
    allow_pickle: bool = False,
    expected_hash: Optional[str] = None,
) -> SecurityReport:
    pol = (policy or production_policy()).normalized()
    report = check_path(path, policy=pol, must_exist=True, purpose="npz_load")
    if allow_pickle or pol.load.allow_pickle:
        report.add("pickle_load_denied", "NumPy allow_pickle=True is denied by Tensorized Blockchain policy", severity=SecuritySeverity.ERROR, recoverable=False)
    if not report.ok:
        return report
    p = Path(str(report.metadata["resolved_path"]))
    if p.suffix.lower() not in {".npz", ".npy"}:
        report.add("npz_extension_invalid", "Expected .npz or .npy checkpoint/artifact", severity=SecuritySeverity.ERROR)
    size_report = check_file_size(p, policy=pol, max_bytes=pol.memory.max_checkpoint_bytes)
    report.extend(size_report)
    digest = hash_file(p)
    report.metadata["sha256"] = digest
    if expected_hash and digest.lower() != str(expected_hash).lower():
        report.add("npz_hash_mismatch", "NPZ file hash does not match expected hash", severity=SecuritySeverity.ERROR, metadata={"expected": expected_hash, "actual": digest})
    elif pol.load.require_hash_verification and not expected_hash and pol.runtime.mode not in {RuntimeSecurityMode.DEMO.value, RuntimeSecurityMode.TEST.value}:
        report.add("npz_expected_hash_missing", "Production NPZ loads should provide expected_hash or verified manifest", severity=SecuritySeverity.WARNING, decision=SecurityDecision.WARN)
    return report


def check_checkpoint_manifest(
    manifest: Mapping[str, Any],
    *,
    artifact_path: Optional[Union[str, os.PathLike[str]]] = None,
    policy: Optional[SecurityPolicy] = None,
) -> SecurityReport:
    pol = (policy or production_policy()).normalized()
    report = SecurityReport(ok=True, component="checkpoint_manifest", mode=pol.runtime.mode)
    m = dict(manifest or {})
    if not m:
        if pol.load.require_manifest:
            report.add("manifest_missing", "Checkpoint manifest is required", severity=SecuritySeverity.ERROR)
        else:
            report.add("manifest_missing_demo", "Checkpoint manifest missing", severity=SecuritySeverity.WARNING, decision=SecurityDecision.WARN)
        return report

    artifact_hash = str(m.get("artifact_hash") or m.get("sha256") or m.get("checkpoint_hash") or "")
    manifest_hash = str(m.get("manifest_hash") or "")
    if pol.checkpoints.require_integrity_hash and not artifact_hash:
        report.add("artifact_hash_missing", "Checkpoint manifest lacks artifact/checkpoint hash", severity=SecuritySeverity.ERROR)
    if pol.checkpoints.require_manifest_hash and not manifest_hash:
        report.add("manifest_hash_missing", "Checkpoint manifest lacks manifest_hash", severity=SecuritySeverity.ERROR)
    if pol.checkpoints.require_lobe_manifest and str(m.get("kind", "")).lower().find("lobe") >= 0 and not m.get("lobe_manifest"):
        report.add("lobe_manifest_missing", "Lobe checkpoint manifest lacks lobe_manifest payload", severity=SecuritySeverity.ERROR)

    if artifact_path is not None:
        path_report = check_path(artifact_path, policy=pol, must_exist=True, purpose="checkpoint_artifact")
        report.extend(path_report)
        if path_report.ok and artifact_hash:
            digest = hash_file(Path(str(path_report.metadata["resolved_path"])))
            if digest.lower() != artifact_hash.lower():
                report.add("checkpoint_hash_mismatch", "Checkpoint artifact hash mismatch", severity=SecuritySeverity.ERROR, metadata={"expected": artifact_hash, "actual": digest})
            report.metadata["artifact_sha256"] = digest

    if manifest_hash:
        m_without_hash = {k: v for k, v in m.items() if k != "manifest_hash"}
        actual_manifest_hash = hash_object(m_without_hash, domain="TB_CHECKPOINT_MANIFEST_V1")
        # Some stores may use a raw canonical manifest hash rather than domain hash;
        # warn instead of deny unless this hash was clearly generated by this module.
        if actual_manifest_hash.lower() != manifest_hash.lower():
            report.add(
                "manifest_hash_unverified",
                "Manifest hash did not match tb_security domain hash; ensure producer uses same manifest-hash scheme",
                severity=SecuritySeverity.WARNING,
                decision=SecurityDecision.WARN,
                metadata={"expected": manifest_hash, "tb_security_actual": actual_manifest_hash},
            )
    report.metadata["manifest_keys"] = sorted(str(k) for k in m.keys())
    return report


def create_checkpoint_manifest(
    artifact_path: Union[str, os.PathLike[str]],
    *,
    kind: str,
    producer: str = "tb_security",
    metadata: Optional[Mapping[str, Any]] = None,
    policy: Optional[SecurityPolicy] = None,
) -> Dict[str, Any]:
    pol = (policy or production_policy()).normalized()
    p = resolve_safe_path(artifact_path, policy=pol, must_exist=True, purpose="checkpoint_manifest_create")
    artifact_hash = hash_file(p)
    manifest = {
        "format": "tb_checkpoint_manifest_v1",
        "kind": str(kind),
        "producer": str(producer),
        "created_at": now_iso_utc(),
        "artifact_name": p.name,
        "artifact_hash_algorithm": DEFAULT_HASH_ALGORITHM,
        "artifact_hash": artifact_hash,
        "artifact_size_bytes": int(p.stat().st_size),
        "metadata": _json_safe(metadata or {}),
    }
    manifest["manifest_hash"] = hash_object(manifest, domain="TB_CHECKPOINT_MANIFEST_V1")
    return manifest


# =============================================================================
# Runtime/consensus safety checks
# =============================================================================

def check_consensus_runtime(
    *,
    device: str = "cpu",
    dtype: str = "float32",
    uses_external_cortex: bool = False,
    uses_wall_clock: bool = False,
    uses_unseeded_randomness: bool = False,
    degraded_mode: bool = False,
    policy: Optional[SecurityPolicy] = None,
) -> SecurityReport:
    pol = (policy or production_policy()).normalized()
    report = SecurityReport(ok=True, component="consensus_runtime", mode=pol.runtime.mode)
    dev = str(device or "cpu").lower().strip()
    dt = str(dtype or "").lower().strip()
    if pol.runtime.consensus_safe:
        if pol.runtime.require_cpu_for_consensus and dev != "cpu" and not pol.runtime.allow_gpu_consensus:
            report.add("gpu_consensus_denied", "Consensus-safe mode requires CPU execution", severity=SecuritySeverity.ERROR, metadata={"device": dev})
        if dt not in set(pol.runtime.allowed_consensus_dtypes):
            report.add("dtype_not_consensus_safe", "Dtype is not allowlisted for consensus-safe execution", severity=SecuritySeverity.ERROR, metadata={"dtype": dt})
        if uses_external_cortex:
            report.add("external_cortex_consensus_denied", "External cortex/LLM is denied in consensus-safe mode", severity=SecuritySeverity.ERROR)
        if uses_wall_clock:
            report.add("wall_clock_consensus_denied", "Wall-clock time is denied in deterministic consensus paths", severity=SecuritySeverity.ERROR)
        if uses_unseeded_randomness:
            report.add("unseeded_randomness_denied", "Unseeded randomness is denied in consensus-safe mode", severity=SecuritySeverity.ERROR)
        if degraded_mode and not pol.runtime.allow_degraded_mode_in_production:
            report.add("degraded_mode_denied", "Degraded fallback mode is denied for production consensus", severity=SecuritySeverity.ERROR)
    report.metadata.update({
        "device": dev,
        "dtype": dt,
        "uses_external_cortex": bool(uses_external_cortex),
        "uses_wall_clock": bool(uses_wall_clock),
        "uses_unseeded_randomness": bool(uses_unseeded_randomness),
        "degraded_mode": bool(degraded_mode),
    })
    return report


def check_external_cortex_use(
    *,
    consensus_path: bool,
    output_committed: bool = False,
    prompt_bytes: int = 0,
    response_bytes: int = 0,
    network_calls: bool = False,
    policy: Optional[SecurityPolicy] = None,
) -> SecurityReport:
    pol = (policy or production_policy()).normalized()
    report = SecurityReport(ok=True, component="external_ai", mode=pol.runtime.mode)
    if not pol.external_ai.allow_external_cortex:
        report.add("external_cortex_disabled", "External cortex/LLM use is disabled by policy", severity=SecuritySeverity.ERROR)
    if consensus_path and not pol.external_ai.allow_external_cortex_in_consensus:
        report.add("external_cortex_consensus_denied", "External cortex/LLM use is denied in consensus path", severity=SecuritySeverity.ERROR)
    if consensus_path and pol.external_ai.require_committed_outputs_for_consensus and not output_committed:
        report.add("external_output_uncommitted", "External AI output must be committed before consensus use", severity=SecuritySeverity.ERROR)
    if network_calls and not pol.external_ai.allow_live_network_calls:
        report.add("live_network_call_denied", "Live network calls are denied by policy", severity=SecuritySeverity.ERROR)
    if int(prompt_bytes) > pol.external_ai.max_prompt_bytes:
        report.add("prompt_too_large", "External AI prompt exceeds policy limit", severity=SecuritySeverity.ERROR, metadata={"prompt_bytes": int(prompt_bytes)})
    if int(response_bytes) > pol.external_ai.max_response_bytes:
        report.add("response_too_large", "External AI response exceeds policy limit", severity=SecuritySeverity.ERROR, metadata={"response_bytes": int(response_bytes)})
    report.metadata.update({
        "consensus_path": bool(consensus_path),
        "output_committed": bool(output_committed),
        "prompt_bytes": int(prompt_bytes),
        "response_bytes": int(response_bytes),
        "network_calls": bool(network_calls),
    })
    return report


# =============================================================================
# Signature and key handling checks
# =============================================================================

def check_signature_policy(
    *,
    signature_scheme: str,
    public_key: Optional[str] = None,
    key_fingerprint: Optional[str] = None,
    signature: Optional[str] = None,
    purpose: str = "transaction",
    policy: Optional[SecurityPolicy] = None,
) -> SecurityReport:
    pol = (policy or production_policy()).normalized()
    report = SecurityReport(ok=True, component="signature", mode=pol.runtime.mode)
    scheme = str(signature_scheme or "").lower().strip()
    if scheme not in set(pol.signatures.allowed_schemes):
        report.add("signature_scheme_denied", f"Signature scheme {scheme!r} is not allowed", severity=SecuritySeverity.ERROR, metadata={"allowed_schemes": pol.signatures.allowed_schemes})
    if scheme == "hmac_demo" and not pol.signatures.allow_hmac_demo:
        report.add("hmac_demo_denied", "hmac_demo is not allowed outside demo/test policy", severity=SecuritySeverity.ERROR)
    if pol.signatures.require_public_key and not public_key:
        report.add("public_key_missing", "Public key is required", severity=SecuritySeverity.ERROR)
    if scheme == "ed25519" and public_key:
        pk = str(public_key).strip().lower().replace("0x", "")
        if len(pk) < pol.signatures.min_ed25519_public_key_hex_len:
            report.add("ed25519_public_key_too_short", "Ed25519 public key appears too short", severity=SecuritySeverity.ERROR, metadata={"public_key_length": len(pk)})
        if not re.fullmatch(r"[0-9a-f]+", pk):
            report.add("public_key_not_hex", "Public key should be hex encoded", severity=SecuritySeverity.ERROR)
    if pol.signatures.require_key_fingerprint and not key_fingerprint:
        report.add("key_fingerprint_missing", "Key fingerprint is required", severity=SecuritySeverity.WARNING, decision=SecurityDecision.WARN)
    if purpose == "validator_vote" and pol.signatures.require_validator_vote_signatures and not signature:
        report.add("validator_vote_signature_missing", "Validator vote signature is required", severity=SecuritySeverity.ERROR)
    if purpose == "transaction" and pol.signatures.require_transaction_signatures and not signature:
        report.add("transaction_signature_missing", "Transaction signature is required", severity=SecuritySeverity.ERROR)
    report.metadata.update({
        "signature_scheme": scheme,
        "purpose": str(purpose),
        "has_public_key": bool(public_key),
        "has_key_fingerprint": bool(key_fingerprint),
        "has_signature": bool(signature),
    })
    return report


def classify_key_material(value: Any) -> KeyMaterialKind:
    s = str(value or "").lower()
    if any(p in s for p in ("private", "secret", "seed", "mnemonic", "privkey")):
        return KeyMaterialKind.PRIVATE
    if any(p in s for p in ("public", "pubkey", "fingerprint", "address")):
        return KeyMaterialKind.PUBLIC
    return KeyMaterialKind.UNKNOWN


def key_fingerprint(public_key: Union[str, bytes], *, scheme: str = "ed25519") -> str:
    if isinstance(public_key, bytes):
        b = public_key
    else:
        s = str(public_key).strip().lower().replace("0x", "")
        try:
            b = bytes.fromhex(s)
        except Exception:
            b = s.encode("utf-8", errors="ignore")
    digest = hash_bytes(b)
    return f"{str(scheme).lower().strip()}:{digest[:32]}"


def check_key_storage_path(
    path: Union[str, os.PathLike[str]],
    *,
    key_kind: Union[str, KeyMaterialKind] = KeyMaterialKind.UNKNOWN,
    policy: Optional[SecurityPolicy] = None,
    must_exist: bool = False,
) -> SecurityReport:
    pol = (policy or production_policy()).normalized()
    report = check_path(path, policy=pol, must_exist=must_exist, purpose="key_storage")
    kind = key_kind if isinstance(key_kind, KeyMaterialKind) else classify_key_material(key_kind)
    p = Path(str(path)).expanduser()
    if kind == KeyMaterialKind.PRIVATE:
        if not pol.key_storage.allow_plaintext_private_keys:
            report.add("plaintext_private_key_warning", "Plaintext private-key storage is discouraged/disabled by policy", severity=SecuritySeverity.WARNING, decision=SecurityDecision.WARN)
        if p.exists() and pol.key_storage.require_private_key_file_permissions:
            mode = stat.S_IMODE(p.stat().st_mode)
            # Warn/deny if group/world readable/writable/executable.
            if mode & 0o077:
                severity = SecuritySeverity.ERROR if pol.runtime.mode in {RuntimeSecurityMode.PRODUCTION.value, RuntimeSecurityMode.CONSENSUS_SAFE.value} else SecuritySeverity.WARNING
                decision = SecurityDecision.DENY if severity == SecuritySeverity.ERROR else SecurityDecision.WARN
                report.add(
                    "private_key_permissions_unsafe",
                    f"Private key file permissions are too broad: {oct(mode)}; recommended {pol.key_storage.recommended_private_key_mode_octal}",
                    severity=severity,
                    decision=decision,
                    metadata={"mode": oct(mode), "recommended": pol.key_storage.recommended_private_key_mode_octal},
                )
    report.metadata.update({"key_kind": kind.value, "path": _public_path(p, pol)})
    return report


def redact_key_material(payload: Any) -> Any:
    """Return payload with obvious key material redacted. Does not mutate input."""
    if isinstance(payload, Mapping):
        out: Dict[str, Any] = {}
        for k, v in payload.items():
            kl = str(k).lower()
            if any(p in kl for p in ("private", "secret", "seed", "mnemonic", "password", "token")):
                out[str(k)] = "<redacted>"
            else:
                out[str(k)] = redact_key_material(v)
        return out
    if isinstance(payload, list):
        return [redact_key_material(x) for x in payload]
    if isinstance(payload, tuple):
        return tuple(redact_key_material(x) for x in payload)
    return payload


# =============================================================================
# Aggregate security checks
# =============================================================================

def validate_security_policy(policy: SecurityPolicy) -> SecurityReport:
    pol = policy.normalized()
    report = SecurityReport(ok=True, component="security_policy", mode=pol.runtime.mode)
    if pol.runtime.consensus_safe:
        if pol.load.allow_pickle:
            report.add("pickle_allowed_consensus", "Pickle loading cannot be enabled in consensus-safe mode", severity=SecuritySeverity.CRITICAL, recoverable=False)
        if pol.external_ai.allow_external_cortex_in_consensus:
            report.add("external_ai_allowed_consensus", "External AI cannot be enabled in consensus-safe mode", severity=SecuritySeverity.CRITICAL, recoverable=False)
        if pol.runtime.allow_gpu_consensus and pol.runtime.require_cpu_for_consensus:
            report.add("gpu_policy_conflict", "GPU consensus allowed while CPU is required", severity=SecuritySeverity.ERROR)
        if pol.signatures.allow_hmac_demo and pol.runtime.mode not in {RuntimeSecurityMode.DEMO.value, RuntimeSecurityMode.TEST.value}:
            report.add("hmac_demo_enabled_production", "hmac_demo must not be enabled in production/consensus policy", severity=SecuritySeverity.ERROR)
    if not pol.files.allowed_roots and pol.runtime.mode not in {RuntimeSecurityMode.DEMO.value, RuntimeSecurityMode.TEST.value}:
        report.add("allowed_roots_missing", "Production file policy should define allowed_roots", severity=SecuritySeverity.WARNING, decision=SecurityDecision.WARN)
    report.metadata.update({"policy_hash": pol.policy_hash(), "policy": pol.to_dict()})
    return report


def security_status(policy: Optional[SecurityPolicy] = None) -> Dict[str, Any]:
    pol = (policy or production_policy()).normalized()
    validation = validate_security_policy(pol)
    return {
        "ok": bool(validation.ok),
        "version": TB_SECURITY_VERSION,
        "ts": now_iso_utc(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "policy_hash": pol.policy_hash(),
        "mode": pol.runtime.mode,
        "consensus_safe": bool(pol.runtime.consensus_safe),
        "deterministic_replay": bool(pol.runtime.deterministic_replay),
        "validation": validation.to_dict(),
        "limits": {
            "max_dense_entries": pol.memory.max_dense_entries,
            "max_dense_bytes": pol.memory.max_dense_bytes,
            "max_file_bytes": pol.files.max_file_bytes,
            "max_checkpoint_bytes": pol.memory.max_checkpoint_bytes,
            "max_lobe_parameters": pol.memory.max_lobe_parameters,
            "max_active_parameters_total": pol.memory.max_active_parameters_total,
        },
    }


def require_ok(report: SecurityReport) -> SecurityReport:
    report.raise_if_failed()
    return report


# =============================================================================
# CLI and self-test
# =============================================================================

def run_self_test() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        pol = demo_policy(allowed_roots=[root])
        r_policy = validate_security_policy(pol)
        if not r_policy.ok:
            # Demo policy may warn, but should not deny.
            deny = [i for i in r_policy.issues if i.decision == SecurityDecision.DENY.value]
            if deny:
                raise AssertionError(r_policy.to_dict())

        f = root / "artifact.json"
        f.write_text('{"ok":true}', encoding="utf-8")
        r_path = check_path(f, policy=pol, must_exist=True)
        r_path.raise_if_failed()

        bad = root / ".." / "outside.json"
        r_bad = check_path(bad, policy=pol, must_exist=False)
        if r_bad.ok:
            raise AssertionError("Path traversal/outside-root check failed")

        digest, r_hash = hash_local_file_if_allowed(f, policy=pol)
        r_hash.raise_if_failed()
        if not digest or len(digest) != 64:
            raise AssertionError("File hash failed")

        r_dense_ok = check_dense_materialization((100, 100), policy=pol)
        r_dense_ok.raise_if_failed()
        strict_pol = production_policy(allowed_roots=[root])
        r_dense_bad = check_dense_materialization((10_000, 10_000), policy=strict_pol)
        if r_dense_bad.ok:
            raise AssertionError("Dense guard failed")

        r_sig_demo = check_signature_policy(
            signature_scheme="hmac_demo",
            public_key="00" * 32,
            key_fingerprint="hmac_demo:test",
            signature="aa",
            purpose="transaction",
            policy=pol,
        )
        r_sig_demo.raise_if_failed()
        r_sig_prod = check_signature_policy(
            signature_scheme="hmac_demo",
            public_key="00" * 32,
            key_fingerprint="hmac_demo:test",
            signature="aa",
            purpose="transaction",
            policy=strict_pol,
        )
        if r_sig_prod.ok:
            raise AssertionError("HMAC demo production denial failed")

        r_runtime = check_consensus_runtime(device="gpu", dtype="float32", policy=strict_pol)
        if r_runtime.ok:
            raise AssertionError("GPU consensus denial failed")

        ckpt = root / "checkpoint.npz"
        ckpt.write_bytes(b"dummy")
        manifest = create_checkpoint_manifest(ckpt, kind="lobe_checkpoint", metadata={"lobe_manifest": {"id": "demo"}}, policy=pol)
        r_manifest = check_checkpoint_manifest(manifest, artifact_path=ckpt, policy=pol)
        # Demo policy may warn about manifest hash scheme, but must not deny.
        if not r_manifest.ok:
            denies = [i for i in r_manifest.issues if i.decision == SecurityDecision.DENY.value]
            if denies:
                raise AssertionError(r_manifest.to_dict())

        return {
            "ok": True,
            "version": TB_SECURITY_VERSION,
            "policy_hash": pol.policy_hash(),
            "file_hash": digest,
            "status": security_status(pol),
            "reports": {
                "path": r_path.to_dict(),
                "dense_bad": r_dense_bad.to_dict(),
                "runtime_bad": r_runtime.to_dict(),
                "manifest": r_manifest.to_dict(),
            },
        }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tensorized Blockchain security policy utilities")
    p.add_argument("--mode", choices=["status", "self-test", "write-policy", "check-path"], default="status")
    p.add_argument("--runtime", choices=[m.value for m in RuntimeSecurityMode], default=RuntimeSecurityMode.PRODUCTION.value)
    p.add_argument("--root", action="append", default=[], help="Allowed root. Can be repeated.")
    p.add_argument("--path", default="", help="Path to check for --mode check-path.")
    p.add_argument("--output", default="", help="Output path for --mode write-policy.")
    p.add_argument("--pretty", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        policy = build_security_policy(mode=args.runtime, allowed_roots=args.root)
        if args.mode == "self-test":
            out = run_self_test()
        elif args.mode == "write-policy":
            if not args.output:
                raise ValueError("--output required for write-policy")
            save_security_policy(policy, args.output, pretty=True)
            out = {"ok": True, "saved": str(Path(args.output).resolve()), "policy_hash": policy.policy_hash()}
        elif args.mode == "check-path":
            if not args.path:
                raise ValueError("--path required for check-path")
            out = check_path(args.path, policy=policy, must_exist=False).to_dict()
        else:
            out = security_status(policy)
        print(json.dumps(_json_safe(out), indent=2 if args.pretty else None, sort_keys=True, ensure_ascii=False))
        return 0 if bool(out.get("ok", False)) else 2
    except Exception as exc:
        err = {"ok": False, "error": repr(exc), "version": TB_SECURITY_VERSION}
        print(json.dumps(err, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
