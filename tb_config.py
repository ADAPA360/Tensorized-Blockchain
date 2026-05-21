#!/usr/bin/env python3
# tb_config.py
r"""
Tensorized Blockchain central configuration
==========================================

Project Chimera / Akkurat - Tensorized Blockchain configuration layer.

This module is intentionally dependency-light and import-safe. It should be
importable by every other project module without creating circular dependencies.
It owns shared constants, runtime-mode policy, file-system safety policy, tensor
materialization limits, chain identity, lobe budgets, and consensus-safe flags.

Design goals
------------
- One source of truth for runtime assumptions.
- Safe defaults for deterministic replay and production consensus paths.
- Explicit separation between demo convenience and production safety.
- JSON-serializable dataclasses for app/demo/report integration.
- No NumPy, no local project imports, and no side effects at import time.

Public API
----------
- RuntimeMode
- DevicePolicy
- SignatureScheme
- ProjectIdentity
- ChainIdentity
- RuntimePolicy
- DenseMaterializationPolicy
- TensorPolicy
- LobeBudgetPolicy
- LobeScaleProfile
- FileSystemSafetyPolicy
- LedgerPolicy
- ConsensusPolicy
- TwinPolicy
- TensorizedBlockchainConfig
- build_default_config(...)
- runtime_profile(...)
- load_config(...)
- save_config(...)
- config_status(...)
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


# =============================================================================
# Version constants
# =============================================================================

TB_CONFIG_VERSION = "tb-config-v1.0-production"
TENSORIZED_BLOCKCHAIN_STACK_VERSION = "tensorized-blockchain-stack-v1.0-production"
TENSORIZED_BLOCKCHAIN_PROTOCOL_VERSION = "tb-protocol-v1"

EXPECTED_SCRIPT_NAMES: Tuple[str, ...] = (
    "tb_config.py",
    "tn.py",
    "digital_twin_kernel.py",
    "twin_anything.py",
    "tensorized_blockchain_kernel.py",
    "ncp.py",
    "cfc.py",
)

# These are policy labels, not hard imports. Individual modules may expose more
# detailed version strings internally.
EXPECTED_MODULE_ROLES: Dict[str, str] = {
    "tb_config.py": "central configuration and runtime policies",
    "tn.py": "deterministic tensor algebra substrate",
    "digital_twin_kernel.py": "canonical governed world-state engine",
    "twin_anything.py": "canonical twin materialization layer",
    "tensorized_blockchain_kernel.py": "cryptographic ledger and tensorized commitment layer",
    "ncp.py": "Liquid Reflex Lobe substrate",
    "cfc.py": "Predictive Transition Lobe substrate",
}

DEFAULT_CHAIN_ID = "tb-local-devnet-v1"
DEFAULT_NETWORK_ID = "local-devnet"
DEFAULT_PROJECT_NAME = "Project Chimera / Akkurat Tensorized Blockchain"

# 2.5B parameter budget per lobe as discussed in the dualistic brain design.
MAX_PARAMETERS_PER_LOBE = 2_500_000_000
DEFAULT_MAX_ACTIVE_PARAMETERS_TOTAL = 5_000_000_000

# Conservative default dense materialization guard. Individual tests may override.
DEFAULT_MAX_DENSE_ENTRIES = 1_000_000
DEFAULT_MAX_DENSE_BYTES = 512 * 1024 * 1024  # 512 MiB

# Default file safety limit for source manifests and app uploads.
DEFAULT_MAX_LOCAL_FILE_BYTES = 512 * 1024 * 1024  # 512 MiB


# =============================================================================
# Helpers
# =============================================================================

class ConfigError(ValueError):
    """Raised when configuration values are inconsistent or unsafe."""


def now_iso_utc() -> str:
    """Return an ISO-like UTC timestamp without requiring timezone packages."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _clip01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return float(max(0.0, min(1.0, v)))


def _positive_int(x: Any, default: int, *, minimum: int = 1, maximum: Optional[int] = None) -> int:
    try:
        v = int(x)
    except Exception:
        v = int(default)
    v = max(int(minimum), v)
    if maximum is not None:
        v = min(v, int(maximum))
    return int(v)


def _positive_float(x: Any, default: float, *, minimum: float = 0.0, maximum: Optional[float] = None) -> float:
    try:
        v = float(x)
    except Exception:
        v = float(default)
    if not math.isfinite(v):
        v = float(default)
    v = max(float(minimum), v)
    if maximum is not None:
        v = min(v, float(maximum))
    return float(v)


def slug(value: Any, *, default: str = "item", max_len: int = 96) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value or "").strip().lower())
    s = re.sub(r"-+", "-", s).strip("-._")
    if not s:
        s = default
    return s[: int(max_len)]


def json_safe(obj: Any) -> Any:
    """JSON-safe conversion used across config/status/receipt payloads."""
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return json_safe(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return json_safe(vars(obj))
        except Exception:
            pass
    return str(obj)


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(stable_json_dumps(obj).encode("utf-8", errors="ignore")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def atomic_write_text(path: Union[str, os.PathLike[str]], text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def _coerce_path_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _resolve_if_possible(path: Union[str, os.PathLike[str]]) -> Optional[Path]:
    try:
        return Path(path).expanduser().resolve()
    except Exception:
        return None


def _is_relative_safe(path_str: str) -> bool:
    if not path_str:
        return True
    p = Path(path_str)
    if p.is_absolute():
        return False
    parts = [x for x in p.parts if x not in ("", ".")]
    return ".." not in parts


def _redact_home(path: str) -> str:
    try:
        home = str(Path.home().resolve())
        p = str(Path(path).expanduser().resolve())
        if p.startswith(home):
            return "~" + p[len(home):]
    except Exception:
        pass
    return str(path)


# =============================================================================
# Enums
# =============================================================================

class RuntimeMode(str, Enum):
    DEMO = "demo"
    TEST = "test"
    PRODUCTION = "production"
    DETERMINISTIC_REPLAY = "deterministic_replay"


class DevicePolicy(str, Enum):
    CPU_ONLY = "cpu_only"
    CPU_FOR_COMMITMENTS_GPU_ALLOWED_FOR_INFERENCE = "cpu_commitments_gpu_inference"
    GPU_ALLOWED = "gpu_allowed"


class SignatureScheme(str, Enum):
    ED25519 = "ed25519"
    SECP256K1 = "secp256k1"
    HMAC_DEMO = "hmac_demo"
    UNSIGNED_DEMO = "unsigned_demo"


class TensorDType(str, Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    FLOAT16 = "float16"


class CheckpointPolicy(str, Enum):
    OFF = "off"
    STATE_ONLY = "state_only"
    FULL_WITH_MANIFEST = "full_with_manifest"
    FULL_WITH_MANIFEST_AND_HASH = "full_with_manifest_and_hash"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# Diagnostics
# =============================================================================

@dataclass
class DiagnosticEvent:
    code: str
    severity: str = Severity.INFO.value
    component: str = "tb_config"
    message: str = ""
    recoverable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "DiagnosticEvent":
        d = copy.deepcopy(self)
        d.code = str(d.code or "diagnostic")
        sev = str(d.severity or Severity.INFO.value).lower().strip()
        if sev not in {x.value for x in Severity}:
            sev = Severity.INFO.value
        d.severity = sev
        d.component = str(d.component or "tb_config")
        d.message = str(d.message or "")
        d.recoverable = bool(d.recoverable)
        d.metadata = dict(d.metadata or {})
        return d

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self.normalized()))


# =============================================================================
# Core config structures
# =============================================================================

@dataclass
class ProjectIdentity:
    project_name: str = DEFAULT_PROJECT_NAME
    stack_version: str = TENSORIZED_BLOCKCHAIN_STACK_VERSION
    protocol_version: str = TENSORIZED_BLOCKCHAIN_PROTOCOL_VERSION
    config_version: str = TB_CONFIG_VERSION
    license_label: str = "permissive"
    heritage_label: str = "AI heritage project"

    def normalized(self) -> "ProjectIdentity":
        p = copy.deepcopy(self)
        p.project_name = str(p.project_name or DEFAULT_PROJECT_NAME)
        p.stack_version = str(p.stack_version or TENSORIZED_BLOCKCHAIN_STACK_VERSION)
        p.protocol_version = str(p.protocol_version or TENSORIZED_BLOCKCHAIN_PROTOCOL_VERSION)
        p.config_version = str(p.config_version or TB_CONFIG_VERSION)
        p.license_label = str(p.license_label or "permissive")
        p.heritage_label = str(p.heritage_label or "AI heritage project")
        return p


@dataclass
class ChainIdentity:
    chain_id: str = DEFAULT_CHAIN_ID
    network_id: str = DEFAULT_NETWORK_ID
    genesis_label: str = "tensorized-blockchain-genesis-v1"
    namespace: str = "project-chimera-akkurat"
    operator_id: str = "local-operator"

    def normalized(self) -> "ChainIdentity":
        c = copy.deepcopy(self)
        c.chain_id = slug(c.chain_id, default=DEFAULT_CHAIN_ID, max_len=128)
        c.network_id = slug(c.network_id, default=DEFAULT_NETWORK_ID, max_len=128)
        c.genesis_label = str(c.genesis_label or "tensorized-blockchain-genesis-v1")
        c.namespace = slug(c.namespace, default="project-chimera-akkurat", max_len=128)
        c.operator_id = slug(c.operator_id, default="local-operator", max_len=128)
        return c

    def commitment_payload(self) -> Dict[str, Any]:
        c = self.normalized()
        return {
            "chain_id": c.chain_id,
            "network_id": c.network_id,
            "genesis_label": c.genesis_label,
            "namespace": c.namespace,
            "operator_id": c.operator_id,
        }

    def identity_hash(self) -> str:
        return sha256_hex({"domain": "TB_CHAIN_IDENTITY_V1", **self.commitment_payload()})


@dataclass
class RuntimePolicy:
    mode: str = RuntimeMode.DEMO.value
    deterministic: bool = False
    consensus_safe: bool = False
    allow_wall_clock: bool = True
    allow_external_cortex: bool = True
    allow_unseeded_rng: bool = True
    allow_gpu: bool = False
    cpu_required_for_commitments: bool = True
    allow_dense_fallback: bool = True
    strict_load_validation: bool = True
    strict_topology_validation: bool = True
    strict_signature_validation: bool = True
    require_transaction_signatures: bool = False
    default_signature_scheme: str = SignatureScheme.HMAC_DEMO.value
    allow_hmac_demo_signatures: bool = True
    allow_unsigned_demo_transactions: bool = False
    verify_npz_manifests: bool = True
    verify_checkpoints: bool = True
    fail_on_degraded_dependency: bool = False
    structured_diagnostics: bool = True

    def normalized(self) -> "RuntimePolicy":
        r = copy.deepcopy(self)
        mode = str(r.mode or RuntimeMode.DEMO.value).lower().strip()
        aliases = {
            "prod": RuntimeMode.PRODUCTION.value,
            "production": RuntimeMode.PRODUCTION.value,
            "replay": RuntimeMode.DETERMINISTIC_REPLAY.value,
            "deterministic": RuntimeMode.DETERMINISTIC_REPLAY.value,
            "deterministic-replay": RuntimeMode.DETERMINISTIC_REPLAY.value,
            "demo": RuntimeMode.DEMO.value,
            "test": RuntimeMode.TEST.value,
        }
        mode = aliases.get(mode, mode)
        if mode not in {x.value for x in RuntimeMode}:
            mode = RuntimeMode.DEMO.value
        r.mode = mode

        # Enforce safety implications by mode.
        if mode == RuntimeMode.PRODUCTION.value:
            r.deterministic = True
            r.consensus_safe = True
            r.allow_wall_clock = False
            r.allow_external_cortex = False
            r.allow_unseeded_rng = False
            r.cpu_required_for_commitments = True
            r.strict_load_validation = True
            r.strict_topology_validation = True
            r.strict_signature_validation = True
            r.require_transaction_signatures = True
            r.allow_hmac_demo_signatures = False
            r.allow_unsigned_demo_transactions = False
            r.verify_npz_manifests = True
            r.verify_checkpoints = True
            r.fail_on_degraded_dependency = True
            if r.default_signature_scheme in {SignatureScheme.HMAC_DEMO.value, SignatureScheme.UNSIGNED_DEMO.value}:
                r.default_signature_scheme = SignatureScheme.ED25519.value

        elif mode == RuntimeMode.DETERMINISTIC_REPLAY.value:
            r.deterministic = True
            r.consensus_safe = True
            r.allow_wall_clock = False
            r.allow_external_cortex = False
            r.allow_unseeded_rng = False
            r.allow_gpu = False
            r.cpu_required_for_commitments = True
            r.strict_load_validation = True
            r.strict_topology_validation = True
            r.verify_npz_manifests = True
            r.verify_checkpoints = True
            r.allow_unsigned_demo_transactions = False

        elif mode == RuntimeMode.TEST.value:
            r.deterministic = True
            r.consensus_safe = True
            r.allow_wall_clock = False
            r.allow_external_cortex = False
            r.allow_unseeded_rng = False
            r.cpu_required_for_commitments = True
            r.strict_load_validation = True
            r.strict_topology_validation = True
            r.verify_npz_manifests = True
            r.verify_checkpoints = True

        else:  # demo
            r.deterministic = bool(r.deterministic)
            r.consensus_safe = bool(r.consensus_safe)

        sig = str(r.default_signature_scheme or SignatureScheme.HMAC_DEMO.value).lower().strip()
        if sig not in {x.value for x in SignatureScheme}:
            sig = SignatureScheme.ED25519.value if r.require_transaction_signatures else SignatureScheme.HMAC_DEMO.value
        r.default_signature_scheme = sig

        for name in (
            "deterministic", "consensus_safe", "allow_wall_clock", "allow_external_cortex", "allow_unseeded_rng",
            "allow_gpu", "cpu_required_for_commitments", "allow_dense_fallback", "strict_load_validation",
            "strict_topology_validation", "strict_signature_validation", "require_transaction_signatures",
            "allow_hmac_demo_signatures", "allow_unsigned_demo_transactions", "verify_npz_manifests",
            "verify_checkpoints", "fail_on_degraded_dependency", "structured_diagnostics",
        ):
            setattr(r, name, bool(getattr(r, name)))
        return r

    def assert_consensus_safe(self) -> None:
        r = self.normalized()
        if not r.consensus_safe:
            raise ConfigError("Runtime policy is not consensus-safe.")
        if r.allow_wall_clock:
            raise ConfigError("Consensus-safe runtime cannot allow wall-clock inputs in consensus paths.")
        if r.allow_external_cortex:
            raise ConfigError("Consensus-safe runtime cannot allow external cortex/LLM calls in consensus paths.")
        if r.allow_unseeded_rng:
            raise ConfigError("Consensus-safe runtime cannot allow unseeded RNG.")
        if not r.cpu_required_for_commitments:
            raise ConfigError("Consensus-safe runtime must require CPU/device-independent tensor commitments.")


@dataclass
class DenseMaterializationPolicy:
    max_dense_entries: int = DEFAULT_MAX_DENSE_ENTRIES
    max_dense_bytes: int = DEFAULT_MAX_DENSE_BYTES
    default_to_matrix_guard: int = DEFAULT_MAX_DENSE_ENTRIES
    allow_dense_materialization_in_production: bool = False

    def normalized(self) -> "DenseMaterializationPolicy":
        d = copy.deepcopy(self)
        d.max_dense_entries = _positive_int(d.max_dense_entries, DEFAULT_MAX_DENSE_ENTRIES, minimum=1)
        d.max_dense_bytes = _positive_int(d.max_dense_bytes, DEFAULT_MAX_DENSE_BYTES, minimum=1024)
        d.default_to_matrix_guard = _positive_int(d.default_to_matrix_guard, d.max_dense_entries, minimum=1)
        d.default_to_matrix_guard = min(d.default_to_matrix_guard, d.max_dense_entries)
        d.allow_dense_materialization_in_production = bool(d.allow_dense_materialization_in_production)
        return d

    def check_entries(self, entries: int, *, dtype_bytes: int = 4, production: bool = False) -> None:
        d = self.normalized()
        entries = _positive_int(entries, 0, minimum=0)
        dtype_bytes = _positive_int(dtype_bytes, 4, minimum=1)
        if production and not d.allow_dense_materialization_in_production:
            raise ConfigError("Dense materialization is disabled in production mode.")
        if entries > d.max_dense_entries:
            raise MemoryError(f"Dense materialization would create {entries:,} entries, limit={d.max_dense_entries:,}.")
        byte_count = entries * dtype_bytes
        if byte_count > d.max_dense_bytes:
            raise MemoryError(f"Dense materialization would use {byte_count:,} bytes, limit={d.max_dense_bytes:,}.")


@dataclass
class TensorPolicy:
    default_dtype: str = TensorDType.FLOAT32.value
    commitment_dtype: str = TensorDType.FLOAT32.value
    allowed_commitment_dtypes: Tuple[str, ...] = (TensorDType.FLOAT32.value, TensorDType.FLOAT64.value)
    device_policy: str = DevicePolicy.CPU_FOR_COMMITMENTS_GPU_ALLOWED_FOR_INFERENCE.value
    default_tt_rank: int = 8
    max_tt_rank: int = 256
    default_bond_dim: int = 8
    max_bond_dim: int = 1024
    default_energy_tol: float = 0.999
    min_energy_tol: float = 0.90
    require_tensor_commitments: bool = True
    require_compression_reports: bool = True
    cpu_only_commitments: bool = True

    def normalized(self) -> "TensorPolicy":
        t = copy.deepcopy(self)
        allowed_dtypes = {x.value for x in TensorDType}
        t.default_dtype = str(t.default_dtype or TensorDType.FLOAT32.value).lower().strip()
        if t.default_dtype not in allowed_dtypes:
            t.default_dtype = TensorDType.FLOAT32.value
        t.commitment_dtype = str(t.commitment_dtype or TensorDType.FLOAT32.value).lower().strip()
        if t.commitment_dtype not in allowed_dtypes:
            t.commitment_dtype = TensorDType.FLOAT32.value
        t.allowed_commitment_dtypes = tuple(
            x for x in [str(v).lower().strip() for v in t.allowed_commitment_dtypes] if x in allowed_dtypes
        ) or (TensorDType.FLOAT32.value,)
        if t.commitment_dtype not in t.allowed_commitment_dtypes:
            t.allowed_commitment_dtypes = tuple(list(t.allowed_commitment_dtypes) + [t.commitment_dtype])
        dp = str(t.device_policy or DevicePolicy.CPU_ONLY.value).lower().strip()
        if dp not in {x.value for x in DevicePolicy}:
            dp = DevicePolicy.CPU_ONLY.value
        t.device_policy = dp
        t.default_tt_rank = _positive_int(t.default_tt_rank, 8, minimum=1, maximum=t.max_tt_rank or None)
        t.max_tt_rank = _positive_int(t.max_tt_rank, max(1, t.default_tt_rank), minimum=t.default_tt_rank)
        t.default_bond_dim = _positive_int(t.default_bond_dim, 8, minimum=1, maximum=t.max_bond_dim or None)
        t.max_bond_dim = _positive_int(t.max_bond_dim, max(1, t.default_bond_dim), minimum=t.default_bond_dim)
        t.min_energy_tol = _positive_float(t.min_energy_tol, 0.90, minimum=0.0, maximum=1.0)
        t.default_energy_tol = _positive_float(t.default_energy_tol, 0.999, minimum=t.min_energy_tol, maximum=1.0)
        t.require_tensor_commitments = bool(t.require_tensor_commitments)
        t.require_compression_reports = bool(t.require_compression_reports)
        t.cpu_only_commitments = bool(t.cpu_only_commitments)
        if t.device_policy == DevicePolicy.CPU_ONLY.value:
            t.cpu_only_commitments = True
        return t

    def assert_commitment_safe(self, *, dtype: Optional[str] = None, device: str = "cpu") -> None:
        t = self.normalized()
        dt = str(dtype or t.commitment_dtype).lower().strip()
        if dt not in t.allowed_commitment_dtypes:
            raise ConfigError(f"Tensor dtype {dt!r} is not allowed for commitments.")
        dev = str(device or "cpu").lower().strip()
        if t.cpu_only_commitments and dev != "cpu":
            raise ConfigError("Tensor commitments must be CPU/device-independent under this policy.")


@dataclass
class LobeScaleProfile:
    lobe_id: str
    lobe_kind: str  # "ncp" | "cfc" | future lobe kinds
    active: bool = True
    parameter_budget: int = 25_000_000
    dense_equivalent_budget: int = 250_000_000
    hidden_size: int = 1024
    num_cells: Optional[int] = None
    tt_bond_rank: int = 8
    precision: str = TensorDType.FLOAT32.value
    update_frequency_hz: float = 1.0
    checkpoint_policy: str = CheckpointPolicy.FULL_WITH_MANIFEST_AND_HASH.value
    consensus_safe: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "LobeScaleProfile":
        p = copy.deepcopy(self)
        p.lobe_id = slug(p.lobe_id, default="lobe", max_len=128)
        p.lobe_kind = slug(p.lobe_kind, default="generic_lobe", max_len=64)
        p.active = bool(p.active)
        p.parameter_budget = _positive_int(p.parameter_budget, 25_000_000, minimum=0, maximum=MAX_PARAMETERS_PER_LOBE)
        p.dense_equivalent_budget = _positive_int(p.dense_equivalent_budget, max(1, p.parameter_budget), minimum=0)
        p.hidden_size = _positive_int(p.hidden_size, 1024, minimum=1)
        p.num_cells = None if p.num_cells is None else _positive_int(p.num_cells, 1, minimum=1)
        p.tt_bond_rank = _positive_int(p.tt_bond_rank, 8, minimum=1)
        p.precision = str(p.precision or TensorDType.FLOAT32.value).lower().strip()
        if p.precision not in {x.value for x in TensorDType} | {"int8", "int4"}:
            p.precision = TensorDType.FLOAT32.value
        p.update_frequency_hz = _positive_float(p.update_frequency_hz, 1.0, minimum=0.0)
        cp = str(p.checkpoint_policy or CheckpointPolicy.FULL_WITH_MANIFEST_AND_HASH.value).lower().strip()
        if cp not in {x.value for x in CheckpointPolicy}:
            cp = CheckpointPolicy.FULL_WITH_MANIFEST_AND_HASH.value
        p.checkpoint_policy = cp
        p.consensus_safe = bool(p.consensus_safe)
        p.metadata = dict(p.metadata or {})
        return p

    def commitment_payload(self) -> Dict[str, Any]:
        p = self.normalized()
        return {
            "domain": "TB_LOBE_SCALE_PROFILE_V1",
            "lobe_id": p.lobe_id,
            "lobe_kind": p.lobe_kind,
            "active": p.active,
            "parameter_budget": p.parameter_budget,
            "dense_equivalent_budget": p.dense_equivalent_budget,
            "hidden_size": p.hidden_size,
            "num_cells": p.num_cells,
            "tt_bond_rank": p.tt_bond_rank,
            "precision": p.precision,
            "update_frequency_hz": p.update_frequency_hz,
            "checkpoint_policy": p.checkpoint_policy,
            "consensus_safe": p.consensus_safe,
            "metadata": json_safe(p.metadata),
        }

    def profile_hash(self) -> str:
        return sha256_hex(self.commitment_payload())

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self.normalized()))


@dataclass
class LobeBudgetPolicy:
    max_parameters_per_lobe: int = MAX_PARAMETERS_PER_LOBE
    max_active_parameters_total: int = DEFAULT_MAX_ACTIVE_PARAMETERS_TOTAL
    default_demo_ncp_budget: int = 25_000_000
    default_demo_cfc_budget: int = 25_000_000
    default_production_ncp_budget: int = 250_000_000
    default_production_cfc_budget: int = 250_000_000
    allow_elastic_scaling: bool = True
    require_lobe_scale_events: bool = True
    require_lobe_health_receipts: bool = True
    require_lobe_checkpoint_hashes: bool = True

    def normalized(self) -> "LobeBudgetPolicy":
        p = copy.deepcopy(self)
        p.max_parameters_per_lobe = _positive_int(p.max_parameters_per_lobe, MAX_PARAMETERS_PER_LOBE, minimum=1)
        p.max_active_parameters_total = _positive_int(p.max_active_parameters_total, DEFAULT_MAX_ACTIVE_PARAMETERS_TOTAL, minimum=1)
        p.default_demo_ncp_budget = _positive_int(p.default_demo_ncp_budget, 25_000_000, minimum=0, maximum=p.max_parameters_per_lobe)
        p.default_demo_cfc_budget = _positive_int(p.default_demo_cfc_budget, 25_000_000, minimum=0, maximum=p.max_parameters_per_lobe)
        p.default_production_ncp_budget = _positive_int(p.default_production_ncp_budget, 250_000_000, minimum=0, maximum=p.max_parameters_per_lobe)
        p.default_production_cfc_budget = _positive_int(p.default_production_cfc_budget, 250_000_000, minimum=0, maximum=p.max_parameters_per_lobe)
        p.allow_elastic_scaling = bool(p.allow_elastic_scaling)
        p.require_lobe_scale_events = bool(p.require_lobe_scale_events)
        p.require_lobe_health_receipts = bool(p.require_lobe_health_receipts)
        p.require_lobe_checkpoint_hashes = bool(p.require_lobe_checkpoint_hashes)
        return p

    def validate_profiles(self, profiles: Sequence[LobeScaleProfile]) -> None:
        p = self.normalized()
        total = 0
        for profile in profiles:
            pr = profile.normalized()
            if pr.parameter_budget > p.max_parameters_per_lobe:
                raise ConfigError(f"Lobe {pr.lobe_id} parameter budget exceeds per-lobe limit.")
            if pr.active:
                total += int(pr.parameter_budget)
        if total > p.max_active_parameters_total:
            raise ConfigError(f"Active lobe parameter budget {total:,} exceeds total limit {p.max_active_parameters_total:,}.")

    def default_profiles(self, *, mode: str = RuntimeMode.DEMO.value) -> List[LobeScaleProfile]:
        p = self.normalized()
        prod_like = str(mode) in {RuntimeMode.PRODUCTION.value, RuntimeMode.DETERMINISTIC_REPLAY.value}
        ncp_budget = p.default_production_ncp_budget if prod_like else p.default_demo_ncp_budget
        cfc_budget = p.default_production_cfc_budget if prod_like else p.default_demo_cfc_budget
        return [
            LobeScaleProfile(
                lobe_id="liquid-reflex-ncp",
                lobe_kind="ncp",
                parameter_budget=ncp_budget,
                dense_equivalent_budget=min(p.max_parameters_per_lobe, max(ncp_budget, ncp_budget * 10)),
                hidden_size=4096 if prod_like else 1024,
                num_cells=None,
                tt_bond_rank=16 if prod_like else 8,
                update_frequency_hz=5.0 if prod_like else 1.0,
                consensus_safe=prod_like,
            ).normalized(),
            LobeScaleProfile(
                lobe_id="predictive-transition-cfc",
                lobe_kind="cfc",
                parameter_budget=cfc_budget,
                dense_equivalent_budget=min(p.max_parameters_per_lobe, max(cfc_budget, cfc_budget * 10)),
                hidden_size=2048 if prod_like else 512,
                num_cells=16 if prod_like else 4,
                tt_bond_rank=16 if prod_like else 8,
                update_frequency_hz=5.0 if prod_like else 1.0,
                consensus_safe=prod_like,
            ).normalized(),
        ]


@dataclass
class FileSystemSafetyPolicy:
    project_root: str = ""
    allowed_roots: List[str] = field(default_factory=list)
    allow_absolute_paths: bool = False
    allow_local_file_hashing: bool = False
    redact_absolute_paths: bool = True
    max_local_file_bytes: int = DEFAULT_MAX_LOCAL_FILE_BYTES
    allowed_manifest_extensions: Tuple[str, ...] = (
        ".txt", ".md", ".json", ".csv", ".yaml", ".yml", ".py", ".npz", ".npy", ".pdf",
    )

    def normalized(self) -> "FileSystemSafetyPolicy":
        f = copy.deepcopy(self)
        f.project_root = _coerce_path_str(f.project_root)
        roots = []
        if f.project_root:
            roots.append(f.project_root)
        roots.extend([_coerce_path_str(x) for x in list(f.allowed_roots or []) if _coerce_path_str(x)])
        # Preserve order but remove duplicates.
        seen = set()
        deduped = []
        for r in roots:
            rr = str(r)
            if rr not in seen:
                seen.add(rr)
                deduped.append(rr)
        f.allowed_roots = deduped
        f.allow_absolute_paths = bool(f.allow_absolute_paths)
        f.allow_local_file_hashing = bool(f.allow_local_file_hashing)
        f.redact_absolute_paths = bool(f.redact_absolute_paths)
        f.max_local_file_bytes = _positive_int(f.max_local_file_bytes, DEFAULT_MAX_LOCAL_FILE_BYTES, minimum=1024)
        f.allowed_manifest_extensions = tuple(
            str(x).lower().strip() for x in f.allowed_manifest_extensions if str(x).strip()
        ) or (".txt", ".json", ".py")
        return f

    def resolved_allowed_roots(self) -> List[Path]:
        f = self.normalized()
        out: List[Path] = []
        for root in f.allowed_roots:
            rp = _resolve_if_possible(root)
            if rp is not None:
                out.append(rp)
        return out

    def is_path_allowed(self, path: Union[str, os.PathLike[str]]) -> bool:
        f = self.normalized()
        s = _coerce_path_str(path)
        if not s:
            return False
        p_raw = Path(s)
        if p_raw.is_absolute() and not f.allow_absolute_paths:
            # Absolute paths are allowed only if they are under an allowed root.
            rp = _resolve_if_possible(s)
            if rp is None:
                return False
            return any(str(rp).startswith(str(root)) for root in f.resolved_allowed_roots())
        if not p_raw.is_absolute() and not _is_relative_safe(s):
            return False
        rp = _resolve_if_possible(s)
        if rp is None:
            return False
        roots = f.resolved_allowed_roots()
        if roots:
            return any(str(rp).startswith(str(root)) for root in roots)
        return not p_raw.is_absolute()

    def validate_manifest_file(self, path: Union[str, os.PathLike[str]], *, require_exists: bool = False) -> Dict[str, Any]:
        f = self.normalized()
        s = _coerce_path_str(path)
        p = Path(s)
        suffix = p.suffix.lower()
        diagnostics: List[DiagnosticEvent] = []
        ok = True
        if suffix and suffix not in f.allowed_manifest_extensions:
            ok = False
            diagnostics.append(DiagnosticEvent("extension_not_allowed", Severity.ERROR.value, "filesystem", f"Extension {suffix!r} is not allowed.", False))
        if not f.is_path_allowed(s):
            ok = False
            diagnostics.append(DiagnosticEvent("path_not_allowed", Severity.ERROR.value, "filesystem", "Path is outside allowed roots or violates path policy.", False))
        if require_exists:
            rp = _resolve_if_possible(s)
            if rp is None or not rp.exists() or not rp.is_file():
                ok = False
                diagnostics.append(DiagnosticEvent("file_missing", Severity.ERROR.value, "filesystem", "File does not exist.", False))
            elif rp.stat().st_size > f.max_local_file_bytes:
                ok = False
                diagnostics.append(DiagnosticEvent("file_too_large", Severity.ERROR.value, "filesystem", "File exceeds max_local_file_bytes.", False))
        return {
            "ok": bool(ok),
            "path": f.public_path(s),
            "diagnostics": [d.to_dict() for d in diagnostics],
        }

    def public_path(self, path: Union[str, os.PathLike[str]]) -> str:
        f = self.normalized()
        s = _coerce_path_str(path)
        if not s:
            return ""
        p = Path(s)
        if not f.redact_absolute_paths:
            return s
        if p.is_absolute():
            # Try relative-to allowed roots first, then redact home.
            rp = _resolve_if_possible(s)
            if rp is not None:
                for root in f.resolved_allowed_roots():
                    try:
                        rel = rp.relative_to(root)
                        return str(rel).replace("\\", "/")
                    except Exception:
                        pass
            return _redact_home(s)
        return s.replace("\\", "/")

    def hash_file_if_allowed(self, path: Union[str, os.PathLike[str]]) -> Optional[str]:
        f = self.normalized()
        if not f.allow_local_file_hashing:
            return None
        validation = f.validate_manifest_file(path, require_exists=True)
        if not validation.get("ok"):
            return None
        rp = _resolve_if_possible(path)
        if rp is None:
            return None
        h = hashlib.sha256()
        with rp.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()


@dataclass
class ConsensusPolicy:
    quorum_ratio: float = 0.66
    min_coherence_ok: float = 0.85
    max_validator_clock_skew_s: int = 5
    require_consensus_certificate: bool = True
    require_transition_receipts: bool = True
    require_pre_post_state_roots: bool = True
    require_policy_receipts: bool = True
    require_lobe_receipts_for_high_risk: bool = True
    max_peer_snapshot_age_s: int = 300

    def normalized(self) -> "ConsensusPolicy":
        c = copy.deepcopy(self)
        c.quorum_ratio = _clip01(c.quorum_ratio, 0.66)
        if c.quorum_ratio <= 0:
            c.quorum_ratio = 0.66
        c.min_coherence_ok = _clip01(c.min_coherence_ok, 0.85)
        c.max_validator_clock_skew_s = _positive_int(c.max_validator_clock_skew_s, 5, minimum=0)
        c.require_consensus_certificate = bool(c.require_consensus_certificate)
        c.require_transition_receipts = bool(c.require_transition_receipts)
        c.require_pre_post_state_roots = bool(c.require_pre_post_state_roots)
        c.require_policy_receipts = bool(c.require_policy_receipts)
        c.require_lobe_receipts_for_high_risk = bool(c.require_lobe_receipts_for_high_risk)
        c.max_peer_snapshot_age_s = _positive_int(c.max_peer_snapshot_age_s, 300, minimum=1)
        return c

    def quorum_required(self, validator_count: int) -> int:
        c = self.normalized()
        n = _positive_int(validator_count, 1, minimum=1)
        return max(1, int(math.ceil(float(n) * c.quorum_ratio)))


@dataclass
class LedgerPolicy:
    max_transactions_per_block: int = 10_000
    max_block_bytes: int = 32 * 1024 * 1024
    max_transaction_bytes: int = 512 * 1024
    require_domain_separated_merkle: bool = True
    require_public_key_signatures: bool = False
    reject_replay_transactions: bool = True
    enforce_sender_sequence: bool = True
    validate_on_load: bool = True
    include_tensor_commitment: bool = True
    include_compression_report_commitment: bool = True
    include_lobe_receipts: bool = True

    def normalized(self) -> "LedgerPolicy":
        l = copy.deepcopy(self)
        l.max_transactions_per_block = _positive_int(l.max_transactions_per_block, 10_000, minimum=1)
        l.max_block_bytes = _positive_int(l.max_block_bytes, 32 * 1024 * 1024, minimum=1024)
        l.max_transaction_bytes = _positive_int(l.max_transaction_bytes, 512 * 1024, minimum=128)
        for name in (
            "require_domain_separated_merkle", "require_public_key_signatures", "reject_replay_transactions",
            "enforce_sender_sequence", "validate_on_load", "include_tensor_commitment",
            "include_compression_report_commitment", "include_lobe_receipts",
        ):
            setattr(l, name, bool(getattr(l, name)))
        return l


@dataclass
class TwinPolicy:
    schema_version: str = "thing_twin_schema_v1"
    deterministic_creation_default: bool = True
    require_twin_creation_receipt: bool = True
    require_state_commitment: bool = True
    include_fast_sketch_root: bool = True
    include_cryptographic_state_root: bool = True
    strict_topology_validation: bool = True
    allow_runtime_metadata_in_consensus_root: bool = False
    source_manifest_root_required: bool = True

    def normalized(self) -> "TwinPolicy":
        t = copy.deepcopy(self)
        t.schema_version = slug(t.schema_version, default="thing_twin_schema_v1", max_len=96)
        for name in (
            "deterministic_creation_default", "require_twin_creation_receipt", "require_state_commitment",
            "include_fast_sketch_root", "include_cryptographic_state_root", "strict_topology_validation",
            "allow_runtime_metadata_in_consensus_root", "source_manifest_root_required",
        ):
            setattr(t, name, bool(getattr(t, name)))
        return t


@dataclass
class DemoPolicy:
    default_vector_dim: int = 64
    default_sketch_dim: int = 32
    default_out_dim: int = 8
    default_in_dim: int = 8
    default_scenario: str = "alicante_fire_dept_wiser"
    enable_dual_lobes: bool = True
    enable_tensor_compression: bool = True
    save_demo_artifacts: bool = True
    pretty_json: bool = True

    def normalized(self) -> "DemoPolicy":
        d = copy.deepcopy(self)
        d.default_vector_dim = _positive_int(d.default_vector_dim, 64, minimum=8)
        d.default_sketch_dim = _positive_int(d.default_sketch_dim, min(32, d.default_vector_dim), minimum=8, maximum=d.default_vector_dim)
        d.default_out_dim = _positive_int(d.default_out_dim, 8, minimum=1)
        d.default_in_dim = _positive_int(d.default_in_dim, 8, minimum=1)
        d.default_scenario = slug(d.default_scenario, default="demo", max_len=128)
        d.enable_dual_lobes = bool(d.enable_dual_lobes)
        d.enable_tensor_compression = bool(d.enable_tensor_compression)
        d.save_demo_artifacts = bool(d.save_demo_artifacts)
        d.pretty_json = bool(d.pretty_json)
        return d


@dataclass
class TensorizedBlockchainConfig:
    project: ProjectIdentity = field(default_factory=ProjectIdentity)
    chain: ChainIdentity = field(default_factory=ChainIdentity)
    runtime: RuntimePolicy = field(default_factory=RuntimePolicy)
    dense: DenseMaterializationPolicy = field(default_factory=DenseMaterializationPolicy)
    tensor: TensorPolicy = field(default_factory=TensorPolicy)
    lobes: LobeBudgetPolicy = field(default_factory=LobeBudgetPolicy)
    filesystem: FileSystemSafetyPolicy = field(default_factory=FileSystemSafetyPolicy)
    consensus: ConsensusPolicy = field(default_factory=ConsensusPolicy)
    ledger: LedgerPolicy = field(default_factory=LedgerPolicy)
    twin: TwinPolicy = field(default_factory=TwinPolicy)
    demo: DemoPolicy = field(default_factory=DemoPolicy)
    lobe_profiles: List[LobeScaleProfile] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "TensorizedBlockchainConfig":
        c = copy.deepcopy(self)
        c.project = c.project.normalized() if isinstance(c.project, ProjectIdentity) else ProjectIdentity(**dict(c.project)).normalized()
        c.chain = c.chain.normalized() if isinstance(c.chain, ChainIdentity) else ChainIdentity(**dict(c.chain)).normalized()
        c.runtime = c.runtime.normalized() if isinstance(c.runtime, RuntimePolicy) else RuntimePolicy(**dict(c.runtime)).normalized()
        c.dense = c.dense.normalized() if isinstance(c.dense, DenseMaterializationPolicy) else DenseMaterializationPolicy(**dict(c.dense)).normalized()
        c.tensor = c.tensor.normalized() if isinstance(c.tensor, TensorPolicy) else TensorPolicy(**dict(c.tensor)).normalized()
        c.lobes = c.lobes.normalized() if isinstance(c.lobes, LobeBudgetPolicy) else LobeBudgetPolicy(**dict(c.lobes)).normalized()
        c.filesystem = c.filesystem.normalized() if isinstance(c.filesystem, FileSystemSafetyPolicy) else FileSystemSafetyPolicy(**dict(c.filesystem)).normalized()
        c.consensus = c.consensus.normalized() if isinstance(c.consensus, ConsensusPolicy) else ConsensusPolicy(**dict(c.consensus)).normalized()
        c.ledger = c.ledger.normalized() if isinstance(c.ledger, LedgerPolicy) else LedgerPolicy(**dict(c.ledger)).normalized()
        c.twin = c.twin.normalized() if isinstance(c.twin, TwinPolicy) else TwinPolicy(**dict(c.twin)).normalized()
        c.demo = c.demo.normalized() if isinstance(c.demo, DemoPolicy) else DemoPolicy(**dict(c.demo)).normalized()
        profiles: List[LobeScaleProfile] = []
        for p in list(c.lobe_profiles or []):
            if isinstance(p, LobeScaleProfile):
                profiles.append(p.normalized())
            elif isinstance(p, Mapping):
                profiles.append(LobeScaleProfile(**dict(p)).normalized())
        if not profiles:
            profiles = c.lobes.default_profiles(mode=c.runtime.mode)
        c.lobe_profiles = profiles
        c.lobes.validate_profiles(c.lobe_profiles)
        c.metadata = dict(c.metadata or {})

        # Cross-policy implications.
        if c.runtime.mode == RuntimeMode.PRODUCTION.value:
            c.ledger.require_public_key_signatures = True
            c.ledger.validate_on_load = True
            c.filesystem.allow_local_file_hashing = False if not c.filesystem.allowed_roots else c.filesystem.allow_local_file_hashing
            c.filesystem.allow_absolute_paths = False
            c.tensor.cpu_only_commitments = True
            c.dense.allow_dense_materialization_in_production = False
        if c.runtime.consensus_safe:
            c.runtime.assert_consensus_safe()
            c.tensor.assert_commitment_safe(device="cpu")
        return c

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self.normalized()))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TensorizedBlockchainConfig":
        p = dict(payload or {})
        def sub(name: str) -> Dict[str, Any]:
            value = p.get(name, {})
            return dict(value) if isinstance(value, Mapping) else {}
        profiles = []
        for item in list(p.get("lobe_profiles", []) or []):
            if isinstance(item, Mapping):
                profiles.append(LobeScaleProfile(**dict(item)))
        return cls(
            project=ProjectIdentity(**sub("project")),
            chain=ChainIdentity(**sub("chain")),
            runtime=RuntimePolicy(**sub("runtime")),
            dense=DenseMaterializationPolicy(**sub("dense")),
            tensor=TensorPolicy(**sub("tensor")),
            lobes=LobeBudgetPolicy(**sub("lobes")),
            filesystem=FileSystemSafetyPolicy(**sub("filesystem")),
            consensus=ConsensusPolicy(**sub("consensus")),
            ledger=LedgerPolicy(**sub("ledger")),
            twin=TwinPolicy(**sub("twin")),
            demo=DemoPolicy(**sub("demo")),
            lobe_profiles=profiles,
            metadata=dict(p.get("metadata", {}) if isinstance(p.get("metadata", {}), Mapping) else {}),
        ).normalized()

    def config_hash(self) -> str:
        return sha256_hex({"domain": "TB_CONFIG_V1", "config": self.to_dict()})

    def commitment_payload(self) -> Dict[str, Any]:
        c = self.normalized()
        return {
            "domain": "TB_CONFIG_COMMITMENT_V1",
            "project": c.project,
            "chain": c.chain.commitment_payload(),
            "runtime": c.runtime,
            "dense": c.dense,
            "tensor": c.tensor,
            "lobes": c.lobes,
            "consensus": c.consensus,
            "ledger": c.ledger,
            "twin": c.twin,
            "lobe_profile_hashes": [p.profile_hash() for p in c.lobe_profiles],
        }

    def commitment_hash(self) -> str:
        return sha256_hex(self.commitment_payload())

    def save_json(self, path: Union[str, os.PathLike[str]], *, pretty: bool = True) -> None:
        text = json.dumps(self.to_dict(), indent=2 if pretty else None, ensure_ascii=False, sort_keys=bool(pretty))
        atomic_write_text(path, text)

    @classmethod
    def load_json(cls, path: Union[str, os.PathLike[str]], *, validate: bool = True) -> "TensorizedBlockchainConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls.from_dict(payload)
        if validate:
            cfg.validate()
        return cfg

    def validate(self) -> Dict[str, Any]:
        c = self.normalized()
        diagnostics: List[DiagnosticEvent] = []
        ok = True
        try:
            c.lobes.validate_profiles(c.lobe_profiles)
        except Exception as exc:
            ok = False
            diagnostics.append(DiagnosticEvent("lobe_budget_invalid", Severity.ERROR.value, "lobes", str(exc), False))
        if c.runtime.consensus_safe:
            try:
                c.runtime.assert_consensus_safe()
                c.tensor.assert_commitment_safe(device="cpu")
            except Exception as exc:
                ok = False
                diagnostics.append(DiagnosticEvent("consensus_policy_invalid", Severity.ERROR.value, "runtime", str(exc), False))
        if c.runtime.mode == RuntimeMode.PRODUCTION.value and c.runtime.default_signature_scheme in {SignatureScheme.HMAC_DEMO.value, SignatureScheme.UNSIGNED_DEMO.value}:
            ok = False
            diagnostics.append(DiagnosticEvent("signature_policy_invalid", Severity.ERROR.value, "ledger", "Production mode requires public-key signatures.", False))
        return {
            "ok": bool(ok),
            "config_hash": c.config_hash(),
            "commitment_hash": c.commitment_hash(),
            "diagnostics": [d.normalized().to_dict() for d in diagnostics],
        }


# =============================================================================
# Builders / profiles
# =============================================================================

def runtime_profile(mode: Union[str, RuntimeMode] = RuntimeMode.DEMO.value) -> RuntimePolicy:
    """Return a normalized RuntimePolicy for a named mode."""
    m = mode.value if isinstance(mode, RuntimeMode) else str(mode or RuntimeMode.DEMO.value)
    if m == RuntimeMode.PRODUCTION.value:
        return RuntimePolicy(mode=m).normalized()
    if m == RuntimeMode.DETERMINISTIC_REPLAY.value:
        return RuntimePolicy(
            mode=m,
            require_transaction_signatures=True,
            default_signature_scheme=SignatureScheme.ED25519.value,
            allow_hmac_demo_signatures=False,
        ).normalized()
    if m == RuntimeMode.TEST.value:
        return RuntimePolicy(
            mode=m,
            require_transaction_signatures=False,
            default_signature_scheme=SignatureScheme.HMAC_DEMO.value,
            allow_hmac_demo_signatures=True,
        ).normalized()
    return RuntimePolicy(mode=RuntimeMode.DEMO.value).normalized()


def build_default_config(
    *,
    mode: Union[str, RuntimeMode] = RuntimeMode.DEMO.value,
    project_root: Optional[Union[str, os.PathLike[str]]] = None,
    chain_id: str = DEFAULT_CHAIN_ID,
    network_id: str = DEFAULT_NETWORK_ID,
    operator_id: str = "local-operator",
) -> TensorizedBlockchainConfig:
    """Build a complete normalized configuration for demo/test/production/replay."""
    rp = runtime_profile(mode)
    root_s = "" if project_root is None else str(Path(project_root).expanduser())
    fs = FileSystemSafetyPolicy(
        project_root=root_s,
        allowed_roots=[root_s] if root_s else [],
        allow_absolute_paths=False,
        allow_local_file_hashing=False if rp.mode == RuntimeMode.PRODUCTION.value else bool(root_s),
        redact_absolute_paths=True,
    )
    tensor_policy = TensorPolicy(
        device_policy=DevicePolicy.CPU_ONLY.value if rp.mode in {RuntimeMode.PRODUCTION.value, RuntimeMode.DETERMINISTIC_REPLAY.value, RuntimeMode.TEST.value} else DevicePolicy.CPU_FOR_COMMITMENTS_GPU_ALLOWED_FOR_INFERENCE.value,
        cpu_only_commitments=True,
    )
    ledger = LedgerPolicy(
        require_public_key_signatures=bool(rp.require_transaction_signatures and rp.default_signature_scheme != SignatureScheme.HMAC_DEMO.value),
        validate_on_load=True,
    )
    cfg = TensorizedBlockchainConfig(
        chain=ChainIdentity(chain_id=chain_id, network_id=network_id, operator_id=operator_id),
        runtime=rp,
        filesystem=fs,
        tensor=tensor_policy,
        ledger=ledger,
        metadata={"created_by": "tb_config.build_default_config", "created_at": now_iso_utc()},
    ).normalized()
    return cfg


def load_config(path: Union[str, os.PathLike[str]], *, validate: bool = True) -> TensorizedBlockchainConfig:
    return TensorizedBlockchainConfig.load_json(path, validate=validate)


def save_config(config: TensorizedBlockchainConfig, path: Union[str, os.PathLike[str]], *, pretty: bool = True) -> None:
    config.save_json(path, pretty=pretty)


def config_status(config: Optional[TensorizedBlockchainConfig] = None) -> Dict[str, Any]:
    cfg = (config or build_default_config()).normalized()
    validation = cfg.validate()
    return {
        "ok": bool(validation.get("ok", False)),
        "ts": now_iso_utc(),
        "tb_config_version": TB_CONFIG_VERSION,
        "stack_version": TENSORIZED_BLOCKCHAIN_STACK_VERSION,
        "protocol_version": TENSORIZED_BLOCKCHAIN_PROTOCOL_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "expected_scripts": list(EXPECTED_SCRIPT_NAMES),
        "expected_module_roles": dict(EXPECTED_MODULE_ROLES),
        "mode": cfg.runtime.mode,
        "chain_id": cfg.chain.chain_id,
        "network_id": cfg.chain.network_id,
        "config_hash": cfg.config_hash(),
        "commitment_hash": cfg.commitment_hash(),
        "validation": validation,
        "lobe_profiles": [p.to_dict() for p in cfg.lobe_profiles],
    }


# =============================================================================
# CLI / self-test
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tensorized Blockchain central configuration")
    p.add_argument("--mode", choices=[m.value for m in RuntimeMode], default=RuntimeMode.DEMO.value)
    p.add_argument("--project-root", default="")
    p.add_argument("--chain-id", default=DEFAULT_CHAIN_ID)
    p.add_argument("--network-id", default=DEFAULT_NETWORK_ID)
    p.add_argument("--operator-id", default="local-operator")
    p.add_argument("--output", default="")
    p.add_argument("--status", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--compact", action="store_true")
    return p


def _self_test() -> Dict[str, Any]:
    import tempfile

    modes = [RuntimeMode.DEMO.value, RuntimeMode.TEST.value, RuntimeMode.PRODUCTION.value, RuntimeMode.DETERMINISTIC_REPLAY.value]
    reports = []
    for mode in modes:
        with tempfile.TemporaryDirectory() as td:
            cfg = build_default_config(mode=mode, project_root=td, chain_id=f"test-{mode}", network_id="testnet")
            validation = cfg.validate()
            if not validation["ok"]:
                raise AssertionError(f"config validation failed for {mode}: {validation}")
            if mode in {RuntimeMode.PRODUCTION.value, RuntimeMode.DETERMINISTIC_REPLAY.value, RuntimeMode.TEST.value}:
                cfg.runtime.assert_consensus_safe()
            out = Path(td) / "tb_config.json"
            cfg.save_json(out)
            loaded = TensorizedBlockchainConfig.load_json(out)
            if loaded.config_hash() != cfg.config_hash():
                raise AssertionError("config hash mismatch after load")
            # File-system safety smoke test.
            f = Path(td) / "sample.txt"
            f.write_text("hello", encoding="utf-8")
            fs_report = cfg.filesystem.validate_manifest_file(f, require_exists=True)
            reports.append({"mode": mode, "hash": cfg.config_hash(), "fs_ok": fs_report["ok"]})
    return {"ok": True, "reports": reports, "status": config_status(build_default_config(mode=RuntimeMode.TEST.value))}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        if args.self_test:
            payload = _self_test()
        else:
            cfg = build_default_config(
                mode=args.mode,
                project_root=args.project_root or None,
                chain_id=args.chain_id,
                network_id=args.network_id,
                operator_id=args.operator_id,
            )
            if args.output:
                cfg.save_json(args.output, pretty=not args.compact)
            payload = config_status(cfg) if args.status else cfg.to_dict()
            if args.output:
                payload = {"ok": True, "saved_config": str(Path(args.output).expanduser().resolve()), "config_hash": cfg.config_hash(), "commitment_hash": cfg.commitment_hash()}
        print(json.dumps(json_safe(payload), indent=None if args.compact else 2, ensure_ascii=False))
        return 0 if bool(payload.get("ok", True)) else 2
    except Exception as exc:
        err = {"ok": False, "error": repr(exc), "component": "tb_config"}
        print(json.dumps(err, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
