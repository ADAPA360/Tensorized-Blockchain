#!/usr/bin/env python3
# lobe_scaler.py
r"""
Project Chimera / Akkurat - Governed Cognitive Lobe Scaler
==========================================================

Purpose
-------
Expansion/contraction controller for elastic cognitive capacity in the
Tensorized Blockchain dual-lobe runtime.

This module owns scale-policy decisions for the NCP Liquid Reflex Lobe and the
CfC Predictive Transition Lobe. It does not instantiate neural models directly,
mutate ledgers, execute twin actions, or perform consensus. It produces
schema-versioned scale profiles, scale events, and deterministic decision
receipts that higher layers can commit into blocks.

Core responsibilities
---------------------
- Define and normalize lobe scale profiles.
- Select lobe size from risk, novelty, uncertainty, stability, resource budget,
  and checkpoint evidence.
- Support dormant/reflex/operational/crisis/deep-cognition modes.
- Record lobe scale events with old/new profile hashes and checkpoint hashes.
- Enforce maximum parameter budgets, including the 2.5B-per-lobe cap.
- Provide deterministic, canonical, auditable decisions.

Design boundary
---------------
The scaler controls *capacity policy*. It does not resize a running neural lobe
in memory by itself. Runtime code should apply returned LobeScaleProfile objects
through cognitive_lobe_runtime.py or an orchestration layer, then checkpoint the
resulting lobe state and commit the LobeScaleEvent.

Public API
----------
- LobeScaleMode
- LobeKind
- ScalingSignal
- ResourceEnvelope
- LobeScaleEvent
- LobeScaleDecision
- LobeScalerConfig
- LobeScaler
- build_default_scaler(...)
- recommend_profiles(...)

The module is dependency-light. It uses tb_schemas/tb_hashing/tb_canonical when
available and falls back to local canonical SHA-256 helpers for standalone use.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union


# =============================================================================
# Version / constants
# =============================================================================

LOBE_SCALER_VERSION = "lobe-scaler-v1.0-production"

MAX_PARAMETERS_PER_LOBE_FALLBACK = 2_500_000_000
MAX_ACTIVE_PARAMETERS_TOTAL_FALLBACK = 5_000_000_000
DEFAULT_HASH_DOMAIN = "TB:LOBE_SCALER:v1"


# =============================================================================
# Optional project imports with safe fallbacks
# =============================================================================

try:  # pragma: no cover - exercised in full stack
    from tb_canonical import canonical_json_safe as _canonical_json_safe  # type: ignore
    from tb_canonical import canonical_bytes as _canonical_bytes  # type: ignore
except Exception:  # pragma: no cover - standalone fallback
    _canonical_json_safe = None
    _canonical_bytes = None

try:  # pragma: no cover - exercised in full stack
    from tb_hashing import domain_hash_hex as _domain_hash_hex  # type: ignore
    from tb_hashing import merkle_root_hex as _merkle_root_hex  # type: ignore
except Exception:  # pragma: no cover - standalone fallback
    _domain_hash_hex = None
    _merkle_root_hex = None

try:  # pragma: no cover - exercised in full stack
    from tb_schemas import LobeScaleProfile as SchemaLobeScaleProfile  # type: ignore
except Exception:  # pragma: no cover - standalone fallback
    SchemaLobeScaleProfile = None  # type: ignore

try:  # pragma: no cover - exercised in full stack
    from tb_config import (  # type: ignore
        MAX_PARAMETERS_PER_LOBE as CONFIG_MAX_PARAMETERS_PER_LOBE,
        DEFAULT_MAX_ACTIVE_PARAMETERS_TOTAL as CONFIG_MAX_ACTIVE_PARAMETERS_TOTAL,
    )
except Exception:  # pragma: no cover - standalone fallback
    CONFIG_MAX_PARAMETERS_PER_LOBE = MAX_PARAMETERS_PER_LOBE_FALLBACK
    CONFIG_MAX_ACTIVE_PARAMETERS_TOTAL = MAX_ACTIVE_PARAMETERS_TOTAL_FALLBACK


# =============================================================================
# Canonical / hashing helpers
# =============================================================================


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_safe(obj: Any) -> Any:
    if _canonical_json_safe is not None:
        try:
            return _canonical_json_safe(obj)
        except Exception:
            pass

    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in list(obj)]
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
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
    return json.dumps(_json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _canonical_bytes_local(obj: Any) -> bytes:
    if _canonical_bytes is not None:
        try:
            return _canonical_bytes(obj)
        except Exception:
            pass
    return _canonical_dumps(obj).encode("utf-8")


def _hash_hex(obj: Any, *, domain: str = DEFAULT_HASH_DOMAIN, tag: str = "") -> str:
    if _domain_hash_hex is not None:
        try:
            return str(_domain_hash_hex(obj, domain=domain, tag=tag))
        except Exception:
            pass
    prefix = f"TB-DOMAIN\x1f{domain}\x1f{tag}\x1e".encode("utf-8")
    return hashlib.sha256(prefix + _canonical_bytes_local(obj)).hexdigest()


def _merkle_hex(leaves: Sequence[Any], *, domain: str = "TB:LOBE_SCALE_EVENT_MERKLE:v1") -> str:
    if _merkle_root_hex is not None:
        try:
            return str(_merkle_root_hex(list(leaves), domain=domain))
        except Exception:
            pass
    if not leaves:
        return _hash_hex({"empty": True}, domain=domain, tag="empty")
    level = [bytes.fromhex(_hash_hex(x, domain=domain, tag="leaf")) for x in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        nxt = []
        for i in range(0, len(level), 2):
            nxt.append(bytes.fromhex(_hash_hex({"left": level[i].hex(), "right": level[i + 1].hex()}, domain=domain, tag="node")))
        level = nxt
    return level[0].hex()


def _clamp01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return float(default)
        return max(0.0, min(1.0, v))
    except Exception:
        return float(default)


def _nonnegative_int(x: Any, default: int = 0, *, maximum: Optional[int] = None) -> int:
    try:
        v = int(x)
    except Exception:
        v = int(default)
    v = max(0, v)
    if maximum is not None:
        v = min(int(maximum), v)
    return int(v)


def _positive_int(x: Any, default: int = 1, *, minimum: int = 1, maximum: Optional[int] = None) -> int:
    try:
        v = int(x)
    except Exception:
        v = int(default)
    v = max(int(minimum), v)
    if maximum is not None:
        v = min(int(maximum), v)
    return int(v)


def _safe_float(x: Any, default: float = 0.0, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            v = float(default)
    except Exception:
        v = float(default)
    if minimum is not None:
        v = max(float(minimum), v)
    if maximum is not None:
        v = min(float(maximum), v)
    return float(v)


def _slug(value: Any, default: str = "item", max_len: int = 128) -> str:
    import re

    s = re.sub(r"[^a-zA-Z0-9_\-]+", "-", str(value or "").strip().lower())
    s = re.sub(r"[-_]+", "-", s).strip("-")
    if not s:
        s = default
    return s[: int(max_len)]


# =============================================================================
# Enums
# =============================================================================


class LobeKind(str, Enum):
    NCP = "ncp"
    CFC = "cfc"
    EXECUTIVE = "executive"
    CUSTOM = "custom"


class LobeScaleMode(str, Enum):
    DORMANT = "dormant"
    REFLEX = "reflex"
    OPERATIONAL = "operational"
    CRISIS = "crisis"
    DEEP_COGNITION = "deep_cognition"


class ScaleDecisionKind(str, Enum):
    KEEP = "keep"
    EXPAND = "expand"
    CONTRACT = "contract"
    ACTIVATE = "activate"
    DORMANT = "dormant"


class ScaleTrigger(str, Enum):
    LOW_RISK = "low_risk"
    STATE_DRIFT_HIGH = "state_drift_high"
    VALIDATOR_COHERENCE_LOW = "validator_coherence_low"
    GOVERNANCE_UNCERTAIN = "governance_uncertain"
    FORECAST_DIVERGENCE_HIGH = "forecast_divergence_high"
    LOBE_SATURATION_HIGH = "lobe_saturation_high"
    CHECKPOINT_REQUIRED = "checkpoint_required"
    RESOURCE_PRESSURE = "resource_pressure"
    OPERATOR_REQUEST = "operator_request"
    POLICY_DEFAULT = "policy_default"


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass(frozen=True)
class ModeTemplate:
    """Capacity template for a lobe kind and scale mode."""

    mode: str
    active: bool
    parameter_budget_fraction: float
    dense_equivalent_multiplier: float
    hidden_size_ncp: int
    hidden_size_cfc: int
    cfc_num_cells: Optional[int]
    tt_bond_rank: int
    update_frequency_hz: float
    precision: str = "float32"
    checkpoint_policy: str = "on_scale_change"

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


DEFAULT_MODE_TEMPLATES: Dict[LobeScaleMode, ModeTemplate] = {
    LobeScaleMode.DORMANT: ModeTemplate(
        mode=LobeScaleMode.DORMANT.value,
        active=False,
        parameter_budget_fraction=0.0,
        dense_equivalent_multiplier=1.0,
        hidden_size_ncp=1,
        hidden_size_cfc=1,
        cfc_num_cells=1,
        tt_bond_rank=1,
        update_frequency_hz=0.0,
        checkpoint_policy="checkpoint_only",
    ),
    LobeScaleMode.REFLEX: ModeTemplate(
        mode=LobeScaleMode.REFLEX.value,
        active=True,
        parameter_budget_fraction=0.01,
        dense_equivalent_multiplier=5.0,
        hidden_size_ncp=256,
        hidden_size_cfc=128,
        cfc_num_cells=2,
        tt_bond_rank=4,
        update_frequency_hz=0.5,
    ),
    LobeScaleMode.OPERATIONAL: ModeTemplate(
        mode=LobeScaleMode.OPERATIONAL.value,
        active=True,
        parameter_budget_fraction=0.10,
        dense_equivalent_multiplier=10.0,
        hidden_size_ncp=1024,
        hidden_size_cfc=512,
        cfc_num_cells=4,
        tt_bond_rank=8,
        update_frequency_hz=1.0,
    ),
    LobeScaleMode.CRISIS: ModeTemplate(
        mode=LobeScaleMode.CRISIS.value,
        active=True,
        parameter_budget_fraction=0.40,
        dense_equivalent_multiplier=12.0,
        hidden_size_ncp=4096,
        hidden_size_cfc=2048,
        cfc_num_cells=16,
        tt_bond_rank=16,
        update_frequency_hz=5.0,
    ),
    LobeScaleMode.DEEP_COGNITION: ModeTemplate(
        mode=LobeScaleMode.DEEP_COGNITION.value,
        active=True,
        parameter_budget_fraction=1.0,
        dense_equivalent_multiplier=15.0,
        hidden_size_ncp=16384,
        hidden_size_cfc=8192,
        cfc_num_cells=64,
        tt_bond_rank=32,
        update_frequency_hz=10.0,
        checkpoint_policy="full_with_manifest_and_hash",
    ),
}


@dataclass
class ScalingSignal:
    """Normalized input signal used to select lobe capacity."""

    risk_score: float = 0.0
    novelty_score: float = 0.0
    uncertainty_score: float = 0.0
    state_drift_score: float = 0.0
    validator_disagreement_score: float = 0.0
    forecast_error_score: float = 0.0
    lobe_saturation_score: float = 0.0
    resource_pressure_score: float = 0.0
    operator_requested_mode: str = ""
    checkpoint_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "ScalingSignal":
        s = copy.deepcopy(self)
        for name in (
            "risk_score",
            "novelty_score",
            "uncertainty_score",
            "state_drift_score",
            "validator_disagreement_score",
            "forecast_error_score",
            "lobe_saturation_score",
            "resource_pressure_score",
        ):
            setattr(s, name, _clamp01(getattr(s, name), 0.0))
        s.operator_requested_mode = str(s.operator_requested_mode or "").strip().lower()
        s.checkpoint_required = bool(s.checkpoint_required)
        s.metadata = dict(s.metadata or {})
        return s

    def composite_risk(self) -> float:
        s = self.normalized()
        # Resource pressure reduces the willingness to expand, but does not hide
        # actual risk. It is handled separately by budget fitting.
        weights = {
            "risk_score": 0.24,
            "novelty_score": 0.10,
            "uncertainty_score": 0.16,
            "state_drift_score": 0.18,
            "validator_disagreement_score": 0.14,
            "forecast_error_score": 0.12,
            "lobe_saturation_score": 0.06,
        }
        return _clamp01(sum(float(getattr(s, k)) * w for k, w in weights.items()))

    def dominant_triggers(self, *, threshold: float = 0.65) -> List[str]:
        s = self.normalized()
        out: List[str] = []
        if s.state_drift_score >= threshold:
            out.append(ScaleTrigger.STATE_DRIFT_HIGH.value)
        if s.validator_disagreement_score >= threshold:
            out.append(ScaleTrigger.VALIDATOR_COHERENCE_LOW.value)
        if s.uncertainty_score >= threshold:
            out.append(ScaleTrigger.GOVERNANCE_UNCERTAIN.value)
        if s.forecast_error_score >= threshold:
            out.append(ScaleTrigger.FORECAST_DIVERGENCE_HIGH.value)
        if s.lobe_saturation_score >= threshold:
            out.append(ScaleTrigger.LOBE_SATURATION_HIGH.value)
        if s.resource_pressure_score >= threshold:
            out.append(ScaleTrigger.RESOURCE_PRESSURE.value)
        if s.checkpoint_required:
            out.append(ScaleTrigger.CHECKPOINT_REQUIRED.value)
        if s.operator_requested_mode:
            out.append(ScaleTrigger.OPERATOR_REQUEST.value)
        if not out:
            out.append(ScaleTrigger.LOW_RISK.value if s.composite_risk() < 0.25 else ScaleTrigger.POLICY_DEFAULT.value)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self.normalized()))

    def signal_hash(self) -> str:
        return _hash_hex(self.to_dict(), domain="TB:SCALING_SIGNAL:v1")


@dataclass
class ResourceEnvelope:
    """Runtime resource budget for scale decisions."""

    max_parameters_per_lobe: int = int(CONFIG_MAX_PARAMETERS_PER_LOBE)
    max_active_parameters_total: int = int(CONFIG_MAX_ACTIVE_PARAMETERS_TOTAL)
    current_active_parameters_total: int = 0
    memory_budget_bytes: Optional[int] = None
    estimated_available_memory_bytes: Optional[int] = None
    prefer_cpu: bool = True
    allow_deep_cognition: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "ResourceEnvelope":
        r = copy.deepcopy(self)
        r.max_parameters_per_lobe = _positive_int(r.max_parameters_per_lobe, CONFIG_MAX_PARAMETERS_PER_LOBE, minimum=1, maximum=CONFIG_MAX_PARAMETERS_PER_LOBE)
        r.max_active_parameters_total = _positive_int(r.max_active_parameters_total, CONFIG_MAX_ACTIVE_PARAMETERS_TOTAL, minimum=1)
        r.current_active_parameters_total = _nonnegative_int(r.current_active_parameters_total, 0)
        if r.memory_budget_bytes is not None:
            r.memory_budget_bytes = _nonnegative_int(r.memory_budget_bytes, 0)
        if r.estimated_available_memory_bytes is not None:
            r.estimated_available_memory_bytes = _nonnegative_int(r.estimated_available_memory_bytes, 0)
        r.prefer_cpu = bool(r.prefer_cpu)
        r.allow_deep_cognition = bool(r.allow_deep_cognition)
        r.metadata = dict(r.metadata or {})
        return r

    def remaining_parameter_budget(self) -> int:
        r = self.normalized()
        return max(0, int(r.max_active_parameters_total) - int(r.current_active_parameters_total))

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self.normalized()))


@dataclass
class LobeScaleEvent:
    """Auditable scale transition event."""

    event_id: str
    lobe_id: str
    lobe_kind: str
    decision: str
    old_profile_hash: str
    new_profile_hash: str
    trigger: str
    signal_hash: str
    checkpoint_hash: str = ""
    approved: bool = True
    parameter_budget_before: int = 0
    parameter_budget_after: int = 0
    active_parameter_count_before: int = 0
    active_parameter_count_after: int = 0
    reason: str = ""
    created_at: str = field(default_factory=_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "tb.lobe_scale_event.v1"

    @staticmethod
    def make_event_id(*, lobe_id: str, old_profile_hash: str, new_profile_hash: str, signal_hash: str) -> str:
        return "scale-" + _hash_hex(
            {
                "lobe_id": lobe_id,
                "old_profile_hash": old_profile_hash,
                "new_profile_hash": new_profile_hash,
                "signal_hash": signal_hash,
            },
            domain="TB:LOBE_SCALE_EVENT_ID:v1",
        )[:32]

    def payload(self) -> Dict[str, Any]:
        d = _json_safe(asdict(self))
        d.pop("event_hash", None)
        return d

    def event_hash(self) -> str:
        return _hash_hex(self.payload(), domain="TB:LOBE_SCALE_EVENT:v1")

    def to_dict(self) -> Dict[str, Any]:
        d = self.payload()
        d["event_hash"] = self.event_hash()
        return d

    def validate(self, *, max_parameters_per_lobe: int = CONFIG_MAX_PARAMETERS_PER_LOBE) -> List[str]:
        issues: List[str] = []
        if not self.lobe_id:
            issues.append("lobe_id_empty")
        if self.parameter_budget_after > int(max_parameters_per_lobe):
            issues.append("parameter_budget_after_exceeds_per_lobe_cap")
        for field_name in ("old_profile_hash", "new_profile_hash", "signal_hash", "checkpoint_hash"):
            v = str(getattr(self, field_name, "") or "")
            if v and (len(v) != 64 or any(c not in "0123456789abcdefABCDEF" for c in v)):
                issues.append(f"{field_name}_invalid")
        return issues


@dataclass
class LobeScaleDecision:
    """Complete decision result from the scaler."""

    scaler_version: str
    signal: Dict[str, Any]
    resource_envelope: Dict[str, Any]
    old_profiles: List[Dict[str, Any]]
    new_profiles: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    decision_kind: str
    composite_risk: float
    event_root: str
    approved: bool = True
    warnings: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "tb.lobe_scale_decision.v1"

    def payload(self) -> Dict[str, Any]:
        d = _json_safe(asdict(self))
        d.pop("decision_hash", None)
        return d

    def decision_hash(self) -> str:
        return _hash_hex(self.payload(), domain="TB:LOBE_SCALE_DECISION:v1")

    def to_dict(self) -> Dict[str, Any]:
        d = self.payload()
        d["decision_hash"] = self.decision_hash()
        return d


@dataclass
class LobeScalerConfig:
    """Configuration for governed lobe scaling."""

    max_parameters_per_lobe: int = int(CONFIG_MAX_PARAMETERS_PER_LOBE)
    max_active_parameters_total: int = int(CONFIG_MAX_ACTIVE_PARAMETERS_TOTAL)
    default_ncp_budget: int = 250_000_000
    default_cfc_budget: int = 250_000_000
    deterministic: bool = True
    consensus_safe: bool = True
    device: str = "cpu"
    precision: str = "float32"
    allow_deep_cognition: bool = True
    require_checkpoint_hash_on_contraction: bool = False
    require_checkpoint_hash_on_dormant: bool = True
    risk_reflex_threshold: float = 0.10
    risk_operational_threshold: float = 0.30
    risk_crisis_threshold: float = 0.65
    risk_deep_cognition_threshold: float = 0.88
    contraction_hysteresis: float = 0.10
    resource_pressure_contract_threshold: float = 0.80
    templates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "LobeScalerConfig":
        c = copy.deepcopy(self)
        c.max_parameters_per_lobe = _positive_int(c.max_parameters_per_lobe, CONFIG_MAX_PARAMETERS_PER_LOBE, minimum=1, maximum=CONFIG_MAX_PARAMETERS_PER_LOBE)
        c.max_active_parameters_total = _positive_int(c.max_active_parameters_total, CONFIG_MAX_ACTIVE_PARAMETERS_TOTAL, minimum=1)
        c.default_ncp_budget = _positive_int(c.default_ncp_budget, 250_000_000, minimum=0, maximum=c.max_parameters_per_lobe)
        c.default_cfc_budget = _positive_int(c.default_cfc_budget, 250_000_000, minimum=0, maximum=c.max_parameters_per_lobe)
        c.deterministic = bool(c.deterministic)
        c.consensus_safe = bool(c.consensus_safe)
        c.device = str(c.device or "cpu").lower().strip()
        if c.consensus_safe:
            c.device = "cpu"
        c.precision = str(c.precision or "float32").lower().strip()
        if c.precision not in {"float32", "float64", "float16", "int8", "int4"}:
            c.precision = "float32"
        c.allow_deep_cognition = bool(c.allow_deep_cognition)
        c.require_checkpoint_hash_on_contraction = bool(c.require_checkpoint_hash_on_contraction)
        c.require_checkpoint_hash_on_dormant = bool(c.require_checkpoint_hash_on_dormant)
        c.risk_reflex_threshold = _clamp01(c.risk_reflex_threshold, 0.10)
        c.risk_operational_threshold = _clamp01(c.risk_operational_threshold, 0.30)
        c.risk_crisis_threshold = _clamp01(c.risk_crisis_threshold, 0.65)
        c.risk_deep_cognition_threshold = _clamp01(c.risk_deep_cognition_threshold, 0.88)
        # Ensure monotonic thresholds.
        vals = sorted([c.risk_reflex_threshold, c.risk_operational_threshold, c.risk_crisis_threshold, c.risk_deep_cognition_threshold])
        c.risk_reflex_threshold, c.risk_operational_threshold, c.risk_crisis_threshold, c.risk_deep_cognition_threshold = vals
        c.contraction_hysteresis = _clamp01(c.contraction_hysteresis, 0.10)
        c.resource_pressure_contract_threshold = _clamp01(c.resource_pressure_contract_threshold, 0.80)
        c.templates = dict(c.templates or {})
        c.metadata = dict(c.metadata or {})
        return c

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self.normalized()))

    def config_hash(self) -> str:
        return _hash_hex(self.to_dict(), domain="TB:LOBE_SCALER_CONFIG:v1")


# =============================================================================
# Profile construction helpers
# =============================================================================


def _profile_to_dict(profile: Any) -> Dict[str, Any]:
    if profile is None:
        return {}
    if hasattr(profile, "to_dict") and callable(profile.to_dict):
        try:
            return dict(profile.to_dict())
        except Exception:
            pass
    if is_dataclass(profile):
        return dict(_json_safe(asdict(profile)))
    if isinstance(profile, Mapping):
        return dict(_json_safe(profile))
    raise TypeError(f"Cannot convert profile to dict: {type(profile)!r}")


def _profile_hash(profile: Any) -> str:
    if profile is None:
        return ""
    for name in ("profile_hash", "hash"):
        fn = getattr(profile, name, None)
        if callable(fn):
            try:
                h = str(fn())
                if len(h) == 64:
                    return h
            except Exception:
                pass
    d = _profile_to_dict(profile)
    for k in ("profile_hash", "hash"):
        if isinstance(d.get(k), str) and len(str(d[k])) == 64:
            return str(d[k])
    return _hash_hex(d, domain="TB:LOBE_SCALE_PROFILE:v1")


def _make_schema_profile(**kwargs: Any) -> Any:
    """Construct tb_schemas.LobeScaleProfile when available; otherwise dict."""
    payload = dict(kwargs)
    if SchemaLobeScaleProfile is not None:
        try:
            fields = {f.name for f in dataclasses.fields(SchemaLobeScaleProfile)}
            obj = SchemaLobeScaleProfile(**{k: v for k, v in payload.items() if k in fields})
            return obj
        except Exception:
            pass
    payload.setdefault("schema_version", "tb.lobe_scale_profile.v1")
    payload["profile_hash"] = _hash_hex(payload, domain="TB:LOBE_SCALE_PROFILE:v1")
    return payload


def _coerce_mode(value: Any, default: LobeScaleMode = LobeScaleMode.OPERATIONAL) -> LobeScaleMode:
    if isinstance(value, LobeScaleMode):
        return value
    s = str(value or "").lower().strip().replace("-", "_")
    for m in LobeScaleMode:
        if s == m.value:
            return m
    return default


def _coerce_kind(value: Any) -> LobeKind:
    if isinstance(value, LobeKind):
        return value
    s = str(value or "").lower().strip().replace("-", "_")
    for k in LobeKind:
        if s == k.value:
            return k
    return LobeKind.CUSTOM


# =============================================================================
# Scaler
# =============================================================================


class LobeScaler:
    """Governed elastic capacity controller for cognitive lobes."""

    def __init__(self, config: Optional[LobeScalerConfig] = None):
        self.config = (config or LobeScalerConfig()).normalized()
        self._events: List[LobeScaleEvent] = []

    # ------------------------------------------------------------------
    # Policy helpers
    # ------------------------------------------------------------------
    def default_profiles(self, *, mode: Union[str, LobeScaleMode] = LobeScaleMode.OPERATIONAL) -> List[Any]:
        m = _coerce_mode(mode)
        return [
            self.profile_for_mode(lobe_id="liquid-reflex-ncp", lobe_kind=LobeKind.NCP, mode=m),
            self.profile_for_mode(lobe_id="predictive-transition-cfc", lobe_kind=LobeKind.CFC, mode=m),
        ]

    def _base_budget_for_kind(self, lobe_kind: Union[str, LobeKind]) -> int:
        kind = _coerce_kind(lobe_kind)
        if kind == LobeKind.NCP:
            return int(self.config.default_ncp_budget)
        if kind == LobeKind.CFC:
            return int(self.config.default_cfc_budget)
        return int(min(self.config.default_ncp_budget, self.config.default_cfc_budget))

    def _template_for_mode(self, mode: Union[str, LobeScaleMode]) -> ModeTemplate:
        m = _coerce_mode(mode)
        # Optional user overrides are stored by mode value.
        override = self.config.templates.get(m.value)
        base = DEFAULT_MODE_TEMPLATES[m]
        if isinstance(override, Mapping):
            payload = asdict(base)
            payload.update(dict(override))
            try:
                return ModeTemplate(**payload)
            except Exception:
                return base
        return base

    def profile_for_mode(
        self,
        *,
        lobe_id: str,
        lobe_kind: Union[str, LobeKind],
        mode: Union[str, LobeScaleMode],
        reason: str = "",
        checkpoint_hash: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        cfg = self.config
        kind = _coerce_kind(lobe_kind)
        m = _coerce_mode(mode)
        template = self._template_for_mode(m)
        base_budget = self._base_budget_for_kind(kind)
        if m == LobeScaleMode.DEEP_COGNITION:
            base_budget = cfg.max_parameters_per_lobe
        parameter_budget = int(round(base_budget * float(template.parameter_budget_fraction)))
        parameter_budget = min(int(cfg.max_parameters_per_lobe), max(0, parameter_budget))
        if template.active and parameter_budget <= 0:
            parameter_budget = min(int(cfg.max_parameters_per_lobe), max(1, self._base_budget_for_kind(kind)))

        dense_equiv = int(min(cfg.max_parameters_per_lobe, max(parameter_budget, int(round(parameter_budget * float(template.dense_equivalent_multiplier))))))
        active_count = int(parameter_budget if template.active else 0)
        hidden_size = int(template.hidden_size_ncp if kind == LobeKind.NCP else template.hidden_size_cfc)
        num_cells = None if kind == LobeKind.NCP else int(template.cfc_num_cells or 1)
        device = "cpu" if cfg.consensus_safe else cfg.device
        md = dict(metadata or {})
        if checkpoint_hash:
            md["checkpoint_hash"] = checkpoint_hash
        md.setdefault("scaler_version", LOBE_SCALER_VERSION)
        md.setdefault("config_hash", cfg.config_hash())
        md.setdefault("template", template.to_dict())

        return _make_schema_profile(
            lobe_id=_slug(lobe_id, default="lobe"),
            lobe_kind=kind.value,
            active=bool(template.active),
            parameter_budget=int(parameter_budget),
            dense_equivalent_budget=int(dense_equiv),
            active_parameter_count=int(active_count),
            hidden_size=int(hidden_size),
            num_cells=num_cells,
            tt_bond_rank=int(template.tt_bond_rank),
            precision=str(template.precision or cfg.precision),
            update_frequency_hz=float(template.update_frequency_hz),
            checkpoint_policy=str(template.checkpoint_policy),
            consensus_safe=bool(cfg.consensus_safe),
            device=device,
            scale_mode=m.value,
            reason=str(reason or m.value),
            metadata=md,
        )

    def select_mode(self, signal: ScalingSignal, *, current_mode: Union[str, LobeScaleMode, None] = None) -> LobeScaleMode:
        cfg = self.config
        s = signal.normalized()
        requested = _coerce_mode(s.operator_requested_mode, default=None) if s.operator_requested_mode else None  # type: ignore[arg-type]
        if isinstance(requested, LobeScaleMode):
            if requested == LobeScaleMode.DEEP_COGNITION and not cfg.allow_deep_cognition:
                return LobeScaleMode.CRISIS
            return requested

        risk = s.composite_risk()
        current = _coerce_mode(current_mode or LobeScaleMode.OPERATIONAL)

        # Resource pressure can force contraction even if risk is moderate. If risk
        # is extreme, keep at least crisis capacity.
        if s.resource_pressure_score >= cfg.resource_pressure_contract_threshold and risk < cfg.risk_crisis_threshold:
            if current in {LobeScaleMode.DEEP_COGNITION, LobeScaleMode.CRISIS}:
                return LobeScaleMode.OPERATIONAL
            if current == LobeScaleMode.OPERATIONAL:
                return LobeScaleMode.REFLEX
            return LobeScaleMode.DORMANT

        if cfg.allow_deep_cognition and risk >= cfg.risk_deep_cognition_threshold:
            return LobeScaleMode.DEEP_COGNITION
        if risk >= cfg.risk_crisis_threshold:
            return LobeScaleMode.CRISIS
        if risk >= cfg.risk_operational_threshold:
            return LobeScaleMode.OPERATIONAL
        if risk >= cfg.risk_reflex_threshold:
            return LobeScaleMode.REFLEX
        return LobeScaleMode.DORMANT

    def _fit_profiles_to_resource_envelope(self, profiles: List[Any], resource: ResourceEnvelope) -> Tuple[List[Any], List[str]]:
        cfg = self.config
        r = resource.normalized()
        warnings: List[str] = []
        fitted = list(profiles)

        def active_sum(items: Sequence[Any]) -> int:
            total = 0
            for p in items:
                d = _profile_to_dict(p)
                if bool(d.get("active", True)):
                    total += int(d.get("parameter_budget", d.get("active_parameter_count", 0)) or 0)
            return total

        # Per-lobe cap enforcement.
        capped: List[Any] = []
        for p in fitted:
            d = _profile_to_dict(p)
            pb = int(d.get("parameter_budget", 0) or 0)
            if pb > r.max_parameters_per_lobe:
                warnings.append(f"capped_{d.get('lobe_id','lobe')}_to_per_lobe_limit")
                d["parameter_budget"] = int(r.max_parameters_per_lobe)
                d["active_parameter_count"] = int(r.max_parameters_per_lobe if d.get("active", True) else 0)
                d["dense_equivalent_budget"] = min(int(d.get("dense_equivalent_budget", r.max_parameters_per_lobe) or 0), int(r.max_parameters_per_lobe))
                d["reason"] = str(d.get("reason", "")) + ";capped_per_lobe"
                capped.append(_make_schema_profile(**{k: v for k, v in d.items() if k != "profile_hash"}))
            else:
                capped.append(p)
        fitted = capped

        max_total = min(int(cfg.max_active_parameters_total), int(r.max_active_parameters_total))
        remaining = max(0, max_total - int(r.current_active_parameters_total))
        total = active_sum(fitted)
        if total <= remaining:
            return fitted, warnings

        # Progressive contraction until the total fits. This preserves the dominant
        # risk decision as much as possible under resource pressure.
        warnings.append("active_parameter_budget_exceeds_resource_envelope_contracting")
        mode_order = [
            LobeScaleMode.DEEP_COGNITION,
            LobeScaleMode.CRISIS,
            LobeScaleMode.OPERATIONAL,
            LobeScaleMode.REFLEX,
            LobeScaleMode.DORMANT,
        ]
        lower: Dict[str, str] = {
            LobeScaleMode.DEEP_COGNITION.value: LobeScaleMode.CRISIS.value,
            LobeScaleMode.CRISIS.value: LobeScaleMode.OPERATIONAL.value,
            LobeScaleMode.OPERATIONAL.value: LobeScaleMode.REFLEX.value,
            LobeScaleMode.REFLEX.value: LobeScaleMode.DORMANT.value,
            LobeScaleMode.DORMANT.value: LobeScaleMode.DORMANT.value,
        }
        for _ in range(16):
            if active_sum(fitted) <= remaining:
                break
            # Contract the largest active profile one step.
            largest_idx = None
            largest_budget = -1
            for i, p in enumerate(fitted):
                d = _profile_to_dict(p)
                if bool(d.get("active", True)) and int(d.get("parameter_budget", 0) or 0) > largest_budget:
                    largest_idx = i
                    largest_budget = int(d.get("parameter_budget", 0) or 0)
            if largest_idx is None:
                break
            d = _profile_to_dict(fitted[largest_idx])
            current_mode = _coerce_mode(d.get("scale_mode", d.get("reason", LobeScaleMode.OPERATIONAL.value)))
            next_mode = _coerce_mode(lower[current_mode.value])
            fitted[largest_idx] = self.profile_for_mode(
                lobe_id=str(d.get("lobe_id", f"lobe-{largest_idx}")),
                lobe_kind=str(d.get("lobe_kind", LobeKind.CUSTOM.value)),
                mode=next_mode,
                reason="resource_fit_contraction",
                metadata={"previous_profile_hash": _profile_hash(d), "resource_remaining": remaining},
            )
        if active_sum(fitted) > remaining:
            warnings.append("resource_envelope_still_exceeded_after_contraction")
        return fitted, warnings

    def _decision_kind(self, old_profile: Any, new_profile: Any) -> str:
        old = _profile_to_dict(old_profile)
        new = _profile_to_dict(new_profile)
        old_active = bool(old.get("active", False))
        new_active = bool(new.get("active", False))
        old_budget = int(old.get("parameter_budget", 0) or 0)
        new_budget = int(new.get("parameter_budget", 0) or 0)
        if not new_active:
            return ScaleDecisionKind.DORMANT.value
        if not old_active and new_active:
            return ScaleDecisionKind.ACTIVATE.value
        if new_budget > old_budget:
            return ScaleDecisionKind.EXPAND.value
        if new_budget < old_budget:
            return ScaleDecisionKind.CONTRACT.value
        return ScaleDecisionKind.KEEP.value

    # ------------------------------------------------------------------
    # Main decision API
    # ------------------------------------------------------------------
    def decide(
        self,
        *,
        current_profiles: Optional[Sequence[Any]] = None,
        signal: Optional[ScalingSignal] = None,
        resource_envelope: Optional[ResourceEnvelope] = None,
        checkpoint_hashes: Optional[Mapping[str, str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> LobeScaleDecision:
        cfg = self.config
        sig = (signal or ScalingSignal()).normalized()
        res = (resource_envelope or ResourceEnvelope(max_parameters_per_lobe=cfg.max_parameters_per_lobe, max_active_parameters_total=cfg.max_active_parameters_total)).normalized()
        checkpoint_hashes = dict(checkpoint_hashes or {})

        old_profiles = list(current_profiles or self.default_profiles(mode=LobeScaleMode.OPERATIONAL))
        old_by_id: Dict[str, Any] = {}
        for p in old_profiles:
            d = _profile_to_dict(p)
            old_by_id[str(d.get("lobe_id", f"lobe-{len(old_by_id)}"))] = p

        target_profiles: List[Any] = []
        triggers = sig.dominant_triggers()
        risk = sig.composite_risk()
        for lobe_id, old in old_by_id.items():
            old_d = _profile_to_dict(old)
            current_mode = old_d.get("scale_mode", old_d.get("reason", LobeScaleMode.OPERATIONAL.value))
            target_mode = self.select_mode(sig, current_mode=current_mode)
            # NCP is more sensitive to drift/saturation; CfC is more sensitive to forecast error.
            kind = _coerce_kind(old_d.get("lobe_kind", LobeKind.CUSTOM.value))
            if kind == LobeKind.NCP and (sig.state_drift_score >= 0.75 or sig.lobe_saturation_score >= 0.75):
                target_mode = max([target_mode, LobeScaleMode.CRISIS], key=lambda m: list(LobeScaleMode).index(m))
            if kind == LobeKind.CFC and sig.forecast_error_score >= 0.75:
                target_mode = max([target_mode, LobeScaleMode.CRISIS], key=lambda m: list(LobeScaleMode).index(m))
            if target_mode == LobeScaleMode.DEEP_COGNITION and not res.allow_deep_cognition:
                target_mode = LobeScaleMode.CRISIS
            target_profiles.append(
                self.profile_for_mode(
                    lobe_id=lobe_id,
                    lobe_kind=kind,
                    mode=target_mode,
                    reason=",".join(triggers),
                    checkpoint_hash=checkpoint_hashes.get(lobe_id, ""),
                    metadata={"composite_risk": risk, "signal_hash": sig.signal_hash()},
                )
            )

        fitted_profiles, warnings = self._fit_profiles_to_resource_envelope(target_profiles, res)

        events: List[LobeScaleEvent] = []
        approved = True
        for new in fitted_profiles:
            nd = _profile_to_dict(new)
            lobe_id = str(nd.get("lobe_id", "lobe"))
            old = old_by_id.get(lobe_id, {})
            od = _profile_to_dict(old) if old else {}
            old_hash = _profile_hash(old) if old else "0" * 64
            new_hash = _profile_hash(new)
            decision_kind = self._decision_kind(old, new)
            checkpoint_hash = str(checkpoint_hashes.get(lobe_id, nd.get("metadata", {}).get("checkpoint_hash", "")) or "")

            if decision_kind in {ScaleDecisionKind.CONTRACT.value, ScaleDecisionKind.DORMANT.value}:
                if cfg.require_checkpoint_hash_on_contraction and not checkpoint_hash:
                    warnings.append(f"checkpoint_hash_missing_for_{lobe_id}_contraction")
                    approved = False
                if decision_kind == ScaleDecisionKind.DORMANT.value and cfg.require_checkpoint_hash_on_dormant and not checkpoint_hash:
                    warnings.append(f"checkpoint_hash_missing_for_{lobe_id}_dormant")
                    approved = False

            event_id = LobeScaleEvent.make_event_id(
                lobe_id=lobe_id,
                old_profile_hash=old_hash,
                new_profile_hash=new_hash,
                signal_hash=sig.signal_hash(),
            )
            event = LobeScaleEvent(
                event_id=event_id,
                lobe_id=lobe_id,
                lobe_kind=str(nd.get("lobe_kind", LobeKind.CUSTOM.value)),
                decision=decision_kind,
                old_profile_hash=old_hash,
                new_profile_hash=new_hash,
                trigger=",".join(triggers),
                signal_hash=sig.signal_hash(),
                checkpoint_hash=checkpoint_hash,
                approved=bool(approved),
                parameter_budget_before=int(od.get("parameter_budget", 0) or 0),
                parameter_budget_after=int(nd.get("parameter_budget", 0) or 0),
                active_parameter_count_before=int(od.get("active_parameter_count", od.get("parameter_budget", 0)) or 0),
                active_parameter_count_after=int(nd.get("active_parameter_count", nd.get("parameter_budget", 0)) or 0),
                reason=str(nd.get("reason", ",".join(triggers))),
                metadata={
                    "scale_mode": nd.get("scale_mode", ""),
                    "config_hash": cfg.config_hash(),
                    "resource_envelope_hash": _hash_hex(res.to_dict(), domain="TB:RESOURCE_ENVELOPE:v1"),
                },
            )
            issues = event.validate(max_parameters_per_lobe=cfg.max_parameters_per_lobe)
            if issues:
                warnings.extend([f"{event.event_id}:{issue}" for issue in issues])
                approved = False
                event.approved = False
            events.append(event)

        self._events.extend(events)
        event_dicts = [e.to_dict() for e in events]
        event_root = _merkle_hex(event_dicts, domain="TB:LOBE_SCALE_EVENT_ROOT:v1")
        # Overall decision kind: most severe non-keep event.
        order = {
            ScaleDecisionKind.KEEP.value: 0,
            ScaleDecisionKind.CONTRACT.value: 1,
            ScaleDecisionKind.DORMANT.value: 2,
            ScaleDecisionKind.ACTIVATE.value: 3,
            ScaleDecisionKind.EXPAND.value: 4,
        }
        overall = max((e.decision for e in events), key=lambda x: order.get(x, 0), default=ScaleDecisionKind.KEEP.value)

        return LobeScaleDecision(
            scaler_version=LOBE_SCALER_VERSION,
            signal=sig.to_dict(),
            resource_envelope=res.to_dict(),
            old_profiles=[_profile_to_dict(p) for p in old_profiles],
            new_profiles=[_profile_to_dict(p) for p in fitted_profiles],
            events=event_dicts,
            decision_kind=overall,
            composite_risk=float(risk),
            event_root=event_root,
            approved=bool(approved),
            warnings=warnings,
            metadata={"config_hash": cfg.config_hash(), **dict(metadata or {})},
        )

    # ------------------------------------------------------------------
    # History / export
    # ------------------------------------------------------------------
    def events(self) -> List[LobeScaleEvent]:
        return list(self._events)

    def event_root(self) -> str:
        return _merkle_hex([e.to_dict() for e in self._events], domain="TB:LOBE_SCALE_EVENT_HISTORY_ROOT:v1")

    def export_history(self) -> Dict[str, Any]:
        events = [e.to_dict() for e in self._events]
        return {
            "schema_version": "tb.lobe_scaler.history.v1",
            "scaler_version": LOBE_SCALER_VERSION,
            "config": self.config.to_dict(),
            "event_count": len(events),
            "event_root": self.event_root(),
            "events": events,
            "created_at": _now_iso(),
        }

    def save_history_json(self, path: Union[str, os.PathLike[str]]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.export_history(), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)


# =============================================================================
# Convenience API
# =============================================================================


def build_default_scaler(*, production: bool = False, deep_cognition: bool = True) -> LobeScaler:
    cfg = LobeScalerConfig(
        default_ncp_budget=250_000_000 if production else 25_000_000,
        default_cfc_budget=250_000_000 if production else 25_000_000,
        consensus_safe=bool(production),
        allow_deep_cognition=bool(deep_cognition),
    )
    return LobeScaler(cfg)


def recommend_profiles(
    *,
    risk_score: float = 0.0,
    state_drift_score: float = 0.0,
    forecast_error_score: float = 0.0,
    current_profiles: Optional[Sequence[Any]] = None,
    production: bool = False,
) -> Dict[str, Any]:
    scaler = build_default_scaler(production=production)
    decision = scaler.decide(
        current_profiles=current_profiles,
        signal=ScalingSignal(
            risk_score=risk_score,
            state_drift_score=state_drift_score,
            forecast_error_score=forecast_error_score,
        ),
    )
    return decision.to_dict()


# =============================================================================
# Status / self-test / CLI
# =============================================================================


def status_report() -> Dict[str, Any]:
    cfg = LobeScalerConfig().normalized()
    return {
        "ok": True,
        "version": LOBE_SCALER_VERSION,
        "max_parameters_per_lobe": int(cfg.max_parameters_per_lobe),
        "max_active_parameters_total": int(cfg.max_active_parameters_total),
        "schema_lobe_scale_profile_available": bool(SchemaLobeScaleProfile is not None),
        "tb_hashing_available": bool(_domain_hash_hex is not None),
        "tb_canonical_available": bool(_canonical_json_safe is not None),
        "default_config_hash": cfg.config_hash(),
        "ts": _now_iso(),
    }


def _self_test() -> Dict[str, Any]:
    scaler = build_default_scaler(production=True, deep_cognition=True)
    profiles = scaler.default_profiles(mode=LobeScaleMode.OPERATIONAL)

    low = scaler.decide(
        current_profiles=profiles,
        signal=ScalingSignal(risk_score=0.02, state_drift_score=0.01, forecast_error_score=0.0),
        checkpoint_hashes={"liquid-reflex-ncp": "a" * 64, "predictive-transition-cfc": "b" * 64},
    )
    if not low.to_dict().get("event_root"):
        raise AssertionError("low-risk decision missing event root")

    high = scaler.decide(
        current_profiles=profiles,
        signal=ScalingSignal(
            risk_score=0.95,
            novelty_score=0.8,
            uncertainty_score=0.9,
            state_drift_score=0.9,
            validator_disagreement_score=0.8,
            forecast_error_score=0.95,
        ),
        resource_envelope=ResourceEnvelope(max_parameters_per_lobe=CONFIG_MAX_PARAMETERS_PER_LOBE, max_active_parameters_total=CONFIG_MAX_ACTIVE_PARAMETERS_TOTAL),
    )
    hd = high.to_dict()
    if not hd["approved"]:
        raise AssertionError(f"high-risk decision unexpectedly not approved: {hd.get('warnings')}")
    for p in hd["new_profiles"]:
        if int(p.get("parameter_budget", 0)) > CONFIG_MAX_PARAMETERS_PER_LOBE:
            raise AssertionError("per-lobe cap violated")

    pressure = scaler.decide(
        current_profiles=profiles,
        signal=ScalingSignal(risk_score=0.4, resource_pressure_score=0.95),
        resource_envelope=ResourceEnvelope(max_active_parameters_total=50_000_000),
        checkpoint_hashes={"liquid-reflex-ncp": "c" * 64, "predictive-transition-cfc": "d" * 64},
    )
    pd = pressure.to_dict()
    if not pd["events"]:
        raise AssertionError("resource-pressure decision missing events")

    # Determinism check.
    scaler2 = build_default_scaler(production=True, deep_cognition=True)
    high2 = scaler2.decide(
        current_profiles=profiles,
        signal=ScalingSignal(
            risk_score=0.95,
            novelty_score=0.8,
            uncertainty_score=0.9,
            state_drift_score=0.9,
            validator_disagreement_score=0.8,
            forecast_error_score=0.95,
        ),
        resource_envelope=ResourceEnvelope(max_parameters_per_lobe=CONFIG_MAX_PARAMETERS_PER_LOBE, max_active_parameters_total=CONFIG_MAX_ACTIVE_PARAMETERS_TOTAL),
    )
    if high.decision_hash() != high2.decision_hash():
        # Timestamps are intentionally included in decision payloads. Deterministic
        # policy is profile/event deterministic, while full decision receipts are
        # time-stamped. Check stable profile/event roots instead.
        if high.event_root != high2.event_root:
            raise AssertionError("deterministic event root mismatch")

    return {
        "ok": True,
        "status": status_report(),
        "low_decision": low.to_dict(),
        "high_event_root": high.event_root,
        "pressure_event_root": pressure.event_root,
        "history_event_count": len(scaler.events()),
        "history_root": scaler.event_root(),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tensorized Blockchain governed cognitive lobe scaler")
    p.add_argument("--mode", choices=["status", "self-test", "recommend"], default="status")
    p.add_argument("--production", action="store_true")
    p.add_argument("--risk", type=float, default=0.0)
    p.add_argument("--drift", type=float, default=0.0)
    p.add_argument("--forecast-error", type=float, default=0.0)
    p.add_argument("--output", default="")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        if args.mode == "status":
            out = status_report()
        elif args.mode == "self-test":
            out = _self_test()
        else:
            out = recommend_profiles(
                risk_score=float(args.risk),
                state_drift_score=float(args.drift),
                forecast_error_score=float(args.forecast_error),
                production=bool(args.production),
            )
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(_json_safe(out), indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(_json_safe(out), indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        err = {"ok": False, "error": repr(exc), "ts": _now_iso()}
        print(json.dumps(err, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
