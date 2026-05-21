#!/usr/bin/env python3
# tb_demo_scenarios.py
r"""
Tensorized Blockchain - Canonical Demo Scenarios and Fixture Generation
======================================================================

Purpose
-------
Deterministic fixture layer for Tensorized Blockchain demos, tests, and app
smoke-runs. This module defines canonical scenario payloads rather than a live
runtime. It deliberately avoids importing the heavy twin/ledger/lobe stack at
module import time so it can be used safely by CLIs, tests, frontend generators,
and CI jobs.

Production role
---------------
- Creates deterministic scenario fixtures.
- Creates deterministic demo Ed25519 identities.
- Produces signed transaction envelopes for demos and tests.
- Provides twin creation payloads for twin_anything.py.
- Provides validator fixtures for tb_consensus.py.
- Exports stable JSON fixture bundles.

Design notes
------------
Demo private keys are deterministic by construction. They are only for local
fixtures and must never be used for real deployments. Public-key signing here is
real Ed25519 when the cryptography package is installed, but the key material is
intentionally predictable from fixture seeds.

This module is dependency-light:
    required: stdlib
    optional: cryptography, tb_canonical, tb_hashing, tb_crypto
"""

from __future__ import annotations

import argparse
import base64
import copy
import dataclasses
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# =============================================================================
# Version / constants
# =============================================================================

TB_DEMO_SCENARIOS_VERSION = "tb-demo-scenarios-v1.0-production"
DEFAULT_CHAIN_ID = "tensorized-blockchain-demo-chain-v1"
DEFAULT_CREATED_AT = "2027-01-01T00:00:00Z"
DEFAULT_BASE_SEED = 2027
DEMO_SIGNATURE_SCHEME = "ed25519-demo-deterministic"


# =============================================================================
# Optional shared-stack imports
# =============================================================================

try:  # pragma: no cover - available when full project stack is present
    from tb_canonical import canonical_json_dumps as _shared_canonical_json_dumps  # type: ignore
except Exception:  # pragma: no cover
    _shared_canonical_json_dumps = None

try:  # pragma: no cover
    from tb_hashing import sha256_hex as _shared_sha256_hex  # type: ignore
except Exception:  # pragma: no cover
    _shared_sha256_hex = None

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

    _CRYPTOGRAPHY_ED25519_OK = True
    _CRYPTOGRAPHY_ED25519_ERROR = ""
except Exception as exc:  # pragma: no cover - cryptography is expected but optional
    Ed25519PrivateKey = None  # type: ignore
    Ed25519PublicKey = None  # type: ignore
    Encoding = NoEncryption = PrivateFormat = PublicFormat = None  # type: ignore
    _CRYPTOGRAPHY_ED25519_OK = False
    _CRYPTOGRAPHY_ED25519_ERROR = repr(exc)


# =============================================================================
# Canonical helpers
# =============================================================================


def _json_safe(obj: Any) -> Any:
    """Return a deterministic JSON-safe representation."""
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return str(obj)
        return obj
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
    return str(obj)


def canonical_json_dumps(obj: Any) -> str:
    if _shared_canonical_json_dumps is not None:
        try:
            return str(_shared_canonical_json_dumps(obj))
        except Exception:
            pass
    return json.dumps(_json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_bytes(obj: Any) -> bytes:
    return canonical_json_dumps(obj).encode("utf-8", errors="strict")


def sha256_hex(obj: Any) -> str:
    if _shared_sha256_hex is not None:
        try:
            # Some project versions accept bytes, some accept arbitrary payloads.
            return str(_shared_sha256_hex(canonical_bytes(obj)))
        except Exception:
            try:
                return str(_shared_sha256_hex(obj))
            except Exception:
                pass
    if isinstance(obj, bytes):
        data = obj
    else:
        data = canonical_bytes(obj)
    return hashlib.sha256(data).hexdigest()


def b64e(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64d(text: str) -> bytes:
    return base64.b64decode(str(text).encode("ascii"), validate=True)


def stable_id(prefix: str, payload: Any, *, n: int = 24) -> str:
    return f"{prefix}_{sha256_hex(payload)[: int(n)]}"


def atomic_write_text(path: os.PathLike[str] | str, text: str) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


# =============================================================================
# Demo key material
# =============================================================================


@dataclass(frozen=True)
class DemoKeyPair:
    """Deterministic Ed25519 fixture identity. Never use outside demos/tests."""

    key_id: str
    owner_id: str
    public_key_b64: str
    private_key_b64: str
    fingerprint: str
    signature_scheme: str = DEMO_SIGNATURE_SCHEME
    warning: str = "DEMO ONLY: deterministic private key; never use in production."

    def public_dict(self) -> Dict[str, Any]:
        return {
            "key_id": self.key_id,
            "owner_id": self.owner_id,
            "public_key_b64": self.public_key_b64,
            "fingerprint": self.fingerprint,
            "signature_scheme": self.signature_scheme,
        }

    def to_dict(self, *, include_private: bool = False) -> Dict[str, Any]:
        out = self.public_dict()
        if include_private:
            out["private_key_b64"] = self.private_key_b64
            out["warning"] = self.warning
        return out


def _seed_bytes(namespace: str, seed: int | str) -> bytes:
    return hashlib.sha256(f"TB_DEMO_KEY_V1::{namespace}::{seed}".encode("utf-8")).digest()


def generate_demo_keypair(owner_id: str, *, seed: int | str = DEFAULT_BASE_SEED) -> DemoKeyPair:
    if not _CRYPTOGRAPHY_ED25519_OK:
        raise RuntimeError(f"cryptography Ed25519 unavailable: {_CRYPTOGRAPHY_ED25519_ERROR}")
    private_seed = _seed_bytes(str(owner_id), seed)[:32]
    priv = Ed25519PrivateKey.from_private_bytes(private_seed)  # type: ignore[union-attr]
    pub = priv.public_key()
    priv_raw = priv.private_bytes(  # type: ignore[union-attr]
        encoding=Encoding.Raw,
        format=PrivateFormat.Raw,
        encryption_algorithm=NoEncryption(),
    )
    pub_raw = pub.public_bytes(encoding=Encoding.Raw, format=PublicFormat.Raw)  # type: ignore[union-attr]
    fingerprint = hashlib.sha256(b"TB_KEY_FINGERPRINT_V1" + pub_raw).hexdigest()[:32]
    key_id = f"key_{fingerprint[:16]}"
    return DemoKeyPair(
        key_id=key_id,
        owner_id=str(owner_id),
        public_key_b64=b64e(pub_raw),
        private_key_b64=b64e(priv_raw),
        fingerprint=fingerprint,
    )


def sign_payload(payload: Mapping[str, Any], keypair: DemoKeyPair) -> str:
    if not _CRYPTOGRAPHY_ED25519_OK:
        raise RuntimeError(f"cryptography Ed25519 unavailable: {_CRYPTOGRAPHY_ED25519_ERROR}")
    priv = Ed25519PrivateKey.from_private_bytes(b64d(keypair.private_key_b64))  # type: ignore[union-attr]
    sig = priv.sign(canonical_bytes(payload))
    return b64e(sig)


def verify_payload_signature(payload: Mapping[str, Any], signature_b64: str, public_key_b64: str) -> bool:
    if not _CRYPTOGRAPHY_ED25519_OK:
        return False
    try:
        pub = Ed25519PublicKey.from_public_bytes(b64d(public_key_b64))  # type: ignore[union-attr]
        pub.verify(b64d(signature_b64), canonical_bytes(payload))
        return True
    except Exception:
        return False


# =============================================================================
# Scenario dataclasses
# =============================================================================


@dataclass(frozen=True)
class DemoValidatorFixture:
    validator_id: str
    display_name: str
    jurisdiction: str
    keypair: DemoKeyPair
    voting_power: int = 1
    role: str = "validator"

    def to_dict(self, *, include_private_keys: bool = False) -> Dict[str, Any]:
        return {
            "validator_id": self.validator_id,
            "display_name": self.display_name,
            "jurisdiction": self.jurisdiction,
            "voting_power": int(self.voting_power),
            "role": self.role,
            "key": self.keypair.to_dict(include_private=include_private_keys),
        }


@dataclass(frozen=True)
class DemoActorFixture:
    actor_id: str
    display_name: str
    actor_type: str
    keypair: DemoKeyPair
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_private_keys: bool = False) -> Dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "display_name": self.display_name,
            "actor_type": self.actor_type,
            "key": self.keypair.to_dict(include_private=include_private_keys),
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class DemoTelemetryEvent:
    event_id: str
    label: str
    node_id: str
    timestamp: str
    payload: Dict[str, Any]
    note: str
    risk_level: str = "nominal"

    def action(self) -> Dict[str, Any]:
        return {
            "kind": "update_node_data",
            "node_id": self.node_id,
            "payload": _json_safe(self.payload),
            "note": self.note,
        }

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class DemoScenarioFixture:
    scenario_id: str
    title: str
    description: str
    chain_id: str
    created_at: str
    seed: int
    twin_input: Dict[str, Any]
    actors: List[DemoActorFixture]
    validators: List[DemoValidatorFixture]
    telemetry_events: List[DemoTelemetryEvent]
    transactions: List[Dict[str, Any]]
    expected_flow: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_private_keys: bool = False) -> Dict[str, Any]:
        payload = {
            "schema": "tb.demo_scenario_fixture.v1",
            "version": TB_DEMO_SCENARIOS_VERSION,
            "scenario_id": self.scenario_id,
            "title": self.title,
            "description": self.description,
            "chain_id": self.chain_id,
            "created_at": self.created_at,
            "seed": int(self.seed),
            "twin_input": _json_safe(self.twin_input),
            "actors": [a.to_dict(include_private_keys=include_private_keys) for a in self.actors],
            "validators": [v.to_dict(include_private_keys=include_private_keys) for v in self.validators],
            "telemetry_events": [e.to_dict() for e in self.telemetry_events],
            "transactions": copy.deepcopy(self.transactions),
            "expected_flow": list(self.expected_flow),
            "metadata": _json_safe(self.metadata),
        }
        payload["fixture_hash"] = sha256_hex({k: v for k, v in payload.items() if k != "fixture_hash"})
        return payload

    def public_dict(self) -> Dict[str, Any]:
        return self.to_dict(include_private_keys=False)


# =============================================================================
# Transaction fixture generation
# =============================================================================


def make_signed_transaction(
    *,
    chain_id: str,
    sender: DemoActorFixture,
    sender_sequence: int,
    action: Mapping[str, Any],
    timestamp: str,
    scenario_id: str,
    event_id: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a deterministic signed transaction fixture."""
    envelope_unsigned = {
        "schema": "tb.transaction_envelope.v1",
        "chain_id": str(chain_id),
        "sender_id": sender.actor_id,
        "sender_public_key_b64": sender.keypair.public_key_b64,
        "sender_key_fingerprint": sender.keypair.fingerprint,
        "signature_scheme": sender.keypair.signature_scheme,
        "sender_sequence": int(sender_sequence),
        "timestamp": str(timestamp),
        "nonce": stable_id("nonce", [chain_id, sender.actor_id, sender_sequence, scenario_id, event_id], n=32),
        "action": _json_safe(action),
        "metadata": _json_safe({
            "scenario_id": scenario_id,
            "event_id": event_id,
            **dict(metadata or {}),
        }),
    }
    payload_hash = sha256_hex(envelope_unsigned)
    tx_id = stable_id("tx", [chain_id, sender.actor_id, sender_sequence, payload_hash], n=32)
    signing_payload = {**envelope_unsigned, "tx_id": tx_id, "payload_hash": payload_hash}
    signature = sign_payload(signing_payload, sender.keypair)
    out = {
        **signing_payload,
        "signature": signature,
    }
    out["signature_valid_self_check"] = verify_payload_signature(signing_payload, signature, sender.keypair.public_key_b64)
    return out


# =============================================================================
# Shared fixture helpers
# =============================================================================


def make_demo_validators(*, seed: int = DEFAULT_BASE_SEED, count: int = 4) -> List[DemoValidatorFixture]:
    jurisdictions = ["Alicante", "Madrid", "Oslo", "Canberra", "Brussels", "Helsinki"]
    validators: List[DemoValidatorFixture] = []
    for i in range(int(count)):
        vid = f"validator_{i + 1:02d}"
        kp = generate_demo_keypair(vid, seed=f"{seed}:validator:{i}")
        validators.append(
            DemoValidatorFixture(
                validator_id=vid,
                display_name=f"Demo Validator {i + 1}",
                jurisdiction=jurisdictions[i % len(jurisdictions)],
                keypair=kp,
                voting_power=1,
            )
        )
    return validators


def validator_quorum_config(validators: Sequence[DemoValidatorFixture], *, quorum_ratio: float = 0.67) -> Dict[str, Any]:
    total_power = int(sum(max(0, int(v.voting_power)) for v in validators))
    threshold = int(max(1, (total_power * float(quorum_ratio) + 0.999999)))
    return {
        "validator_count": len(validators),
        "total_voting_power": total_power,
        "quorum_ratio": float(quorum_ratio),
        "quorum_power_required": threshold,
    }


def _build_transactions_from_events(
    *,
    chain_id: str,
    scenario_id: str,
    actor: DemoActorFixture,
    events: Sequence[DemoTelemetryEvent],
) -> List[Dict[str, Any]]:
    return [
        make_signed_transaction(
            chain_id=chain_id,
            sender=actor,
            sender_sequence=i + 1,
            action=event.action(),
            timestamp=event.timestamp,
            scenario_id=scenario_id,
            event_id=event.event_id,
            metadata={"risk_level": event.risk_level, "label": event.label},
        )
        for i, event in enumerate(events)
    ]


# =============================================================================
# Scenario builders
# =============================================================================


def build_alicante_fire_department_scenario(
    *,
    chain_id: str = DEFAULT_CHAIN_ID,
    seed: int = DEFAULT_BASE_SEED,
    include_private_keys: bool = False,
) -> DemoScenarioFixture:
    del include_private_keys  # output selection happens in to_dict
    scenario_id = "alicante_fire_department_robot"
    robot = DemoActorFixture(
        actor_id="alicante_robot_01",
        display_name="Alicante Robot 01",
        actor_type="edge_robot",
        keypair=generate_demo_keypair("alicante_robot_01", seed=f"{seed}:robot"),
        metadata={"organization": "Alicante Fire Department", "deployment": "search_and_rescue"},
    )
    validators = make_demo_validators(seed=seed, count=4)
    twin_input = {
        "thing_name": "Hazard Robot Alpha",
        "thing_type": "industrial_robot",
        "description": "Search-and-rescue quadruped digital product passport for Alicante Fire Department operations.",
        "observables": {"temperature_c": 22.0, "gas_ppm": 0.0, "battery_pct": 91.0, "status": "nominal"},
        "components": {"platform": "quadruped", "payloads": ["thermal_camera", "gas_sensor", "lidar"]},
        "environment": {"city": "Alicante", "scenario": "fire_zone_entry", "connectivity": "edge_intermittent"},
        "success_criteria": {"minimum_probability": 0.80, "requires_governance_approval": True, "requires_audit_trail": True},
        "constraints": [
            "Actions must pass governed simulation.",
            "Thermal and gas telemetry must be committed before mission-state changes.",
            "Robot egress must be prioritized when thermal risk is high.",
        ],
        "risks": [
            {"name": "thermal_overload", "probability": 0.30, "impact": 0.85},
            {"name": "gas_exposure", "probability": 0.25, "impact": 0.80},
            {"name": "connectivity_loss", "probability": 0.20, "impact": 0.55},
        ],
        "actions": [
            {"kind": "observe", "description": "Commit telemetry snapshot."},
            {"kind": "simulate", "description": "Run governed mission-state simulation."},
            {"kind": "egress", "description": "Leave high-risk area when policy requires."},
        ],
        "tags": ["wiser", "alicante", "edge_robotics", "digital_product_passport", "tensorized_blockchain"],
        "metadata": {"demo_seed": seed, "canonical_scenario": scenario_id},
    }
    events = [
        DemoTelemetryEvent(
            event_id="fire_entry_telemetry",
            label="Fire-zone entry telemetry",
            node_id="5.4",
            timestamp="2027-01-01T00:01:00Z",
            payload={"temperature_c": 450.0, "gas_ppm": 80.0, "battery_pct": 87.0, "status": "fire_zone"},
            note="fire_zone_entry_telemetry",
            risk_level="high",
        ),
        DemoTelemetryEvent(
            event_id="thermal_peak_telemetry",
            label="Thermal peak telemetry",
            node_id="5.4",
            timestamp="2027-01-01T00:02:00Z",
            payload={"temperature_c": 510.0, "gas_ppm": 96.0, "battery_pct": 83.0, "status": "thermal_peak"},
            note="thermal_peak_telemetry",
            risk_level="critical",
        ),
        DemoTelemetryEvent(
            event_id="egress_telemetry",
            label="Egress telemetry",
            node_id="5.4",
            timestamp="2027-01-01T00:03:00Z",
            payload={"temperature_c": 390.0, "gas_ppm": 65.0, "battery_pct": 80.0, "status": "egress"},
            note="fire_zone_egress_telemetry",
            risk_level="elevated",
        ),
    ]
    txs = _build_transactions_from_events(chain_id=chain_id, scenario_id=scenario_id, actor=robot, events=events)
    return DemoScenarioFixture(
        scenario_id=scenario_id,
        title="Alicante Fire Department Robot",
        description="Deterministic edge-robot digital twin, telemetry transaction, lobe, and validator fixture.",
        chain_id=chain_id,
        created_at=DEFAULT_CREATED_AT,
        seed=seed,
        twin_input=twin_input,
        actors=[robot],
        validators=validators,
        telemetry_events=events,
        transactions=txs,
        expected_flow=[
            "create_twin_from_twin_input",
            "submit_signed_telemetry_transactions_to_mempool",
            "execute_governed_digital_twin_transition",
            "run_ncp_liquid_reflex_health_pass",
            "run_cfc_predictive_transition_forecast",
            "collect_transition_consensus_votes",
            "append_tensorized_block_with_pre_post_state_roots",
            "export_demo_artifacts",
        ],
        metadata={
            "validator_quorum": validator_quorum_config(validators),
            "primary_actor_id": robot.actor_id,
            "demo_profile": "edge_robotics_high_risk",
        },
    )


def build_industrial_machine_dpp_scenario(
    *,
    chain_id: str = DEFAULT_CHAIN_ID,
    seed: int = DEFAULT_BASE_SEED,
    include_private_keys: bool = False,
) -> DemoScenarioFixture:
    del include_private_keys
    scenario_id = "industrial_machine_digital_product_passport"
    machine_agent = DemoActorFixture(
        actor_id="cnc_mill_01_agent",
        display_name="CNC Mill 01 Agent",
        actor_type="machine_agent",
        keypair=generate_demo_keypair("cnc_mill_01_agent", seed=f"{seed}:cnc"),
        metadata={"site": "demo_factory", "asset_class": "cnc_mill"},
    )
    validators = make_demo_validators(seed=seed + 17, count=4)
    twin_input = {
        "thing_name": "CNC Mill 01",
        "thing_type": "industrial_machine",
        "description": "Digital Product Passport fixture for a CNC mill with maintenance and telemetry commitments.",
        "observables": {"spindle_temp_c": 41.0, "vibration_mm_s": 0.7, "hours_since_service": 120, "status": "nominal"},
        "components": {
            "machine": "CNC Mill",
            "serial": "CNC-DEMO-0001",
            "major_components": ["spindle", "controller", "coolant_system", "axis_drive"],
        },
        "environment": {"site": "demo_factory", "line": "cell_A", "operator_model": "supervised_autonomy"},
        "source_files": [
            {
                "filename": "cnc_mill_01_passport.json",
                "sha256": sha256_hex({"passport": "cnc_mill_01", "version": 1}),
                "mime_type": "application/json",
                "role": "digital_product_passport",
            }
        ],
        "success_criteria": {"minimum_probability": 0.82, "requires_maintenance_trace": True, "requires_audit_trail": True},
        "constraints": [
            "Maintenance events must be source-hash committed.",
            "High vibration requires governed alert before production continuation.",
            "Passport updates must not mutate historical component records.",
        ],
        "risks": [
            {"name": "spindle_wear", "probability": 0.18, "impact": 0.70},
            {"name": "quality_drift", "probability": 0.16, "impact": 0.65},
            {"name": "unverified_maintenance", "probability": 0.08, "impact": 0.80},
        ],
        "tags": ["digital_product_passport", "industrial_machine", "maintenance", "tensorized_blockchain"],
        "metadata": {"demo_seed": seed, "canonical_scenario": scenario_id},
    }
    events = [
        DemoTelemetryEvent(
            event_id="normal_operation_snapshot",
            label="Normal operation snapshot",
            node_id="5.4",
            timestamp="2027-01-01T01:00:00Z",
            payload={"spindle_temp_c": 43.0, "vibration_mm_s": 0.8, "hours_since_service": 124, "status": "nominal"},
            note="normal_operation_telemetry",
            risk_level="nominal",
        ),
        DemoTelemetryEvent(
            event_id="vibration_alert_snapshot",
            label="Vibration alert snapshot",
            node_id="5.4",
            timestamp="2027-01-01T01:10:00Z",
            payload={"spindle_temp_c": 52.0, "vibration_mm_s": 3.9, "hours_since_service": 134, "status": "inspect"},
            note="vibration_alert_telemetry",
            risk_level="high",
        ),
        DemoTelemetryEvent(
            event_id="maintenance_commitment_snapshot",
            label="Maintenance commitment snapshot",
            node_id="5.11",
            timestamp="2027-01-01T01:40:00Z",
            payload={
                "maintenance_action": "spindle_inspection",
                "technician_id": "tech_demo_01",
                "source_hash": sha256_hex({"inspection": "spindle", "result": "within_tolerance", "ts": "2027-01-01T01:35:00Z"}),
                "status": "maintenance_committed",
            },
            note="maintenance_source_commitment",
            risk_level="controlled",
        ),
    ]
    txs = _build_transactions_from_events(chain_id=chain_id, scenario_id=scenario_id, actor=machine_agent, events=events)
    return DemoScenarioFixture(
        scenario_id=scenario_id,
        title="Industrial Machine Digital Product Passport",
        description="Deterministic CNC mill DPP fixture with telemetry, maintenance source commitment, and validator set.",
        chain_id=chain_id,
        created_at=DEFAULT_CREATED_AT,
        seed=seed,
        twin_input=twin_input,
        actors=[machine_agent],
        validators=validators,
        telemetry_events=events,
        transactions=txs,
        expected_flow=[
            "create_machine_twin",
            "attach_passport_source_manifest",
            "submit_machine_telemetry_transactions",
            "detect_vibration_risk_with_lobes",
            "commit_maintenance_source_hash",
            "append_passport_state_transition_block",
        ],
        metadata={
            "validator_quorum": validator_quorum_config(validators),
            "primary_actor_id": machine_agent.actor_id,
            "demo_profile": "industrial_dpp_maintenance",
        },
    )


def build_lk20_curriculum_scenario(
    *,
    chain_id: str = DEFAULT_CHAIN_ID,
    seed: int = DEFAULT_BASE_SEED,
    include_private_keys: bool = False,
) -> DemoScenarioFixture:
    del include_private_keys
    scenario_id = "lk20_curriculum_alignment"
    curriculum_agent = DemoActorFixture(
        actor_id="lk20_alignment_agent",
        display_name="LK20 Alignment Agent",
        actor_type="curriculum_agent",
        keypair=generate_demo_keypair("lk20_alignment_agent", seed=f"{seed}:lk20"),
        metadata={"domain": "education", "region": "norway"},
    )
    validators = make_demo_validators(seed=seed + 31, count=4)
    twin_input = {
        "thing_name": "LK20 Local Curriculum Alignment Demo",
        "thing_type": "curriculum",
        "description": "Deterministic curriculum-alignment fixture for later LK20/Grep integration demos.",
        "observables": {"coverage_ratio": 0.0, "alignment_confidence": 0.0, "version_status": "pending"},
        "components": {
            "layers": ["canonical_curriculum", "subject", "grade", "competence_aim", "local_overlay", "assessment"],
        },
        "source_files": [
            {
                "filename": "local_curriculum_overlay_demo.json",
                "sha256": sha256_hex({"local_overlay": "demo", "version": 1}),
                "mime_type": "application/json",
                "role": "local_curriculum_overlay",
            }
        ],
        "success_criteria": {"minimum_coverage_ratio": 0.90, "requires_subject_binding": True, "requires_version_match": True},
        "constraints": [
            "Local overlays do not mutate canonical curriculum nodes.",
            "Alignment evidence must include source hashes.",
            "Low-confidence mappings must be quarantined before ledger commitment.",
        ],
        "risks": [
            {"name": "coverage_gap", "probability": 0.20, "impact": 0.60},
            {"name": "version_drift", "probability": 0.15, "impact": 0.55},
        ],
        "tags": ["lk20", "curriculum", "alignment", "tensorized_blockchain"],
        "metadata": {"demo_seed": seed, "canonical_scenario": scenario_id},
    }
    events = [
        DemoTelemetryEvent(
            event_id="overlay_registered",
            label="Local overlay registered",
            node_id="5.11",
            timestamp="2027-01-01T02:00:00Z",
            payload={"source_hash": sha256_hex({"overlay": "demo"}), "status": "registered"},
            note="curriculum_overlay_registered",
            risk_level="nominal",
        ),
        DemoTelemetryEvent(
            event_id="alignment_scored",
            label="Alignment scored",
            node_id="5.4",
            timestamp="2027-01-01T02:05:00Z",
            payload={"coverage_ratio": 0.93, "alignment_confidence": 0.88, "version_status": "matched"},
            note="curriculum_alignment_scored",
            risk_level="controlled",
        ),
    ]
    txs = _build_transactions_from_events(chain_id=chain_id, scenario_id=scenario_id, actor=curriculum_agent, events=events)
    return DemoScenarioFixture(
        scenario_id=scenario_id,
        title="LK20 Curriculum Alignment",
        description="Deterministic curriculum-alignment fixture reserved for later LK20/Grep demo integration.",
        chain_id=chain_id,
        created_at=DEFAULT_CREATED_AT,
        seed=seed,
        twin_input=twin_input,
        actors=[curriculum_agent],
        validators=validators,
        telemetry_events=events,
        transactions=txs,
        expected_flow=[
            "create_curriculum_twin",
            "register_local_overlay_source_hash",
            "score_alignment",
            "commit_alignment_receipt",
        ],
        metadata={
            "validator_quorum": validator_quorum_config(validators),
            "primary_actor_id": curriculum_agent.actor_id,
            "demo_profile": "curriculum_alignment_reserved",
        },
    )


SCENARIO_BUILDERS = {
    "alicante_fire_department_robot": build_alicante_fire_department_scenario,
    "alicante": build_alicante_fire_department_scenario,
    "fire_robot": build_alicante_fire_department_scenario,
    "industrial_machine_digital_product_passport": build_industrial_machine_dpp_scenario,
    "industrial_dpp": build_industrial_machine_dpp_scenario,
    "dpp": build_industrial_machine_dpp_scenario,
    "lk20_curriculum_alignment": build_lk20_curriculum_scenario,
    "lk20": build_lk20_curriculum_scenario,
    "curriculum": build_lk20_curriculum_scenario,
}

CANONICAL_SCENARIO_IDS = [
    "alicante_fire_department_robot",
    "industrial_machine_digital_product_passport",
    "lk20_curriculum_alignment",
]


def list_scenarios() -> List[str]:
    return list(CANONICAL_SCENARIO_IDS)


def build_scenario(
    scenario: str,
    *,
    chain_id: str = DEFAULT_CHAIN_ID,
    seed: int = DEFAULT_BASE_SEED,
) -> DemoScenarioFixture:
    key = str(scenario or "").lower().strip()
    builder = SCENARIO_BUILDERS.get(key)
    if builder is None:
        raise ValueError(f"Unknown scenario {scenario!r}. Available: {', '.join(CANONICAL_SCENARIO_IDS)}")
    return builder(chain_id=chain_id, seed=seed)


def build_all_scenarios(*, chain_id: str = DEFAULT_CHAIN_ID, seed: int = DEFAULT_BASE_SEED) -> Dict[str, DemoScenarioFixture]:
    return {sid: build_scenario(sid, chain_id=chain_id, seed=seed) for sid in CANONICAL_SCENARIO_IDS}


# =============================================================================
# Export helpers
# =============================================================================


def export_scenario_json(
    path: os.PathLike[str] | str,
    scenario: str,
    *,
    chain_id: str = DEFAULT_CHAIN_ID,
    seed: int = DEFAULT_BASE_SEED,
    include_private_keys: bool = False,
    pretty: bool = True,
) -> Path:
    fixture = build_scenario(scenario, chain_id=chain_id, seed=seed)
    payload = fixture.to_dict(include_private_keys=include_private_keys)
    text = json.dumps(payload, indent=2 if pretty else None, ensure_ascii=False, sort_keys=True)
    p = Path(path).expanduser().resolve()
    atomic_write_text(p, text + "\n")
    return p


def export_all_fixtures(
    output_dir: os.PathLike[str] | str,
    *,
    chain_id: str = DEFAULT_CHAIN_ID,
    seed: int = DEFAULT_BASE_SEED,
    include_private_keys: bool = False,
    pretty: bool = True,
) -> Dict[str, Any]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}
    hashes: Dict[str, str] = {}
    for sid in CANONICAL_SCENARIO_IDS:
        p = out / f"{sid}.json"
        export_scenario_json(
            p,
            sid,
            chain_id=chain_id,
            seed=seed,
            include_private_keys=include_private_keys,
            pretty=pretty,
        )
        data = p.read_bytes()
        files[sid] = str(p)
        hashes[sid] = hashlib.sha256(data).hexdigest()
    manifest = {
        "schema": "tb.demo_fixture_export_manifest.v1",
        "version": TB_DEMO_SCENARIOS_VERSION,
        "chain_id": chain_id,
        "seed": int(seed),
        "include_private_keys": bool(include_private_keys),
        "scenario_ids": CANONICAL_SCENARIO_IDS,
        "files": files,
        "file_hashes": hashes,
    }
    manifest["manifest_hash"] = sha256_hex(manifest)
    manifest_path = out / "demo_fixtures_manifest.json"
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


# =============================================================================
# Optional runtime adapters
# =============================================================================


def try_materialize_twin_fixture(scenario_fixture: DemoScenarioFixture) -> Dict[str, Any]:
    """
    Best-effort adapter for twin_anything.py. This is intentionally optional and
    returns a structured diagnostic rather than failing imports at module load.
    """
    try:
        import twin_anything as ta  # type: ignore
    except Exception as exc:
        return {"ok": False, "stage": "import_twin_anything", "error": repr(exc)}

    try:
        if hasattr(ta, "build_anything_twin"):
            result = ta.build_anything_twin(**scenario_fixture.twin_input)
        else:
            factory = ta.TwinAnythingFactory()  # type: ignore[attr-defined]
            result = factory.create_twin(**scenario_fixture.twin_input)
        snapshot = result.get("snapshot") if isinstance(result, Mapping) else None
        metadata = result.get("metadata") if isinstance(result, Mapping) else None
        return {"ok": True, "snapshot": _json_safe(snapshot), "metadata": _json_safe(metadata)}
    except Exception as exc:
        return {"ok": False, "stage": "create_twin", "error": repr(exc)}


# =============================================================================
# Self-test
# =============================================================================


def _assert_transactions_valid(fixture: DemoScenarioFixture) -> None:
    seen = set()
    seq_by_sender: Dict[str, int] = {}
    for tx in fixture.transactions:
        tx_id = str(tx.get("tx_id", ""))
        if not tx_id or tx_id in seen:
            raise AssertionError("duplicate or empty tx_id")
        seen.add(tx_id)
        sender = str(tx.get("sender_id", ""))
        seq = int(tx.get("sender_sequence", 0))
        if seq != seq_by_sender.get(sender, 0) + 1:
            raise AssertionError(f"sender sequence gap for {sender}: {seq}")
        seq_by_sender[sender] = seq
        signing_payload = {k: v for k, v in tx.items() if k not in {"signature", "signature_valid_self_check"}}
        if not verify_payload_signature(signing_payload, str(tx["signature"]), str(tx["sender_public_key_b64"])):
            raise AssertionError(f"signature verification failed for {tx_id}")


def run_self_test() -> Dict[str, Any]:
    if not _CRYPTOGRAPHY_ED25519_OK:
        raise RuntimeError(f"cryptography Ed25519 unavailable: {_CRYPTOGRAPHY_ED25519_ERROR}")

    a1 = build_scenario("alicante_fire_department_robot")
    a2 = build_scenario("alicante_fire_department_robot")
    if canonical_json_dumps(a1.to_dict(include_private_keys=True)) != canonical_json_dumps(a2.to_dict(include_private_keys=True)):
        raise AssertionError("scenario generation is not deterministic")

    for fixture in build_all_scenarios().values():
        _assert_transactions_valid(fixture)
        public_payload = fixture.to_dict(include_private_keys=False)
        private_payload = fixture.to_dict(include_private_keys=True)
        if "private_key_b64" in canonical_json_dumps(public_payload):
            raise AssertionError("public fixture leaked private_key_b64")
        if "private_key_b64" not in canonical_json_dumps(private_payload):
            raise AssertionError("private fixture missing private_key_b64")
        if not public_payload.get("fixture_hash"):
            raise AssertionError("fixture_hash missing")

    with tempfile.TemporaryDirectory() as td:
        manifest = export_all_fixtures(td, include_private_keys=False)
        if len(manifest.get("files", {})) != len(CANONICAL_SCENARIO_IDS):
            raise AssertionError("export manifest missing files")
        for p in manifest["files"].values():
            if not Path(p).exists():
                raise AssertionError(f"missing export file {p}")

    return {
        "ok": True,
        "version": TB_DEMO_SCENARIOS_VERSION,
        "scenario_ids": CANONICAL_SCENARIO_IDS,
        "crypto_ed25519": _CRYPTOGRAPHY_ED25519_OK,
        "alicante_fixture_hash": a1.public_dict()["fixture_hash"],
    }


# =============================================================================
# CLI
# =============================================================================


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tensorized Blockchain deterministic demo scenarios")
    p.add_argument("--mode", choices=["list", "scenario", "export-all", "self-test"], default="list")
    p.add_argument("--scenario", default="alicante_fire_department_robot")
    p.add_argument("--chain-id", default=DEFAULT_CHAIN_ID)
    p.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    p.add_argument("--output", default="")
    p.add_argument("--include-private-keys", action="store_true")
    p.add_argument("--compact", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.mode == "list":
            payload: Any = {"ok": True, "version": TB_DEMO_SCENARIOS_VERSION, "scenario_ids": list_scenarios()}
        elif args.mode == "scenario":
            fixture = build_scenario(args.scenario, chain_id=args.chain_id, seed=args.seed)
            payload = fixture.to_dict(include_private_keys=bool(args.include_private_keys))
            if args.output:
                export_scenario_json(
                    args.output,
                    args.scenario,
                    chain_id=args.chain_id,
                    seed=args.seed,
                    include_private_keys=bool(args.include_private_keys),
                    pretty=not bool(args.compact),
                )
                payload = {"ok": True, "saved": str(Path(args.output).expanduser().resolve()), "fixture_hash": payload["fixture_hash"]}
        elif args.mode == "export-all":
            if not args.output:
                raise ValueError("--output directory is required for --mode export-all")
            payload = export_all_fixtures(
                args.output,
                chain_id=args.chain_id,
                seed=args.seed,
                include_private_keys=bool(args.include_private_keys),
                pretty=not bool(args.compact),
            )
            payload["ok"] = True
        else:
            payload = run_self_test()
        print(json.dumps(_json_safe(payload), indent=None if args.compact else 2, sort_keys=True, ensure_ascii=False))
        return 0
    except Exception as exc:
        err = {"ok": False, "error": repr(exc), "version": TB_DEMO_SCENARIOS_VERSION}
        print(json.dumps(err, indent=2, sort_keys=True), file=os.sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
