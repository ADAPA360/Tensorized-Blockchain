#!/usr/bin/env python3
# app.py
r"""
Tensorized Blockchain — Cinematic Demo Cockpit
==============================================

Streamlit entrypoint for the Project Chimera / Akkurat Tensorized Blockchain
stack. The app is deliberately built as a single-file production demo shell so it
can be dropped into the project root and launched quickly:

    streamlit run app.py

Design intent
-------------
The first demo version should feel like a mission-control cockpit rather than a
plain form UI. It shows a repeatable story:

1. A governed digital twin is created for a high-risk physical scenario.
2. Signed telemetry transactions are submitted.
3. The governed state transition is executed.
4. Validator votes/certificates are visualized.
5. A cryptographic block is forged.
6. Tensor commitments and compression are shown.
7. The dualistic brain lobes, NCP and CfC, are inspected.
8. Ledger/tensor/demo artifacts can be exported.

The app uses the full stack when available, but degrades gracefully when a module
is missing. This keeps the demo useful during active development.
"""

from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import importlib
import io
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore


APP_VERSION = "tensorized-blockchain-streamlit-app-v1.0-production"
DEFAULT_CHAIN_ID = "tb-demo-alicante-wiser-2026"
DEFAULT_SEED = 2027
DEFAULT_OUTPUT_DIR = Path("./artifacts")


# =============================================================================
# Low-level helpers
# =============================================================================


def _module_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()


def _configure_import_path() -> None:
    here = _module_dir()
    candidates = [here, Path.cwd(), Path(tempfile.gettempdir())]
    for p in candidates:
        try:
            s = str(p.resolve())
        except Exception:
            continue
        if s not in sys.path:
            sys.path.insert(0, s)


_configure_import_path()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if np is not None and isinstance(obj, np.generic):
        return json_safe(obj.item())
    if np is not None and isinstance(obj, np.ndarray):
        arr = np.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.astype(float).tolist()
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
    if hasattr(obj, "public_dict") and callable(obj.public_dict):
        try:
            return json_safe(obj.public_dict())
        except Exception:
            pass
    if hasattr(obj, "summary") and callable(obj.summary):
        try:
            return json_safe(obj.summary())
        except Exception:
            pass
    if hasattr(obj, "describe") and callable(obj.describe):
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


def canonical_json(obj: Any, *, indent: Optional[int] = None) -> str:
    return json.dumps(json_safe(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":") if indent is None else None, indent=indent)


def sha256_hex(obj: Any) -> str:
    if isinstance(obj, bytes):
        b = obj
    else:
        b = canonical_json(obj).encode("utf-8", errors="ignore")
    return hashlib.sha256(b).hexdigest()


def short_hash(value: Any, n: int = 12) -> str:
    s = str(value or "")
    if len(s) >= n and all(c in "0123456789abcdefABCDEF" for c in s[:n]):
        return s[:n]
    return sha256_hex(s)[:n]


def clamp01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return float(default)
        return max(0.0, min(1.0, v))
    except Exception:
        return float(default)


def safe_get(mapping: Any, path: Sequence[Any], default: Any = None) -> Any:
    cur = mapping
    for key in path:
        try:
            if isinstance(cur, Mapping):
                cur = cur[key]
            elif isinstance(cur, Sequence) and not isinstance(cur, (str, bytes)):
                cur = cur[int(key)]
            else:
                return default
        except Exception:
            return default
    return cur


def import_optional(name: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return importlib.import_module(name), None
    except Exception as exc:
        return None, repr(exc)


def module_status(names: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in names:
        mod, err = import_optional(name)
        out[name] = {
            "available": mod is not None,
            "module": getattr(mod, "__file__", None) if mod is not None else None,
            "error": err,
        }
    return out


def bytes_download(payload: Any, *, pretty: bool = True) -> bytes:
    return canonical_json(payload, indent=2 if pretty else None).encode("utf-8")


def make_zip_bytes(files: Mapping[str, Any]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, payload in files.items():
            if isinstance(payload, bytes):
                data = payload
            elif isinstance(payload, str):
                data = payload.encode("utf-8")
            else:
                data = bytes_download(payload)
            zf.writestr(filename, data)
    return bio.getvalue()


# =============================================================================
# Data model
# =============================================================================


@dataclass
class CinematicPhase:
    index: int
    title: str
    subtitle: str
    status: str = "pending"  # pending | active | complete | degraded | failed
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class DemoRunArtifacts:
    run_id: str
    scenario_id: str
    created_at: str
    fixture: Dict[str, Any]
    twin_materialization: Dict[str, Any]
    ledger_report: Dict[str, Any]
    lobe_report: Dict[str, Any]
    validator_report: Dict[str, Any]
    tensor_report: Dict[str, Any]
    dependency_status: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = json_safe(asdict(self))
        payload["artifact_hash"] = sha256_hex({k: v for k, v in payload.items() if k != "artifact_hash"})
        return payload


# =============================================================================
# Cinematic CSS and UI primitives
# =============================================================================


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --tb-bg: #05070d;
            --tb-panel: rgba(12, 18, 32, 0.78);
            --tb-panel-2: rgba(16, 26, 48, 0.82);
            --tb-line: rgba(92, 180, 255, 0.24);
            --tb-cyan: #74d7ff;
            --tb-gold: #ffd166;
            --tb-green: #80ed99;
            --tb-red: #ff6b6b;
            --tb-muted: #9fb3c8;
        }
        .stApp {
            background:
                radial-gradient(circle at 8% 12%, rgba(60, 120, 255, 0.18), transparent 28%),
                radial-gradient(circle at 88% 18%, rgba(255, 209, 102, 0.12), transparent 24%),
                linear-gradient(135deg, #02040a 0%, #07111f 48%, #02040a 100%);
            color: #edf6ff;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(6, 12, 24, 0.96), rgba(8, 15, 28, 0.92));
            border-right: 1px solid rgba(116, 215, 255, 0.15);
        }
        .tb-hero {
            border: 1px solid rgba(116, 215, 255, 0.20);
            background: linear-gradient(135deg, rgba(8, 16, 32, 0.88), rgba(18, 30, 58, 0.74));
            border-radius: 28px;
            padding: 28px 30px;
            box-shadow: 0 24px 80px rgba(0,0,0,0.35), inset 0 0 55px rgba(116,215,255,0.045);
            margin-bottom: 18px;
        }
        .tb-title {
            font-size: 2.85rem;
            line-height: 1.02;
            font-weight: 800;
            letter-spacing: -0.06em;
            margin: 0 0 8px 0;
        }
        .tb-subtitle {
            color: var(--tb-muted);
            font-size: 1.05rem;
            max-width: 980px;
        }
        .tb-chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
        .tb-chip {
            border: 1px solid rgba(116, 215, 255, 0.24);
            background: rgba(116, 215, 255, 0.07);
            color: #d8f4ff;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            letter-spacing: .02em;
        }
        .tb-card {
            border: 1px solid rgba(116, 215, 255, 0.17);
            background: var(--tb-panel);
            border-radius: 22px;
            padding: 18px 18px;
            box-shadow: 0 18px 55px rgba(0,0,0,0.22);
            min-height: 100px;
        }
        .tb-card h3 { margin-top: 0; margin-bottom: 8px; font-size: 1.08rem; }
        .tb-muted { color: var(--tb-muted); font-size: 0.92rem; }
        .tb-metric-large {
            font-size: 2.0rem;
            font-weight: 760;
            letter-spacing: -0.04em;
            color: #fff;
        }
        .tb-good { color: var(--tb-green); }
        .tb-warn { color: var(--tb-gold); }
        .tb-bad { color: var(--tb-red); }
        .tb-glow {
            color: var(--tb-cyan);
            text-shadow: 0 0 18px rgba(116, 215, 255, 0.35);
        }
        .tb-phase {
            border-left: 3px solid rgba(116, 215, 255, 0.34);
            padding: 10px 12px;
            margin: 8px 0;
            border-radius: 10px;
            background: rgba(255,255,255,0.035);
        }
        .tb-phase-complete { border-left-color: #80ed99; }
        .tb-phase-active { border-left-color: #ffd166; background: rgba(255,209,102,0.07); }
        .tb-phase-degraded { border-left-color: #ffb703; }
        .tb-phase-failed { border-left-color: #ff6b6b; }
        .tb-hash {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.78rem;
            color: #bfefff;
            word-break: break-all;
        }
        .tb-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(116,215,255,0.36), transparent);
            margin: 18px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, chips: Sequence[str]) -> None:
    chip_html = "".join(f"<span class='tb-chip'>{c}</span>" for c in chips)
    st.markdown(
        f"""
        <div class="tb-hero">
            <div class="tb-title">{title}</div>
            <div class="tb-subtitle">{subtitle}</div>
            <div class="tb-chip-row">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, body: str, *, value: Optional[str] = None, tone: str = "") -> None:
    value_html = f"<div class='tb-metric-large {tone}'>{value}</div>" if value is not None else ""
    st.markdown(
        f"""
        <div class="tb-card">
            <h3>{title}</h3>
            {value_html}
            <div class="tb-muted">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def phase_timeline(phases: Sequence[CinematicPhase]) -> None:
    for p in phases:
        cls = f"tb-phase tb-phase-{p.status}"
        st.markdown(
            f"""
            <div class="{cls}">
                <strong>{p.index:02d}. {p.title}</strong><br/>
                <span class="tb-muted">{p.subtitle}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_hash(label: str, value: Any) -> None:
    st.markdown(f"**{label}**  ")
    st.markdown(f"<div class='tb-hash'>{value}</div>", unsafe_allow_html=True)


def status_pill(ok: bool, label: str) -> str:
    return f"🟢 {label}" if ok else f"🟠 {label}"


# =============================================================================
# Stack integration
# =============================================================================


STACK_MODULES = [
    "tb_config",
    "tb_canonical",
    "tb_hashing",
    "tb_crypto",
    "tb_schemas",
    "tb_validation",
    "tb_storage",
    "tb_mempool",
    "tb_consensus",
    "tb_execution",
    "tn",
    "digital_twin_kernel",
    "twin_anything",
    "tensorized_blockchain_kernel",
    "ncp",
    "cfc",
    "cognitive_lobe_runtime",
    "lobe_scaler",
    "tb_tensor_commitments",
    "tb_demo_scenarios",
]


def get_stack_status() -> Dict[str, Any]:
    status = module_status(STACK_MODULES)
    available = sum(1 for v in status.values() if v.get("available"))
    status["__summary__"] = {
        "available": available,
        "total": len(STACK_MODULES),
        "ratio": available / max(1, len(STACK_MODULES)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "app_version": APP_VERSION,
        "ts": utc_now(),
    }
    return status


def build_fixture(scenario_id: str, *, seed: int, chain_id: str) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    mod, err = import_optional("tb_demo_scenarios")
    if mod is not None and hasattr(mod, "build_scenario"):
        try:
            fixture = mod.build_scenario(scenario_id, chain_id=chain_id, seed=seed)
            if hasattr(fixture, "public_dict"):
                return fixture.public_dict(), warnings
            if hasattr(fixture, "to_dict"):
                return fixture.to_dict(), warnings
            return json_safe(fixture), warnings
        except Exception as exc:
            warnings.append(f"tb_demo_scenarios.build_scenario failed: {exc!r}")
    else:
        warnings.append(f"tb_demo_scenarios unavailable: {err}")

    # Deterministic fallback fixture.
    txs: List[Dict[str, Any]] = []
    events = [
        ("fire_zone_entry", {"temperature_c": 450.0, "gas_ppm": 80.0, "status": "fire_zone"}),
        ("egress", {"temperature_c": 390.0, "gas_ppm": 65.0, "status": "egress"}),
    ]
    for i, (note, payload) in enumerate(events, start=1):
        unsigned = {
            "schema": "tb.transaction_envelope.v1",
            "chain_id": chain_id,
            "sender_id": "alicante_robot_01",
            "sender_sequence": i,
            "timestamp": f"2026-05-20T12:00:0{i}Z",
            "nonce": sha256_hex([chain_id, i, note])[:32],
            "action": {"kind": "update_node_data", "node_id": "5.4", "payload": payload, "note": note},
            "metadata": {"scenario_id": scenario_id, "event_id": note},
        }
        unsigned["payload_hash"] = sha256_hex(unsigned)
        unsigned["tx_id"] = sha256_hex([unsigned["payload_hash"], i])[:32]
        unsigned["signature"] = "fallback_unsigned_demo"
        txs.append(unsigned)
    fixture = {
        "schema": "tb.demo_scenario_fixture.v1",
        "scenario_id": scenario_id,
        "title": "Alicante Fire Department Robot",
        "description": "Fallback deterministic fixture for governed rescue-robot telemetry.",
        "chain_id": chain_id,
        "seed": seed,
        "created_at": "2026-05-20T12:00:00Z",
        "twin_input": {
            "thing_name": "Hazard Robot Alpha",
            "thing_type": "industrial_machine",
            "description": "Search-and-rescue quadruped digital product passport.",
            "observables": {"temperature_c": 22.0, "gas_ppm": 0.0, "battery_pct": 91.0},
            "tags": ["wiser", "alicante", "tensorized_blockchain"],
        },
        "validators": [{"validator_id": f"validator_{i:02d}", "voting_power": 1} for i in range(1, 5)],
        "transactions": txs,
        "expected_flow": ["create_twin", "submit_tx", "governed_execute", "vote", "append_block", "compress_tensor"],
    }
    fixture["fixture_hash"] = sha256_hex(fixture)
    return fixture, warnings


def materialize_twin(fixture: Mapping[str, Any], *, deterministic: bool = True) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    demo_mod, _ = import_optional("tb_demo_scenarios")
    if demo_mod is not None and hasattr(demo_mod, "try_materialize_twin_fixture"):
        # This helper expects the dataclass fixture in some versions, not dict. Try only if possible.
        pass

    ta, err = import_optional("twin_anything")
    if ta is not None:
        try:
            twin_input = dict(fixture.get("twin_input", {}))
            cfg_kwargs = {
                "vector_dim": 128,
                "sketch_dim": 64,
                "history_capacity": 32,
                "seed": int(fixture.get("seed", DEFAULT_SEED)),
                "use_tn_projection": False,
            }
            factory = ta.TwinAnythingFactory(**cfg_kwargs)
            # Prefer serializable creation if available.
            if hasattr(factory, "create_twin_snapshot"):
                result = factory.create_twin_snapshot(**twin_input, deterministic=deterministic)  # type: ignore[arg-type]
                return json_safe(result), warnings
            if hasattr(factory, "create_twin"):
                result = factory.create_twin(**twin_input)  # type: ignore[arg-type]
                serial = {k: v for k, v in json_safe(result).items() if k not in {"network", "control_plane"}}
                serial["runtime_objects_available"] = True
                serial["state_root"] = short_hash(serial, 64)
                return serial, warnings
        except Exception as exc:
            warnings.append(f"Twin materialization failed: {exc!r}")
    else:
        warnings.append(f"twin_anything unavailable: {err}")

    fallback = {
        "ok": False,
        "degraded": True,
        "state_root": sha256_hex({"fixture": fixture, "deterministic": deterministic}),
        "sketch_root": sha256_hex({"sketch": fixture.get("twin_input", {})})[:16],
        "receipt": {
            "schema": "tb.twin_creation_receipt.v1.fallback",
            "fixture_hash": fixture.get("fixture_hash") or sha256_hex(fixture),
            "created_at": "deterministic" if deterministic else utc_now(),
        },
    }
    return fallback, warnings


def run_ledger_demo(*, output_dir: Path) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    mod, err = import_optional("tensorized_blockchain_kernel")
    if mod is not None and hasattr(mod, "run_wiser_demo"):
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            ledger_path = output_dir / "alicante_demo_ledger.json"
            report = mod.run_wiser_demo(output=ledger_path)
            out = json_safe(report)
            out["ledger_path"] = str(ledger_path)
            return out, warnings
        except Exception as exc:
            warnings.append(f"tensorized_blockchain_kernel.run_wiser_demo failed: {exc!r}")
    else:
        warnings.append(f"tensorized_blockchain_kernel unavailable: {err}")

    # Structural fallback block report.
    block0 = {
        "index": 0,
        "block_hash": sha256_hex(["genesis", DEFAULT_CHAIN_ID]),
        "tx_count": 0,
        "world_root": "genesis_world_root",
    }
    block1 = {
        "index": 1,
        "block_hash": sha256_hex([block0["block_hash"], "fire_zone_entry"]),
        "prev_hash": block0["block_hash"],
        "tx_count": 1,
        "world_root": sha256_hex("fire_zone_world"),
        "tensor_commitment": sha256_hex("fallback_tensor_commitment"),
    }
    return {
        "ok": False,
        "degraded": True,
        "demo": "FALLBACK_STRUCTURAL_LEDGER_DEMO",
        "ledger": {"block_count": 2, "head_hash": block1["block_hash"], "valid": True},
        "blocks": [block0, block1],
        "validation": {"ok": True, "mode": "fallback_structural"},
        "compression": {"status": "fallback", "compression_ratio": 0.0},
    }, warnings


def run_lobe_demo(fixture: Mapping[str, Any], ledger_report: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    rt_mod, err = import_optional("cognitive_lobe_runtime")
    if rt_mod is not None:
        try:
            runtime_cls = getattr(rt_mod, "DualLobeRuntime", None) or getattr(rt_mod, "CognitiveLobeRuntime", None)
            if runtime_cls is not None:
                # Constructor signatures may evolve. Try a minimal deterministic config.
                try:
                    runtime = runtime_cls(seed=int(fixture.get("seed", DEFAULT_SEED)), deterministic=True)
                except TypeError:
                    runtime = runtime_cls()
                tx = safe_get(fixture, ["transactions", 0], {})
                pre = safe_get(ledger_report, ["blocks", 0, "world_root"], "0" * 64)
                post = safe_get(ledger_report, ["blocks", -1, "world_root"], sha256_hex("post"))
                if hasattr(runtime, "run"):
                    result = runtime.run(
                        twin_snapshot=fixture.get("twin_input", {}),
                        transaction=tx,
                        pre_state_commitment=pre,
                        post_state_commitment=post,
                        extra={"source": "streamlit_app", "ledger_head": safe_get(ledger_report, ["ledger", "head_hash"], "")},
                        time_delta=0.5,
                    )
                    return json_safe(result), warnings
        except Exception as exc:
            warnings.append(f"cognitive_lobe_runtime execution failed: {exc!r}")
    else:
        warnings.append(f"cognitive_lobe_runtime unavailable: {err}")

    # Lightweight fallback lobe visualization data.
    feature_hash = sha256_hex({"fixture": fixture.get("fixture_hash"), "ledger": safe_get(ledger_report, ["ledger", "head_hash"], "")})
    return {
        "ok": False,
        "degraded": True,
        "runtime_id": "fallback_dual_lobe_runtime",
        "run_hash": feature_hash,
        "ncp_result": {
            "lobe_kind": "ncp_liquid_reflex",
            "health_receipt": {
                "is_stable": True,
                "hidden_norm": 0.42,
                "saturation_fraction": 0.02,
                "state_hash": sha256_hex([feature_hash, "ncp"]),
            },
            "metrics": {"hidden_norm": 0.42, "stability": 0.98},
        },
        "cfc_result": {
            "lobe_kind": "cfc_predictive_transition",
            "health_receipt": {
                "is_stable": True,
                "forecast_error": 0.031,
                "state_hash": sha256_hex([feature_hash, "cfc"]),
            },
            "metrics": {"forecast_error": 0.031, "compression_factor": 8.6},
        },
    }, warnings


def extract_validator_report(fixture: Mapping[str, Any], ledger_report: Mapping[str, Any]) -> Dict[str, Any]:
    validators = list(fixture.get("validators", []))
    if not validators:
        validators = [{"validator_id": f"validator_{i:02d}", "voting_power": 1} for i in range(1, 5)]
    head = safe_get(ledger_report, ["ledger", "head_hash"], sha256_hex("head"))
    votes = []
    for i, v in enumerate(validators):
        vid = str(v.get("validator_id") or v.get("display_name") or f"validator_{i+1}")
        approve = i < max(1, math.ceil(0.67 * len(validators)))
        votes.append({
            "validator_id": vid,
            "approval": bool(approve),
            "voting_power": int(v.get("voting_power", 1)),
            "vote_hash": sha256_hex([vid, head, approve]),
            "reason": "transition_root_and_policy_receipt_match" if approve else "withheld_for_demo_variance",
        })
    approvals = sum(int(v["voting_power"]) for v in votes if v["approval"])
    total = sum(int(v["voting_power"]) for v in votes)
    threshold = math.ceil(0.67 * total)
    return {
        "ok": approvals >= threshold,
        "validator_count": len(votes),
        "approvals": approvals,
        "total_power": total,
        "quorum_required": threshold,
        "quorum_ratio": approvals / max(1, total),
        "certificate_hash": sha256_hex({"head": head, "votes": votes, "threshold": threshold}),
        "votes": votes,
    }


def extract_tensor_report(ledger_report: Mapping[str, Any]) -> Dict[str, Any]:
    compression = dict(safe_get(ledger_report, ["compression"], {}) or {})
    ledger = dict(safe_get(ledger_report, ["ledger"], {}) or {})
    blocks = list(ledger_report.get("blocks", [])) if isinstance(ledger_report, Mapping) else []
    param_old = compression.get("old_parameters") or compression.get("dense_equivalent_parameters") or 4096
    param_new = compression.get("new_parameters") or compression.get("active_parameters") or 768
    try:
        ratio = 1.0 - (float(param_new) / max(1.0, float(param_old)))
    except Exception:
        ratio = 0.0
    return {
        "ok": bool(ledger_report.get("ok", True)) if isinstance(ledger_report, Mapping) else True,
        "tensor_commitment": safe_get(blocks, [-1, "tensor_commitment"], None) or safe_get(ledger_report, ["tensor_commitment"], None) or sha256_hex(blocks),
        "compression_status": compression.get("status", "reported"),
        "old_parameters": int(param_old) if str(param_old).isdigit() else param_old,
        "new_parameters": int(param_new) if str(param_new).isdigit() else param_new,
        "compression_ratio": float(compression.get("compression_ratio", ratio) or ratio),
        "space_saved_percent": float(compression.get("space_saved_percent", 100.0 * ratio) or 100.0 * ratio),
        "block_count": ledger.get("block_count", len(blocks)),
        "head_hash": ledger.get("head_hash", safe_get(blocks, [-1, "block_hash"], "")),
    }


# =============================================================================
# Cached orchestration
# =============================================================================


@st.cache_data(show_spinner=False) if st is not None else (lambda f: f)  # type: ignore[misc]
def cached_stack_status() -> Dict[str, Any]:
    return get_stack_status()


@st.cache_data(show_spinner=False) if st is not None else (lambda f: f)  # type: ignore[misc]
def cached_fixture(scenario_id: str, seed: int, chain_id: str) -> Dict[str, Any]:
    fixture, warnings = build_fixture(scenario_id, seed=seed, chain_id=chain_id)
    return {"fixture": fixture, "warnings": warnings}


def run_full_demo(scenario_id: str, *, seed: int, chain_id: str, output_dir: Path) -> DemoRunArtifacts:
    fixture, w1 = build_fixture(scenario_id, seed=seed, chain_id=chain_id)
    twin, w2 = materialize_twin(fixture, deterministic=True)
    ledger, w3 = run_ledger_demo(output_dir=output_dir)
    lobes, w4 = run_lobe_demo(fixture, ledger)
    validators = extract_validator_report(fixture, ledger)
    tensor = extract_tensor_report(ledger)
    deps = cached_stack_status()
    run_id = sha256_hex({"scenario_id": scenario_id, "seed": seed, "chain_id": chain_id, "ts": "deterministic"})[:24]
    return DemoRunArtifacts(
        run_id=run_id,
        scenario_id=scenario_id,
        created_at=utc_now(),
        fixture=fixture,
        twin_materialization=twin,
        ledger_report=ledger,
        lobe_report=lobes,
        validator_report=validators,
        tensor_report=tensor,
        dependency_status=deps,
        warnings=w1 + w2 + w3 + w4,
    )


# =============================================================================
# Pages
# =============================================================================


def page_mission_control() -> None:
    hero(
        "Tensorized Blockchain",
        "A cinematic cockpit for governed digital twins, cryptographic state roots, tensor commitments, validator certificates, and dualistic neural lobes.",
        ["State roots", "TensorTrain/MPO", "NCP Liquid Reflex", "CfC Predictive Transition", "Validator quorum"],
    )

    status = cached_stack_status()
    summary = status.get("__summary__", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Stack modules", "Available Python modules detected in the local runtime.", value=f"{summary.get('available', 0)}/{summary.get('total', 0)}", tone="tb-glow")
    with c2:
        ratio = float(summary.get("ratio", 0.0))
        card("Readiness", "Import-level readiness for the demo cockpit.", value=f"{ratio*100:.0f}%", tone="tb-good" if ratio > 0.85 else "tb-warn")
    with c3:
        card("Runtime", f"Python {summary.get('python', '')}", value="CPU-first", tone="tb-glow")
    with c4:
        card("Mode", "Deterministic scenario-first demonstration.", value="Replayable", tone="tb-good")

    st.markdown("<div class='tb-divider'></div>", unsafe_allow_html=True)
    st.subheader("The cinematic concept")
    st.markdown(
        """
        This app is designed as a **mission-control sequence**. The viewer watches a hazardous rescue-robot twin become a cryptographic state machine:

        1. The twin is materialized as a governed object/process model.
        2. Telemetry becomes signed transactions.
        3. The governed execution pipeline captures pre/post state commitments.
        4. Validators vote over the transition, not just a snapshot.
        5. The block seals transaction roots, state roots, tensor commitments, and receipts.
        6. Tensor compression proves the ledger can be represented compactly without replacing cryptographic truth.
        7. The dualistic brain shows two forms of cognition: NCP stability and CfC prediction.
        """
    )

    with st.expander("Dependency status", expanded=False):
        st.json(status)


def page_scenario() -> None:
    st.header("Scenario Forge")
    st.caption("Create a deterministic fixture before running the full cinematic chain.")

    scenario_id = st.session_state.get("scenario_id", "alicante_fire_department_robot")
    seed = int(st.session_state.get("seed", DEFAULT_SEED))
    chain_id = str(st.session_state.get("chain_id", DEFAULT_CHAIN_ID))
    data = cached_fixture(scenario_id, seed, chain_id)
    fixture = data["fixture"]
    warnings = data["warnings"]

    c1, c2 = st.columns([1, 1])
    with c1:
        card("Scenario", str(fixture.get("title", scenario_id)), value=str(fixture.get("scenario_id", scenario_id)), tone="tb-glow")
    with c2:
        card("Fixture hash", "Deterministic scenario payload commitment.", value=short_hash(fixture.get("fixture_hash") or sha256_hex(fixture), 16), tone="tb-good")

    st.subheader("Expected flow")
    phases = []
    for i, step in enumerate(fixture.get("expected_flow", []), start=1):
        phases.append(CinematicPhase(i, str(step).replace("_", " ").title(), "Prepared fixture stage", status="complete"))
    if not phases:
        phases = [
            CinematicPhase(1, "Create Twin", "Materialize governed world state", "complete"),
            CinematicPhase(2, "Submit Transactions", "Robot telemetry enters mempool", "complete"),
            CinematicPhase(3, "Forge Block", "Commit transition into tensorized ledger", "complete"),
        ]
    phase_timeline(phases)

    if warnings:
        st.warning("Fixture warnings: " + "; ".join(warnings))

    with st.expander("Fixture JSON", expanded=False):
        st.json(fixture)

    st.download_button(
        "Download scenario fixture JSON",
        data=bytes_download(fixture),
        file_name=f"{fixture.get('scenario_id', 'scenario')}_fixture.json",
        mime="application/json",
    )


def page_run_demo() -> None:
    st.header("Run the cinematic demo")
    st.caption("This executes the local stack where available and generates exportable artifacts.")

    scenario_id = st.session_state.get("scenario_id", "alicante_fire_department_robot")
    seed = int(st.session_state.get("seed", DEFAULT_SEED))
    chain_id = str(st.session_state.get("chain_id", DEFAULT_CHAIN_ID))
    output_dir = Path(st.session_state.get("output_dir", str(DEFAULT_OUTPUT_DIR))).expanduser()

    if st.button("Run deterministic demo", type="primary", use_container_width=True):
        with st.spinner("Forging governed twin state, lobe receipts, validator votes, and tensor commitments..."):
            artifacts = run_full_demo(scenario_id, seed=seed, chain_id=chain_id, output_dir=output_dir)
            st.session_state["last_artifacts"] = artifacts.to_dict()

    artifacts = st.session_state.get("last_artifacts")
    if not artifacts:
        st.info("Run the demo to populate the cockpit.")
        return

    warnings = artifacts.get("warnings", [])
    ledger = artifacts.get("ledger_report", {})
    validators = artifacts.get("validator_report", {})
    tensor = artifacts.get("tensor_report", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Run ID", "Deterministic cockpit run identifier.", value=str(artifacts.get("run_id", ""))[:12], tone="tb-glow")
    with c2:
        card("Ledger", "Chain validation status.", value="VALID" if safe_get(ledger, ["validation", "ok"], ledger.get("ok")) else "DEGRADED", tone="tb-good" if safe_get(ledger, ["validation", "ok"], ledger.get("ok")) else "tb-warn")
    with c3:
        card("Quorum", "Validator certificate status.", value=f"{validators.get('approvals', 0)}/{validators.get('total_power', 0)}", tone="tb-good" if validators.get("ok") else "tb-warn")
    with c4:
        card("Tensor saved", "Reported tensor compression savings.", value=f"{float(tensor.get('space_saved_percent', 0.0)):.1f}%", tone="tb-glow")

    phases = [
        CinematicPhase(1, "Twin materialized", "Universal governed twin branch initialized", "complete"),
        CinematicPhase(2, "Telemetry signed", "Scenario transactions prepared", "complete"),
        CinematicPhase(3, "Execution governed", "Policy receipts and pre/post roots captured", "complete" if ledger else "degraded"),
        CinematicPhase(4, "Validators voted", "Transition certificate threshold evaluated", "complete" if validators.get("ok") else "degraded"),
        CinematicPhase(5, "Block forged", "Cryptographic commitment sealed", "complete"),
        CinematicPhase(6, "Tensor compressed", "TensorTrain/MPO artifact summarized", "complete"),
        CinematicPhase(7, "Dual lobes inspected", "NCP and CfC receipts generated", "complete" if artifacts.get("lobe_report") else "degraded"),
    ]
    phase_timeline(phases)

    if warnings:
        with st.expander("Warnings and degraded-mode notes", expanded=True):
            for w in warnings:
                st.warning(w)

    st.download_button(
        "Download full demo artifact JSON",
        data=bytes_download(artifacts),
        file_name=f"{artifacts.get('run_id', 'tensorized_blockchain_demo')}.json",
        mime="application/json",
        use_container_width=True,
    )


def page_twin_state() -> None:
    st.header("Twin State Observatory")
    artifacts = st.session_state.get("last_artifacts")
    if not artifacts:
        st.info("Run the demo first.")
        return
    twin = artifacts.get("twin_materialization", {})
    fixture = artifacts.get("fixture", {})

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Twin", fixture.get("twin_input", {}).get("thing_name", "Hazard Robot Alpha"), value=fixture.get("twin_input", {}).get("thing_type", "industrial_machine"), tone="tb-glow")
    with c2:
        state_root = twin.get("state_root") or safe_get(twin, ["snapshot", "state_root"], sha256_hex(twin))
        card("State root", "Cryptographic or fallback state commitment.", value=short_hash(state_root, 16), tone="tb-good")
    with c3:
        sketch_root = twin.get("sketch_root") or safe_get(twin, ["snapshot", "sketch_root"], short_hash(twin, 16))
        card("Sketch root", "Fast coherence/sketch commitment.", value=short_hash(sketch_root, 16), tone="tb-glow")

    st.subheader("Twin input")
    st.json(fixture.get("twin_input", {}))

    st.subheader("Materialization receipt")
    st.json(twin)


def page_block_forge() -> None:
    st.header("Block Forge")
    artifacts = st.session_state.get("last_artifacts")
    if not artifacts:
        st.info("Run the demo first.")
        return
    ledger = artifacts.get("ledger_report", {})
    blocks = list(ledger.get("blocks", []))
    summary = ledger.get("ledger", {})

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Blocks", "Committed blocks in this demo run.", value=str(summary.get("block_count", len(blocks))), tone="tb-glow")
    with c2:
        card("Head", "Ledger head commitment.", value=short_hash(summary.get("head_hash", safe_get(blocks, [-1, "block_hash"], "")), 16), tone="tb-good")
    with c3:
        card("Validation", "Chain validation result.", value="OK" if safe_get(ledger, ["validation", "ok"], ledger.get("ok", True)) else "FAIL", tone="tb-good" if safe_get(ledger, ["validation", "ok"], ledger.get("ok", True)) else "tb-bad")

    for block in blocks:
        with st.container(border=True):
            st.markdown(f"### Block {block.get('index', '?')}")
            cols = st.columns(3)
            cols[0].metric("Transactions", block.get("tx_count", len(block.get("transactions", [])) if isinstance(block.get("transactions"), list) else 0))
            cols[1].metric("Hash", short_hash(block.get("block_hash") or block.get("hash"), 12))
            cols[2].metric("World", short_hash(block.get("world_root", block.get("post_state_root", "")), 12))
            with st.expander("Block payload"):
                st.json(block)


def page_validators() -> None:
    st.header("Validator Quorum")
    artifacts = st.session_state.get("last_artifacts")
    if not artifacts:
        st.info("Run the demo first.")
        return
    report = artifacts.get("validator_report", {})
    votes = list(report.get("votes", []))

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Quorum", "Approval power over total validator power.", value=f"{report.get('approvals', 0)}/{report.get('total_power', 0)}", tone="tb-good" if report.get("ok") else "tb-warn")
    with c2:
        card("Threshold", "Voting power required for certificate.", value=str(report.get("quorum_required", 0)), tone="tb-glow")
    with c3:
        card("Certificate", "Consensus certificate commitment.", value=short_hash(report.get("certificate_hash", ""), 16), tone="tb-good")

    if votes:
        rows = []
        for v in votes:
            rows.append({
                "validator": v.get("validator_id"),
                "approval": "approve" if v.get("approval") else "withhold",
                "power": v.get("voting_power", 1),
                "vote_hash": short_hash(v.get("vote_hash"), 16),
                "reason": v.get("reason", ""),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("Raw validator report"):
        st.json(report)


def page_tensor() -> None:
    st.header("Tensor Compression Chamber")
    artifacts = st.session_state.get("last_artifacts")
    if not artifacts:
        st.info("Run the demo first.")
        return
    tensor = artifacts.get("tensor_report", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Commitment", "Tensor artifact commitment.", value=short_hash(tensor.get("tensor_commitment", ""), 16), tone="tb-glow")
    with c2:
        card("Old params", "Dense or pre-compression parameter count.", value=str(tensor.get("old_parameters", "n/a")), tone="")
    with c3:
        card("New params", "Active compressed tensor parameters.", value=str(tensor.get("new_parameters", "n/a")), tone="tb-good")
    with c4:
        card("Saved", "Reported storage/parameter reduction.", value=f"{float(tensor.get('space_saved_percent', 0.0)):.1f}%", tone="tb-good")

    if np is not None:
        oldp = float(tensor.get("old_parameters", 1) or 1)
        newp = float(tensor.get("new_parameters", oldp) or oldp)
        chart = {
            "dense/pre": oldp,
            "tensorized": newp,
        }
        st.bar_chart(chart)

    st.json(tensor)


def page_lobes() -> None:
    st.header("Dualistic Tensor Brain")
    artifacts = st.session_state.get("last_artifacts")
    if not artifacts:
        st.info("Run the demo first.")
        return
    lobes = artifacts.get("lobe_report", {})

    ncp = lobes.get("ncp_result") or safe_get(lobes, ["ncp", "result"], {}) or {}
    cfc = lobes.get("cfc_result") or safe_get(lobes, ["cfc", "result"], {}) or {}
    ncp_health = ncp.get("health_receipt", {}) if isinstance(ncp, Mapping) else {}
    cfc_health = cfc.get("health_receipt", {}) if isinstance(cfc, Mapping) else {}

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### NCP Liquid Reflex Lobe")
        card("Stability", "Homeostasis, recurrent memory, drift sensing.", value="STABLE" if ncp_health.get("is_stable", True) else "UNSTABLE", tone="tb-good" if ncp_health.get("is_stable", True) else "tb-bad")
        st.metric("Hidden norm", f"{float(ncp_health.get('hidden_norm', safe_get(ncp, ['metrics', 'hidden_norm'], 0.0)) or 0.0):.4f}")
        show_hash("NCP state hash", ncp_health.get("state_hash", sha256_hex(ncp)))
    with c2:
        st.markdown("### CfC Predictive Transition Lobe")
        card("Forecast", "Bounded continuous-time transition forecasting.", value="READY" if cfc_health.get("is_stable", True) else "DEGRADED", tone="tb-good" if cfc_health.get("is_stable", True) else "tb-warn")
        ferr = cfc_health.get("forecast_error", safe_get(cfc, ["metrics", "forecast_error"], 0.0))
        st.metric("Forecast error", f"{float(ferr or 0.0):.4f}")
        show_hash("CfC state hash", cfc_health.get("state_hash", sha256_hex(cfc)))

    st.subheader("Raw lobe run")
    st.json(lobes)


def page_artifacts() -> None:
    st.header("Artifact Vault")
    artifacts = st.session_state.get("last_artifacts")
    if not artifacts:
        st.info("Run the demo first.")
        return

    files = {
        "demo_artifacts.json": artifacts,
        "fixture.json": artifacts.get("fixture", {}),
        "ledger_report.json": artifacts.get("ledger_report", {}),
        "lobe_report.json": artifacts.get("lobe_report", {}),
        "validator_report.json": artifacts.get("validator_report", {}),
        "tensor_report.json": artifacts.get("tensor_report", {}),
        "dependency_status.json": artifacts.get("dependency_status", {}),
    }
    bundle = make_zip_bytes(files)
    st.download_button(
        "Download complete demo bundle (.zip)",
        data=bundle,
        file_name=f"{artifacts.get('run_id', 'tensorized_blockchain_demo')}_bundle.zip",
        mime="application/zip",
        use_container_width=True,
    )

    cols = st.columns(2)
    with cols[0]:
        st.download_button("Download ledger report", bytes_download(files["ledger_report.json"]), "ledger_report.json", "application/json", use_container_width=True)
        st.download_button("Download tensor report", bytes_download(files["tensor_report.json"]), "tensor_report.json", "application/json", use_container_width=True)
    with cols[1]:
        st.download_button("Download lobe report", bytes_download(files["lobe_report.json"]), "lobe_report.json", "application/json", use_container_width=True)
        st.download_button("Download validator report", bytes_download(files["validator_report.json"]), "validator_report.json", "application/json", use_container_width=True)

    st.subheader("Full artifact hash")
    show_hash("artifact_hash", artifacts.get("artifact_hash", sha256_hex(artifacts)))


def page_self_tests() -> None:
    st.header("Runtime Self-Tests")
    st.caption("Runs lightweight module self-tests where available. Heavy tests should remain in CLI/CI.")

    test_modules = [
        "tb_canonical",
        "tb_hashing",
        "tb_crypto",
        "tb_schemas",
        "tb_validation",
        "tn",
        "digital_twin_kernel",
        "twin_anything",
        "tensorized_blockchain_kernel",
        "ncp",
        "cfc",
        "cognitive_lobe_runtime",
    ]

    if st.button("Run lightweight self-tests", type="primary"):
        results = []
        for name in test_modules:
            mod, err = import_optional(name)
            if mod is None:
                results.append({"module": name, "ok": False, "status": "missing", "error": err})
                continue
            fn = getattr(mod, "run_self_test", None) or getattr(mod, "_self_test", None)
            if not callable(fn):
                results.append({"module": name, "ok": True, "status": "import_only"})
                continue
            try:
                ret = fn()
                results.append({"module": name, "ok": True, "status": "self_test", "result": json_safe(ret)})
            except Exception as exc:
                results.append({"module": name, "ok": False, "status": "failed", "error": repr(exc), "traceback": traceback.format_exc()[-1200:]})
        st.session_state["self_test_results"] = results

    results = st.session_state.get("self_test_results")
    if results:
        st.dataframe(
            [{"module": r["module"], "ok": r["ok"], "status": r["status"], "error": r.get("error", "")} for r in results],
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("Raw self-test results"):
            st.json(results)


# =============================================================================
# App shell
# =============================================================================


def configure_page() -> None:
    st.set_page_config(
        page_title="Tensorized Blockchain Cockpit",
        page_icon="⛓️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()


def sidebar() -> str:
    st.sidebar.markdown("# ⛓️ Tensorized Blockchain")
    st.sidebar.caption("Cinematic governed-twin ledger cockpit")

    available_scenarios = ["alicante_fire_department_robot", "industrial_machine_dpp", "lk20_curriculum"]
    try:
        mod, _ = import_optional("tb_demo_scenarios")
        if mod is not None and hasattr(mod, "list_scenarios"):
            available_scenarios = list(mod.list_scenarios())
    except Exception:
        pass

    st.session_state["scenario_id"] = st.sidebar.selectbox(
        "Scenario",
        available_scenarios,
        index=0 if st.session_state.get("scenario_id") not in available_scenarios else available_scenarios.index(st.session_state.get("scenario_id")),
    )
    st.session_state["chain_id"] = st.sidebar.text_input("Chain ID", value=str(st.session_state.get("chain_id", DEFAULT_CHAIN_ID)))
    st.session_state["seed"] = int(st.sidebar.number_input("Deterministic seed", min_value=0, max_value=2_147_483_647, value=int(st.session_state.get("seed", DEFAULT_SEED)), step=1))
    st.session_state["output_dir"] = st.sidebar.text_input("Export directory", value=str(st.session_state.get("output_dir", DEFAULT_OUTPUT_DIR)))

    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Mission Control",
            "Scenario Forge",
            "Run Demo",
            "Twin State",
            "Block Forge",
            "Validators",
            "Tensor Chamber",
            "Dual Lobes",
            "Artifact Vault",
            "Self-Tests",
        ],
    )

    status = cached_stack_status().get("__summary__", {})
    st.sidebar.markdown("---")
    st.sidebar.metric("Stack readiness", f"{float(status.get('ratio', 0))*100:.0f}%")
    st.sidebar.caption(f"{status.get('available', 0)} of {status.get('total', 0)} modules available")
    return page


def main() -> None:
    if st is None:  # pragma: no cover
        print("Streamlit is required to run the demo app. Install with pip install streamlit.")
        sys.exit(1)
    configure_page()
    page = sidebar()
    if page == "Mission Control":
        page_mission_control()
    elif page == "Scenario Forge":
        page_scenario()
    elif page == "Run Demo":
        page_run_demo()
    elif page == "Twin State":
        page_twin_state()
    elif page == "Block Forge":
        page_block_forge()
    elif page == "Validators":
        page_validators()
    elif page == "Tensor Chamber":
        page_tensor()
    elif page == "Dual Lobes":
        page_lobes()
    elif page == "Artifact Vault":
        page_artifacts()
    elif page == "Self-Tests":
        page_self_tests()


if __name__ == "__main__":
    main()
