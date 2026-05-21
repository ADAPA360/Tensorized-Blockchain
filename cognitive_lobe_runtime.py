#!/usr/bin/env python3
# cognitive_lobe_runtime.py
r"""
Project Chimera / Akkurat - Dualistic Tensor Cognitive Lobe Runtime
===================================================================

Production coordinator over the low-level neural substrates:

    ncp.py  -> NCP Liquid Reflex Lobe
    cfc.py  -> CfC Predictive Transition Lobe

Design boundary
---------------
This module coordinates neural lobe execution, feature encoding, lobe receipts,
snapshots, and checkpoints. It intentionally does not append ledger blocks, make
consensus decisions, mutate digital-twin policy directly, or perform tool routing.
Those responsibilities belong to digital_twin_kernel.py, tb_execution.py,
tb_consensus.py, and tensorized_blockchain_kernel.py.

Core responsibilities
---------------------
- Instantiate NCP Liquid Reflex Lobe.
- Instantiate CfC Predictive Transition Lobe.
- Encode twin/transaction/state features into deterministic numeric vectors.
- Run NCP stability/homeostasis pass.
- Run CfC transition forecast and optional actual-step pass.
- Generate schema-compatible lobe health and transition receipts.
- Expose snapshots, manifests, checkpoint export, and checkpoint loading.

Compatibility
-------------
The underlying ncp.py and cfc.py APIs remain authoritative. This coordinator
uses their production methods when present and falls back to conservative local
receipts when running with older compatible substrates.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


# =============================================================================
# Version / path bootstrap
# =============================================================================

COGNITIVE_LOBE_RUNTIME_VERSION = "cognitive-lobe-runtime-v1.0-production"


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


def configure_paths(*, project_root: Optional[Union[str, os.PathLike]] = None) -> Dict[str, Optional[str]]:
    import tempfile
    here = _module_dir()
    candidates = [
        project_root,
        os.environ.get("TENSORIZED_BLOCKCHAIN_ROOT"),
        os.environ.get("TB_ROOT"),
        os.environ.get("AKKURAT_ROOT"),
        os.environ.get("LK20_ROOT"),
        here,
        here.parent,
        Path.cwd(),
        Path(tempfile.gettempdir()),
    ]
    found: Dict[str, Optional[str]] = {
        "module_dir": str(here),
        "project_root": None if project_root is None else str(Path(project_root).expanduser()),
        "ncp_root": None,
        "cfc_root": None,
        "tb_schemas_root": None,
        "tb_hashing_root": None,
        "tb_canonical_root": None,
        "tb_config_root": None,
    }
    for candidate in candidates:
        p = _add_path(candidate, prepend=True)
        if p is None:
            continue
        if found["ncp_root"] is None and (p / "ncp.py").exists():
            found["ncp_root"] = str(p)
        if found["cfc_root"] is None and (p / "cfc.py").exists():
            found["cfc_root"] = str(p)
        if found["tb_schemas_root"] is None and (p / "tb_schemas.py").exists():
            found["tb_schemas_root"] = str(p)
        if found["tb_hashing_root"] is None and (p / "tb_hashing.py").exists():
            found["tb_hashing_root"] = str(p)
        if found["tb_canonical_root"] is None and (p / "tb_canonical.py").exists():
            found["tb_canonical_root"] = str(p)
        if found["tb_config_root"] is None and (p / "tb_config.py").exists():
            found["tb_config_root"] = str(p)
    _add_path(here, prepend=True)
    return found


_PATHS = configure_paths()


# =============================================================================
# Optional shared-stack imports
# =============================================================================

_IMPORT_ERRORS: Dict[str, str] = {}

try:
    import tb_canonical as tbc  # type: ignore
except Exception as exc:  # pragma: no cover
    tbc = None  # type: ignore
    _IMPORT_ERRORS["tb_canonical"] = repr(exc)

try:
    import tb_hashing as tbh  # type: ignore
except Exception as exc:  # pragma: no cover
    tbh = None  # type: ignore
    _IMPORT_ERRORS["tb_hashing"] = repr(exc)

try:
    import tb_schemas as tbs  # type: ignore
except Exception as exc:  # pragma: no cover
    tbs = None  # type: ignore
    _IMPORT_ERRORS["tb_schemas"] = repr(exc)

try:
    import tb_config as tbcfg  # type: ignore
except Exception as exc:  # pragma: no cover
    tbcfg = None  # type: ignore
    _IMPORT_ERRORS["tb_config"] = repr(exc)

try:
    import ncp  # type: ignore
except Exception as exc:  # pragma: no cover
    ncp = None  # type: ignore
    _IMPORT_ERRORS["ncp"] = repr(exc)

try:
    import cfc  # type: ignore
except Exception as exc:  # pragma: no cover
    cfc = None  # type: ignore
    _IMPORT_ERRORS["cfc"] = repr(exc)


# =============================================================================
# Helpers
# =============================================================================

_EPS = 1e-12


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_safe(obj: Any) -> Any:
    if tbc is not None and hasattr(tbc, "canonical_json_safe"):
        try:
            return tbc.canonical_json_safe(obj)  # type: ignore[attr-defined]
        except Exception:
            pass
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
    if hasattr(obj, "payload") and callable(obj.payload):
        try:
            return _json_safe(obj.payload())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return _json_safe(vars(obj))
        except Exception:
            pass
    return str(obj)


def _canonical_dumps(obj: Any) -> str:
    if tbc is not None and hasattr(tbc, "canonical_dumps"):
        try:
            return tbc.canonical_dumps(obj)  # type: ignore[attr-defined]
        except Exception:
            pass
    return json.dumps(_json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_bytes(obj: Any) -> bytes:
    return _canonical_dumps(obj).encode("utf-8", errors="ignore")


def _sha256_hex_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_object(obj: Any, *, domain: str = "TB:COGNITIVE_LOBE_RUNTIME:OBJECT:v1") -> str:
    if tbh is not None and hasattr(tbh, "domain_hash_hex"):
        try:
            return str(tbh.domain_hash_hex(domain, obj))  # type: ignore[attr-defined]
        except Exception:
            pass
    if tbs is not None and hasattr(tbs, "domain_hash"):
        try:
            return str(tbs.domain_hash(domain, obj))  # type: ignore[attr-defined]
        except Exception:
            pass
    return _sha256_hex_bytes(domain.encode("utf-8") + b"\x00" + _canonical_bytes(obj))


def _array_hash(arr: Any, *, dtype: np.dtype = np.float32, tag: str = "array") -> str:
    x = np.asarray(arr, dtype=np.dtype(dtype)).reshape(-1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.dtype(dtype), copy=False)
    payload = {
        "tag": str(tag),
        "dtype": str(np.dtype(dtype)),
        "shape": list(np.asarray(arr).shape),
        "flat_size": int(x.size),
        "bytes_sha256": _sha256_hex_bytes(x.tobytes(order="C")),
    }
    return _hash_object(payload, domain="TB:COGNITIVE_LOBE_RUNTIME:ARRAY:v1")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(float(lo), min(float(hi), float(x))))


def _coerce_vector(x: Any, *, dim: int, dtype: np.dtype = np.float32, clip: Optional[float] = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.dtype(dtype)).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.dtype(dtype), copy=False)
    if arr.size != int(dim):
        out = np.zeros((int(dim),), dtype=np.dtype(dtype))
        n = min(arr.size, int(dim))
        if n:
            out[:n] = arr[:n]
        arr = out
    if clip is not None and float(clip) > 0:
        arr = np.clip(arr, -float(clip), float(clip)).astype(np.dtype(dtype), copy=False)
    return arr


def _read_hash_file(path: Union[str, os.PathLike]) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Union[str, os.PathLike], payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(payload), sort_keys=True, indent=2, ensure_ascii=False))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def _receipt_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(_json_safe(obj))
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return dict(_json_safe(obj.to_dict()))
    if hasattr(obj, "payload") and callable(obj.payload):
        out = dict(_json_safe(obj.payload()))
        if hasattr(obj, "compute_hash"):
            try:
                out.setdefault("receipt_hash", obj.compute_hash())
            except Exception:
                pass
        return out
    return dict(_json_safe(obj)) if isinstance(_json_safe(obj), Mapping) else {"value": _json_safe(obj)}


def _health_score_from_metrics(metrics: Mapping[str, Any]) -> float:
    score = 1.0
    if bool(metrics.get("has_nan", False)):
        score -= 0.45
    if bool(metrics.get("is_saturated", False)):
        score -= 0.25
    if not bool(metrics.get("is_stable", True)):
        score -= 0.30
    sat = _safe_float(metrics.get("hidden_saturation_fraction", metrics.get("state_saturation_fraction", 0.0)), 0.0)
    score -= 0.25 * _clip(sat, 0.0, 1.0)
    return _clip(score, 0.0, 1.0)


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class FeatureEncoderConfig:
    """Deterministic feature encoder configuration."""

    feature_dim: int = 128
    dtype: str = "float32"
    hash_buckets: int = 96
    numeric_slots: int = 32
    normalize: bool = True
    clip: float = 8.0
    include_payload_length_features: bool = True
    schema_version: str = "tb.cognitive_lobe.feature_encoder.v1"

    def normalized(self) -> "FeatureEncoderConfig":
        cfg = FeatureEncoderConfig(**_json_safe(asdict(self)))
        cfg.feature_dim = int(max(8, cfg.feature_dim))
        cfg.hash_buckets = int(max(1, min(cfg.hash_buckets, cfg.feature_dim)))
        cfg.numeric_slots = int(max(0, min(cfg.numeric_slots, cfg.feature_dim)))
        cfg.dtype = str(cfg.dtype or "float32").lower().strip()
        if cfg.dtype not in {"float32", "float64"}:
            cfg.dtype = "float32"
        cfg.clip = float(max(0.0, cfg.clip))
        cfg.normalize = bool(cfg.normalize)
        cfg.include_payload_length_features = bool(cfg.include_payload_length_features)
        return cfg


@dataclass
class CognitiveLobeRuntimeConfig:
    """Configuration for the dual-lobe runtime."""

    runtime_id: str = "dualistic_tensor_brain"
    deterministic: bool = True
    consensus_safe: bool = False
    rng_seed: int = 2027
    dtype: str = "float32"
    device: str = "cpu"

    feature_encoder: FeatureEncoderConfig = field(default_factory=FeatureEncoderConfig)

    enable_ncp: bool = True
    ncp_policy_id: str = "ncp_liquid_reflex_lobe"
    ncp_hidden_size: int = 64
    ncp_output_size: int = 32
    ncp_bond_dim: int = 4
    ncp_leaky_lambda: float = 0.25

    enable_cfc: bool = True
    cfc_policy_id: str = "cfc_predictive_transition_lobe"
    cfc_hidden_size: int = 32
    cfc_num_cells: int = 4
    cfc_bond_dim: int = 4
    cfc_time_scale: float = 1.0
    cfc_leaky_lambda: float = 1.0

    force_dense: bool = False
    tt_strict: bool = False
    max_parameter_budget_per_lobe: int = 2_500_000_000
    update_frequency_hz: float = 1.0
    checkpoint_policy: str = "hash_on_checkpoint"
    health_receipt_include_expensive: bool = False
    schema_version: str = "tb.cognitive_lobe.runtime_config.v1"

    def normalized(self) -> "CognitiveLobeRuntimeConfig":
        cfg = CognitiveLobeRuntimeConfig(**_json_safe(asdict(self)))
        cfg.runtime_id = str(cfg.runtime_id or "dualistic_tensor_brain")
        cfg.deterministic = bool(cfg.deterministic)
        cfg.consensus_safe = bool(cfg.consensus_safe)
        cfg.rng_seed = int(cfg.rng_seed)
        cfg.dtype = str(cfg.dtype or "float32").lower().strip()
        if cfg.dtype not in {"float32", "float64"}:
            cfg.dtype = "float32"
        cfg.device = str(cfg.device or "cpu").lower().strip()
        if cfg.consensus_safe:
            cfg.deterministic = True
            cfg.device = "cpu"
            cfg.force_dense = False
        cfg.feature_encoder = FeatureEncoderConfig(**dict(cfg.feature_encoder)).normalized() if isinstance(cfg.feature_encoder, Mapping) else cfg.feature_encoder.normalized()
        cfg.ncp_hidden_size = int(max(1, cfg.ncp_hidden_size))
        cfg.ncp_output_size = int(max(1, cfg.ncp_output_size))
        cfg.ncp_bond_dim = int(max(1, cfg.ncp_bond_dim))
        cfg.ncp_leaky_lambda = float(_clip(cfg.ncp_leaky_lambda, 0.0, 1.0))
        cfg.cfc_hidden_size = int(max(1, cfg.cfc_hidden_size))
        cfg.cfc_num_cells = int(max(1, cfg.cfc_num_cells))
        cfg.cfc_bond_dim = int(max(1, cfg.cfc_bond_dim))
        cfg.cfc_time_scale = float(max(_EPS, cfg.cfc_time_scale))
        cfg.cfc_leaky_lambda = float(_clip(cfg.cfc_leaky_lambda, 0.0, 1.0))
        cfg.max_parameter_budget_per_lobe = int(max(0, cfg.max_parameter_budget_per_lobe))
        cfg.update_frequency_hz = float(max(0.0, cfg.update_frequency_hz))
        return cfg

    def config_hash(self) -> str:
        return _hash_object(asdict(self.normalized()), domain="TB:COGNITIVE_LOBE_RUNTIME_CONFIG:v1")


@dataclass
class EncodedFeatureVector:
    vector: List[float]
    vector_hash: str
    source_hash: str
    feature_dim: int
    dtype: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "tb.cognitive_lobe.encoded_features.v1"

    def to_array(self) -> np.ndarray:
        return np.asarray(self.vector, dtype=np.float32).reshape(-1)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class LobePassResult:
    lobe_id: str
    lobe_kind: str
    output_hash: str
    state_hash: str
    health_receipt: Dict[str, Any]
    snapshot: Dict[str, Any]
    metrics: Dict[str, Any]
    output_vector: List[float] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "tb.cognitive_lobe.pass_result.v1"

    def result_hash(self) -> str:
        return _hash_object(asdict(self), domain="TB:COGNITIVE_LOBE_PASS_RESULT:v1")

    def to_dict(self) -> Dict[str, Any]:
        out = _json_safe(asdict(self))
        out["result_hash"] = self.result_hash()
        return out


@dataclass
class DualLobeRunResult:
    runtime_id: str
    encoded_features: Dict[str, Any]
    ncp_result: Optional[Dict[str, Any]] = None
    cfc_result: Optional[Dict[str, Any]] = None
    cfc_transition_receipt: Optional[Dict[str, Any]] = None
    run_hash: str = ""
    created_at: str = field(default_factory=_now_iso)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "tb.cognitive_lobe.run_result.v1"

    def payload(self) -> Dict[str, Any]:
        d = _json_safe(asdict(self))
        d.pop("run_hash", None)
        return d

    def compute_hash(self) -> str:
        return _hash_object(self.payload(), domain="TB:COGNITIVE_LOBE_RUN_RESULT:v1")

    def to_dict(self) -> Dict[str, Any]:
        d = self.payload()
        d["run_hash"] = self.run_hash or self.compute_hash()
        return d


@dataclass
class DualLobeCheckpointManifest:
    runtime_id: str
    created_at: str
    runtime_config_hash: str
    checkpoint_root: str
    files: List[Dict[str, Any]] = field(default_factory=list)
    snapshots: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "tb.cognitive_lobe.checkpoint_manifest.v1"

    def manifest_hash(self) -> str:
        return _hash_object(asdict(self), domain="TB:COGNITIVE_LOBE_CHECKPOINT_MANIFEST:v1")

    def to_dict(self) -> Dict[str, Any]:
        out = _json_safe(asdict(self))
        out["manifest_hash"] = self.manifest_hash()
        return out


# =============================================================================
# Feature encoder
# =============================================================================

class CognitiveFeatureEncoder:
    """Deterministic encoder from heterogeneous twin/transaction context to numeric lobe vector."""

    def __init__(self, cfg: Optional[FeatureEncoderConfig] = None):
        self.cfg = (cfg or FeatureEncoderConfig()).normalized()
        self.dtype = np.dtype(self.cfg.dtype)

    def _iter_numeric(self, obj: Any, *, prefix: str = "") -> Iterable[Tuple[str, float]]:
        if obj is None:
            return
        if isinstance(obj, bool):
            yield prefix or "bool", 1.0 if obj else 0.0
            return
        if isinstance(obj, (int, float, np.number)):
            v = _safe_float(obj, 0.0)
            yield prefix or "number", v
            return
        if isinstance(obj, np.ndarray):
            arr = np.asarray(obj, dtype=np.float64).reshape(-1)
            for i, v in enumerate(arr[:4096]):
                yield f"{prefix}[{i}]", _safe_float(v, 0.0)
            return
        if is_dataclass(obj):
            yield from self._iter_numeric(asdict(obj), prefix=prefix)
            return
        if isinstance(obj, Mapping):
            for k in sorted(obj.keys(), key=lambda z: str(z)):
                key = f"{prefix}.{k}" if prefix else str(k)
                yield from self._iter_numeric(obj[k], prefix=key)
            return
        if isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj[:4096]):
                key = f"{prefix}[{i}]" if prefix else f"[{i}]"
                yield from self._iter_numeric(item, prefix=key)
            return

    def encode(self, payload: Any, *, metadata: Optional[Mapping[str, Any]] = None) -> EncodedFeatureVector:
        cfg = self.cfg
        D = int(cfg.feature_dim)
        vec = np.zeros((D,), dtype=self.dtype)
        safe_payload = _json_safe(payload)
        source_hash = _hash_object(safe_payload, domain="TB:COGNITIVE_LOBE_FEATURE_SOURCE:v1")
        canonical = _canonical_bytes(safe_payload)

        # Byte-distribution features hashed into deterministic buckets.
        bucket_count = min(int(cfg.hash_buckets), D)
        if canonical:
            for i, b in enumerate(canonical[:65536]):
                h = hashlib.sha256(f"byte:{i}:{b}:{source_hash}".encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "little") % bucket_count
                sign = -1.0 if (h[4] & 1) else 1.0
                vec[idx] += self.dtype.type(sign * (1.0 + (b / 255.0)))

        # Numeric slots preserve direct magnitudes in a deterministic tail region.
        numeric_values = list(self._iter_numeric(safe_payload))[: max(0, int(cfg.numeric_slots))]
        if cfg.numeric_slots > 0 and numeric_values:
            start = max(0, D - int(cfg.numeric_slots))
            for j, (key, value) in enumerate(numeric_values[: int(cfg.numeric_slots)]):
                idx = start + j
                if idx >= D:
                    break
                # Signed log compression keeps large telemetry bounded but useful.
                v = _safe_float(value, 0.0)
                vec[idx] = self.dtype.type(math.copysign(math.log1p(abs(v)), v))
                # Key identity still influences the hash bucket region.
                kh = hashlib.sha256(str(key).encode("utf-8", errors="ignore")).digest()
                bi = int.from_bytes(kh[:4], "little") % bucket_count
                vec[bi] += self.dtype.type(0.25 if (kh[4] & 1) == 0 else -0.25)

        if cfg.include_payload_length_features and D >= 4:
            stats = np.asarray(
                [
                    len(canonical),
                    canonical.count(b"{"),
                    canonical.count(b"["),
                    len(numeric_values),
                ],
                dtype=self.dtype,
            )
            stats = np.sign(stats) * np.log1p(np.abs(stats))
            vec[: min(4, D)] += stats[: min(4, D)]

        if cfg.clip > 0:
            vec = np.clip(vec, -float(cfg.clip), float(cfg.clip)).astype(self.dtype, copy=False)
        if cfg.normalize:
            nrm = float(np.linalg.norm(vec.astype(np.float64)))
            if math.isfinite(nrm) and nrm > _EPS:
                vec = (vec / nrm).astype(self.dtype, copy=False)

        return EncodedFeatureVector(
            vector=vec.astype(float).tolist(),
            vector_hash=_array_hash(vec, dtype=self.dtype, tag="encoded_feature_vector"),
            source_hash=source_hash,
            feature_dim=D,
            dtype=str(self.dtype),
            metadata=dict(metadata or {}),
        )


# =============================================================================
# Runtime
# =============================================================================

class CognitiveLobeRuntime:
    """Dual-lobe runtime coordinating NCP and CfC neural substrates."""

    def __init__(
        self,
        config: Optional[CognitiveLobeRuntimeConfig] = None,
        *,
        ncp_lobe: Optional[Any] = None,
        cfc_lobe: Optional[Any] = None,
    ):
        self.config = (config or CognitiveLobeRuntimeConfig()).normalized()
        self.encoder = CognitiveFeatureEncoder(self.config.feature_encoder)
        self.dtype = np.dtype(self.config.dtype)
        self.runtime_id = self.config.runtime_id
        self.warnings: List[str] = []

        self.ncp_lobe = ncp_lobe if ncp_lobe is not None else (self._build_ncp_lobe() if self.config.enable_ncp else None)
        self.cfc_lobe = cfc_lobe if cfc_lobe is not None else (self._build_cfc_lobe() if self.config.enable_cfc else None)
        self._last_result: Optional[DualLobeRunResult] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_ncp_lobe(self) -> Any:
        if ncp is None or not hasattr(ncp, "TensorizedNCP"):
            raise RuntimeError(f"ncp.py is unavailable or missing TensorizedNCP: {_IMPORT_ERRORS.get('ncp', '')}")
        seed = int(self.config.rng_seed) + 101
        return ncp.TensorizedNCP(  # type: ignore[attr-defined]
            policy_id=self.config.ncp_policy_id,
            num_inputs=self.config.feature_encoder.feature_dim,
            num_hidden=self.config.ncp_hidden_size,
            num_outputs=self.config.ncp_output_size,
            bond_dim=self.config.ncp_bond_dim,
            rng_seed=seed,
            deterministic=bool(self.config.deterministic),
            consensus_safe=bool(self.config.consensus_safe),
            verbose=False,
            output_activation="tanh",
            leaky_lambda=self.config.ncp_leaky_lambda,
            tt_strict=bool(self.config.tt_strict),
            force_dense=bool(self.config.force_dense),
            check_finite=True,
            tt_device="cpu" if self.config.consensus_safe else self.config.device,
        )

    def _build_cfc_lobe(self) -> Any:
        if cfc is None or not hasattr(cfc, "TCfC_Policy"):
            raise RuntimeError(f"cfc.py is unavailable or missing TCfC_Policy: {_IMPORT_ERRORS.get('cfc', '')}")
        seed = int(self.config.rng_seed) + 202
        return cfc.TCfC_Policy(  # type: ignore[attr-defined]
            policy_id=self.config.cfc_policy_id,
            input_size=self.config.feature_encoder.feature_dim,
            hidden_size=self.config.cfc_hidden_size,
            num_cells=self.config.cfc_num_cells,
            bond_dim=self.config.cfc_bond_dim,
            rng_seed=seed,
            deterministic=bool(self.config.deterministic),
            consensus_safe=bool(self.config.consensus_safe),
            dtype=np.dtype(self.config.dtype),
            verbose=False,
            tt_strict=bool(self.config.tt_strict),
            force_dense=bool(self.config.force_dense),
            time_scale=self.config.cfc_time_scale,
            leaky_lambda=self.config.cfc_leaky_lambda,
        )

    # ------------------------------------------------------------------
    # Feature encoding
    # ------------------------------------------------------------------
    def encode_features(
        self,
        *,
        twin_snapshot: Optional[Any] = None,
        transaction: Optional[Any] = None,
        pre_state_commitment: Optional[Any] = None,
        post_state_commitment: Optional[Any] = None,
        policy_receipt: Optional[Any] = None,
        extra: Optional[Mapping[str, Any]] = None,
        raw_vector: Optional[Any] = None,
    ) -> EncodedFeatureVector:
        if raw_vector is not None:
            vec = _coerce_vector(raw_vector, dim=self.config.feature_encoder.feature_dim, dtype=self.dtype, clip=self.config.feature_encoder.clip)
            return EncodedFeatureVector(
                vector=vec.astype(float).tolist(),
                vector_hash=_array_hash(vec, dtype=self.dtype, tag="raw_feature_vector"),
                source_hash=_hash_object({"raw_vector_hash": _array_hash(vec, dtype=self.dtype, tag="raw_source")}, domain="TB:COGNITIVE_LOBE_FEATURE_SOURCE:v1"),
                feature_dim=int(vec.size),
                dtype=str(self.dtype),
                metadata={"source": "raw_vector"},
            )

        payload = {
            "runtime_id": self.runtime_id,
            "twin_snapshot": _json_safe(twin_snapshot),
            "transaction": _json_safe(transaction),
            "pre_state_commitment": _json_safe(pre_state_commitment),
            "post_state_commitment": _json_safe(post_state_commitment),
            "policy_receipt": _json_safe(policy_receipt),
            "extra": _json_safe(dict(extra or {})),
        }
        return self.encoder.encode(payload, metadata={"source": "structured_context"})

    # ------------------------------------------------------------------
    # Receipt conversion helpers
    # ------------------------------------------------------------------
    def _schema_lobe_health_receipt(self, *, lobe: Any, lobe_kind: str, native: Optional[Any] = None) -> Dict[str, Any]:
        native_dict = _receipt_to_dict(native)
        metrics = {}
        try:
            if hasattr(lobe, "health_metrics"):
                metrics = dict(_json_safe(lobe.health_metrics(include_expensive=bool(self.config.health_receipt_include_expensive))))
        except TypeError:
            try:
                metrics = dict(_json_safe(lobe.health_metrics()))
            except Exception:
                metrics = {}
        except Exception:
            metrics = {}

        lobe_id = str(getattr(lobe, "lobe_id", getattr(lobe, "policy_id", lobe_kind)))
        state_hash = ""
        if hasattr(lobe, "state_hash"):
            try:
                state_hash = str(lobe.state_hash())
            except Exception:
                state_hash = ""
        if not state_hash and hasattr(lobe, "get_state"):
            try:
                state_hash = _array_hash(lobe.get_state(), dtype=self.dtype, tag=f"{lobe_kind}:state")
            except Exception:
                pass

        checkpoint_hash = str(native_dict.get("checkpoint_hash", ""))
        scale_profile_hash = str(
            native_dict.get("scale_profile_hash")
            or native_dict.get("scale_hash")
            or self.scale_profile_hash(lobe_kind)
        )
        parameter_count = _safe_int(native_dict.get("parameter_count", 0), 0)
        if parameter_count <= 0 and hasattr(lobe, "parameter_count"):
            try:
                parameter_count = int(lobe.parameter_count())
            except Exception:
                parameter_count = 0
        dense_count = _safe_int(native_dict.get("dense_equivalent_parameter_count", 0), 0)
        if dense_count <= 0 and hasattr(lobe, "equivalent_dense_parameter_count"):
            try:
                dense_count = int(lobe.equivalent_dense_parameter_count())
            except Exception:
                dense_count = 0

        stable = bool(metrics.get("is_stable", native_dict.get("is_stable", native_dict.get("stable", True))))
        degraded = bool(metrics.get("degraded_mode", native_dict.get("degraded", False)))
        health_score = _health_score_from_metrics(metrics)
        recurrent_mode = str(metrics.get("recurrent_mode", native_dict.get("recurrent_mode", "")))

        if tbs is not None and hasattr(tbs, "LobeHealthReceipt"):
            try:
                receipt = tbs.LobeHealthReceipt(  # type: ignore[attr-defined]
                    lobe_id=lobe_id,
                    lobe_kind=lobe_kind,
                    stable=stable,
                    degraded=degraded,
                    health_score=health_score,
                    state_hash=state_hash,
                    checkpoint_hash=checkpoint_hash,
                    scale_profile_hash=scale_profile_hash,
                    parameter_count=parameter_count,
                    dense_equivalent_parameter_count=dense_count,
                    recurrent_mode=recurrent_mode,
                    metrics=metrics,
                    warnings=[] if stable else ["lobe_unstable"],
                    metadata={"native_receipt": native_dict},
                )
                return receipt.to_dict()
            except Exception:
                pass

        payload = {
            "lobe_id": lobe_id,
            "lobe_kind": lobe_kind,
            "stable": stable,
            "degraded": degraded,
            "health_score": health_score,
            "state_hash": state_hash,
            "checkpoint_hash": checkpoint_hash,
            "scale_profile_hash": scale_profile_hash,
            "parameter_count": parameter_count,
            "dense_equivalent_parameter_count": dense_count,
            "recurrent_mode": recurrent_mode,
            "metrics": metrics,
            "metadata": {"native_receipt": native_dict},
            "created_at": _now_iso(),
            "schema_version": "tb.lobe_health_receipt.v1",
        }
        payload["receipt_hash"] = _hash_object(payload, domain="TB:LOBE_HEALTH_RECEIPT:v1")
        return payload

    def _schema_cfc_transition_receipt(
        self,
        *,
        native: Optional[Any],
        pre_state_root: str,
        tx_hash: str,
        predicted_state: Any,
        actual_state: Optional[Any],
        control_vector: Any,
        health_receipt: Mapping[str, Any],
        time_delta: float,
        forecast_error: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        native_dict = _receipt_to_dict(native)
        lobe_id = str(getattr(self.cfc_lobe, "lobe_id", self.config.cfc_policy_id))
        predicted_hash = _array_hash(predicted_state, dtype=self.dtype, tag="cfc_predicted_post_state")
        actual_hash = "" if actual_state is None else _array_hash(actual_state, dtype=self.dtype, tag="cfc_actual_post_state")
        control_hash = _array_hash(control_vector, dtype=self.dtype, tag="cfc_control_vector")
        lobe_health_hash = str(health_receipt.get("receipt_hash") or health_receipt.get("health_hash") or _hash_object(health_receipt, domain="TB:LOBE_HEALTH_RECEIPT:v1"))
        scale_hash = str(health_receipt.get("scale_profile_hash") or self.scale_profile_hash("cfc"))
        backbone_hashes: List[str] = []
        if self.cfc_lobe is not None and hasattr(self.cfc_lobe, "tensor_backbone_commitments"):
            try:
                backbone_hashes = [str(x) for x in self.cfc_lobe.tensor_backbone_commitments()]
            except Exception:
                backbone_hashes = []
        if not backbone_hashes and native_dict.get("tensor_backbone_root"):
            backbone_hashes = [str(native_dict.get("tensor_backbone_root"))]

        err_val: Optional[float] = None
        cos_val: Optional[float] = None
        accepted: Optional[bool] = None
        if forecast_error:
            err_val = _safe_float(forecast_error.get("relative_error", forecast_error.get("l2_error", forecast_error.get("error", None))), None)  # type: ignore[arg-type]
            cos_raw = forecast_error.get("cosine_similarity", forecast_error.get("cosine", None))
            if cos_raw is not None:
                cos_val = _safe_float(cos_raw, None)  # type: ignore[arg-type]
            if "ok" in forecast_error:
                accepted = bool(forecast_error.get("ok"))
        elif isinstance(native_dict.get("forecast_error"), Mapping):
            fe = native_dict.get("forecast_error")
            err_val = _safe_float(fe.get("relative_error", fe.get("l2_error", 0.0)), 0.0)
            cos_val = _safe_float(fe.get("cosine_similarity", 0.0), 0.0) if "cosine_similarity" in fe else None
            accepted = bool(fe.get("ok")) if "ok" in fe else None

        if tbs is not None and hasattr(tbs, "CfCTransitionReceipt"):
            try:
                receipt = tbs.CfCTransitionReceipt(  # type: ignore[attr-defined]
                    lobe_id=lobe_id,
                    pre_state_root=str(pre_state_root or native_dict.get("pre_state_hash", "") or "0" * 64),
                    tx_hash=str(tx_hash or native_dict.get("input_hash", "") or "0" * 64),
                    predicted_post_state_hash=predicted_hash,
                    actual_post_state_hash=actual_hash,
                    forecast_error=err_val,
                    cosine_similarity=cos_val,
                    time_delta=float(time_delta),
                    control_vector_hash=control_hash,
                    lobe_health_hash=lobe_health_hash,
                    scale_profile_hash=scale_hash,
                    tensor_backbone_commitment_hashes=backbone_hashes,
                    accepted=accepted,
                    metadata={"native_receipt": native_dict, "forecast_error_report": dict(forecast_error or {})},
                )
                return receipt.to_dict()
            except Exception:
                pass

        payload = {
            "lobe_id": lobe_id,
            "pre_state_root": str(pre_state_root or native_dict.get("pre_state_hash", "") or "0" * 64),
            "tx_hash": str(tx_hash or native_dict.get("input_hash", "") or "0" * 64),
            "predicted_post_state_hash": predicted_hash,
            "actual_post_state_hash": actual_hash,
            "forecast_error": err_val,
            "cosine_similarity": cos_val,
            "time_delta": float(time_delta),
            "control_vector_hash": control_hash,
            "lobe_health_hash": lobe_health_hash,
            "scale_profile_hash": scale_hash,
            "tensor_backbone_commitment_hashes": backbone_hashes,
            "accepted": accepted,
            "metadata": {"native_receipt": native_dict, "forecast_error_report": dict(forecast_error or {})},
            "created_at": _now_iso(),
            "schema_version": "tb.cfc_transition_receipt.v1",
        }
        payload["receipt_hash"] = _hash_object(payload, domain="TB:CFC_TRANSITION_RECEIPT:v1")
        return payload

    # ------------------------------------------------------------------
    # Scale profiles / snapshots
    # ------------------------------------------------------------------
    def scale_profile(self, lobe_kind: str) -> Dict[str, Any]:
        lobe_kind = str(lobe_kind).lower().strip()
        lobe = self.ncp_lobe if lobe_kind.startswith("ncp") else self.cfc_lobe
        if lobe is not None and hasattr(lobe, "scale_profile"):
            try:
                sp = getattr(lobe, "scale_profile")
                if hasattr(sp, "to_dict"):
                    return dict(_json_safe(sp.to_dict()))
                return dict(_json_safe(sp))
            except Exception:
                pass
        if tbs is not None and hasattr(tbs, "LobeScaleProfile"):
            try:
                if lobe_kind.startswith("ncp"):
                    prof = tbs.LobeScaleProfile(  # type: ignore[attr-defined]
                        lobe_id=str(getattr(lobe, "lobe_id", self.config.ncp_policy_id)),
                        lobe_kind="ncp_liquid_reflex",
                        active=bool(self.config.enable_ncp),
                        parameter_budget=int(self.config.max_parameter_budget_per_lobe),
                        dense_equivalent_budget=int(self._dense_equivalent_count(lobe)),
                        active_parameter_count=int(self._parameter_count(lobe)),
                        hidden_size=int(self.config.ncp_hidden_size),
                        num_cells=None,
                        tt_bond_rank=int(self.config.ncp_bond_dim),
                        precision=str(self.dtype),
                        update_frequency_hz=float(self.config.update_frequency_hz),
                        checkpoint_policy=self.config.checkpoint_policy,
                        consensus_safe=bool(self.config.consensus_safe),
                        device=self.config.device,
                        scale_mode="operational",
                    )
                else:
                    prof = tbs.LobeScaleProfile(  # type: ignore[attr-defined]
                        lobe_id=str(getattr(lobe, "lobe_id", self.config.cfc_policy_id)),
                        lobe_kind="cfc_predictive_transition",
                        active=bool(self.config.enable_cfc),
                        parameter_budget=int(self.config.max_parameter_budget_per_lobe),
                        dense_equivalent_budget=int(self._dense_equivalent_count(lobe)),
                        active_parameter_count=int(self._parameter_count(lobe)),
                        hidden_size=int(self.config.cfc_hidden_size),
                        num_cells=int(self.config.cfc_num_cells),
                        tt_bond_rank=int(self.config.cfc_bond_dim),
                        precision=str(self.dtype),
                        update_frequency_hz=float(self.config.update_frequency_hz),
                        checkpoint_policy=self.config.checkpoint_policy,
                        consensus_safe=bool(self.config.consensus_safe),
                        device=self.config.device,
                        scale_mode="operational",
                    )
                return prof.to_dict()
            except Exception:
                pass
        payload = {
            "lobe_id": str(getattr(lobe, "lobe_id", lobe_kind)),
            "lobe_kind": lobe_kind,
            "active": lobe is not None,
            "parameter_budget": int(self.config.max_parameter_budget_per_lobe),
            "dense_equivalent_budget": int(self._dense_equivalent_count(lobe)),
            "active_parameter_count": int(self._parameter_count(lobe)),
            "hidden_size": int(getattr(lobe, "num_hidden", getattr(lobe, "hidden_size", 0))) if lobe is not None else 0,
            "num_cells": getattr(lobe, "num_cells", None),
            "tt_bond_rank": int(self.config.ncp_bond_dim if lobe_kind.startswith("ncp") else self.config.cfc_bond_dim),
            "precision": str(self.dtype),
            "update_frequency_hz": float(self.config.update_frequency_hz),
            "checkpoint_policy": self.config.checkpoint_policy,
            "consensus_safe": bool(self.config.consensus_safe),
            "device": self.config.device,
            "schema_version": "tb.lobe_scale_profile.v1",
        }
        payload["profile_hash"] = _hash_object(payload, domain="TB:LOBE_SCALE_PROFILE:v1")
        return payload

    def scale_profile_hash(self, lobe_kind: str) -> str:
        profile = self.scale_profile(lobe_kind)
        return str(profile.get("profile_hash") or profile.get("hash") or _hash_object(profile, domain="TB:LOBE_SCALE_PROFILE:v1"))

    @staticmethod
    def _parameter_count(lobe: Any) -> int:
        if lobe is None:
            return 0
        if hasattr(lobe, "parameter_count"):
            try:
                return int(lobe.parameter_count())
            except Exception:
                return 0
        return 0

    @staticmethod
    def _dense_equivalent_count(lobe: Any) -> int:
        if lobe is None:
            return 0
        for name in ("equivalent_dense_parameter_count", "recurrent_dense_equivalent_parameter_count"):
            if hasattr(lobe, name):
                try:
                    return int(getattr(lobe, name)())
                except Exception:
                    pass
        return CognitiveLobeRuntime._parameter_count(lobe)

    def snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "runtime_id": self.runtime_id,
            "version": COGNITIVE_LOBE_RUNTIME_VERSION,
            "created_at": _now_iso(),
            "config": _json_safe(asdict(self.config.normalized())),
            "config_hash": self.config.config_hash(),
            "imports": dependency_status(),
            "scale_profiles": {
                "ncp": self.scale_profile("ncp") if self.ncp_lobe is not None else None,
                "cfc": self.scale_profile("cfc") if self.cfc_lobe is not None else None,
            },
            "lobes": {},
            "last_result_hash": None if self._last_result is None else self._last_result.compute_hash(),
        }
        if self.ncp_lobe is not None:
            try:
                out["lobes"]["ncp"] = self.ncp_lobe.snapshot()
            except Exception as exc:
                out["lobes"]["ncp"] = {"error": repr(exc)}
        if self.cfc_lobe is not None:
            try:
                out["lobes"]["cfc"] = self.cfc_lobe.snapshot()
            except Exception as exc:
                out["lobes"]["cfc"] = {"error": repr(exc)}
        out["snapshot_hash"] = _hash_object(out, domain="TB:COGNITIVE_LOBE_RUNTIME_SNAPSHOT:v1")
        return _json_safe(out)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run_ncp_pass(self, features: Union[EncodedFeatureVector, Sequence[float], np.ndarray], *, time_delta: float = 1.0) -> Optional[LobePassResult]:
        if self.ncp_lobe is None:
            return None
        x = features.to_array() if isinstance(features, EncodedFeatureVector) else _coerce_vector(features, dim=self.config.feature_encoder.feature_dim, dtype=self.dtype)
        warnings: List[str] = []
        try:
            try:
                y, aux = self.ncp_lobe.step(x, dt=float(time_delta), return_state=True)
            except TypeError:
                y, aux = self.ncp_lobe.step(x, return_state=True)
            y = np.asarray(y, dtype=self.dtype).reshape(-1)
            native_health = None
            if hasattr(self.ncp_lobe, "lobe_health_receipt"):
                try:
                    native_health = self.ncp_lobe.lobe_health_receipt(include_expensive=bool(self.config.health_receipt_include_expensive))
                except TypeError:
                    native_health = self.ncp_lobe.lobe_health_receipt()
            health = self._schema_lobe_health_receipt(lobe=self.ncp_lobe, lobe_kind="ncp_liquid_reflex", native=native_health)
            metrics = dict(_json_safe(aux.get("health", {}))) if isinstance(aux, Mapping) else {}
            if not metrics and hasattr(self.ncp_lobe, "health_metrics"):
                metrics = dict(_json_safe(self.ncp_lobe.health_metrics()))
            try:
                snap = self.ncp_lobe.snapshot()
            except Exception:
                snap = {}
            state_hash = str(health.get("state_hash") or _array_hash(getattr(self.ncp_lobe, "get_state")(), dtype=self.dtype, tag="ncp_state"))
            return LobePassResult(
                lobe_id=str(getattr(self.ncp_lobe, "lobe_id", self.config.ncp_policy_id)),
                lobe_kind="ncp_liquid_reflex",
                output_hash=_array_hash(y, dtype=self.dtype, tag="ncp_output"),
                state_hash=state_hash,
                health_receipt=health,
                snapshot=dict(_json_safe(snap)),
                metrics=metrics,
                output_vector=y.astype(float).tolist(),
                warnings=warnings,
                metadata={"aux": _json_safe(aux)},
            )
        except Exception as exc:
            warnings.append(f"ncp_pass_failed:{exc!r}")
            self.warnings.extend(warnings)
            raise

    def run_cfc_pass(
        self,
        features: Union[EncodedFeatureVector, Sequence[float], np.ndarray],
        *,
        time_delta: float = 1.0,
        advance: bool = True,
        actual_state: Optional[Any] = None,
        pre_state_root: str = "",
        tx_hash: str = "",
    ) -> Tuple[Optional[LobePassResult], Optional[Dict[str, Any]]]:
        if self.cfc_lobe is None:
            return None, None
        x = features.to_array() if isinstance(features, EncodedFeatureVector) else _coerce_vector(features, dim=self.config.feature_encoder.feature_dim, dtype=self.dtype)
        warnings: List[str] = []
        pre_state = None
        try:
            if hasattr(self.cfc_lobe, "get_state"):
                pre_state = self.cfc_lobe.get_state()
        except Exception:
            pre_state = None

        # First run a dry forecast so the receipt can refer to the predicted future without mutation.
        native_transition = None
        try:
            if hasattr(self.cfc_lobe, "forecast"):
                predicted, native_transition = self.cfc_lobe.forecast(x, time_delta=float(time_delta), return_receipt=True)
            else:
                predicted = self.cfc_lobe.step(x, time_delta=float(time_delta))
        except TypeError:
            predicted = self.cfc_lobe.step(x, float(time_delta))
        predicted_arr = np.asarray(predicted, dtype=self.dtype).reshape(-1)

        actual_arr: Optional[np.ndarray] = None
        aux: Dict[str, Any] = {}
        if actual_state is not None:
            actual_arr = np.asarray(actual_state, dtype=self.dtype).reshape(-1)
        elif advance:
            try:
                actual, aux_obj = self.cfc_lobe.step(x, time_delta=float(time_delta), return_state=True)
                actual_arr = np.asarray(actual, dtype=self.dtype).reshape(-1)
                aux = dict(_json_safe(aux_obj)) if isinstance(aux_obj, Mapping) else {}
            except TypeError:
                actual = self.cfc_lobe.step(x, float(time_delta))
                actual_arr = np.asarray(actual, dtype=self.dtype).reshape(-1)
        else:
            actual_arr = None

        forecast_error_report: Optional[Dict[str, Any]] = None
        if actual_arr is not None and hasattr(self.cfc_lobe, "compare_forecast_actual"):
            try:
                err = self.cfc_lobe.compare_forecast_actual(
                    predicted_arr,
                    actual_arr,
                    input_vector=x,
                    time_delta=float(time_delta),
                    pre_state=pre_state,
                )
                forecast_error_report = _receipt_to_dict(err)
            except Exception as exc:
                warnings.append(f"forecast_error_compare_failed:{exc!r}")
        elif actual_arr is not None:
            diff = actual_arr.astype(np.float64) - predicted_arr.astype(np.float64)
            denom = float(np.linalg.norm(predicted_arr.astype(np.float64))) + _EPS
            forecast_error_report = {
                "l2_error": float(np.linalg.norm(diff)),
                "relative_error": float(np.linalg.norm(diff) / denom),
                "ok": bool(float(np.linalg.norm(diff) / denom) <= 1e-3),
            }

        native_health = None
        if hasattr(self.cfc_lobe, "lobe_health_receipt"):
            try:
                native_health = self.cfc_lobe.lobe_health_receipt(include_expensive=bool(self.config.health_receipt_include_expensive))
            except TypeError:
                native_health = self.cfc_lobe.lobe_health_receipt()
        health = self._schema_lobe_health_receipt(lobe=self.cfc_lobe, lobe_kind="cfc_predictive_transition", native=native_health)
        try:
            control = self.cfc_lobe.get_control_vector(out_dim=min(32, int(getattr(self.cfc_lobe, "hidden_size", 32))), state=predicted_arr)
        except TypeError:
            try:
                control = self.cfc_lobe.get_control_vector(out_dim=min(32, int(getattr(self.cfc_lobe, "hidden_size", 32))))
            except Exception:
                control = predicted_arr[: min(32, predicted_arr.size)]
        except Exception:
            control = predicted_arr[: min(32, predicted_arr.size)]

        transition_receipt = self._schema_cfc_transition_receipt(
            native=native_transition,
            pre_state_root=pre_state_root or ("0" * 64),
            tx_hash=tx_hash or _array_hash(x, dtype=self.dtype, tag="cfc_input_tx_hash"),
            predicted_state=predicted_arr,
            actual_state=actual_arr,
            control_vector=control,
            health_receipt=health,
            time_delta=float(time_delta),
            forecast_error=forecast_error_report,
        )

        try:
            snap = self.cfc_lobe.snapshot()
        except Exception:
            snap = {}
        metrics = {}
        try:
            metrics = dict(_json_safe(self.cfc_lobe.health_metrics()))
        except Exception:
            if isinstance(aux, Mapping):
                metrics = dict(_json_safe(aux.get("health", {})))
        state_hash = str(health.get("state_hash") or _array_hash(actual_arr if actual_arr is not None else predicted_arr, dtype=self.dtype, tag="cfc_state"))
        result = LobePassResult(
            lobe_id=str(getattr(self.cfc_lobe, "lobe_id", self.config.cfc_policy_id)),
            lobe_kind="cfc_predictive_transition",
            output_hash=_array_hash(control, dtype=self.dtype, tag="cfc_control_output"),
            state_hash=state_hash,
            health_receipt=health,
            snapshot=dict(_json_safe(snap)),
            metrics=metrics,
            output_vector=np.asarray(control, dtype=self.dtype).reshape(-1).astype(float).tolist(),
            warnings=warnings,
            metadata={
                "predicted_state_hash": _array_hash(predicted_arr, dtype=self.dtype, tag="cfc_predicted"),
                "actual_state_hash": "" if actual_arr is None else _array_hash(actual_arr, dtype=self.dtype, tag="cfc_actual"),
                "forecast_error_report": forecast_error_report or {},
                "aux": aux,
            },
        )
        self.warnings.extend(warnings)
        return result, transition_receipt

    def run(
        self,
        *,
        twin_snapshot: Optional[Any] = None,
        transaction: Optional[Any] = None,
        pre_state_commitment: Optional[Any] = None,
        post_state_commitment: Optional[Any] = None,
        policy_receipt: Optional[Any] = None,
        extra: Optional[Mapping[str, Any]] = None,
        raw_vector: Optional[Any] = None,
        time_delta: float = 1.0,
        advance_cfc: bool = True,
    ) -> DualLobeRunResult:
        encoded = self.encode_features(
            twin_snapshot=twin_snapshot,
            transaction=transaction,
            pre_state_commitment=pre_state_commitment,
            post_state_commitment=post_state_commitment,
            policy_receipt=policy_receipt,
            extra=extra,
            raw_vector=raw_vector,
        )
        warnings: List[str] = []
        ncp_result = self.run_ncp_pass(encoded, time_delta=time_delta) if self.ncp_lobe is not None else None
        pre_root = self._extract_root(pre_state_commitment) or "0" * 64
        txh = self._extract_tx_hash(transaction) or encoded.vector_hash
        cfc_result, cfc_receipt = self.run_cfc_pass(
            encoded,
            time_delta=time_delta,
            advance=advance_cfc,
            pre_state_root=pre_root,
            tx_hash=txh,
        ) if self.cfc_lobe is not None else (None, None)
        warnings.extend(self.warnings[-10:])
        result = DualLobeRunResult(
            runtime_id=self.runtime_id,
            encoded_features=encoded.to_dict(),
            ncp_result=None if ncp_result is None else ncp_result.to_dict(),
            cfc_result=None if cfc_result is None else cfc_result.to_dict(),
            cfc_transition_receipt=cfc_receipt,
            warnings=warnings,
            metadata={
                "runtime_config_hash": self.config.config_hash(),
                "time_delta": float(time_delta),
                "advance_cfc": bool(advance_cfc),
            },
        )
        result.run_hash = result.compute_hash()
        self._last_result = result
        return result

    @staticmethod
    def _extract_root(obj: Any) -> str:
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj if len(obj) == 64 else ""
        if isinstance(obj, Mapping):
            for key in ("state_root", "post_state_root", "pre_state_root", "commitment_hash"):
                val = obj.get(key)
                if isinstance(val, str) and len(val) == 64:
                    return val
        if hasattr(obj, "state_root"):
            try:
                val = str(getattr(obj, "state_root"))
                return val if len(val) == 64 else ""
            except Exception:
                pass
        if hasattr(obj, "to_dict"):
            try:
                return CognitiveLobeRuntime._extract_root(obj.to_dict())
            except Exception:
                pass
        return ""

    @staticmethod
    def _extract_tx_hash(obj: Any) -> str:
        if obj is None:
            return ""
        if isinstance(obj, Mapping):
            for key in ("tx_hash", "transaction_hash", "hash"):
                val = obj.get(key)
                if isinstance(val, str) and len(val) == 64:
                    return val
            return _hash_object(obj, domain="TB:COGNITIVE_LOBE_TX_CONTEXT:v1")
        if hasattr(obj, "tx_hash") and callable(obj.tx_hash):
            try:
                return str(obj.tx_hash())
            except Exception:
                pass
        if hasattr(obj, "hash") and callable(obj.hash):
            try:
                return str(obj.hash())
            except Exception:
                pass
        if hasattr(obj, "to_dict"):
            try:
                return _hash_object(obj.to_dict(), domain="TB:COGNITIVE_LOBE_TX_CONTEXT:v1")
            except Exception:
                pass
        return _hash_object(_json_safe(obj), domain="TB:COGNITIVE_LOBE_TX_CONTEXT:v1")

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def save_checkpoints(self, directory: Union[str, os.PathLike]) -> DualLobeCheckpointManifest:
        root = Path(directory).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        files: List[Dict[str, Any]] = []
        if self.ncp_lobe is not None and hasattr(self.ncp_lobe, "save_checkpoint"):
            path = root / "ncp_liquid_reflex_lobe.npz"
            self.ncp_lobe.save_checkpoint(path)
            files.append({"kind": "ncp", "path": path.name, "sha256": _read_hash_file(path), "size_bytes": path.stat().st_size})
        if self.cfc_lobe is not None and hasattr(self.cfc_lobe, "save_checkpoint"):
            path = root / "cfc_predictive_transition_lobe.npz"
            self.cfc_lobe.save_checkpoint(path)
            files.append({"kind": "cfc", "path": path.name, "sha256": _read_hash_file(path), "size_bytes": path.stat().st_size})
        manifest_seed = {"files": files, "runtime_config_hash": self.config.config_hash(), "runtime_id": self.runtime_id}
        checkpoint_root = _hash_object(manifest_seed, domain="TB:COGNITIVE_LOBE_CHECKPOINT_ROOT:v1")
        manifest = DualLobeCheckpointManifest(
            runtime_id=self.runtime_id,
            created_at=_now_iso(),
            runtime_config_hash=self.config.config_hash(),
            checkpoint_root=checkpoint_root,
            files=files,
            snapshots=self.snapshot(),
        )
        _atomic_write_json(root / "cognitive_lobe_checkpoint_manifest.json", manifest.to_dict())
        return manifest

    @classmethod
    def load_from_checkpoints(
        cls,
        directory: Union[str, os.PathLike],
        *,
        config: Optional[CognitiveLobeRuntimeConfig] = None,
    ) -> "CognitiveLobeRuntime":
        root = Path(directory).expanduser().resolve()
        cfg = (config or CognitiveLobeRuntimeConfig()).normalized()
        ncp_lobe = None
        cfc_lobe = None
        ncp_path = root / "ncp_liquid_reflex_lobe.npz"
        cfc_path = root / "cfc_predictive_transition_lobe.npz"
        if ncp_path.exists() and ncp is not None and hasattr(ncp, "TensorizedNCP"):
            ncp_lobe = ncp.TensorizedNCP.load_checkpoint(ncp_path)  # type: ignore[attr-defined]
        if cfc_path.exists() and cfc is not None and hasattr(cfc, "TCfC_Policy"):
            cfc_lobe = cfc.TCfC_Policy.load_checkpoint(cfc_path, verbose=False, tt_strict=cfg.tt_strict)  # type: ignore[attr-defined]
        return cls(cfg, ncp_lobe=ncp_lobe, cfc_lobe=cfc_lobe)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_consensus_safe(self, *, raise_on_error: bool = False) -> Dict[str, Any]:
        errors: List[str] = []
        if not self.config.consensus_safe:
            errors.append("runtime.consensus_safe_false")
        if not self.config.deterministic:
            errors.append("runtime.deterministic_false")
        if self.config.device != "cpu":
            errors.append("runtime.device_not_cpu")
        for name, lobe in (("ncp", self.ncp_lobe), ("cfc", self.cfc_lobe)):
            if lobe is None:
                continue
            if hasattr(lobe, "validate_consensus_safe"):
                try:
                    res = lobe.validate_consensus_safe()
                    if not bool(res.get("ok", False)):
                        errors.extend([f"{name}.{x}" for x in res.get("errors", [])])
                except Exception as exc:
                    errors.append(f"{name}.consensus_safe_validation_failed:{exc!r}")
            else:
                if not bool(getattr(lobe, "deterministic", self.config.deterministic)):
                    errors.append(f"{name}.deterministic_false")
                if not bool(getattr(lobe, "consensus_safe", self.config.consensus_safe)):
                    errors.append(f"{name}.consensus_safe_false")
        out = {"ok": len(errors) == 0, "errors": errors, "runtime_id": self.runtime_id}
        if raise_on_error and errors:
            raise RuntimeError(f"CognitiveLobeRuntime consensus-safe validation failed: {errors}")
        return out


# =============================================================================
# Status / demo / CLI
# =============================================================================


def dependency_status() -> Dict[str, Any]:
    return {
        "version": COGNITIVE_LOBE_RUNTIME_VERSION,
        "ts": _now_iso(),
        "paths": configure_paths(),
        "imports": {
            "ncp": {"available": ncp is not None, "module": getattr(ncp, "__file__", None)},
            "cfc": {"available": cfc is not None, "module": getattr(cfc, "__file__", None)},
            "tb_schemas": {"available": tbs is not None, "module": getattr(tbs, "__file__", None)},
            "tb_hashing": {"available": tbh is not None, "module": getattr(tbh, "__file__", None)},
            "tb_canonical": {"available": tbc is not None, "module": getattr(tbc, "__file__", None)},
            "tb_config": {"available": tbcfg is not None, "module": getattr(tbcfg, "__file__", None)},
        },
        "errors": dict(_IMPORT_ERRORS),
    }


def run_self_test() -> Dict[str, Any]:
    cfg = CognitiveLobeRuntimeConfig(
        runtime_id="selftest_dualistic_tensor_brain",
        deterministic=True,
        consensus_safe=False,
        rng_seed=123,
        feature_encoder=FeatureEncoderConfig(feature_dim=32, hash_buckets=24, numeric_slots=8),
        ncp_hidden_size=32,
        ncp_output_size=8,
        ncp_bond_dim=2,
        cfc_hidden_size=16,
        cfc_num_cells=2,
        cfc_bond_dim=2,
        tt_strict=False,
        force_dense=False,
    ).normalized()
    rt = CognitiveLobeRuntime(cfg)
    result = rt.run(
        twin_snapshot={"state_root": "a" * 64, "temperature_c": 22.5, "status": "nominal"},
        transaction={"tx_hash": "b" * 64, "kind": "update_node_data", "node_id": "5.4", "payload": {"temperature_c": 44.0}},
        pre_state_commitment={"state_root": "a" * 64},
        extra={"scenario": "self_test"},
        time_delta=0.25,
    )
    rd = result.to_dict()
    if not rd.get("run_hash"):
        raise AssertionError("run_hash missing")
    if cfg.enable_ncp and not rd.get("ncp_result"):
        raise AssertionError("ncp_result missing")
    if cfg.enable_cfc and not rd.get("cfc_transition_receipt"):
        raise AssertionError("cfc_transition_receipt missing")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        manifest = rt.save_checkpoints(td)
        if not manifest.files:
            raise AssertionError("checkpoint files missing")
        rt2 = CognitiveLobeRuntime.load_from_checkpoints(td, config=cfg)
        snap2 = rt2.snapshot()
        if not snap2.get("snapshot_hash"):
            raise AssertionError("loaded snapshot hash missing")
    return {
        "ok": True,
        "dependency_status": dependency_status(),
        "run": rd,
        "snapshot_hash": rt.snapshot().get("snapshot_hash"),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Akkurat dualistic tensor cognitive lobe runtime")
    p.add_argument("--mode", choices=["status", "self-test", "demo"], default="self-test")
    p.add_argument("--pretty", action="store_true")
    p.add_argument("--checkpoint-dir", default="")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        if args.mode == "status":
            payload = dependency_status()
        elif args.mode in {"self-test", "demo"}:
            payload = run_self_test()
            if args.checkpoint_dir:
                rt = CognitiveLobeRuntime(CognitiveLobeRuntimeConfig().normalized())
                manifest = rt.save_checkpoints(args.checkpoint_dir)
                payload["checkpoint_manifest"] = manifest.to_dict()
        else:
            raise ValueError(f"unknown mode: {args.mode}")
        print(json.dumps(_json_safe(payload), indent=2 if args.pretty else None, ensure_ascii=False, sort_keys=True))
        return 0 if bool(payload.get("ok", True)) else 2
    except Exception as exc:
        err = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc(), "dependency_status": dependency_status()}
        print(json.dumps(_json_safe(err), indent=2, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
