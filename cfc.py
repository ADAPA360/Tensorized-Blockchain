#!/usr/bin/env python3
# cfc.py
r"""
Project Chimera / Akkurat - Production Predictive Transition Lobe v4.0
======================================================================

Tensorized Closed-form Continuous-time substrate for the Tensorized Blockchain
stack.

Production role
---------------
This module provides the Predictive Transition Lobe: a bounded, deterministic,
dt-aware, TensorTrain/MPO-compatible continuous-time forecasting substrate. It
is intentionally a neural substrate only. It does not route tools, execute
ledger transactions, or make governance decisions. Higher layers consume its
outputs as auditable transition receipts.

Compatibility
-------------
The primary class preserves the original constructor and stepping style:

    policy = TCfC_Policy(policy_id, input_size, hidden_size, num_cells, ...)
    state = policy.step(x, time_delta=1.0)
    state, aux = policy.step(x, time_delta=1.0, return_state=True)

Production upgrades
-------------------
- CfCTransitionReceipt for forecast/pre/post transition auditing.
- ForecastErrorReport for predicted-vs-actual state comparisons.
- LobeScaleProfile compatibility with the Liquid Reflex Lobe.
- Deterministic lobe commitment manifest and tensor backbone commitments.
- Consensus-safe deterministic mode with CPU/dtype/runtime guardrails.
- Checkpoint integrity manifest/hash verification.
- Dense-equivalent parameter accounting and compression factor reporting.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

CFC_VERSION = "cfc-predictive-transition-lobe-v4.0-production"
CFC_COMMITMENT_SCHEME = "akkurat.lobe.cfc.commitment.v1"
CFC_CHECKPOINT_FORMAT = "akkurat.cfc.checkpoint.v2"
CFC_TRANSITION_RECEIPT_SCHEME = "akkurat.lobe.cfc.transition_receipt.v1"
_EPS = 1e-12


# =============================================================================
# TensorTrain import
# =============================================================================

TENSORTRAIN_AVAILABLE = False
TT_IMPORT_ERROR: Optional[Exception] = None

try:
    try:
        from .tn import TensorTrain, TTConfig, get_factors, tensor_train_svd  # type: ignore
        TENSORTRAIN_AVAILABLE = True
    except Exception:
        try:
            from tn import TensorTrain, TTConfig, get_factors, tensor_train_svd  # type: ignore
            TENSORTRAIN_AVAILABLE = True
        except Exception:
            root = Path(__file__).resolve().parent
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from tn import TensorTrain, TTConfig, get_factors, tensor_train_svd  # type: ignore
            TENSORTRAIN_AVAILABLE = True
except Exception as exc:  # pragma: no cover - exercised only without tn.py
    TT_IMPORT_ERROR = exc

    def get_factors(n: int) -> Tuple[int, int]:  # type: ignore[no-redef]
        n = int(n)
        if n <= 0:
            return (1, 1)
        a = int(math.isqrt(n))
        while a > 0:
            if n % a == 0:
                return (a, n // a)
            a -= 1
        return (1, n)

    @dataclass(frozen=True)
    class TTConfig:  # type: ignore[no-redef]
        dtype: np.dtype = np.float32
        device: str = "cpu"
        check_finite: bool = True
        seed: Optional[int] = None
        deterministic: bool = False
        consensus_safe: bool = False

    class TensorTrain:  # type: ignore[no-redef]
        pass

    def tensor_train_svd(*args: Any, **kwargs: Any) -> Optional[Any]:  # type: ignore[no-redef]
        return None


def _np_from_any_array(x: Any) -> np.ndarray:
    """Return a NumPy array from NumPy/CuPy-like arrays."""
    if hasattr(x, "__cuda_array_interface__"):
        try:
            import cupy as cp  # type: ignore

            return cp.asnumpy(x)
        except Exception:
            pass
    return np.asarray(x)


# =============================================================================
# Canonical serialization / hashing helpers
# =============================================================================


def _utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, np.dtype):
        return str(obj)
    if isinstance(obj, np.ndarray):
        arr = np.nan_to_num(np.asarray(obj), nan=0.0, posinf=0.0, neginf=0.0)
        return arr.astype(float).tolist()
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in sorted(dict(obj).items(), key=lambda kv: str(kv[0]))}
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


def _json_dumps_stable(obj: Any) -> str:
    return json.dumps(_json_safe(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(obj: Any) -> str:
    return _sha256_bytes(_json_dumps_stable(obj).encode("utf-8", errors="ignore"))


def _array_hash(arr: Any, *, dtype: np.dtype = np.float32, tag: str = "array") -> str:
    x = np.nan_to_num(np.asarray(arr, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
    h = hashlib.sha256()
    h.update(str(tag).encode("utf-8"))
    h.update(b"\x00")
    h.update(_json_dumps_stable({"shape": list(map(int, x.shape)), "dtype": str(np.dtype(dtype))}).encode("utf-8"))
    h.update(b"\x00")
    h.update(np.ascontiguousarray(x).tobytes(order="C"))
    return h.hexdigest()


def _atomic_save_npz(path: Union[str, os.PathLike], arrays: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez_compressed(tmp, **dict(arrays))
        actual_npz = Path(str(tmp) + ".npz")
        actual = actual_npz if actual_npz.exists() else tmp
        os.replace(actual, p)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    except Exception:
        for candidate in (tmp, Path(str(tmp) + ".npz")):
            try:
                if candidate.exists():
                    candidate.unlink()
            except Exception:
                pass
        raise


# =============================================================================
# Numerical helpers
# =============================================================================


def two_factors(n: int) -> Tuple[int, int]:
    n = int(n)
    if n <= 0:
        return (1, 1)
    try:
        a, b = get_factors(n)
        a, b = int(a), int(b)
        if a > 0 and b > 0 and a * b == n:
            return a, b
    except Exception:
        pass
    r = int(math.isqrt(n))
    for f in range(r, 0, -1):
        if n % f == 0:
            return (f, n // f)
    return (1, n)


def _prod(values: Sequence[int]) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return int(out)


def _coerce_vector(
    x: Any,
    *,
    expected_dim: Optional[int] = None,
    dtype: np.dtype = np.float32,
    name: str = "vector",
    clip: Optional[float] = None,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.dtype(dtype)).reshape(-1).copy()
    if expected_dim is not None and int(arr.size) != int(expected_dim):
        raise ValueError(f"{name} size mismatch: expected {expected_dim}, got {arr.size}")
    if arr.size == 0 and expected_dim not in (None, 0):
        raise ValueError(f"{name} cannot be empty")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.dtype(dtype), copy=False)
    if clip is not None and float(clip) > 0:
        np.clip(arr, -float(clip), float(clip), out=arr)
    return arr


def _sanitize(arr: Any, dtype: np.dtype = np.float32, fill: float = 0.0) -> np.ndarray:
    out = np.asarray(arr, dtype=np.dtype(dtype)).copy()
    mask = ~np.isfinite(out)
    if np.any(mask):
        out[mask] = np.dtype(dtype).type(fill)
    return out.astype(np.dtype(dtype), copy=False)


def _safe_norm(x: Any) -> float:
    try:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return 0.0
        return float(np.linalg.norm(arr))
    except Exception:
        return 0.0


def _cosine(a: Any, b: Any) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size == 0 or aa.size != bb.size:
        return 0.0
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if not np.isfinite(na) or not np.isfinite(nb) or na <= _EPS or nb <= _EPS:
        return 0.0
    return float(np.dot(aa, bb) / (na * nb))


def _count_nonfinite(x: Any) -> int:
    arr = np.asarray(x)
    return int(arr.size - np.count_nonzero(np.isfinite(arr)))


def _saturation_fraction(x: Any, threshold: float = 0.98) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr) >= float(threshold)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))).astype(np.float32, copy=False)


def _activation(x: np.ndarray, mode: str) -> np.ndarray:
    mode = str(mode).lower().strip()
    if mode == "tanh":
        return np.tanh(x).astype(x.dtype, copy=False)
    if mode == "clip":
        return np.clip(x, -1.0, 1.0).astype(x.dtype, copy=False)
    return x


def _clamp_dt(dt: float, *, lo: float = 0.0, hi: float = 1e3) -> float:
    try:
        v = float(dt)
    except Exception:
        return 1.0
    if not math.isfinite(v):
        return 1.0
    return float(np.clip(v, float(lo), float(hi)))


def _check_factors(size: int, f1: int, f2: int, name: str) -> None:
    if int(f1) * int(f2) != int(size):
        raise ValueError(f"[{name}] factorization mismatch: ({f1} x {f2}) != {size}.")


def _bond_dims_from_spec(spec: Union[int, Sequence[int]], num_cores: int) -> List[int]:
    if num_cores <= 1:
        return []
    needed = num_cores - 1
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        vals = [int(max(1, x)) for x in spec]
        if len(vals) == needed:
            return vals
        if len(vals) == 0:
            return [4] * needed
        return [vals[0]] * needed
    return [int(max(1, int(spec)))] * needed


def _estimate_dense_spectral_norm(mat: np.ndarray, *, iters: int = 16) -> Optional[float]:
    try:
        W = np.asarray(mat, dtype=np.float64)
        if W.ndim != 2 or W.size == 0:
            return 0.0
        rng = np.random.default_rng(12345)
        v = rng.normal(size=(W.shape[1],))
        v /= np.linalg.norm(v) + _EPS
        for _ in range(max(1, int(iters))):
            u = W @ v
            un = np.linalg.norm(u)
            if not np.isfinite(un) or un < _EPS:
                return 0.0
            u /= un
            v = W.T @ u
            vn = np.linalg.norm(v)
            if not np.isfinite(vn) or vn < _EPS:
                return 0.0
            v /= vn
        return float(np.linalg.norm(W @ v))
    except Exception:
        return None


def _stable_dense_init_rect(
    out_dim: int,
    in_dim: int,
    rng: np.random.Generator,
    dtype: np.dtype,
    *,
    base_scale: float = 0.9,
    max_sv: float = 0.95,
) -> np.ndarray:
    out_dim = int(out_dim)
    in_dim = int(in_dim)
    dt = np.dtype(dtype)
    if out_dim <= 0 or in_dim <= 0:
        return np.zeros((max(0, out_dim), max(0, in_dim)), dtype=dt)

    scale = float(base_scale) / math.sqrt(max(1, in_dim))
    W = rng.normal(0.0, scale, size=(out_dim, in_dim)).astype(np.float64)
    if out_dim == in_dim:
        try:
            A = rng.normal(0.0, 1.0, size=(out_dim, out_dim))
            q, r = np.linalg.qr(A)
            d = np.sign(np.diag(r))
            d[d == 0] = 1.0
            W = q * d * float(base_scale)
        except Exception:
            pass
    smax = _estimate_dense_spectral_norm(W, iters=12)
    if smax is not None and smax > float(max_sv) and smax > 0.0:
        W *= float(max_sv) / float(smax)
    return W.astype(dt, copy=False)


def _tt_param_count(tt_obj: Any) -> Optional[int]:
    for name in ("parameter_count", "num_parameters", "parameters"):
        fn = getattr(tt_obj, name, None)
        try:
            if callable(fn):
                val = int(fn())
                if val >= 0:
                    return val
        except Exception:
            pass
    cores = getattr(tt_obj, "cores_data", None)
    if isinstance(cores, (list, tuple)):
        try:
            return int(sum(int(np.asarray(_np_from_any_array(c)).size) for c in cores))
        except Exception:
            return None
    return None


def _tt_commitment_hash(tt_obj: Any) -> str:
    fn = getattr(tt_obj, "commitment", None)
    if callable(fn):
        try:
            c = fn()
            if hasattr(c, "commitment_hash"):
                return str(c.commitment_hash)
            if isinstance(c, Mapping):
                return str(c.get("commitment_hash", _sha256_json(c)))
            return _sha256_json(c)
        except Exception:
            pass
    cores = getattr(tt_obj, "cores_data", None)
    if isinstance(cores, (list, tuple)):
        return _sha256_json([_array_hash(_np_from_any_array(c), tag=f"tt_core_{i}") for i, c in enumerate(cores)])
    return ""


def _tt_apply(tt_obj: Any, vec: np.ndarray, *, out_dim: int, dtype: np.dtype, strict: bool = True) -> np.ndarray:
    dt = np.dtype(dtype)
    x = np.asarray(vec, dtype=dt).reshape(-1)
    last_err: Optional[BaseException] = None
    for method in ("apply", "contract_with_vector", "matvec", "__call__"):
        fn = getattr(tt_obj, method, None)
        if fn is None:
            continue
        try:
            y = fn(x)
            y = np.asarray(_np_from_any_array(y), dtype=dt).reshape(-1)
            if y.size == int(out_dim):
                return _sanitize(y, dt)
            if strict:
                raise ValueError(f"TT output size {y.size} != expected {out_dim}")
            out = np.zeros((int(out_dim),), dtype=dt)
            out[: min(out.size, y.size)] = y[: min(out.size, y.size)]
            return _sanitize(out, dt)
        except Exception as exc:
            last_err = exc
    tm = getattr(tt_obj, "to_matrix", None)
    if callable(tm):
        try:
            try:
                W = np.asarray(tm(force_cpu=True), dtype=dt)
            except TypeError:
                W = np.asarray(tm(), dtype=dt)
            if W.ndim == 2 and W.shape[1] == x.size:
                y = W @ x
                y = np.asarray(y, dtype=dt).reshape(-1)
                if y.size == int(out_dim):
                    return _sanitize(y, dt)
                if strict:
                    raise ValueError(f"TT dense output size {y.size} != expected {out_dim}")
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"TensorTrain contraction failed; last error: {last_err}")


# =============================================================================
# Production dataclasses
# =============================================================================


@dataclass
class LobeScaleProfile:
    lobe_id: str
    lobe_kind: str = "cfc_predictive_transition"
    active: bool = True
    parameter_budget: int = 2_500_000_000
    dense_equivalent_budget: int = 2_500_000_000
    hidden_size: int = 128
    num_cells: int = 4
    tt_bond_rank: int = 8
    precision: str = "float32"
    update_frequency_hz: float = 1.0
    checkpoint_policy: str = "hash_on_checkpoint"
    consensus_safe: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "LobeScaleProfile":
        p = LobeScaleProfile(**_json_safe(asdict(self)))
        p.lobe_id = str(p.lobe_id or "cfc_predictive_transition")
        p.lobe_kind = str(p.lobe_kind or "cfc_predictive_transition")
        p.active = bool(p.active)
        p.parameter_budget = int(max(0, p.parameter_budget))
        p.dense_equivalent_budget = int(max(0, p.dense_equivalent_budget))
        p.hidden_size = int(max(1, p.hidden_size))
        p.num_cells = int(max(1, p.num_cells))
        p.tt_bond_rank = int(max(1, p.tt_bond_rank))
        p.precision = str(p.precision or "float32").lower().strip()
        if p.precision not in {"float16", "float32", "float64", "int8"}:
            p.precision = "float32"
        p.update_frequency_hz = float(max(0.0, p.update_frequency_hz))
        p.checkpoint_policy = str(p.checkpoint_policy or "hash_on_checkpoint")
        p.consensus_safe = bool(p.consensus_safe)
        p.metadata = dict(p.metadata or {})
        return p

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self.normalized()))

    def compute_hash(self) -> str:
        return _sha256_json({"domain": "LobeScaleProfile", "payload": self.to_dict()})


@dataclass
class LobeHealthReceipt:
    lobe_id: str
    lobe_kind: str
    step_count: int
    health: Dict[str, Any]
    scale_profile_hash: str
    state_hash: str
    manifest_hash: str
    receipt_hash: str = ""

    def payload(self) -> Dict[str, Any]:
        return {
            "lobe_id": self.lobe_id,
            "lobe_kind": self.lobe_kind,
            "step_count": int(self.step_count),
            "health": _json_safe(self.health),
            "scale_profile_hash": self.scale_profile_hash,
            "state_hash": self.state_hash,
            "manifest_hash": self.manifest_hash,
        }

    def compute_hash(self) -> str:
        return _sha256_json({"domain": "LobeHealthReceipt", "payload": self.payload()})

    def to_dict(self) -> Dict[str, Any]:
        out = self.payload()
        out["receipt_hash"] = self.receipt_hash or self.compute_hash()
        return out


@dataclass
class ForecastErrorReport:
    predicted_hash: str
    actual_hash: str
    l2_error: float
    relative_l2_error: float
    cosine_similarity: float
    max_abs_error: float
    ok: bool

    @classmethod
    def compare(cls, predicted: Any, actual: Any, *, tolerance: float = 1e-3, dtype: np.dtype = np.float32) -> "ForecastErrorReport":
        p = np.asarray(predicted, dtype=dtype).reshape(-1)
        a = np.asarray(actual, dtype=dtype).reshape(-1)
        if p.size != a.size:
            n = min(p.size, a.size)
            p2 = np.zeros(max(p.size, a.size), dtype=dtype)
            a2 = np.zeros(max(p.size, a.size), dtype=dtype)
            p2[:p.size] = p
            a2[:a.size] = a
            p, a = p2, a2
        diff = p.astype(np.float64) - a.astype(np.float64)
        l2 = float(np.linalg.norm(diff))
        denom = float(np.linalg.norm(a.astype(np.float64))) + _EPS
        rel = float(l2 / denom)
        return cls(
            predicted_hash=_array_hash(p, dtype=dtype, tag="predicted_state"),
            actual_hash=_array_hash(a, dtype=dtype, tag="actual_state"),
            l2_error=l2,
            relative_l2_error=rel,
            cosine_similarity=_cosine(p, a),
            max_abs_error=float(np.max(np.abs(diff))) if diff.size else 0.0,
            ok=bool(rel <= float(tolerance)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class CfCTransitionReceipt:
    scheme: str
    version: str
    policy_id: str
    lobe_id: str
    step_count: int
    time_delta: float
    input_hash: str
    pre_state_hash: str
    predicted_state_hash: str
    actual_state_hash: str = ""
    forecast_error: Optional[Dict[str, Any]] = None
    control_hash: str = ""
    health_hash: str = ""
    manifest_hash: str = ""
    scale_profile_hash: str = ""
    tensor_backbone_root: str = ""
    receipt_hash: str = ""

    def payload(self) -> Dict[str, Any]:
        return {
            "scheme": self.scheme,
            "version": self.version,
            "policy_id": self.policy_id,
            "lobe_id": self.lobe_id,
            "step_count": int(self.step_count),
            "time_delta": float(self.time_delta),
            "input_hash": self.input_hash,
            "pre_state_hash": self.pre_state_hash,
            "predicted_state_hash": self.predicted_state_hash,
            "actual_state_hash": self.actual_state_hash,
            "forecast_error": _json_safe(self.forecast_error or {}),
            "control_hash": self.control_hash,
            "health_hash": self.health_hash,
            "manifest_hash": self.manifest_hash,
            "scale_profile_hash": self.scale_profile_hash,
            "tensor_backbone_root": self.tensor_backbone_root,
        }

    def compute_hash(self) -> str:
        return _sha256_json({"domain": CFC_TRANSITION_RECEIPT_SCHEME, "payload": self.payload()})

    def to_dict(self) -> Dict[str, Any]:
        out = self.payload()
        out["receipt_hash"] = self.receipt_hash or self.compute_hash()
        return out


@dataclass
class CfCLobeCommitmentManifest:
    scheme: str
    version: str
    policy_id: str
    lobe_id: str
    config_hash: str
    state_hash: str
    scale_profile_hash: str
    tensor_backbone_root: str
    tensor_backbone_commitments: List[str]
    dense_backbone_hashes: List[str]
    head_hashes: List[str]
    parameter_count: int
    dense_equivalent_parameter_count: int
    compression_factor: Optional[float]
    consensus_safe: bool
    checkpoint_hash: str = ""
    manifest_hash: str = ""

    def payload(self) -> Dict[str, Any]:
        return {
            "scheme": self.scheme,
            "version": self.version,
            "policy_id": self.policy_id,
            "lobe_id": self.lobe_id,
            "config_hash": self.config_hash,
            "state_hash": self.state_hash,
            "scale_profile_hash": self.scale_profile_hash,
            "tensor_backbone_root": self.tensor_backbone_root,
            "tensor_backbone_commitments": list(self.tensor_backbone_commitments),
            "dense_backbone_hashes": list(self.dense_backbone_hashes),
            "head_hashes": list(self.head_hashes),
            "parameter_count": int(self.parameter_count),
            "dense_equivalent_parameter_count": int(self.dense_equivalent_parameter_count),
            "compression_factor": None if self.compression_factor is None else float(self.compression_factor),
            "consensus_safe": bool(self.consensus_safe),
            "checkpoint_hash": self.checkpoint_hash,
        }

    def compute_hash(self) -> str:
        return _sha256_json({"domain": CFC_COMMITMENT_SCHEME, "payload": self.payload()})

    def to_dict(self) -> Dict[str, Any]:
        out = self.payload()
        out["manifest_hash"] = self.manifest_hash or self.compute_hash()
        return out


@dataclass
class TCfCCellConfig:
    input_size: int
    hidden_size: int
    bond_dim: Union[int, Sequence[int]] = 8
    dtype: str = "float32"
    tt_strict: bool = True
    force_dense: bool = False
    energy_tol: float = 0.999
    input_clip: float = 5.0
    state_clip: float = 1.0
    preactivation_clip: float = 10.0
    candidate_activation: str = "tanh"
    backbone_activation: str = "tanh"
    time_scale: float = 1.0
    leaky_lambda: float = 1.0
    strict_runtime_checks: bool = True
    deterministic: bool = False
    rng_seed: Optional[int] = None
    consensus_safe: bool = False


@dataclass
class TCfCPolicyConfig:
    policy_id: str
    input_size: int
    hidden_size: int
    num_cells: int
    bond_dim: Union[int, Sequence[int]] = 8
    dtype: str = "float32"
    tt_strict: bool = True
    force_dense: bool = False
    energy_tol: float = 0.999
    input_clip: float = 5.0
    state_clip: float = 1.0
    preactivation_clip: float = 10.0
    candidate_activation: str = "tanh"
    backbone_activation: str = "tanh"
    time_scale: float = 1.0
    leaky_lambda: float = 1.0
    strict_runtime_checks: bool = True
    verbose: bool = True
    rng_seed: Optional[int] = None
    deterministic: bool = False
    consensus_safe: bool = False

    def normalized(self) -> "TCfCPolicyConfig":
        cfg = TCfCPolicyConfig(**_json_safe(asdict(self)))
        cfg.policy_id = str(cfg.policy_id or "tcfc_policy")
        cfg.input_size = int(max(1, cfg.input_size))
        cfg.hidden_size = int(max(1, cfg.hidden_size))
        cfg.num_cells = int(max(1, cfg.num_cells))
        cfg.dtype = str(cfg.dtype or "float32").lower().strip()
        if cfg.dtype not in {"float32", "float64", "float16"}:
            cfg.dtype = "float32"
        if bool(cfg.consensus_safe):
            cfg.deterministic = True
            if cfg.rng_seed is None:
                raise ValueError("consensus_safe=True requires rng_seed for deterministic construction")
            if cfg.dtype not in {"float32", "float64"}:
                raise ValueError("consensus_safe=True requires dtype float32 or float64")
        if bool(cfg.deterministic) and cfg.rng_seed is None:
            raise ValueError("deterministic=True requires rng_seed")
        cfg.energy_tol = float(cfg.energy_tol)
        cfg.input_clip = float(max(_EPS, cfg.input_clip))
        cfg.state_clip = float(max(_EPS, cfg.state_clip))
        cfg.preactivation_clip = float(max(0.0, cfg.preactivation_clip))
        cfg.time_scale = float(max(_EPS, cfg.time_scale))
        cfg.leaky_lambda = float(np.clip(cfg.leaky_lambda, 0.0, 1.0))
        return cfg


# =============================================================================
# Production Tensorized CfC cell
# =============================================================================


class TCfC_Cell:
    """Tensorized Closed-form Continuous-time cell."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bond_dim: Union[int, Sequence[int]] = 8,
        rng: Optional[np.random.Generator] = None,
        dtype: np.dtype = np.float32,
        *,
        tt_strict: bool = True,
        force_dense: Optional[bool] = None,
        energy_tol: float = 0.999,
        input_clip: float = 5.0,
        state_clip: float = 1.0,
        preactivation_clip: float = 10.0,
        candidate_activation: str = "tanh",
        backbone_activation: str = "tanh",
        time_scale: float = 1.0,
        leaky_lambda: float = 1.0,
        strict_runtime_checks: bool = True,
        deterministic: bool = False,
        rng_seed: Optional[int] = None,
        consensus_safe: bool = False,
        cell_index: int = 0,
        policy_id: str = "",
    ):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.bond_dim = bond_dim
        self.dtype = np.dtype(dtype)
        self.tt_strict = bool(tt_strict)
        self.energy_tol = float(energy_tol)
        self.input_clip = float(input_clip)
        self.state_clip = float(state_clip)
        self.preactivation_clip = float(preactivation_clip)
        self.candidate_activation = str(candidate_activation).lower().strip()
        self.backbone_activation = str(backbone_activation).lower().strip()
        self.time_scale = float(max(_EPS, time_scale))
        self.leaky_lambda = float(np.clip(leaky_lambda, 0.0, 1.0))
        self.strict_runtime_checks = bool(strict_runtime_checks)
        self.deterministic = bool(deterministic)
        self.rng_seed = None if rng_seed is None else int(rng_seed)
        self.consensus_safe = bool(consensus_safe)
        self.cell_index = int(cell_index)
        self.policy_id = str(policy_id or "tcfc_policy")

        if self.input_size <= 0 or self.hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        if self.consensus_safe:
            self.deterministic = True
            if self.rng_seed is None:
                raise ValueError("consensus_safe cell requires rng_seed")
            if str(self.dtype) not in {"float32", "float64"}:
                raise ValueError("consensus_safe cell requires float32 or float64")

        if rng is not None:
            self.rng = rng
        elif self.rng_seed is not None:
            self.rng = np.random.default_rng(self.rng_seed + 1009 * self.cell_index)
        else:
            if self.deterministic:
                raise ValueError("deterministic=True requires rng_seed or rng")
            self.rng = np.random.default_rng()

        env_force_dense = False if self.consensus_safe else str(os.getenv("CFC_FORCE_DENSE", "")).strip().lower() in {"1", "true", "yes"}
        self.force_dense = bool(force_dense) if force_dense is not None else False
        self._force_dense_effective = bool(self.force_dense or env_force_dense)

        self.combined_size = self.input_size + self.hidden_size
        in_f1, in_f2 = two_factors(self.combined_size)
        out_f1, out_f2 = two_factors(self.hidden_size)
        _check_factors(self.combined_size, in_f1, in_f2, "TCfC_Cell.backbone_input")
        _check_factors(self.hidden_size, out_f1, out_f2, "TCfC_Cell.backbone_output")
        self._tt_in_shape = (int(in_f1), int(in_f2))
        self._tt_out_shape = (int(out_f1), int(out_f2))
        self._tt_bond_dims = _bond_dims_from_spec(self.bond_dim, len(self._tt_in_shape))

        self.backbone: Optional[Any] = None
        self.dense_backbone: Optional[np.ndarray] = None
        self._use_tt = bool(TENSORTRAIN_AVAILABLE) and not self._force_dense_effective
        self._tt_init_error: Optional[str] = None
        self._backbone_param_count = 0
        self._init_backbone()
        self._init_heads()

        self._last_backbone_out = np.zeros((self.hidden_size,), dtype=self.dtype)
        self._last_f = np.zeros((self.hidden_size,), dtype=self.dtype)
        self._last_g = np.zeros((self.hidden_size,), dtype=self.dtype)
        self._last_h = np.zeros((self.hidden_size,), dtype=self.dtype)
        self._last_gate = np.zeros((self.hidden_size,), dtype=self.dtype)
        self._last_candidate = np.zeros((self.hidden_size,), dtype=self.dtype)
        self._last_update_mix = 0.0
        self._last_dt = 0.0

    def _init_backbone(self) -> None:
        dense_init = _stable_dense_init_rect(self.hidden_size, self.combined_size, self.rng, self.dtype, base_scale=0.9, max_sv=0.95)
        if self._force_dense_effective:
            self._use_tt = False
            self.dense_backbone = dense_init
            self.backbone = None
            self._backbone_param_count = int(self.dense_backbone.size)
            return
        if not TENSORTRAIN_AVAILABLE:
            msg = "TensorTrain unavailable"
            if TT_IMPORT_ERROR is not None:
                msg += f": {TT_IMPORT_ERROR}"
            self._tt_init_error = msg
            if self.tt_strict:
                raise RuntimeError(msg)
            self._use_tt = False
            self.dense_backbone = dense_init
            self.backbone = None
            self._backbone_param_count = int(self.dense_backbone.size)
            return
        try:
            max_rank = max(self._tt_bond_dims) if self._tt_bond_dims else int(max(1, self.bond_dim if isinstance(self.bond_dim, int) else 4))
            tt = tensor_train_svd(
                dense_init,
                output_dims=list(self._tt_out_shape),
                input_dims=list(self._tt_in_shape),
                max_bond_dim=int(max_rank),
                dtype=self.dtype,
                energy_tol=self.energy_tol,
                device="cpu",
                check_finite=True,
                seed=self.rng_seed,
                deterministic=self.deterministic,
                consensus_safe=self.consensus_safe,
                metadata={"source": "TCfC_Cell", "policy_id": self.policy_id, "cell_index": self.cell_index},
                commit_metadata={"lobe": "cfc", "policy_id": self.policy_id, "cell_index": self.cell_index},
            )
            if tt is None:
                raise RuntimeError("tensor_train_svd returned None")
            self.backbone = tt
            self.dense_backbone = None
            self._use_tt = True
            self._backbone_param_count = int(_tt_param_count(tt) or 0)
        except TypeError:
            # Compatibility with older tn.py lacking production keyword args.
            try:
                tt = tensor_train_svd(
                    dense_init,
                    output_dims=list(self._tt_out_shape),
                    input_dims=list(self._tt_in_shape),
                    max_bond_dim=int(max(self._tt_bond_dims) if self._tt_bond_dims else 4),
                    dtype=self.dtype,
                    energy_tol=self.energy_tol,
                    device="cpu",
                    check_finite=True,
                )
                if tt is None:
                    raise RuntimeError("tensor_train_svd returned None")
                self.backbone = tt
                self.dense_backbone = None
                self._use_tt = True
                self._backbone_param_count = int(_tt_param_count(tt) or 0)
            except Exception as exc:
                self._handle_tt_init_failure(exc, dense_init)
        except Exception as exc:
            self._handle_tt_init_failure(exc, dense_init)

    def _handle_tt_init_failure(self, exc: Exception, dense_init: np.ndarray) -> None:
        msg = f"TensorTrain backbone init failed: {exc}"
        self._tt_init_error = msg
        if self.tt_strict:
            raise RuntimeError(msg) from exc
        self._use_tt = False
        self.backbone = None
        self.dense_backbone = dense_init
        self._backbone_param_count = int(self.dense_backbone.size)

    def _init_heads(self) -> None:
        self.f_head = _stable_dense_init_rect(self.hidden_size, self.hidden_size, self.rng, self.dtype, base_scale=0.7, max_sv=0.9).T
        self.g_head = _stable_dense_init_rect(self.hidden_size, self.hidden_size, self.rng, self.dtype, base_scale=0.8, max_sv=0.9).T
        self.h_head = _stable_dense_init_rect(self.hidden_size, self.hidden_size, self.rng, self.dtype, base_scale=0.8, max_sv=0.9).T
        self.f_bias = np.zeros(self.hidden_size, dtype=self.dtype)
        self.g_bias = np.zeros(self.hidden_size, dtype=self.dtype)
        self.h_bias = np.zeros(self.hidden_size, dtype=self.dtype)

    def _backbone_apply(self, full_input: np.ndarray) -> np.ndarray:
        x = _coerce_vector(full_input, expected_dim=self.combined_size, dtype=self.dtype, name="full_input", clip=self.input_clip)
        if self._use_tt:
            if self.backbone is None:
                raise RuntimeError("TT backbone is not initialized")
            y = _tt_apply(self.backbone, x, out_dim=self.hidden_size, dtype=self.dtype, strict=True)
        else:
            if self.dense_backbone is None:
                raise RuntimeError("Dense backbone is not initialized")
            y = self.dense_backbone @ x
            y = _sanitize(y, self.dtype)
        return _activation(y.astype(self.dtype, copy=False), self.backbone_activation)

    def forward(self, input_vec: Any, prev_state: Any, time_delta: float, *, return_aux: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        x = _coerce_vector(input_vec, expected_dim=self.input_size, dtype=self.dtype, name="input_vec", clip=self.input_clip)
        prev = _coerce_vector(prev_state, expected_dim=self.hidden_size, dtype=self.dtype, name="prev_state", clip=self.state_clip)
        dt = _clamp_dt(time_delta)
        self._last_dt = float(dt)
        if dt <= 0.0:
            next_state = prev.copy()
            aux = self._aux(next_state=next_state, dt=dt, skipped=True)
            return (next_state, aux) if return_aux else next_state
        full_input = np.concatenate([x, prev]).astype(self.dtype, copy=False)
        z = self._backbone_apply(full_input)
        f_out = z @ self.f_head + self.f_bias
        g_out = z @ self.g_head + self.g_bias
        h_out = z @ self.h_head + self.h_bias
        if self.preactivation_clip > 0:
            f_out = np.clip(f_out, -self.preactivation_clip, self.preactivation_clip).astype(self.dtype, copy=False)
            g_out = np.clip(g_out, -self.preactivation_clip, self.preactivation_clip).astype(self.dtype, copy=False)
            h_out = np.clip(h_out, -self.preactivation_clip, self.preactivation_clip).astype(self.dtype, copy=False)
        g_act = _activation(g_out, self.candidate_activation)
        h_act = _activation(h_out, self.candidate_activation)
        gate = sigmoid(np.clip(-f_out * self.dtype.type(dt), -20.0, 20.0))
        candidate = gate * g_act + (self.dtype.type(1.0) - gate) * h_act
        candidate = _sanitize(candidate, self.dtype)
        temporal_mix = 1.0 - math.exp(-float(dt) / self.time_scale)
        update_mix = float(np.clip(self.leaky_lambda * temporal_mix, 0.0, 1.0))
        next_state = ((1.0 - update_mix) * prev + update_mix * candidate).astype(self.dtype, copy=False)
        if self.state_clip > 0:
            next_state = np.clip(next_state, -self.state_clip, self.state_clip).astype(self.dtype, copy=False)
        next_state = _sanitize(next_state, self.dtype)
        self._last_backbone_out = z.copy()
        self._last_f = _sanitize(f_out, self.dtype)
        self._last_g = _sanitize(g_act, self.dtype)
        self._last_h = _sanitize(h_act, self.dtype)
        self._last_gate = _sanitize(gate, self.dtype)
        self._last_candidate = _sanitize(candidate, self.dtype)
        self._last_update_mix = float(update_mix)
        if self.strict_runtime_checks and _count_nonfinite(next_state) > 0:
            raise FloatingPointError("Non-finite TCfC cell state encountered")
        aux = self._aux(next_state=next_state, dt=dt, skipped=False)
        return (next_state, aux) if return_aux else next_state

    def _aux(self, *, next_state: np.ndarray, dt: float, skipped: bool) -> Dict[str, Any]:
        return {
            "recurrent_mode": "tt" if self._use_tt else "dense",
            "tt_init_error": self._tt_init_error,
            "dt": float(dt),
            "skipped": bool(skipped),
            "update_mix": float(self._last_update_mix),
            "backbone_out": self._last_backbone_out.copy(),
            "f": self._last_f.copy(),
            "g": self._last_g.copy(),
            "h": self._last_h.copy(),
            "gate": self._last_gate.copy(),
            "candidate": self._last_candidate.copy(),
            "next_state_norm": float(np.linalg.norm(next_state)),
            "backbone_commitment_hash": self.backbone_commitment_hash(),
        }

    def backbone_commitment_hash(self) -> str:
        if self._use_tt and self.backbone is not None:
            return _tt_commitment_hash(self.backbone)
        if self.dense_backbone is not None:
            return _array_hash(self.dense_backbone, dtype=self.dtype, tag=f"dense_backbone:{self.policy_id}:{self.cell_index}")
        return ""

    def head_hashes(self) -> List[str]:
        return [
            _array_hash(self.f_head, dtype=self.dtype, tag=f"f_head:{self.cell_index}"),
            _array_hash(self.g_head, dtype=self.dtype, tag=f"g_head:{self.cell_index}"),
            _array_hash(self.h_head, dtype=self.dtype, tag=f"h_head:{self.cell_index}"),
            _array_hash(self.f_bias, dtype=self.dtype, tag=f"f_bias:{self.cell_index}"),
            _array_hash(self.g_bias, dtype=self.dtype, tag=f"g_bias:{self.cell_index}"),
            _array_hash(self.h_bias, dtype=self.dtype, tag=f"h_bias:{self.cell_index}"),
        ]

    def parameter_count(self) -> int:
        return int(self._backbone_param_count + self.f_head.size + self.g_head.size + self.h_head.size + self.f_bias.size + self.g_bias.size + self.h_bias.size)

    def dense_equivalent_parameter_count(self) -> int:
        return int(self.combined_size * self.hidden_size + 3 * (self.hidden_size * self.hidden_size + self.hidden_size))

    def estimate_backbone_norm(self, *, iters: int = 12) -> Optional[float]:
        if not self._use_tt and self.dense_backbone is not None:
            return _estimate_dense_spectral_norm(self.dense_backbone, iters=iters)
        if self._use_tt and self.backbone is not None:
            try:
                rng = np.random.default_rng(2027)
                gains = []
                for _ in range(max(1, int(iters))):
                    v = rng.normal(size=(self.combined_size,)).astype(self.dtype)
                    v /= np.linalg.norm(v) + _EPS
                    y = self._backbone_apply(v)
                    gains.append(float(np.linalg.norm(y)))
                return float(np.mean(gains)) if gains else None
            except Exception:
                return None
        return None

    def apply_regularization(self, target_norm: float = 1.0) -> None:
        target_norm = float(target_norm)
        if target_norm <= 0:
            raise ValueError("target_norm must be positive")
        if not self._use_tt and self.dense_backbone is not None:
            n = _estimate_dense_spectral_norm(self.dense_backbone)
            if n is not None and n > target_norm and n > 0:
                self.dense_backbone *= self.dtype.type(target_norm / n)
            return
        if self._use_tt and self.backbone is not None:
            n = self.estimate_backbone_norm()
            if n is not None and n > target_norm and n > 0 and hasattr(self.backbone, "scale"):
                self.backbone = self.backbone.scale(float(target_norm / n))
                self._backbone_param_count = int(_tt_param_count(self.backbone) or self._backbone_param_count)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "input_size": int(self.input_size),
            "hidden_size": int(self.hidden_size),
            "combined_size": int(self.combined_size),
            "recurrent_mode": "tt" if self._use_tt else "dense",
            "tt_available": bool(TENSORTRAIN_AVAILABLE),
            "tt_in_shape": list(self._tt_in_shape),
            "tt_out_shape": list(self._tt_out_shape),
            "tt_bond_dims": list(self._tt_bond_dims),
            "tt_init_error": self._tt_init_error,
            "parameter_count": int(self.parameter_count()),
            "dense_equivalent_parameter_count": int(self.dense_equivalent_parameter_count()),
            "last_dt": float(self._last_dt),
            "last_update_mix": float(self._last_update_mix),
            "backbone_commitment_hash": self.backbone_commitment_hash(),
        }


# =============================================================================
# Production TCfC policy
# =============================================================================


class TCfC_Policy:
    """Multi-cell Tensorized Closed-form Continuous-time policy."""

    CHECKPOINT_VERSION = 4

    def __init__(
        self,
        policy_id: str,
        input_size: int,
        hidden_size: int,
        num_cells: int,
        bond_dim: Union[int, Sequence[int]] = 8,
        rng: Optional[np.random.Generator] = None,
        dtype: np.dtype = np.float32,
        verbose: bool = True,
        *,
        tt_strict: bool = True,
        force_dense: Optional[bool] = None,
        energy_tol: float = 0.999,
        input_clip: float = 5.0,
        state_clip: float = 1.0,
        preactivation_clip: float = 10.0,
        candidate_activation: str = "tanh",
        backbone_activation: str = "tanh",
        time_scale: float = 1.0,
        leaky_lambda: float = 1.0,
        strict_runtime_checks: bool = True,
        deterministic: bool = False,
        rng_seed: Optional[int] = None,
        consensus_safe: bool = False,
        scale_profile: Optional[LobeScaleProfile] = None,
    ):
        cfg = TCfCPolicyConfig(
            policy_id=str(policy_id),
            input_size=int(input_size),
            hidden_size=int(hidden_size),
            num_cells=int(num_cells),
            bond_dim=list(bond_dim) if isinstance(bond_dim, Sequence) and not isinstance(bond_dim, (str, bytes)) else int(bond_dim),
            dtype=str(np.dtype(dtype)),
            tt_strict=bool(tt_strict),
            force_dense=bool(force_dense) if force_dense is not None else False,
            energy_tol=float(energy_tol),
            input_clip=float(input_clip),
            state_clip=float(state_clip),
            preactivation_clip=float(preactivation_clip),
            candidate_activation=str(candidate_activation),
            backbone_activation=str(backbone_activation),
            time_scale=float(time_scale),
            leaky_lambda=float(leaky_lambda),
            strict_runtime_checks=bool(strict_runtime_checks),
            verbose=bool(verbose),
            rng_seed=None if rng_seed is None else int(rng_seed),
            deterministic=bool(deterministic),
            consensus_safe=bool(consensus_safe),
        ).normalized()
        self.config = cfg
        self.policy_id = cfg.policy_id
        self.lobe_id = f"cfc:{self.policy_id}"
        self.input_size = cfg.input_size
        self.hidden_size = cfg.hidden_size
        self.num_cells = cfg.num_cells
        self.bond_dim = cfg.bond_dim
        self.dtype = np.dtype(cfg.dtype)
        self.verbose = cfg.verbose
        self.tt_strict = cfg.tt_strict
        self.force_dense = cfg.force_dense
        self.energy_tol = cfg.energy_tol
        self.input_clip = cfg.input_clip
        self.state_clip = cfg.state_clip
        self.preactivation_clip = cfg.preactivation_clip
        self.candidate_activation = cfg.candidate_activation
        self.backbone_activation = cfg.backbone_activation
        self.time_scale = cfg.time_scale
        self.leaky_lambda = cfg.leaky_lambda
        self.strict_runtime_checks = cfg.strict_runtime_checks
        self.deterministic = cfg.deterministic
        self.rng_seed = cfg.rng_seed
        self.consensus_safe = cfg.consensus_safe
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(self.rng_seed if self.rng_seed is not None else None)
        if self.consensus_safe and self.force_dense and not self.deterministic:
            raise ValueError("consensus_safe force_dense mode still requires deterministic construction")

        self.scale_profile = (scale_profile or LobeScaleProfile(
            lobe_id=self.lobe_id,
            hidden_size=self.hidden_size,
            num_cells=self.num_cells,
            tt_bond_rank=max(_bond_dims_from_spec(self.bond_dim, 2) or [int(self.bond_dim) if isinstance(self.bond_dim, int) else 8]),
            precision=str(self.dtype),
            consensus_safe=self.consensus_safe,
        )).normalized()

        self.cells: List[TCfC_Cell] = [
            TCfC_Cell(
                self.input_size,
                self.hidden_size,
                bond_dim=self.bond_dim,
                rng=self.rng if self.rng_seed is None else None,
                dtype=self.dtype,
                tt_strict=self.tt_strict,
                force_dense=self.force_dense,
                energy_tol=self.energy_tol,
                input_clip=self.input_clip,
                state_clip=self.state_clip,
                preactivation_clip=self.preactivation_clip,
                candidate_activation=self.candidate_activation,
                backbone_activation=self.backbone_activation,
                time_scale=self.time_scale,
                leaky_lambda=self.leaky_lambda,
                strict_runtime_checks=self.strict_runtime_checks,
                deterministic=self.deterministic,
                rng_seed=(None if self.rng_seed is None else int(self.rng_seed) + 7919 * i),
                consensus_safe=self.consensus_safe,
                cell_index=i,
                policy_id=self.policy_id,
            )
            for i in range(self.num_cells)
        ]

        self.state = np.zeros((self.hidden_size * self.num_cells,), dtype=self.dtype)
        self._step_count = 0
        self._created_at = _utc_ts()
        self._last_input = np.zeros((self.input_size,), dtype=self.dtype)
        self._last_cell_aux: List[Dict[str, Any]] = []
        self._last_time_delta = 0.0
        self._last_transition_receipt: Optional[CfCTransitionReceipt] = None
        self._last_error_report: Optional[ForecastErrorReport] = None

        if self.verbose:
            print(
                f"[TCfC_Policy:{self.policy_id}] cells={self.num_cells} hidden={self.hidden_size} "
                f"input={self.input_size} modes={self.mode_counts()} params={self.parameter_count():,} "
                f"dense_equiv={self.equivalent_dense_parameter_count():,}"
            )

    # ------------------------------------------------------------------
    # State API
    # ------------------------------------------------------------------
    def reset_state(self, value: float = 0.0) -> None:
        v = self.dtype.type(value)
        self.state.fill(v)
        if self.state_clip > 0:
            np.clip(self.state, -self.state_clip, self.state_clip, out=self.state)
        self._step_count = 0
        self._last_input.fill(0)
        self._last_cell_aux = []
        self._last_time_delta = 0.0
        self._last_transition_receipt = None
        self._last_error_report = None

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def set_state(self, new_state: Any) -> None:
        self.state = _coerce_vector(new_state, expected_dim=self.state.size, dtype=self.dtype, name="new_state", clip=self.state_clip)

    def state_hash(self, state: Optional[Any] = None) -> str:
        return _array_hash(self.state if state is None else state, dtype=self.dtype, tag=f"cfc_state:{self.policy_id}")

    def input_hash(self, x: Any) -> str:
        return _array_hash(_coerce_vector(x, expected_dim=self.input_size, dtype=self.dtype, name="input"), dtype=self.dtype, tag=f"cfc_input:{self.policy_id}")

    def serialize_state(self) -> Dict[str, Any]:
        return {
            "schema": "TCfCPolicy.runtime_state.v2",
            "policy_id": self.policy_id,
            "kind": "tcfc_policy_state",
            "input_size": int(self.input_size),
            "hidden_size": int(self.hidden_size),
            "num_cells": int(self.num_cells),
            "dtype": str(self.dtype),
            "state": self.state.astype(np.float32).tolist(),
            "state_hash": self.state_hash(),
            "step_count": int(self._step_count),
            "last_time_delta": float(self._last_time_delta),
            "mode_counts": self.mode_counts(),
            "scale_profile_hash": self.scale_profile.compute_hash(),
            "serialized_at": _utc_ts(),
        }

    def load_state(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        if int(payload.get("hidden_size", self.hidden_size)) != self.hidden_size:
            raise ValueError("hidden_size mismatch in TCfC state payload")
        if int(payload.get("num_cells", self.num_cells)) != self.num_cells:
            raise ValueError("num_cells mismatch in TCfC state payload")
        state = payload.get("state", None)
        if state is None:
            raise ValueError("payload missing 'state'")
        self.set_state(state)
        self._step_count = int(payload.get("step_count", self._step_count))
        self._last_time_delta = float(payload.get("last_time_delta", self._last_time_delta))

    # ------------------------------------------------------------------
    # Core update / forecast API
    # ------------------------------------------------------------------
    def _next_state_from(self, current_state: np.ndarray, x: np.ndarray, dt: float, *, wants_aux: bool = False) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        state_matrix = current_state.reshape(self.num_cells, self.hidden_size)
        new_states = np.empty_like(state_matrix)
        cell_aux: List[Dict[str, Any]] = []
        for i, cell in enumerate(self.cells):
            if wants_aux:
                ns, aux = cell.forward(x, state_matrix[i], dt, return_aux=True)  # type: ignore[misc]
                cell_aux.append(aux)
            else:
                ns = cell.forward(x, state_matrix[i], dt, return_aux=False)  # type: ignore[assignment]
            ns_arr = np.asarray(ns, dtype=self.dtype).reshape(-1)
            if ns_arr.size != self.hidden_size:
                raise RuntimeError(f"TCfC cell {i} returned size {ns_arr.size}; expected {self.hidden_size}")
            new_states[i, :] = ns_arr
        out = new_states.reshape(-1).astype(self.dtype, copy=False)
        out = _sanitize(out, self.dtype)
        if self.state_clip > 0:
            np.clip(out, -self.state_clip, self.state_clip, out=out)
        return out, cell_aux

    def forecast(self, external_input: Any, time_delta: float = 1.0, *, from_state: Optional[Any] = None, return_receipt: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, CfCTransitionReceipt]]:
        x = _coerce_vector(external_input, expected_dim=self.input_size, dtype=self.dtype, name="external_input", clip=self.input_clip)
        dt = _clamp_dt(time_delta)
        base_state = self.state.copy() if from_state is None else _coerce_vector(from_state, expected_dim=self.state.size, dtype=self.dtype, name="from_state", clip=self.state_clip)
        # Preserve runtime cache by restoring after deterministic dry-run.
        predicted, _ = self._next_state_from(base_state, x, dt, wants_aux=False)
        if return_receipt:
            receipt = self._make_transition_receipt(
                x=x,
                dt=dt,
                pre_state=base_state,
                predicted_state=predicted,
                actual_state=None,
                forecast_error=None,
            )
            return predicted, receipt
        return predicted

    def step(self, external_input: Any, time_delta: float = 1.0, *, return_state: bool = False, return_aux: Optional[bool] = None) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        wants_aux = bool(return_state if return_aux is None else return_aux)
        x = _coerce_vector(external_input, expected_dim=self.input_size, dtype=self.dtype, name="external_input", clip=self.input_clip)
        dt = _clamp_dt(time_delta)
        pre_state = self.state.copy()
        self._step_count += 1
        self._last_input = x.copy()
        self._last_time_delta = float(dt)
        predicted_state, cell_aux = self._next_state_from(pre_state, x, dt, wants_aux=wants_aux)
        self.state = predicted_state
        self._last_cell_aux = cell_aux
        if self.strict_runtime_checks and _count_nonfinite(self.state) > 0:
            raise FloatingPointError(f"[{self.policy_id}] non-finite TCfC policy state encountered")
        receipt = self._make_transition_receipt(x=x, dt=dt, pre_state=pre_state, predicted_state=self.state, actual_state=self.state, forecast_error=None)
        self._last_transition_receipt = receipt
        if wants_aux:
            aux_out = {
                "policy_id": self.policy_id,
                "hidden_state": self.state.copy(),
                "control_vector": self.get_control_vector(out_dim=min(32, self.hidden_size)),
                "cell_means": self.get_cell_means(),
                "cell_norms": self.get_norms(),
                "cell_aux": cell_aux,
                "transition_receipt": receipt.to_dict(),
                "runtime_flags": {
                    "mode_counts": self.mode_counts(),
                    "strict_runtime_checks": bool(self.strict_runtime_checks),
                    "tt_available": bool(TENSORTRAIN_AVAILABLE),
                    "deterministic": bool(self.deterministic),
                    "consensus_safe": bool(self.consensus_safe),
                },
                "health": self.health_metrics(),
            }
            return self.state.copy(), aux_out
        return self.state.copy()

    def _make_transition_receipt(
        self,
        *,
        x: np.ndarray,
        dt: float,
        pre_state: np.ndarray,
        predicted_state: np.ndarray,
        actual_state: Optional[np.ndarray],
        forecast_error: Optional[ForecastErrorReport],
    ) -> CfCTransitionReceipt:
        health = self.health_metrics(include_expensive=False)
        health_hash = _sha256_json(health)
        control = self.get_control_vector(out_dim=min(32, self.hidden_size), state=predicted_state)
        receipt = CfCTransitionReceipt(
            scheme=CFC_TRANSITION_RECEIPT_SCHEME,
            version=CFC_VERSION,
            policy_id=self.policy_id,
            lobe_id=self.lobe_id,
            step_count=int(self._step_count),
            time_delta=float(dt),
            input_hash=_array_hash(x, dtype=self.dtype, tag=f"cfc_input:{self.policy_id}"),
            pre_state_hash=_array_hash(pre_state, dtype=self.dtype, tag=f"cfc_pre_state:{self.policy_id}"),
            predicted_state_hash=_array_hash(predicted_state, dtype=self.dtype, tag=f"cfc_predicted_state:{self.policy_id}"),
            actual_state_hash="" if actual_state is None else _array_hash(actual_state, dtype=self.dtype, tag=f"cfc_actual_state:{self.policy_id}"),
            forecast_error=None if forecast_error is None else forecast_error.to_dict(),
            control_hash=_array_hash(control, dtype=self.dtype, tag=f"cfc_control:{self.policy_id}"),
            health_hash=health_hash,
            manifest_hash=self.lobe_commitment_hash(),
            scale_profile_hash=self.scale_profile.compute_hash(),
            tensor_backbone_root=self.tensor_backbone_root(),
        )
        return CfCTransitionReceipt(**{**receipt.payload(), "receipt_hash": receipt.compute_hash()})

    def compare_forecast_actual(
        self,
        predicted_state: Any,
        actual_state: Any,
        *,
        tolerance: float = 1e-3,
        input_vector: Optional[Any] = None,
        time_delta: Optional[float] = None,
        pre_state: Optional[Any] = None,
    ) -> ForecastErrorReport:
        report = ForecastErrorReport.compare(predicted_state, actual_state, tolerance=tolerance, dtype=self.dtype)
        self._last_error_report = report
        if input_vector is not None and time_delta is not None:
            x = _coerce_vector(input_vector, expected_dim=self.input_size, dtype=self.dtype, name="input_vector", clip=self.input_clip)
            pre = self.state.copy() if pre_state is None else _coerce_vector(pre_state, expected_dim=self.state.size, dtype=self.dtype, name="pre_state", clip=self.state_clip)
            receipt = self._make_transition_receipt(
                x=x,
                dt=_clamp_dt(time_delta),
                pre_state=pre,
                predicted_state=_coerce_vector(predicted_state, expected_dim=self.state.size, dtype=self.dtype, name="predicted_state"),
                actual_state=_coerce_vector(actual_state, expected_dim=self.state.size, dtype=self.dtype, name="actual_state"),
                forecast_error=report,
            )
            self._last_transition_receipt = receipt
        return report

    # ------------------------------------------------------------------
    # Readouts / diagnostics
    # ------------------------------------------------------------------
    def get_control_vector(self, out_dim: int = 32, *, state: Optional[Any] = None) -> np.ndarray:
        out_dim = int(max(1, out_dim))
        src = self.state if state is None else _coerce_vector(state, expected_dim=self.state.size, dtype=self.dtype, name="state")
        state_mat = src.reshape(self.num_cells, self.hidden_size)
        summary = state_mat.mean(axis=0).astype(self.dtype, copy=False)
        if out_dim <= summary.size:
            return summary[:out_dim].astype(self.dtype, copy=False)
        out = np.zeros((out_dim,), dtype=self.dtype)
        out[: summary.size] = summary
        return out

    def get_cell_means(self) -> np.ndarray:
        state_mat = self.state.reshape(self.num_cells, self.hidden_size)
        return state_mat.mean(axis=1).astype(self.dtype, copy=False)

    def get_norms(self) -> np.ndarray:
        state_mat = self.state.reshape(self.num_cells, self.hidden_size)
        return np.linalg.norm(state_mat, axis=1).astype(self.dtype, copy=False)

    def mode_counts(self) -> Dict[str, int]:
        tt_count = sum(1 for c in self.cells if c._use_tt)
        return {"tt": int(tt_count), "dense": int(self.num_cells - tt_count)}

    def parameter_count(self) -> int:
        return int(sum(c.parameter_count() for c in self.cells))

    def equivalent_dense_parameter_count(self) -> int:
        return int(sum(c.dense_equivalent_parameter_count() for c in self.cells))

    def compression_factor(self) -> Optional[float]:
        params = self.parameter_count()
        dense = self.equivalent_dense_parameter_count()
        if params <= 0:
            return None
        return float(dense / params)

    def tensor_backbone_commitments(self) -> List[str]:
        return [c.backbone_commitment_hash() for c in self.cells]

    def tensor_backbone_root(self) -> str:
        return _sha256_json({"domain": "TCfCBackboneRoot", "commitments": self.tensor_backbone_commitments()})

    def dense_backbone_hashes(self) -> List[str]:
        out = []
        for i, c in enumerate(self.cells):
            if c.dense_backbone is not None:
                out.append(_array_hash(c.dense_backbone, dtype=self.dtype, tag=f"dense_backbone:{self.policy_id}:{i}"))
        return out

    def head_hashes(self) -> List[str]:
        out: List[str] = []
        for c in self.cells:
            out.extend(c.head_hashes())
        return out

    def config_hash(self) -> str:
        return _sha256_json(self.config.normalized())

    def lobe_commitment_manifest(self, *, checkpoint_hash: str = "") -> CfCLobeCommitmentManifest:
        manifest = CfCLobeCommitmentManifest(
            scheme=CFC_COMMITMENT_SCHEME,
            version=CFC_VERSION,
            policy_id=self.policy_id,
            lobe_id=self.lobe_id,
            config_hash=self.config_hash(),
            state_hash=self.state_hash(),
            scale_profile_hash=self.scale_profile.compute_hash(),
            tensor_backbone_root=self.tensor_backbone_root(),
            tensor_backbone_commitments=self.tensor_backbone_commitments(),
            dense_backbone_hashes=self.dense_backbone_hashes(),
            head_hashes=self.head_hashes(),
            parameter_count=self.parameter_count(),
            dense_equivalent_parameter_count=self.equivalent_dense_parameter_count(),
            compression_factor=self.compression_factor(),
            consensus_safe=bool(self.consensus_safe),
            checkpoint_hash=str(checkpoint_hash or ""),
        )
        return CfCLobeCommitmentManifest(**{**manifest.payload(), "manifest_hash": manifest.compute_hash()})

    def lobe_commitment_hash(self) -> str:
        return self.lobe_commitment_manifest().compute_hash()

    def verify_lobe_commitment(self, manifest: Union[CfCLobeCommitmentManifest, Mapping[str, Any]]) -> bool:
        if isinstance(manifest, Mapping):
            p = dict(manifest)
            expected_hash = str(p.get("manifest_hash", ""))
            expected_checkpoint_hash = str(p.get("checkpoint_hash", ""))
        else:
            expected_hash = manifest.manifest_hash or manifest.compute_hash()
            expected_checkpoint_hash = manifest.checkpoint_hash
        current = self.lobe_commitment_manifest(checkpoint_hash=expected_checkpoint_hash)
        return current.compute_hash() == expected_hash

    def health_metrics(self, *, include_expensive: bool = False) -> Dict[str, Any]:
        state = self.state
        norms = self.get_norms()
        metrics: Dict[str, Any] = {
            "policy_id": self.policy_id,
            "lobe_id": self.lobe_id,
            "step_count": int(self._step_count),
            "mode_counts": self.mode_counts(),
            "state_norm": float(np.linalg.norm(state)),
            "state_mean": float(np.mean(state)) if state.size else 0.0,
            "state_std": float(np.std(state)) if state.size else 0.0,
            "state_min": float(np.min(state)) if state.size else 0.0,
            "state_max": float(np.max(state)) if state.size else 0.0,
            "cell_norm_min": float(np.min(norms)) if norms.size else 0.0,
            "cell_norm_mean": float(np.mean(norms)) if norms.size else 0.0,
            "cell_norm_max": float(np.max(norms)) if norms.size else 0.0,
            "state_saturation_fraction": _saturation_fraction(state, threshold=max(0.5, 0.98 * self.state_clip)),
            "nonfinite_state_count": _count_nonfinite(state),
            "last_input_norm": float(np.linalg.norm(self._last_input)),
            "last_time_delta": float(self._last_time_delta),
            "parameter_count": int(self.parameter_count()),
            "equivalent_dense_parameter_count": int(self.equivalent_dense_parameter_count()),
            "compression_factor": self.compression_factor(),
            "deterministic": bool(self.deterministic),
            "consensus_safe": bool(self.consensus_safe),
            "state_hash": self.state_hash(),
            "tensor_backbone_root": self.tensor_backbone_root(),
        }
        metrics["has_nan"] = bool(metrics["nonfinite_state_count"] > 0)
        metrics["is_saturated"] = bool(metrics["state_saturation_fraction"] >= 0.25)
        metrics["is_stable"] = bool(not metrics["has_nan"] and metrics["state_norm"] < max(1.0, self.state.size * 1.25))
        if include_expensive:
            norms_est: List[Optional[float]] = [c.estimate_backbone_norm() for c in self.cells]
            finite = [float(x) for x in norms_est if x is not None and np.isfinite(x)]
            metrics["backbone_norm_mean"] = float(np.mean(finite)) if finite else None
            metrics["backbone_norm_max"] = float(np.max(finite)) if finite else None
        return metrics

    def lobe_health_receipt(self, *, include_expensive: bool = False) -> LobeHealthReceipt:
        health = self.health_metrics(include_expensive=include_expensive)
        receipt = LobeHealthReceipt(
            lobe_id=self.lobe_id,
            lobe_kind="cfc_predictive_transition",
            step_count=self._step_count,
            health=health,
            scale_profile_hash=self.scale_profile.compute_hash(),
            state_hash=self.state_hash(),
            manifest_hash=self.lobe_commitment_hash(),
        )
        return LobeHealthReceipt(**{**receipt.payload(), "receipt_hash": receipt.compute_hash()})

    def snapshot(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "lobe_id": self.lobe_id,
            "kind": "tcfc_policy",
            "version": CFC_VERSION,
            "input_size": int(self.input_size),
            "hidden_size": int(self.hidden_size),
            "num_cells": int(self.num_cells),
            "dtype": str(self.dtype),
            "mode_counts": self.mode_counts(),
            "parameter_count": int(self.parameter_count()),
            "equivalent_dense_parameter_count": int(self.equivalent_dense_parameter_count()),
            "compression_factor": self.compression_factor(),
            "state_norm": float(np.linalg.norm(self.state)),
            "state_hash": self.state_hash(),
            "cell_norms": self.get_norms().astype(float).tolist(),
            "step_count": int(self._step_count),
            "created_at": self._created_at,
            "config": _json_safe(self.config.normalized()),
            "scale_profile": self.scale_profile.to_dict(),
            "lobe_commitment": self.lobe_commitment_manifest().to_dict(),
            "health_receipt": self.lobe_health_receipt().to_dict(),
            "last_transition_receipt": None if self._last_transition_receipt is None else self._last_transition_receipt.to_dict(),
            "last_forecast_error": None if self._last_error_report is None else self._last_error_report.to_dict(),
            "cells": [c.snapshot() for c in self.cells],
        }

    def apply_regularization(self, target_norm: float = 1.0) -> None:
        for c in self.cells:
            c.apply_regularization(target_norm=target_norm)

    def summary(self) -> str:
        cf = self.compression_factor()
        cf_s = "n/a" if cf is None else f"{cf:.2f}x"
        return (
            f"[TCfC Summary]\n"
            f"- id: {self.policy_id}\n"
            f"- lobe: {self.lobe_id}\n"
            f"- input: {self.input_size}\n"
            f"- hidden: {self.hidden_size}\n"
            f"- cells: {self.num_cells}\n"
            f"- dtype: {self.dtype}\n"
            f"- modes: {self.mode_counts()}\n"
            f"- params: {self.parameter_count():,}\n"
            f"- dense-equivalent params: {self.equivalent_dense_parameter_count():,}\n"
            f"- compression: {cf_s}\n"
            f"- manifest: {self.lobe_commitment_hash()[:16]}...\n"
        )

    # ------------------------------------------------------------------
    # Checkpointing with integrity manifest
    # ------------------------------------------------------------------
    def _checkpoint_arrays_without_manifest(self) -> Dict[str, Any]:
        arrays: Dict[str, Any] = {
            "checkpoint_version": np.asarray(self.CHECKPOINT_VERSION, dtype=np.int64),
            "format": np.asarray(CFC_CHECKPOINT_FORMAT),
            "config_json": np.asarray(_json_dumps_stable(self.config.normalized())),
            "scale_profile_json": np.asarray(_json_dumps_stable(self.scale_profile.to_dict())),
            "state": self.state.astype(self.dtype, copy=False),
            "step_count": np.asarray(self._step_count, dtype=np.int64),
            "last_time_delta": np.asarray(self._last_time_delta, dtype=np.float64),
            "created_at": np.asarray(self._created_at),
        }
        for i, cell in enumerate(self.cells):
            arrays[f"cell_{i}_f_head"] = cell.f_head.astype(self.dtype, copy=False)
            arrays[f"cell_{i}_g_head"] = cell.g_head.astype(self.dtype, copy=False)
            arrays[f"cell_{i}_h_head"] = cell.h_head.astype(self.dtype, copy=False)
            arrays[f"cell_{i}_f_bias"] = cell.f_bias.astype(self.dtype, copy=False)
            arrays[f"cell_{i}_g_bias"] = cell.g_bias.astype(self.dtype, copy=False)
            arrays[f"cell_{i}_h_bias"] = cell.h_bias.astype(self.dtype, copy=False)
            arrays[f"cell_{i}_use_tt"] = np.asarray(1 if cell._use_tt else 0, dtype=np.int8)
            arrays[f"cell_{i}_tt_in_shape"] = np.asarray(cell._tt_in_shape, dtype=np.int64)
            arrays[f"cell_{i}_tt_out_shape"] = np.asarray(cell._tt_out_shape, dtype=np.int64)
            arrays[f"cell_{i}_tt_bond_dims"] = np.asarray(cell._tt_bond_dims, dtype=np.int64)
            if cell._use_tt and cell.backbone is not None:
                cores = getattr(cell.backbone, "cores_data", None)
                if isinstance(cores, (list, tuple)):
                    arrays[f"cell_{i}_tt_num_cores"] = np.asarray(len(cores), dtype=np.int64)
                    for j, core in enumerate(cores):
                        arrays[f"cell_{i}_tt_core_{j}"] = np.asarray(_np_from_any_array(core), dtype=self.dtype)
                else:
                    arrays[f"cell_{i}_tt_num_cores"] = np.asarray(0, dtype=np.int64)
            else:
                arrays[f"cell_{i}_tt_num_cores"] = np.asarray(0, dtype=np.int64)
                if cell.dense_backbone is not None:
                    arrays[f"cell_{i}_dense_backbone"] = cell.dense_backbone.astype(self.dtype, copy=False)
        return arrays

    @staticmethod
    def _arrays_manifest_hash(arrays: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
        items = []
        for k in sorted(arrays.keys()):
            if str(k).startswith("__"):
                continue
            arr = np.asarray(arrays[k])
            items.append({
                "name": str(k),
                "shape": list(map(int, arr.shape)),
                "dtype": str(arr.dtype),
                "sha256": _sha256_bytes(np.ascontiguousarray(arr).tobytes(order="C")),
            })
        manifest = {"format": CFC_CHECKPOINT_FORMAT, "version": CFC_VERSION, "arrays": items}
        return manifest, _sha256_json(manifest)

    def save_checkpoint(self, path: Union[str, os.PathLike[str]]) -> None:
        arrays = self._checkpoint_arrays_without_manifest()
        manifest, checkpoint_hash = self._arrays_manifest_hash(arrays)
        lobe_manifest = self.lobe_commitment_manifest(checkpoint_hash=checkpoint_hash)
        manifest["checkpoint_hash"] = checkpoint_hash
        manifest["lobe_commitment"] = lobe_manifest.to_dict()
        manifest_hash = _sha256_json(manifest)
        arrays["__checkpoint_manifest_json__"] = np.asarray(_json_dumps_stable(manifest))
        arrays["__checkpoint_manifest_hash__"] = np.asarray(manifest_hash)
        arrays["__checkpoint_hash__"] = np.asarray(checkpoint_hash)
        _atomic_save_npz(path, arrays)

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, os.PathLike[str]],
        *,
        rng: Optional[np.random.Generator] = None,
        verbose: Optional[bool] = None,
        force_dense: Optional[bool] = None,
        tt_strict: Optional[bool] = None,
        verify: bool = True,
    ) -> "TCfC_Policy":
        p = Path(path)
        with np.load(p, allow_pickle=False) as data:
            if verify:
                if "__checkpoint_manifest_json__" not in data or "__checkpoint_manifest_hash__" not in data:
                    raise ValueError("checkpoint missing integrity manifest")
                manifest_json = str(data["__checkpoint_manifest_json__"].item())
                manifest_hash = str(data["__checkpoint_manifest_hash__"].item())
                manifest = json.loads(manifest_json)
                if _sha256_json(manifest) != manifest_hash:
                    raise ValueError("checkpoint manifest hash mismatch")
                arrays_for_hash = {k: np.asarray(data[k]) for k in data.files if not k.startswith("__")}
                _, actual_checkpoint_hash = cls._arrays_manifest_hash(arrays_for_hash)
                expected_checkpoint_hash = str(data["__checkpoint_hash__"].item()) if "__checkpoint_hash__" in data.files else str(manifest.get("checkpoint_hash", ""))
                if actual_checkpoint_hash != expected_checkpoint_hash:
                    raise ValueError("checkpoint array integrity hash mismatch")

            cfg = TCfCPolicyConfig(**json.loads(str(data["config_json"].item()))).normalized()
            dtype = np.dtype(cfg.dtype)
            obj = cls(
                policy_id=cfg.policy_id,
                input_size=cfg.input_size,
                hidden_size=cfg.hidden_size,
                num_cells=cfg.num_cells,
                bond_dim=cfg.bond_dim,
                rng=rng,
                dtype=dtype,
                verbose=bool(cfg.verbose) if verbose is None else bool(verbose),
                tt_strict=bool(cfg.tt_strict) if tt_strict is None else bool(tt_strict),
                force_dense=bool(cfg.force_dense) if force_dense is None else bool(force_dense),
                energy_tol=cfg.energy_tol,
                input_clip=cfg.input_clip,
                state_clip=cfg.state_clip,
                preactivation_clip=cfg.preactivation_clip,
                candidate_activation=cfg.candidate_activation,
                backbone_activation=cfg.backbone_activation,
                time_scale=cfg.time_scale,
                leaky_lambda=cfg.leaky_lambda,
                strict_runtime_checks=cfg.strict_runtime_checks,
                deterministic=cfg.deterministic,
                rng_seed=cfg.rng_seed,
                consensus_safe=cfg.consensus_safe,
            )
            if "scale_profile_json" in data.files:
                obj.scale_profile = LobeScaleProfile(**json.loads(str(data["scale_profile_json"].item()))).normalized()
            obj.state = np.asarray(data["state"], dtype=dtype).reshape(obj.hidden_size * obj.num_cells)
            obj._step_count = int(np.asarray(data["step_count"]).item()) if "step_count" in data.files else 0
            obj._last_time_delta = float(np.asarray(data["last_time_delta"]).item()) if "last_time_delta" in data.files else 0.0
            obj._created_at = str(data["created_at"].item()) if "created_at" in data.files else _utc_ts()

            for i, cell in enumerate(obj.cells):
                cell.f_head = np.asarray(data[f"cell_{i}_f_head"], dtype=dtype)
                cell.g_head = np.asarray(data[f"cell_{i}_g_head"], dtype=dtype)
                cell.h_head = np.asarray(data[f"cell_{i}_h_head"], dtype=dtype)
                cell.f_bias = np.asarray(data[f"cell_{i}_f_bias"], dtype=dtype)
                cell.g_bias = np.asarray(data[f"cell_{i}_g_bias"], dtype=dtype)
                cell.h_bias = np.asarray(data[f"cell_{i}_h_bias"], dtype=dtype)
                use_tt = bool(int(np.asarray(data[f"cell_{i}_use_tt"]).item())) and TENSORTRAIN_AVAILABLE and not bool(force_dense)
                cell._tt_in_shape = tuple(int(x) for x in np.asarray(data[f"cell_{i}_tt_in_shape"], dtype=np.int64).tolist())
                cell._tt_out_shape = tuple(int(x) for x in np.asarray(data[f"cell_{i}_tt_out_shape"], dtype=np.int64).tolist())
                cell._tt_bond_dims = [int(x) for x in np.asarray(data[f"cell_{i}_tt_bond_dims"], dtype=np.int64).tolist()]
                ncores = int(np.asarray(data[f"cell_{i}_tt_num_cores"]).item()) if f"cell_{i}_tt_num_cores" in data.files else 0
                if use_tt and ncores > 0:
                    cores = [np.asarray(data[f"cell_{i}_tt_core_{j}"], dtype=dtype) for j in range(ncores)]
                    try:
                        try:
                            cfg_tt = TTConfig(dtype=dtype, device="cpu", check_finite=True, seed=cfg.rng_seed, deterministic=cfg.deterministic, consensus_safe=cfg.consensus_safe)  # type: ignore[call-arg]
                        except TypeError:
                            cfg_tt = TTConfig(dtype=dtype, device="cpu")  # type: ignore[call-arg]
                        cell.backbone = TensorTrain(
                            output_dims=list(cell._tt_out_shape),
                            input_dims=list(cell._tt_in_shape),
                            cores_data=cores,
                            config=cfg_tt,
                            metadata={"source": "TCfC_Policy.load_checkpoint", "cell_index": i},
                        )
                        cell.dense_backbone = None
                        cell._use_tt = True
                        cell._backbone_param_count = int(_tt_param_count(cell.backbone) or 0)
                    except Exception:
                        if bool(tt_strict):
                            raise
                        cell._use_tt = False
                        cell.backbone = None
                        key = f"cell_{i}_dense_backbone"
                        cell.dense_backbone = np.asarray(data[key], dtype=dtype) if key in data.files else _stable_dense_init_rect(cell.hidden_size, cell.combined_size, obj.rng, dtype)
                        cell._backbone_param_count = int(cell.dense_backbone.size)
                else:
                    cell._use_tt = False
                    cell.backbone = None
                    key = f"cell_{i}_dense_backbone"
                    if key in data.files:
                        cell.dense_backbone = np.asarray(data[key], dtype=dtype)
                    elif cell.dense_backbone is None:
                        cell.dense_backbone = _stable_dense_init_rect(cell.hidden_size, cell.combined_size, obj.rng, dtype)
                    cell._backbone_param_count = int(cell.dense_backbone.size)
            return obj

    def __repr__(self) -> str:
        return f"TCfC_Policy(id={self.policy_id!r}, input={self.input_size}, hidden={self.hidden_size}, cells={self.num_cells}, modes={self.mode_counts()}, dtype={self.dtype})"


# =============================================================================
# Legacy compatibility wrappers
# =============================================================================


class QE_CfC_Cell(TCfC_Cell):
    """Legacy alias wrapper retained for backward compatibility."""

    def __init__(self, input_size: int, hidden_size: int, bond_dim: int = 8, dtype: np.dtype = np.float32, rng: Optional[np.random.Generator] = None, tt_strict: bool = False):
        super().__init__(input_size, hidden_size, bond_dim=bond_dim, rng=rng, dtype=dtype, tt_strict=tt_strict, force_dense=False)

    def forward(self, x_prev: np.ndarray, input_vec: np.ndarray, time_delta: float) -> np.ndarray:  # type: ignore[override]
        prev = np.asarray(x_prev, dtype=self.dtype).reshape(-1)
        if prev.size != self.hidden_size:
            prev = np.zeros((self.hidden_size,), dtype=self.dtype)
        return super().forward(input_vec, prev, time_delta, return_aux=False)  # type: ignore[return-value]


class QECfCPolicy:
    """Legacy policy facade around TCfC_Policy."""

    def __init__(self, policy_id: str, input_size: int, hidden_size: int, num_cells: int, bond_dim: int = 8, dtype: np.dtype = np.float32, verbose: bool = True, rng: Optional[np.random.Generator] = None, tt_strict: bool = False):
        self._policy = TCfC_Policy(policy_id, input_size, hidden_size, num_cells, bond_dim=bond_dim, dtype=dtype, verbose=verbose, rng=rng, tt_strict=tt_strict)

    def reset_state(self, value: float = 0.0) -> None:
        self._policy.reset_state(value)

    def get_state(self) -> np.ndarray:
        return self._policy.get_state()

    def set_state(self, new_state: Any) -> None:
        self._policy.set_state(new_state)

    def step(self, external_input: Any, time_delta: float = 1.0) -> np.ndarray:
        return self._policy.step(external_input, time_delta)  # type: ignore[return-value]

    def parameter_count(self) -> int:
        return self._policy.parameter_count()

    def __repr__(self) -> str:
        return repr(self._policy)


TensorizedCfCPolicy = TCfC_Policy
TensorizedCfCCell = TCfC_Cell


# =============================================================================
# Self-test
# =============================================================================


def _self_test() -> None:
    print("=== TCfC production substrate self-test ===")
    policy = TCfC_Policy(
        policy_id="selftest_tcfc",
        input_size=16,
        hidden_size=32,
        num_cells=4,
        bond_dim=4,
        rng_seed=42,
        dtype=np.float32,
        verbose=True,
        tt_strict=False,
        force_dense=False,
        state_clip=1.0,
        time_scale=1.0,
        leaky_lambda=1.0,
        deterministic=True,
    )
    rng = np.random.default_rng(99)
    x = rng.normal(size=(16,)).astype(np.float32)
    s0 = policy.get_state()
    s_zero = policy.step(x, time_delta=0.0)
    assert np.allclose(s0, s_zero), "dt=0 must preserve state"
    state, aux = policy.step(x, time_delta=0.5, return_state=True)
    assert state.shape == (32 * 4,)
    assert np.isfinite(state).all()
    assert "health" in aux and aux["health"]["is_stable"]
    pred, rec = policy.forecast(x, time_delta=0.25, return_receipt=True)
    assert pred.shape == state.shape
    assert rec.compute_hash() == rec.receipt_hash
    err = policy.compare_forecast_actual(pred, pred, input_vector=x, time_delta=0.25)
    assert err.ok
    manifest = policy.lobe_commitment_manifest().to_dict()
    assert policy.verify_lobe_commitment(manifest)
    health = policy.lobe_health_receipt().to_dict()
    assert health["receipt_hash"]
    payload = policy.serialize_state()
    policy.reset_state()
    policy.load_state(payload)
    assert np.allclose(policy.get_state(), state)

    p1 = TCfC_Policy("det", 16, 32, 4, bond_dim=4, rng_seed=2025, deterministic=True, verbose=False, tt_strict=False)
    p2 = TCfC_Policy("det", 16, 32, 4, bond_dim=4, rng_seed=2025, deterministic=True, verbose=False, tt_strict=False)
    y1 = p1.step(x, time_delta=0.25)
    y2 = p2.step(x, time_delta=0.25)
    assert np.allclose(y1, y2, atol=1e-5, rtol=1e-5), "determinism probe failed"

    ckpt = Path(os.getenv("CFC_SELFTEST_CHECKPOINT", "./tcfc_selftest_checkpoint.npz"))
    policy.save_checkpoint(ckpt)
    loaded = TCfC_Policy.load_checkpoint(ckpt, verbose=False, tt_strict=False, verify=True)
    assert np.allclose(loaded.get_state(), policy.get_state(), atol=1e-6)
    assert loaded.lobe_commitment_hash()
    try:
        ckpt.unlink()
    except Exception:
        pass
    print(policy.summary())
    print("Health:", policy.health_metrics())
    print("=== TCfC self-test passed ===")


if __name__ == "__main__":
    _self_test()
