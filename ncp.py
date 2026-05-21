#!/usr/bin/env python3
# ncp.py
r"""
Project Chimera / Akkurat - Production Liquid Reflex Lobe v4.0
==============================================================

Tensorized Neural Circuit Policy substrate for the Tensorized Blockchain stack.

Production role
---------------
This module provides the Liquid Reflex Lobe: a bounded recurrent temporal
substrate for stability monitoring, recurrent memory, drift sensing, and
homeostatic control. It is intentionally a neural substrate only. It does not
route tools, execute ledger transactions, or make governance decisions. Higher
layers consume its outputs as auditable lobe receipts.

Compatibility
-------------
The primary class keeps the previous constructor and stepping style:

    ncp = TensorizedNCP(...)
    y = ncp.step(feature_vector)
    y, aux = ncp.step(feature_vector, return_state=True)

Production upgrades
-------------------
- LobeScaleProfile, LobeHealthReceipt, and LobeCommitmentManifest.
- Deterministic seeded initialization as a first-class configuration option.
- Consensus-safe mode with CPU/dtype/runtime guardrails.
- TensorTrain recurrent operator commitments when tn.py exposes them.
- Explicit dense-equivalent parameter accounting and compression factors.
- Checkpoint integrity manifest and verification on load.
- Runtime state serialization and lobe state/weight commitment hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


NCP_VERSION = "ncp-liquid-reflex-lobe-v4.0-production"
NCP_COMMITMENT_SCHEME = "akkurat.lobe.ncp.commitment.v1"
NCP_CHECKPOINT_FORMAT = "akkurat.ncp.checkpoint.v2"


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
# Utility helpers
# =============================================================================

_EPS = 1e-12


def _utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_safe(obj: Any) -> Any:
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
    payload = {
        "tag": str(tag),
        "shape": list(map(int, x.shape)),
        "dtype": str(np.dtype(dtype)),
        "bytes_sha256": _sha256_bytes(x.astype(dtype, copy=False).tobytes(order="C")),
    }
    return _sha256_json(payload)


def _coerce_float64_vector(x: Any, expected_dim: Optional[int] = None, name: str = "input") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1).copy()
    if expected_dim is not None and arr.size != int(expected_dim):
        raise ValueError(f"{name} size mismatch: expected {expected_dim}, got {arr.size}")
    arr[~np.isfinite(arr)] = 0.0
    return arr


def _safe_clip_inplace(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    np.clip(arr, float(lo), float(hi), out=arr)
    return arr


def _sigmoid(x: Any) -> np.ndarray:
    z = np.clip(np.asarray(x, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _sanitize_vector(arr: Any, fill_value: float = 0.0) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).copy()
    mask = ~np.isfinite(out)
    if np.any(mask):
        out[mask] = float(fill_value)
    return out


def _count_nonfinite(arr: Any) -> int:
    a = np.asarray(arr)
    return int(np.size(a) - np.count_nonzero(np.isfinite(a)))


def _saturation_fraction(arr: Any, threshold: float = 0.98) -> float:
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return 0.0
    return float(np.mean(np.abs(a) >= float(threshold)))


def _safe_norm(arr: Any) -> float:
    try:
        n = float(np.linalg.norm(np.asarray(arr, dtype=np.float64)))
        return n if math.isfinite(n) else 0.0
    except Exception:
        return 0.0


def _safe_spectral_norm(mat: np.ndarray) -> Optional[float]:
    try:
        s = np.linalg.svd(np.asarray(mat, dtype=np.float64), compute_uv=False)
        return float(s[0]) if s.size else 0.0
    except Exception:
        return None


def _stable_dense_init(h: int, rng: np.random.Generator, *, gamma: float = 0.95) -> np.ndarray:
    """Orthogonal-ish recurrent initialization with spectral control."""
    h = int(h)
    if h <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    A = rng.normal(0.0, 1.0, size=(h, h))
    q, r = np.linalg.qr(A)
    d = np.sign(np.diag(r))
    d[d == 0] = 1.0
    W = (q * d * 0.9).astype(np.float64, copy=False)
    smax = _safe_spectral_norm(W)
    if smax is not None and smax > gamma and smax > 0.0:
        W *= float(gamma) / float(smax)
    return W


def _atomic_write_npz(path: Union[str, os.PathLike], arrays: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    np.savez_compressed(str(tmp), **dict(arrays))
    # NumPy appends .npz if suffix is not .npz. Normalize both possibilities.
    candidate = Path(str(tmp) + ".npz")
    if candidate.exists() and not tmp.exists():
        tmp = candidate
    os.replace(tmp, p)


# =============================================================================
# Compatibility legacy neuron / sparse NCP
# =============================================================================

class LTCNeuron:
    """Lightweight compatibility mirror for neuron-like state inspection."""

    def __init__(self, neuron_id: int, membrane_capacitance: float = 1.0, bias: float = 0.0):
        self.neuron_id = int(neuron_id)
        self.state = 0.0
        self.membrane_capacitance = float(membrane_capacitance)
        self.bias = float(bias)

    def __repr__(self) -> str:
        return (
            f"LTCNeuron(id={self.neuron_id}, state={self.state:.4f}, "
            f"C={self.membrane_capacitance:.4f}, bias={self.bias:.4f})"
        )


class NeuralCircuitPolicy:
    """Legacy sparse NCP retained for backward compatibility and comparison tests."""

    def __init__(
        self,
        policy_id: str,
        num_neurons: int,
        num_inputs: int,
        num_outputs: int,
        sparsity: float = 0.8,
        rng: Optional[np.random.Generator] = None,
        verbose: bool = True,
    ):
        self.policy_id = str(policy_id)
        self.num_neurons = int(num_neurons)
        self.num_inputs = int(num_inputs)
        self.num_outputs = int(num_outputs)
        self.verbose = bool(verbose)
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()
        if self.num_inputs <= 0 or self.num_outputs <= 0 or self.num_neurons <= 0:
            raise ValueError("num_neurons, num_inputs, and num_outputs must be positive")
        if self.num_inputs + self.num_outputs > self.num_neurons:
            raise ValueError("num_inputs + num_outputs cannot exceed num_neurons")
        self.neurons = [LTCNeuron(neuron_id=i) for i in range(self.num_neurons)]
        self.input_neuron_indices = list(range(self.num_inputs))
        self.output_neuron_indices = list(range(self.num_neurons - self.num_outputs, self.num_neurons))
        self.synapses = self._create_sparse_wiring(float(sparsity))
        if self.verbose:
            print(f"[LegacySparseNCP:{self.policy_id}] neurons={self.num_neurons} synapses={len(self.synapses)}")

    def _create_sparse_wiring(self, sparsity: float) -> List[Tuple[int, int, float]]:
        sparsity = float(np.clip(sparsity, 0.0, 0.999))
        synapses: List[Tuple[int, int, float]] = []
        fan_in_est = max(1, int(round((1.0 - sparsity) * (self.num_neurons - 1))))
        scale = 1.0 / math.sqrt(fan_in_est)
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i == j or self.rng.random() < sparsity:
                    continue
                synapses.append((i, j, float(self.rng.normal(0.0, 1.0) * scale)))
        return synapses

    def reset_state(self, value: float = 0.0) -> None:
        for n in self.neurons:
            n.state = float(value)

    def get_state(self) -> np.ndarray:
        return np.array([n.state for n in self.neurons], dtype=np.float64)

    def set_state(self, new_state: Any) -> None:
        arr = _coerce_float64_vector(new_state, expected_dim=self.num_neurons, name="new_state")
        for i, n in enumerate(self.neurons):
            n.state = float(arr[i])

    def step(self, inputs: Any, dt: float = 1.0) -> np.ndarray:
        x = _coerce_float64_vector(inputs, expected_dim=self.num_inputs, name="inputs")
        dt = float(dt) if math.isfinite(float(dt)) and float(dt) > 0.0 else 1.0
        input_currents = np.zeros((self.num_neurons,), dtype=np.float64)
        for i, val in enumerate(x):
            input_currents[self.input_neuron_indices[i]] += float(val)
        src_states = np.array([n.state for n in self.neurons], dtype=np.float64)
        src_acts = _sigmoid(src_states)
        for s_idx, t_idx, w in self.synapses:
            input_currents[t_idx] += float(w) * src_acts[s_idx]
        _safe_clip_inplace(input_currents, -10.0, 10.0)
        for i, n in enumerate(self.neurons):
            C = max(1e-8, float(n.membrane_capacitance))
            tau = 10.0
            leakage = C / tau
            d_state = (-leakage * n.state + n.bias + input_currents[i]) / C
            n.state = float(np.tanh(n.state + dt * d_state))
        return np.array([self.neurons[i].state for i in self.output_neuron_indices], dtype=np.float64)

    def snapshot(self) -> Dict[str, Any]:
        state = self.get_state()
        return {
            "policy_id": self.policy_id,
            "kind": "legacy_sparse_ncp",
            "num_neurons": self.num_neurons,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "synapse_count": len(self.synapses),
            "hidden_state_norm": float(np.linalg.norm(state)),
        }


# =============================================================================
# Production dataclasses
# =============================================================================

@dataclass
class LobeScaleProfile:
    lobe_id: str
    lobe_kind: str = "ncp_liquid_reflex"
    active: bool = True
    parameter_budget: int = 0
    dense_equivalent_budget: int = 0
    hidden_size: int = 0
    tt_bond_rank: int = 0
    precision: str = "float32"
    update_frequency_hz: float = 1.0
    checkpoint_policy: str = "hash_on_checkpoint"
    consensus_safe: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "LobeScaleProfile":
        p = LobeScaleProfile(**_json_safe(asdict(self)))
        p.lobe_id = str(p.lobe_id or "ncp_lobe")
        p.lobe_kind = str(p.lobe_kind or "ncp_liquid_reflex")
        p.active = bool(p.active)
        p.parameter_budget = int(max(0, p.parameter_budget))
        p.dense_equivalent_budget = int(max(0, p.dense_equivalent_budget))
        p.hidden_size = int(max(0, p.hidden_size))
        p.tt_bond_rank = int(max(0, p.tt_bond_rank))
        p.precision = str(p.precision or "float32")
        p.update_frequency_hz = float(max(0.0, p.update_frequency_hz))
        p.checkpoint_policy = str(p.checkpoint_policy or "hash_on_checkpoint")
        p.consensus_safe = bool(p.consensus_safe)
        p.metadata = dict(p.metadata or {})
        return p

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self.normalized()))

    def hash(self) -> str:
        return _sha256_json({"scheme": "akkurat.lobe.scale_profile.v1", "profile": self.to_dict()})


@dataclass(frozen=True)
class LobeHealthReceipt:
    scheme: str
    version: str
    lobe_id: str
    lobe_kind: str
    step_count: int
    recurrent_mode: str
    is_stable: bool
    is_saturated: bool
    has_nan: bool
    degraded_mode: bool
    hidden_norm: float
    output_norm: float
    recurrent_gain_estimate: Optional[float]
    parameter_count: int
    dense_equivalent_parameter_count: int
    compression_factor: Optional[float]
    scale_profile_hash: str
    state_hash: str
    weights_hash: str
    receipt_hash: str = ""

    def payload(self) -> Dict[str, Any]:
        d = _json_safe(asdict(self))
        d.pop("receipt_hash", None)
        return d

    def compute_hash(self) -> str:
        return _sha256_json({"domain": "LobeHealthReceipt", "payload": self.payload()})

    def to_dict(self) -> Dict[str, Any]:
        d = _json_safe(asdict(self))
        if not d.get("receipt_hash"):
            d["receipt_hash"] = self.compute_hash()
        return d


@dataclass(frozen=True)
class LobeCommitmentManifest:
    scheme: str
    version: str
    lobe_id: str
    lobe_kind: str
    config_hash: str
    scale_profile_hash: str
    state_hash: str
    weights_hash: str
    recurrent_commitment_hash: str
    checkpoint_hash: str
    parameter_count: int
    dense_equivalent_parameter_count: int
    compression_factor: Optional[float]
    consensus_safe: bool
    manifest_hash: str = ""

    def payload(self) -> Dict[str, Any]:
        d = _json_safe(asdict(self))
        d.pop("manifest_hash", None)
        return d

    def compute_hash(self) -> str:
        return _sha256_json({"domain": "LobeCommitmentManifest", "payload": self.payload()})

    def to_dict(self) -> Dict[str, Any]:
        d = _json_safe(asdict(self))
        if not d.get("manifest_hash"):
            d["manifest_hash"] = self.compute_hash()
        return d


@dataclass
class TensorizedNCPConfig:
    policy_id: str
    num_inputs: int
    num_hidden: int
    num_outputs: int

    bond_dim: Union[int, Sequence[int]] = 4
    tau_min: float = 1.0
    tau_max: float = 50.0
    tt_modes: Optional[Sequence[int]] = None
    output_activation: str = "linear"  # linear | tanh | sigmoid
    output_scale: float = 1.0
    tau_jitter_frac: float = 0.0
    safety_clamp: Optional[float] = None
    tt_strict: bool = True

    leaky_lambda: float = 0.1
    num_freq_gates: int = 0
    num_mvm_outputs: int = 0

    verbose: bool = False
    rng_seed: Optional[int] = None
    deterministic: bool = False
    consensus_safe: bool = False
    strict_runtime_checks: bool = True
    force_dense: bool = False
    check_finite: bool = True
    mirror_neurons_each_step: bool = False

    tt_init_from_dense: bool = True
    tt_energy_tol: Optional[float] = 0.999
    tt_device: str = "cpu"

    input_clip: float = 10.0
    current_clip: float = 5.0
    output_clip: Optional[float] = None
    max_dt: float = 100.0

    lobe_id: str = ""
    lobe_kind: str = "ncp_liquid_reflex"
    max_parameter_budget: int = 2_500_000_000

    def normalized(self) -> "TensorizedNCPConfig":
        cfg = TensorizedNCPConfig(**_json_safe(asdict(self)))
        cfg.policy_id = str(cfg.policy_id or "ncp_lobe")
        cfg.num_inputs = int(max(1, cfg.num_inputs))
        cfg.num_hidden = int(max(1, cfg.num_hidden))
        cfg.num_outputs = int(max(1, cfg.num_outputs))
        if isinstance(cfg.bond_dim, list):
            cfg.bond_dim = [int(max(1, int(x))) for x in cfg.bond_dim]
        else:
            cfg.bond_dim = int(max(1, int(cfg.bond_dim)))
        cfg.tau_min = float(max(_EPS, cfg.tau_min))
        cfg.tau_max = float(max(cfg.tau_min + _EPS, cfg.tau_max))
        cfg.output_activation = str(cfg.output_activation or "linear").lower().strip()
        if cfg.output_activation not in {"linear", "tanh", "sigmoid"}:
            cfg.output_activation = "linear"
        cfg.output_scale = float(cfg.output_scale)
        cfg.tau_jitter_frac = float(max(0.0, cfg.tau_jitter_frac))
        cfg.tt_strict = bool(cfg.tt_strict)
        cfg.leaky_lambda = float(np.clip(cfg.leaky_lambda, 0.0, 1.0))
        cfg.num_freq_gates = int(max(0, cfg.num_freq_gates))
        cfg.num_mvm_outputs = int(max(0, cfg.num_mvm_outputs))
        cfg.deterministic = bool(cfg.deterministic)
        cfg.consensus_safe = bool(cfg.consensus_safe)
        if cfg.consensus_safe:
            cfg.deterministic = True
            cfg.tt_device = "cpu"
            cfg.check_finite = True
            cfg.strict_runtime_checks = True
            if cfg.rng_seed is None:
                raise ValueError("consensus_safe TensorizedNCP requires rng_seed")
        cfg.tt_device = str(cfg.tt_device or "cpu").lower().strip()
        if cfg.tt_device not in {"cpu", "gpu"}:
            cfg.tt_device = "cpu"
        cfg.input_clip = float(max(_EPS, cfg.input_clip))
        cfg.current_clip = float(max(_EPS, cfg.current_clip))
        cfg.max_dt = float(max(_EPS, cfg.max_dt))
        cfg.lobe_id = str(cfg.lobe_id or cfg.policy_id)
        cfg.lobe_kind = str(cfg.lobe_kind or "ncp_liquid_reflex")
        cfg.max_parameter_budget = int(max(0, cfg.max_parameter_budget))
        return cfg


# =============================================================================
# Production TensorizedNCP
# =============================================================================

class TensorizedNCP:
    """
    Production-grade Tensorized Neural Circuit Policy / Liquid Reflex Lobe.

    Core guarantees:
    - Numeric vector input only.
    - Stable bounded hidden-state dynamics.
    - TensorTrain recurrent operator when available/configured.
    - Dense fallback when configured and allowed.
    - State serialization, checkpointing, lobe commitments, and health receipts.
    """

    CHECKPOINT_VERSION = 4

    def __init__(
        self,
        policy_id: str,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
        bond_dim: Union[int, Iterable[int]] = 4,
        tau_min: float = 1.0,
        tau_max: float = 50.0,
        tt_modes: Optional[Iterable[int]] = None,
        rng: Optional[np.random.Generator] = None,
        rng_seed: Optional[int] = None,
        deterministic: bool = False,
        consensus_safe: bool = False,
        verbose: bool = True,
        output_activation: str = "linear",
        output_scale: float = 1.0,
        tau_jitter_frac: float = 0.0,
        safety_clamp: Optional[float] = None,
        tt_strict: bool = True,
        leaky_lambda: float = 0.1,
        num_freq_gates: int = 0,
        num_mvm_outputs: int = 0,
        strict_runtime_checks: bool = True,
        force_dense: Optional[bool] = None,
        check_finite: bool = True,
        mirror_neurons_each_step: bool = False,
        tt_init_from_dense: bool = True,
        tt_energy_tol: Optional[float] = 0.999,
        tt_device: str = "cpu",
        input_clip: float = 10.0,
        current_clip: float = 5.0,
        output_clip: Optional[float] = None,
        max_dt: float = 100.0,
        lobe_id: str = "",
        lobe_kind: str = "ncp_liquid_reflex",
        max_parameter_budget: int = 2_500_000_000,
        scale_profile: Optional[LobeScaleProfile] = None,
    ):
        if isinstance(bond_dim, Iterable) and not isinstance(bond_dim, (str, bytes)):
            cfg_bond: Union[int, List[int]] = [int(max(1, x)) for x in list(bond_dim)]
        else:
            cfg_bond = int(max(1, int(bond_dim)))

        self.config = TensorizedNCPConfig(
            policy_id=str(policy_id),
            num_inputs=int(num_inputs),
            num_hidden=int(num_hidden),
            num_outputs=int(num_outputs),
            bond_dim=cfg_bond,
            tau_min=float(tau_min),
            tau_max=float(tau_max),
            tt_modes=list(tt_modes) if tt_modes is not None else None,
            output_activation=str(output_activation).lower().strip(),
            output_scale=float(output_scale),
            tau_jitter_frac=float(tau_jitter_frac),
            safety_clamp=None if safety_clamp is None else float(safety_clamp),
            tt_strict=bool(tt_strict),
            leaky_lambda=float(leaky_lambda),
            num_freq_gates=int(num_freq_gates),
            num_mvm_outputs=int(num_mvm_outputs),
            verbose=bool(verbose),
            rng_seed=None if rng_seed is None else int(rng_seed),
            deterministic=bool(deterministic),
            consensus_safe=bool(consensus_safe),
            strict_runtime_checks=bool(strict_runtime_checks),
            force_dense=bool(force_dense) if force_dense is not None else False,
            check_finite=bool(check_finite),
            mirror_neurons_each_step=bool(mirror_neurons_each_step),
            tt_init_from_dense=bool(tt_init_from_dense),
            tt_energy_tol=tt_energy_tol,
            tt_device=str(tt_device).lower().strip(),
            input_clip=float(input_clip),
            current_clip=float(current_clip),
            output_clip=None if output_clip is None else float(output_clip),
            max_dt=float(max_dt),
            lobe_id=str(lobe_id or policy_id),
            lobe_kind=str(lobe_kind or "ncp_liquid_reflex"),
            max_parameter_budget=int(max_parameter_budget),
        ).normalized()

        self.policy_id = self.config.policy_id
        self.lobe_id = self.config.lobe_id
        self.lobe_kind = self.config.lobe_kind
        self.num_inputs = self.config.num_inputs
        self.num_hidden = self.config.num_hidden
        self.num_outputs = self.config.num_outputs
        self.verbose = self.config.verbose
        self.output_activation = self.config.output_activation
        self.output_scale = float(self.config.output_scale)
        self.safety_clamp = self.config.safety_clamp
        self.tt_strict = bool(self.config.tt_strict)
        self.strict_runtime_checks = bool(self.config.strict_runtime_checks)
        self.leaky_lambda = float(self.config.leaky_lambda)
        self.num_freq_gates = int(self.config.num_freq_gates)
        self.num_mvm_outputs = int(self.config.num_mvm_outputs)
        self.check_finite = bool(self.config.check_finite)
        self.mirror_neurons_each_step = bool(self.config.mirror_neurons_each_step)
        self.consensus_safe = bool(self.config.consensus_safe)
        self.deterministic = bool(self.config.deterministic)

        if rng is not None and self.config.rng_seed is not None and self.deterministic:
            # Prefer explicit seed for deterministic reproducibility.
            self.rng = np.random.default_rng(int(self.config.rng_seed))
        elif rng is not None:
            self.rng = rng
        elif self.config.rng_seed is not None:
            self.rng = np.random.default_rng(int(self.config.rng_seed))
        elif self.deterministic or self.consensus_safe:
            raise ValueError("deterministic/consensus_safe TensorizedNCP requires rng_seed when rng is not provided")
        else:
            self.rng = np.random.default_rng()

        self._step_count = 0
        self._created_at = _utc_ts()
        self._last_maintenance_at: Optional[str] = None

        if self.num_inputs <= 0 or self.num_hidden <= 0 or self.num_outputs <= 0:
            raise ValueError("num_inputs, num_hidden, and num_outputs must all be > 0")

        if isinstance(self.config.bond_dim, list):
            self.bond_dims_list = [int(max(1, x)) for x in self.config.bond_dim]
            self.bond_dim = int(self.bond_dims_list[0]) if self.bond_dims_list else 4
        else:
            self.bond_dim = int(max(1, self.config.bond_dim))
            self.bond_dims_list = [self.bond_dim]

        self._state = np.zeros((self.num_hidden,), dtype=np.float64)
        self._membrane_capacitance = np.ones((self.num_hidden,), dtype=np.float64)
        self._bias = self.rng.normal(0.0, 0.01, size=(self.num_hidden,)).astype(np.float64)
        self.hidden_neurons = [
            LTCNeuron(neuron_id=i, membrane_capacitance=float(self._membrane_capacitance[i]), bias=float(self._bias[i]))
            for i in range(self.num_hidden)
        ]

        self.input_weights = self.rng.normal(0.0, 0.1, size=(self.num_hidden, self.num_inputs)).astype(np.float64)
        self.output_weights = self.rng.normal(0.0, 0.1, size=(self.num_outputs, self.num_hidden)).astype(np.float64)

        self.freq_gate_weights: Optional[np.ndarray] = None
        if self.num_freq_gates > 0:
            self.freq_gate_weights = self.rng.normal(0.0, 0.1, size=(self.num_freq_gates, self.num_hidden)).astype(np.float64)

        self.mvm_weights: Optional[np.ndarray] = None
        if self.num_mvm_outputs > 0:
            self.mvm_weights = self.rng.normal(0.0, 0.1, size=(self.num_mvm_outputs, self.num_hidden)).astype(np.float64)

        self.tc_W_h = self.rng.normal(0.0, 0.05, size=(self.num_hidden, self.num_hidden)).astype(np.float64)
        self.tc_W_x = self.rng.normal(0.0, 0.05, size=(self.num_hidden, self.num_inputs)).astype(np.float64)
        self.tc_b = self.rng.normal(0.0, 0.2, size=(self.num_hidden,)).astype(np.float64)

        base_tau_min = float(self.config.tau_min)
        base_tau_max = float(self.config.tau_max)
        jitter = float(max(0.0, self.config.tau_jitter_frac))
        if jitter > 0.0:
            eps = self.rng.uniform(-jitter, jitter, size=(self.num_hidden,)).astype(np.float64)
            self._tau_min_arr = np.maximum(base_tau_min * (1.0 + eps), 1e-6)
            self._tau_max_arr = np.maximum(base_tau_max * (1.0 + eps), self._tau_min_arr + 1e-6)
        else:
            self._tau_min_arr = np.full((self.num_hidden,), base_tau_min, dtype=np.float64)
            self._tau_max_arr = np.full((self.num_hidden,), base_tau_max, dtype=np.float64)

        self._infer_tt_shapes()

        env_force_dense = str(os.getenv("NCP_FORCE_DENSE", "")).strip().lower() in ("1", "true", "yes", "on")
        self._force_dense_effective = bool(self.config.force_dense) or (bool(env_force_dense) and not self.consensus_safe)
        self._use_tt = False
        self.recurrent_weights: Optional[Any] = None
        self.recurrent_dense: Optional[np.ndarray] = None
        self._recurrent_param_count: Optional[int] = None
        self._tt_init_error: Optional[str] = None
        self._tt_bond_dims_actual: List[int] = []
        self._init_recurrent_operator()

        self.scale_profile = (scale_profile or LobeScaleProfile(
            lobe_id=self.lobe_id,
            lobe_kind=self.lobe_kind,
            active=True,
            parameter_budget=int(self.total_parameter_count()),
            dense_equivalent_budget=int(self.equivalent_dense_parameter_count()),
            hidden_size=int(self.num_hidden),
            tt_bond_rank=int(max(self._tt_bond_dims_actual) if self._tt_bond_dims_actual else self.bond_dim),
            precision="float64_state_float32_tt",
            consensus_safe=bool(self.consensus_safe),
            metadata={"max_parameter_budget": int(self.config.max_parameter_budget)},
        )).normalized()

        self._last_input = np.zeros((self.num_inputs,), dtype=np.float64)
        self._last_input_signal = np.zeros((self.num_hidden,), dtype=np.float64)
        self._last_recurrent_signal = np.zeros((self.num_hidden,), dtype=np.float64)
        self._last_total_currents = np.zeros((self.num_hidden,), dtype=np.float64)
        self._last_dynamic_tau = np.full((self.num_hidden,), self.config.tau_min, dtype=np.float64)
        self._last_output = np.zeros((self.num_outputs,), dtype=np.float64)
        self._last_aux_outputs: Dict[str, np.ndarray] = {}
        self._last_error: Optional[str] = None

        if self.consensus_safe:
            self.validate_consensus_safe(raise_on_error=True)

        if self.verbose:
            mode = "TT" if self._use_tt else "Dense"
            print(
                f"[TensorizedNCP:{self.policy_id}] mode={mode} hidden={self.num_hidden} "
                f"inputs={self.num_inputs} outputs={self.num_outputs} recurrent_params={self.parameter_count()} "
                f"dense_equiv={self.equivalent_dense_parameter_count()} total_params={self.total_parameter_count()}"
            )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Union[TensorizedNCPConfig, Mapping[str, Any]], *, rng: Optional[np.random.Generator] = None) -> "TensorizedNCP":
        if isinstance(cfg, Mapping):
            cfg = TensorizedNCPConfig(**dict(cfg))
        cfg = cfg.normalized()
        return cls(
            policy_id=cfg.policy_id,
            num_inputs=cfg.num_inputs,
            num_hidden=cfg.num_hidden,
            num_outputs=cfg.num_outputs,
            bond_dim=cfg.bond_dim,
            tau_min=cfg.tau_min,
            tau_max=cfg.tau_max,
            tt_modes=cfg.tt_modes,
            rng=rng,
            rng_seed=cfg.rng_seed,
            deterministic=cfg.deterministic,
            consensus_safe=cfg.consensus_safe,
            verbose=cfg.verbose,
            output_activation=cfg.output_activation,
            output_scale=cfg.output_scale,
            tau_jitter_frac=cfg.tau_jitter_frac,
            safety_clamp=cfg.safety_clamp,
            tt_strict=cfg.tt_strict,
            leaky_lambda=cfg.leaky_lambda,
            num_freq_gates=cfg.num_freq_gates,
            num_mvm_outputs=cfg.num_mvm_outputs,
            strict_runtime_checks=cfg.strict_runtime_checks,
            force_dense=cfg.force_dense,
            check_finite=cfg.check_finite,
            mirror_neurons_each_step=cfg.mirror_neurons_each_step,
            tt_init_from_dense=cfg.tt_init_from_dense,
            tt_energy_tol=cfg.tt_energy_tol,
            tt_device=cfg.tt_device,
            input_clip=cfg.input_clip,
            current_clip=cfg.current_clip,
            output_clip=cfg.output_clip,
            max_dt=cfg.max_dt,
            lobe_id=cfg.lobe_id,
            lobe_kind=cfg.lobe_kind,
            max_parameter_budget=cfg.max_parameter_budget,
        )

    def config_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self.config.normalized()))

    def _infer_tt_shapes(self) -> None:
        if self.config.tt_modes is not None:
            modes = tuple(int(m) for m in self.config.tt_modes if int(m) > 0)
            prod = int(np.prod(modes)) if modes else 0
            if prod != self.num_hidden:
                if self.tt_strict:
                    raise ValueError(f"[{self.policy_id}] tt_modes product ({prod}) must equal num_hidden ({self.num_hidden})")
                f1, f2 = get_factors(self.num_hidden)
                self._tt_in_shape = (int(f1), int(f2))
                self._tt_out_shape = (int(f1), int(f2))
            else:
                self._tt_in_shape = modes
                self._tt_out_shape = modes
        else:
            f1, f2 = get_factors(self.num_hidden)
            self._tt_in_shape = (int(f1), int(f2))
            self._tt_out_shape = (int(f1), int(f2))

    def _resolve_tt_bond_dims(self, num_cores: int) -> List[int]:
        if num_cores <= 1:
            return []
        needed = num_cores - 1
        if isinstance(self.config.bond_dim, list):
            if len(self.bond_dims_list) == needed:
                return [int(max(1, x)) for x in self.bond_dims_list]
            if self.tt_strict:
                raise ValueError(f"[{self.policy_id}] bond_dims length mismatch: expected {needed}, got {len(self.bond_dims_list)}")
            return [self.bond_dim] * needed
        return [self.bond_dim] * needed

    def _make_tt_config(self) -> Any:
        try:
            return TTConfig(
                dtype=np.float32,
                device=self.config.tt_device,
                check_finite=self.check_finite,
                seed=self.config.rng_seed,
                deterministic=bool(self.deterministic),
                consensus_safe=bool(self.consensus_safe),
            )
        except TypeError:
            return TTConfig(dtype=np.float32, device=self.config.tt_device, check_finite=self.check_finite)

    def _init_recurrent_operator(self) -> None:
        if self._force_dense_effective:
            self._use_tt = False
            self.recurrent_dense = _stable_dense_init(self.num_hidden, self.rng)
            self.recurrent_weights = None
            self._recurrent_param_count = int(self.recurrent_dense.size)
            return

        if not TENSORTRAIN_AVAILABLE:
            msg = f"[{self.policy_id}] TensorTrain unavailable"
            if TT_IMPORT_ERROR is not None:
                msg += f": {TT_IMPORT_ERROR}"
            self._tt_init_error = msg
            if self.tt_strict or self.consensus_safe:
                raise RuntimeError(msg)
            self._use_tt = False
            self.recurrent_dense = _stable_dense_init(self.num_hidden, self.rng)
            self.recurrent_weights = None
            self._recurrent_param_count = int(self.recurrent_dense.size)
            return

        num_cores = len(self._tt_in_shape)
        bond_dims = self._resolve_tt_bond_dims(num_cores)
        max_rank = max(bond_dims) if bond_dims else self.bond_dim
        try:
            commit_metadata = {
                "lobe_id": self.lobe_id,
                "lobe_kind": self.lobe_kind,
                "operator": "ncp_recurrent",
                "policy_id": self.policy_id,
                "num_hidden": self.num_hidden,
                "seed": self.config.rng_seed,
            }
            if bool(self.config.tt_init_from_dense):
                dense_init = _stable_dense_init(self.num_hidden, self.rng)
                kwargs = dict(
                    output_dims=list(self._tt_out_shape),
                    input_dims=list(self._tt_in_shape),
                    max_bond_dim=int(max_rank),
                    dtype=np.float32,
                    energy_tol=self.config.tt_energy_tol,
                    device=self.config.tt_device,
                    check_finite=self.check_finite,
                )
                try:
                    tt = tensor_train_svd(
                        dense_init,
                        **kwargs,
                        seed=self.config.rng_seed,
                        deterministic=bool(self.deterministic),
                        consensus_safe=bool(self.consensus_safe),
                        metadata={"source": "TensorizedNCP._init_recurrent_operator"},
                        commit_metadata=commit_metadata,
                    )
                except TypeError:
                    tt = tensor_train_svd(dense_init, **kwargs)
                if tt is None:
                    raise RuntimeError("tensor_train_svd returned None")
                self.recurrent_weights = tt
            else:
                self.recurrent_weights = TensorTrain(
                    output_dims=list(self._tt_out_shape),
                    input_dims=list(self._tt_in_shape),
                    bond_dims=bond_dims,
                    rng=self.rng,
                    config=self._make_tt_config(),
                    init_scale=1e-2,
                    metadata={"source": "TensorizedNCP.random_recurrent"},
                    commit_metadata=commit_metadata,
                )
            self._use_tt = True
            self.recurrent_dense = None
            self._tt_bond_dims_actual = list(self.recurrent_weights.bond_ranks()) if hasattr(self.recurrent_weights, "bond_ranks") else list(bond_dims)
            self._recurrent_param_count = int(self.recurrent_weights.parameter_count()) if hasattr(self.recurrent_weights, "parameter_count") else None
        except Exception as exc:
            msg = f"[{self.policy_id}] TensorTrain recurrent init failed: {exc}"
            self._tt_init_error = msg
            if self.tt_strict or self.consensus_safe:
                raise RuntimeError(msg) from exc
            if self.verbose:
                print(f"{msg}. Falling back to dense recurrent operator.")
            self._use_tt = False
            self.recurrent_weights = None
            self.recurrent_dense = _stable_dense_init(self.num_hidden, self.rng)
            self._recurrent_param_count = int(self.recurrent_dense.size)

    # ------------------------------------------------------------------
    # Core math
    # ------------------------------------------------------------------
    def _recurrent_matvec(self, h_prev: np.ndarray) -> np.ndarray:
        h_prev = _coerce_float64_vector(h_prev, expected_dim=self.num_hidden, name="h_prev")
        if not self._use_tt:
            if self.recurrent_dense is None:
                raise RuntimeError(f"[{self.policy_id}] recurrent_dense is not initialized")
            return _sanitize_vector(self.recurrent_dense @ h_prev)
        if self.recurrent_weights is None:
            raise RuntimeError(f"[{self.policy_id}] recurrent_weights is not initialized")
        try:
            if hasattr(self.recurrent_weights, "apply"):
                out = self.recurrent_weights.apply(h_prev.astype(np.float32, copy=False))
            else:
                out = self.recurrent_weights.contract_with_vector(h_prev.astype(np.float32, copy=False))
        except Exception as exc:
            raise RuntimeError(f"[{self.policy_id}] TensorTrain contraction failed: {exc}") from exc
        out = np.asarray(_np_from_any_array(out), dtype=np.float64).reshape(-1)
        if out.size != self.num_hidden:
            raise ValueError(f"[{self.policy_id}] TT matvec shape mismatch: expected {self.num_hidden}, got {out.size}")
        return _sanitize_vector(out)

    def _dynamic_tau(self, x: np.ndarray) -> np.ndarray:
        gate_pre = self.tc_W_h @ self._state + self.tc_W_x @ x + self.tc_b
        gate = _sigmoid(gate_pre)
        tau = self._tau_min_arr + gate * (self._tau_max_arr - self._tau_min_arr)
        return _sanitize_vector(np.maximum(tau, 1e-6))

    def _vectorized_ltc_update(self, total_currents: np.ndarray, dynamic_tau: np.ndarray, dt: float) -> None:
        total_currents = _coerce_float64_vector(total_currents, expected_dim=self.num_hidden, name="total_currents")
        dynamic_tau = _coerce_float64_vector(dynamic_tau, expected_dim=self.num_hidden, name="dynamic_tau")
        C = np.maximum(_sanitize_vector(self._membrane_capacitance, 1.0), 1e-8)
        bias = _sanitize_vector(self._bias, 0.0)
        tau = np.clip(dynamic_tau, 1e-6, np.inf)
        leakage = C / tau
        d_state = (-leakage * self._state + bias + total_currents) / C
        if self.safety_clamp is not None and self.safety_clamp > 0:
            _safe_clip_inplace(d_state, -float(self.safety_clamp), float(self.safety_clamp))
        potential_new_state = np.tanh(self._state + float(dt) * d_state)
        self._state = (1.0 - self.leaky_lambda) * self._state + self.leaky_lambda * potential_new_state
        self._state = _sanitize_vector(self._state)
        if self.mirror_neurons_each_step:
            self.sync_neuron_mirror_from_vectors()

    def _activate_output(self, out: np.ndarray) -> np.ndarray:
        y = np.asarray(out, dtype=np.float64)
        if self.output_activation == "tanh":
            y = np.tanh(y)
        elif self.output_activation == "sigmoid":
            y = _sigmoid(y)
        if self.output_scale != 1.0:
            y = y * self.output_scale
        if self.config.output_clip is not None and self.config.output_clip > 0:
            _safe_clip_inplace(y, -float(self.config.output_clip), float(self.config.output_clip))
        return _sanitize_vector(y)

    # ------------------------------------------------------------------
    # Public state API
    # ------------------------------------------------------------------
    def sync_neuron_mirror_from_vectors(self) -> None:
        for i, n in enumerate(self.hidden_neurons):
            n.state = float(self._state[i])
            n.membrane_capacitance = float(self._membrane_capacitance[i])
            n.bias = float(self._bias[i])

    def reset_state(self, value: float = 0.0) -> None:
        v = float(value)
        self._state.fill(v)
        for n in self.hidden_neurons:
            n.state = v
        self._step_count = 0
        self._last_input.fill(0.0)
        self._last_input_signal.fill(0.0)
        self._last_recurrent_signal.fill(0.0)
        self._last_total_currents.fill(0.0)
        self._last_dynamic_tau.fill(self.config.tau_min)
        self._last_output.fill(0.0)
        self._last_aux_outputs = {}
        self._last_error = None

    def get_state(self) -> np.ndarray:
        return self._state.copy()

    def set_state(self, new_state: Union[np.ndarray, Sequence[float]]) -> None:
        arr = _coerce_float64_vector(new_state, expected_dim=self.num_hidden, name="new_state")
        self._state = np.clip(arr, -1.0, 1.0).astype(np.float64, copy=True)
        if self.mirror_neurons_each_step:
            self.sync_neuron_mirror_from_vectors()
        else:
            for i, n in enumerate(self.hidden_neurons):
                n.state = float(self._state[i])

    def serialize_state(self) -> Dict[str, Any]:
        return {
            "schema": "TensorizedNCP.runtime_state.v2",
            "policy_id": self.policy_id,
            "lobe_id": self.lobe_id,
            "lobe_kind": self.lobe_kind,
            "num_inputs": self.num_inputs,
            "num_hidden": self.num_hidden,
            "num_outputs": self.num_outputs,
            "recurrent_mode": "tt" if self._use_tt else "dense",
            "hidden_state": self._state.astype(np.float64).tolist(),
            "step_count": int(self._step_count),
            "tau_min": float(self.config.tau_min),
            "tau_max": float(self.config.tau_max),
            "leaky_lambda": float(self.leaky_lambda),
            "output_activation": str(self.output_activation),
            "output_scale": float(self.output_scale),
            "num_freq_gates": int(self.num_freq_gates),
            "num_mvm_outputs": int(self.num_mvm_outputs),
            "tt_in_shape": list(self._tt_in_shape),
            "tt_out_shape": list(self._tt_out_shape),
            "tt_bond_dims": list(self._tt_bond_dims_actual),
            "state_hash": self.state_hash(),
            "scale_profile_hash": self.scale_profile.hash(),
            "created_at": self._created_at,
            "serialized_at": _utc_ts(),
        }

    def load_state(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        nh = int(payload.get("num_hidden", self.num_hidden))
        if nh != self.num_hidden:
            raise ValueError(f"[{self.policy_id}] load_state hidden dim mismatch: expected {self.num_hidden}, got {nh}")
        vec = payload.get("hidden_state", None)
        if vec is None:
            raise ValueError(f"[{self.policy_id}] load_state missing 'hidden_state'")
        self.set_state(vec)
        self._step_count = int(payload.get("step_count", self._step_count))

    # ------------------------------------------------------------------
    # Parameter accounting / commitments
    # ------------------------------------------------------------------
    def parameter_count(self) -> int:
        return int(self._recurrent_param_count) if self._recurrent_param_count is not None else -1

    def recurrent_dense_equivalent_parameter_count(self) -> int:
        return int(self.num_hidden * self.num_hidden)

    def equivalent_dense_parameter_count(self) -> int:
        total = self.recurrent_dense_equivalent_parameter_count()
        total += int(self.input_weights.size)
        total += int(self.output_weights.size)
        total += int(self.tc_W_h.size + self.tc_W_x.size + self.tc_b.size)
        total += int(self._membrane_capacitance.size + self._bias.size)
        if self.freq_gate_weights is not None:
            total += int(self.freq_gate_weights.size)
        if self.mvm_weights is not None:
            total += int(self.mvm_weights.size)
        return int(total)

    def total_parameter_count(self) -> int:
        total = int(self.parameter_count() if self.parameter_count() >= 0 else self.recurrent_dense_equivalent_parameter_count())
        total += int(self.input_weights.size)
        total += int(self.output_weights.size)
        total += int(self.tc_W_h.size + self.tc_W_x.size + self.tc_b.size)
        total += int(self._membrane_capacitance.size + self._bias.size)
        if self.freq_gate_weights is not None:
            total += int(self.freq_gate_weights.size)
        if self.mvm_weights is not None:
            total += int(self.mvm_weights.size)
        return int(total)

    def compression_factor(self) -> Optional[float]:
        active = int(self.total_parameter_count())
        dense = int(self.equivalent_dense_parameter_count())
        if active <= 0:
            return None
        return float(dense / active)

    def auxiliary_head_manifest(self) -> Dict[str, Any]:
        manifest: Dict[str, Any] = {}
        if self.freq_gate_weights is not None:
            manifest["freq_gates"] = {"size": int(self.num_freq_gates), "activation": "sigmoid"}
        if self.mvm_weights is not None:
            manifest["mvm_predictions"] = {"size": int(self.num_mvm_outputs), "activation": "linear"}
        return manifest

    def state_hash(self) -> str:
        return _sha256_json({
            "scheme": "akkurat.ncp.state.v1",
            "policy_id": self.policy_id,
            "lobe_id": self.lobe_id,
            "step_count": int(self._step_count),
            "hidden_state_hash": _array_hash(self._state, dtype=np.float64, tag="hidden_state"),
        })

    def config_hash(self) -> str:
        return _sha256_json({"scheme": "akkurat.ncp.config.v1", "config": self.config_dict()})

    def recurrent_commitment_hash(self) -> str:
        if self._use_tt and self.recurrent_weights is not None:
            cores = getattr(self.recurrent_weights, "cores_data", None)
            if isinstance(cores, (list, tuple)):
                core_hashes = [
                    _array_hash(_np_from_any_array(c), dtype=np.float32, tag=f"tt_core_{i}")
                    for i, c in enumerate(cores)
                ]
                return _sha256_json({
                    "scheme": "akkurat.ncp.recurrent_tt.v1",
                    "output_dims": list(getattr(self.recurrent_weights, "output_dims", list(self._tt_out_shape))),
                    "input_dims": list(getattr(self.recurrent_weights, "input_dims", list(self._tt_in_shape))),
                    "bond_ranks": list(self._tt_bond_dims_actual),
                    "core_hashes": core_hashes,
                })
            if hasattr(self.recurrent_weights, "commitment"):
                try:
                    return str(self.recurrent_weights.commitment().commitment_hash)
                except Exception:
                    pass
        if self.recurrent_dense is not None:
            return _array_hash(self.recurrent_dense, dtype=np.float64, tag="recurrent_dense")
        return ""

    def weights_manifest(self) -> Dict[str, Any]:
        payload = {
            "scheme": "akkurat.ncp.weights_manifest.v1",
            "policy_id": self.policy_id,
            "lobe_id": self.lobe_id,
            "recurrent_mode": "tt" if self._use_tt else "dense",
            "recurrent_commitment_hash": self.recurrent_commitment_hash(),
            "input_weights_hash": _array_hash(self.input_weights, dtype=np.float64, tag="input_weights"),
            "output_weights_hash": _array_hash(self.output_weights, dtype=np.float64, tag="output_weights"),
            "tc_W_h_hash": _array_hash(self.tc_W_h, dtype=np.float64, tag="tc_W_h"),
            "tc_W_x_hash": _array_hash(self.tc_W_x, dtype=np.float64, tag="tc_W_x"),
            "tc_b_hash": _array_hash(self.tc_b, dtype=np.float64, tag="tc_b"),
            "membrane_capacitance_hash": _array_hash(self._membrane_capacitance, dtype=np.float64, tag="membrane_capacitance"),
            "bias_hash": _array_hash(self._bias, dtype=np.float64, tag="bias"),
            "freq_gate_weights_hash": _array_hash(self.freq_gate_weights, dtype=np.float64, tag="freq_gate_weights") if self.freq_gate_weights is not None else "",
            "mvm_weights_hash": _array_hash(self.mvm_weights, dtype=np.float64, tag="mvm_weights") if self.mvm_weights is not None else "",
        }
        payload["weights_hash"] = _sha256_json(payload)
        return payload

    def weights_hash(self) -> str:
        return str(self.weights_manifest()["weights_hash"])

    def lobe_commitment_manifest(self, *, checkpoint_hash: str = "") -> LobeCommitmentManifest:
        manifest = LobeCommitmentManifest(
            scheme=NCP_COMMITMENT_SCHEME,
            version=NCP_VERSION,
            lobe_id=self.lobe_id,
            lobe_kind=self.lobe_kind,
            config_hash=self.config_hash(),
            scale_profile_hash=self.scale_profile.hash(),
            state_hash=self.state_hash(),
            weights_hash=self.weights_hash(),
            recurrent_commitment_hash=self.recurrent_commitment_hash(),
            checkpoint_hash=str(checkpoint_hash or ""),
            parameter_count=int(self.total_parameter_count()),
            dense_equivalent_parameter_count=int(self.equivalent_dense_parameter_count()),
            compression_factor=self.compression_factor(),
            consensus_safe=bool(self.consensus_safe),
        )
        return LobeCommitmentManifest(**{**manifest.payload(), "manifest_hash": manifest.compute_hash()})

    def lobe_commitment_hash(self) -> str:
        return self.lobe_commitment_manifest().compute_hash()

    def verify_lobe_commitment(self, manifest: Union[LobeCommitmentManifest, Mapping[str, Any]]) -> bool:
        if isinstance(manifest, LobeCommitmentManifest):
            expected = manifest
        else:
            p = dict(manifest)
            expected = LobeCommitmentManifest(
                scheme=str(p.get("scheme", NCP_COMMITMENT_SCHEME)),
                version=str(p.get("version", NCP_VERSION)),
                lobe_id=str(p.get("lobe_id", "")),
                lobe_kind=str(p.get("lobe_kind", "")),
                config_hash=str(p.get("config_hash", "")),
                scale_profile_hash=str(p.get("scale_profile_hash", "")),
                state_hash=str(p.get("state_hash", "")),
                weights_hash=str(p.get("weights_hash", "")),
                recurrent_commitment_hash=str(p.get("recurrent_commitment_hash", "")),
                checkpoint_hash=str(p.get("checkpoint_hash", "")),
                parameter_count=int(p.get("parameter_count", 0)),
                dense_equivalent_parameter_count=int(p.get("dense_equivalent_parameter_count", 0)),
                compression_factor=p.get("compression_factor", None),
                consensus_safe=bool(p.get("consensus_safe", False)),
                manifest_hash=str(p.get("manifest_hash", "")),
            )
        current = self.lobe_commitment_manifest(checkpoint_hash=expected.checkpoint_hash)
        return bool(current.to_dict() == expected.to_dict())

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        state = self._state
        tau = self._last_dynamic_tau
        return {
            "policy_id": self.policy_id,
            "lobe_id": self.lobe_id,
            "lobe_kind": self.lobe_kind,
            "kind": "tensorized_ncp_liquid_reflex_lobe",
            "version": NCP_VERSION,
            "num_inputs": self.num_inputs,
            "num_hidden": self.num_hidden,
            "num_outputs": self.num_outputs,
            "recurrent_mode": "tt" if self._use_tt else "dense",
            "tt_available": bool(TENSORTRAIN_AVAILABLE),
            "tt_in_shape": list(self._tt_in_shape),
            "tt_out_shape": list(self._tt_out_shape),
            "tt_bond_dims": list(self._tt_bond_dims_actual),
            "tt_init_error": self._tt_init_error,
            "output_activation": self.output_activation,
            "output_scale": float(self.output_scale),
            "hidden_state_norm": float(np.linalg.norm(state)),
            "hidden_state_mean": float(np.mean(state)) if state.size else 0.0,
            "hidden_state_std": float(np.std(state)) if state.size else 0.0,
            "tau_min": float(np.min(tau)) if tau.size else 0.0,
            "tau_mean": float(np.mean(tau)) if tau.size else 0.0,
            "tau_max": float(np.max(tau)) if tau.size else 0.0,
            "parameter_count": self.parameter_count(),
            "total_parameter_count": self.total_parameter_count(),
            "dense_equivalent_parameter_count": self.equivalent_dense_parameter_count(),
            "compression_factor": self.compression_factor(),
            "auxiliary_heads": self.auxiliary_head_manifest(),
            "scale_profile": self.scale_profile.to_dict(),
            "scale_profile_hash": self.scale_profile.hash(),
            "state_hash": self.state_hash(),
            "weights_hash": self.weights_hash(),
            "lobe_commitment_hash": self.lobe_commitment_hash(),
            "consensus_safe": bool(self.consensus_safe),
            "deterministic": bool(self.deterministic),
            "step_count": int(self._step_count),
            "created_at": self._created_at,
            "last_maintenance_at": self._last_maintenance_at,
            "last_error": self._last_error,
        }

    def estimate_recurrent_gain_power(self, iters: int = 12, seed: Optional[int] = None) -> float:
        rng = np.random.default_rng(seed if seed is not None else 12345)
        v = rng.normal(size=(self.num_hidden,)).astype(np.float64)
        n = _safe_norm(v)
        if n <= _EPS:
            return 0.0
        v /= n
        last_n = 0.0
        for _ in range(int(max(1, iters))):
            w = self._recurrent_matvec(v)
            last_n = _safe_norm(w)
            if last_n <= _EPS:
                return 0.0
            v = w / last_n
        return float(last_n)

    def health_metrics(self, include_expensive: bool = False) -> Dict[str, Any]:
        state = self._state
        tau = self._last_dynamic_tau
        metrics: Dict[str, Any] = {
            "policy_id": self.policy_id,
            "lobe_id": self.lobe_id,
            "lobe_kind": self.lobe_kind,
            "recurrent_mode": "tt" if self._use_tt else "dense",
            "hidden_norm": float(np.linalg.norm(state)),
            "hidden_mean": float(np.mean(state)) if state.size else 0.0,
            "hidden_std": float(np.std(state)) if state.size else 0.0,
            "hidden_saturation_fraction": _saturation_fraction(state),
            "tau_min": float(np.min(tau)) if tau.size else 0.0,
            "tau_mean": float(np.mean(tau)) if tau.size else 0.0,
            "tau_max": float(np.max(tau)) if tau.size else 0.0,
            "tau_span": float(np.max(tau) - np.min(tau)) if tau.size else 0.0,
            "input_signal_norm": float(np.linalg.norm(self._last_input_signal)),
            "recurrent_signal_norm": float(np.linalg.norm(self._last_recurrent_signal)),
            "total_current_norm": float(np.linalg.norm(self._last_total_currents)),
            "output_norm": float(np.linalg.norm(self._last_output)),
            "nonfinite_hidden_count": _count_nonfinite(state),
            "nonfinite_tau_count": _count_nonfinite(tau),
            "nonfinite_output_count": _count_nonfinite(self._last_output),
            "recurrent_gain_estimate": None,
            "is_saturated": bool(_saturation_fraction(state) >= 0.25),
            "has_nan": bool(_count_nonfinite(state) > 0 or _count_nonfinite(tau) > 0 or _count_nonfinite(self._last_output) > 0),
            "degraded_mode": bool(not self._use_tt and self._tt_init_error is not None),
            "parameter_count": int(self.total_parameter_count()),
            "dense_equivalent_parameter_count": int(self.equivalent_dense_parameter_count()),
            "compression_factor": self.compression_factor(),
            "state_hash": self.state_hash(),
            "weights_hash": self.weights_hash(),
            "scale_profile_hash": self.scale_profile.hash(),
            "step_count": int(self._step_count),
        }
        if include_expensive:
            if self.recurrent_dense is not None:
                metrics["recurrent_gain_estimate"] = _safe_spectral_norm(self.recurrent_dense)
            else:
                metrics["recurrent_gain_estimate"] = self.estimate_recurrent_gain_power(iters=12)
        metrics["is_stable"] = bool(
            not metrics["has_nan"]
            and metrics["hidden_norm"] < (self.num_hidden * 1.25)
            and metrics["tau_min"] > 0.0
            and metrics["output_norm"] < max(1.0, self.num_outputs * 10.0)
        )
        return metrics

    def lobe_health_receipt(self, *, include_expensive: bool = False) -> LobeHealthReceipt:
        h = self.health_metrics(include_expensive=include_expensive)
        receipt = LobeHealthReceipt(
            scheme="akkurat.lobe.health_receipt.v1",
            version=NCP_VERSION,
            lobe_id=self.lobe_id,
            lobe_kind=self.lobe_kind,
            step_count=int(self._step_count),
            recurrent_mode=str(h["recurrent_mode"]),
            is_stable=bool(h["is_stable"]),
            is_saturated=bool(h["is_saturated"]),
            has_nan=bool(h["has_nan"]),
            degraded_mode=bool(h["degraded_mode"]),
            hidden_norm=float(h["hidden_norm"]),
            output_norm=float(h["output_norm"]),
            recurrent_gain_estimate=h.get("recurrent_gain_estimate"),
            parameter_count=int(h["parameter_count"]),
            dense_equivalent_parameter_count=int(h["dense_equivalent_parameter_count"]),
            compression_factor=h.get("compression_factor"),
            scale_profile_hash=str(h["scale_profile_hash"]),
            state_hash=str(h["state_hash"]),
            weights_hash=str(h["weights_hash"]),
        )
        return LobeHealthReceipt(**{**receipt.payload(), "receipt_hash": receipt.compute_hash()})

    def validate_consensus_safe(self, *, raise_on_error: bool = False) -> Dict[str, Any]:
        errors: List[str] = []
        if not self.consensus_safe:
            errors.append("config.consensus_safe_false")
        if not self.deterministic:
            errors.append("config.deterministic_false")
        if self.config.rng_seed is None:
            errors.append("missing_rng_seed")
        if self.config.tt_device != "cpu":
            errors.append("tt_device_not_cpu")
        if not self.check_finite:
            errors.append("check_finite_false")
        if self._use_tt and self.recurrent_weights is not None:
            if hasattr(self.recurrent_weights, "is_consensus_safe_backend"):
                try:
                    if not bool(self.recurrent_weights.is_consensus_safe_backend()):
                        errors.append("tt_backend_not_consensus_safe")
                except Exception:
                    errors.append("tt_backend_check_failed")
        elif self._tt_init_error is not None:
            errors.append("tt_degraded_to_dense")
        ok = len(errors) == 0
        out = {"ok": ok, "errors": errors, "lobe_id": self.lobe_id}
        if raise_on_error and not ok:
            raise RuntimeError(f"TensorizedNCP consensus-safe validation failed: {errors}")
        return out

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def apply_regularization(self, target_norm: float = 1.0, *, use_power_estimate: bool = True) -> None:
        target_norm = float(target_norm)
        if target_norm <= 0:
            raise ValueError("target_norm must be > 0")
        if not self._use_tt:
            if self.recurrent_dense is None:
                return
            current_norm = _safe_spectral_norm(self.recurrent_dense)
            if current_norm is not None and current_norm > target_norm and current_norm > 0:
                self.recurrent_dense *= target_norm / current_norm
                self._last_maintenance_at = _utc_ts()
            return
        if self.recurrent_weights is None:
            return
        try:
            current_norm = self.estimate_recurrent_gain_power(iters=12) if use_power_estimate else None
            if current_norm is not None and current_norm > target_norm and current_norm > 0:
                self.recurrent_weights = self.recurrent_weights.scale(float(target_norm / current_norm))
                if hasattr(self.recurrent_weights, "round"):
                    self.recurrent_weights = self.recurrent_weights.round(
                        max_rank=max(self._tt_bond_dims_actual) if self._tt_bond_dims_actual else self.bond_dim,
                        energy_tol=0.999,
                    )
                self._recurrent_param_count = int(self.recurrent_weights.parameter_count()) if hasattr(self.recurrent_weights, "parameter_count") else self._recurrent_param_count
                self._last_maintenance_at = _utc_ts()
        except Exception as exc:
            self._last_error = f"regularization_failed:{exc}"
            if self.strict_runtime_checks:
                raise

    def canonicalize_recurrent(self, direction: str = "left") -> None:
        if not self._use_tt or self.recurrent_weights is None:
            return
        direction = str(direction).lower().strip()
        if direction == "left":
            self.recurrent_weights = self.recurrent_weights.left_orthonormalize()
        elif direction == "right":
            self.recurrent_weights = self.recurrent_weights.right_orthonormalize()
        else:
            raise ValueError("direction must be 'left' or 'right'")
        self._recurrent_param_count = int(self.recurrent_weights.parameter_count())
        self._last_maintenance_at = _utc_ts()

    def round_recurrent(self, max_rank: Optional[int] = None, energy_tol: Optional[float] = 0.999) -> Dict[str, Any]:
        if not self._use_tt or self.recurrent_weights is None:
            return {"status": "skipped", "reason": "not_tt"}
        if hasattr(self.recurrent_weights, "round_with_report"):
            self.recurrent_weights, report = self.recurrent_weights.round_with_report(
                max_rank=max_rank if max_rank is not None else (max(self._tt_bond_dims_actual) if self._tt_bond_dims_actual else self.bond_dim),
                energy_tol=energy_tol,
            )
            out = report.to_dict() if hasattr(report, "to_dict") else _json_safe(report)
        else:
            self.recurrent_weights = self.recurrent_weights.round(
                max_rank=max_rank if max_rank is not None else (max(self._tt_bond_dims_actual) if self._tt_bond_dims_actual else self.bond_dim),
                energy_tol=energy_tol,
            )
            out = {"status": "rounded"}
        self._recurrent_param_count = int(self.recurrent_weights.parameter_count())
        self._tt_bond_dims_actual = list(self.recurrent_weights.bond_ranks()) if hasattr(self.recurrent_weights, "bond_ranks") else self._tt_bond_dims_actual
        self._last_maintenance_at = _utc_ts()
        return out

    # ------------------------------------------------------------------
    # Forward step
    # ------------------------------------------------------------------
    def step(self, inputs: Any, dt: float = 1.0, return_state: bool = False):
        if isinstance(inputs, Mapping):
            raise TypeError(
                f"[{self.policy_id}] TensorizedNCP.step() accepts numeric vectors only. "
                "Structured inputs must be encoded by the calling lobe/runtime."
            )
        x = _coerce_float64_vector(inputs, expected_dim=self.num_inputs, name="inputs")
        _safe_clip_inplace(x, -float(self.config.input_clip), float(self.config.input_clip))
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0:
            dt = 1.0
        dt = float(np.clip(dt, 1e-9, max(1e-9, self.config.max_dt)))
        self._step_count += 1
        self._last_input = x.copy()
        try:
            input_signal = self.input_weights @ x
            recurrent_signal = self._recurrent_matvec(self._state)
            total_currents = input_signal + recurrent_signal
            _safe_clip_inplace(total_currents, -float(self.config.current_clip), float(self.config.current_clip))
            dynamic_tau = self._dynamic_tau(x)
            self._vectorized_ltc_update(total_currents, dynamic_tau, dt)
            out = self.output_weights @ self._state
            out = self._activate_output(out)
            aux_outputs: Dict[str, Any] = {}
            if self.freq_gate_weights is not None:
                aux_outputs["freq_gates"] = _sigmoid(self.freq_gate_weights @ self._state).astype(np.float64, copy=False)
            if self.mvm_weights is not None:
                aux_outputs["mvm_predictions"] = _sanitize_vector(self.mvm_weights @ self._state)
            self._last_input_signal = _sanitize_vector(input_signal)
            self._last_recurrent_signal = _sanitize_vector(recurrent_signal)
            self._last_total_currents = _sanitize_vector(total_currents)
            self._last_dynamic_tau = _sanitize_vector(dynamic_tau)
            self._last_output = out.copy()
            self._last_aux_outputs = {k: np.asarray(v, dtype=np.float64).copy() for k, v in aux_outputs.items() if isinstance(v, np.ndarray)}
            self._last_error = None
            if self.strict_runtime_checks:
                if _count_nonfinite(self._state) > 0 or _count_nonfinite(self._last_dynamic_tau) > 0 or _count_nonfinite(self._last_output) > 0:
                    raise FloatingPointError(f"[{self.policy_id}] non-finite runtime values detected")
        except Exception as exc:
            self._last_error = str(exc)
            raise
        if return_state:
            aux = {
                "hidden_state": self._state.copy(),
                "dynamic_tau": self._last_dynamic_tau.copy(),
                "input_signal": self._last_input_signal.copy(),
                "recurrent_signal": self._last_recurrent_signal.copy(),
                "total_currents": self._last_total_currents.copy(),
                "output_vector": self._last_output.copy(),
                "runtime_flags": {
                    "recurrent_mode": "tt" if self._use_tt else "dense",
                    "strict_runtime_checks": bool(self.strict_runtime_checks),
                    "tt_degraded_to_dense": bool(not self._use_tt and self._tt_init_error is not None),
                    "consensus_safe": bool(self.consensus_safe),
                    "step_count": int(self._step_count),
                },
                "health": self.health_metrics(include_expensive=False),
                "health_receipt": self.lobe_health_receipt(include_expensive=False).to_dict(),
                "lobe_commitment": self.lobe_commitment_manifest().to_dict(),
            }
            aux.update(aux_outputs)
            return out.copy(), aux
        return out.copy()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _checkpoint_arrays_without_manifest(self) -> Dict[str, Any]:
        arrays: Dict[str, Any] = {
            "checkpoint_version": np.asarray(self.CHECKPOINT_VERSION, dtype=np.int64),
            "format": np.asarray(NCP_CHECKPOINT_FORMAT),
            "config_json": np.asarray(_json_dumps_stable(self.config_dict())),
            "scale_profile_json": np.asarray(_json_dumps_stable(self.scale_profile.to_dict())),
            "snapshot_json": np.asarray(_json_dumps_stable(self.snapshot())),
            "state": self._state.astype(np.float64),
            "membrane_capacitance": self._membrane_capacitance.astype(np.float64),
            "bias": self._bias.astype(np.float64),
            "input_weights": self.input_weights.astype(np.float64),
            "output_weights": self.output_weights.astype(np.float64),
            "tc_W_h": self.tc_W_h.astype(np.float64),
            "tc_W_x": self.tc_W_x.astype(np.float64),
            "tc_b": self.tc_b.astype(np.float64),
            "tau_min_arr": self._tau_min_arr.astype(np.float64),
            "tau_max_arr": self._tau_max_arr.astype(np.float64),
            "step_count": np.asarray(self._step_count, dtype=np.int64),
            "created_at": np.asarray(self._created_at),
            "recurrent_mode": np.asarray("tt" if self._use_tt else "dense"),
            "tt_in_shape": np.asarray(self._tt_in_shape, dtype=np.int64),
            "tt_out_shape": np.asarray(self._tt_out_shape, dtype=np.int64),
            "tt_bond_dims_actual": np.asarray(self._tt_bond_dims_actual, dtype=np.int64),
        }
        if self.freq_gate_weights is not None:
            arrays["freq_gate_weights"] = self.freq_gate_weights.astype(np.float64)
        if self.mvm_weights is not None:
            arrays["mvm_weights"] = self.mvm_weights.astype(np.float64)
        if self.recurrent_dense is not None:
            arrays["recurrent_dense"] = self.recurrent_dense.astype(np.float64)
        if self.recurrent_weights is not None and hasattr(self.recurrent_weights, "cores_data"):
            cores = getattr(self.recurrent_weights, "cores_data")
            arrays["tt_num_cores"] = np.asarray(len(cores), dtype=np.int64)
            for i, core in enumerate(cores):
                arrays[f"tt_core_{i}"] = _np_from_any_array(core).astype(np.float32)
        return arrays

    @staticmethod
    def _arrays_manifest_hash(arrays: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
        entries: Dict[str, Any] = {}
        for k in sorted(arrays.keys()):
            if str(k).startswith("__"):
                continue
            v = arrays[k]
            if isinstance(v, np.ndarray):
                arr = v
            else:
                arr = np.asarray(v)
            if arr.dtype.kind in {"U", "S", "O"}:
                raw = str(arr.item() if arr.shape == () else arr.tolist()).encode("utf-8", errors="ignore")
                entries[k] = {"kind": "text", "shape": list(map(int, arr.shape)), "sha256": _sha256_bytes(raw)}
            else:
                raw_arr = np.asarray(arr)
                entries[k] = {
                    "kind": "array",
                    "shape": list(map(int, raw_arr.shape)),
                    "dtype": str(raw_arr.dtype),
                    "sha256": _sha256_bytes(np.ascontiguousarray(raw_arr).tobytes()),
                }
        manifest = {"format": NCP_CHECKPOINT_FORMAT, "version": TensorizedNCP.CHECKPOINT_VERSION, "entries": entries}
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
        _atomic_write_npz(path, arrays)

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, os.PathLike[str]],
        *,
        rng: Optional[np.random.Generator] = None,
        verify: bool = True,
    ) -> "TensorizedNCP":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        with np.load(str(p), allow_pickle=False) as data:
            if verify:
                if "__checkpoint_manifest_json__" not in data or "__checkpoint_manifest_hash__" not in data:
                    raise ValueError("checkpoint missing integrity manifest")
                manifest_json = str(data["__checkpoint_manifest_json__"].item())
                manifest_hash = str(data["__checkpoint_manifest_hash__"].item())
                if _sha256_json(json.loads(manifest_json)) != manifest_hash:
                    raise ValueError("checkpoint manifest hash mismatch")
                arrays_for_hash = {k: data[k] for k in data.files if not k.startswith("__")}
                _, actual_checkpoint_hash = cls._arrays_manifest_hash(arrays_for_hash)
                expected_checkpoint_hash = str(data["__checkpoint_hash__"].item()) if "__checkpoint_hash__" in data.files else json.loads(manifest_json).get("checkpoint_hash", "")
                if actual_checkpoint_hash != expected_checkpoint_hash:
                    raise ValueError("checkpoint array integrity hash mismatch")

            cfg_json = str(data["config_json"].item())
            cfg_dict = json.loads(cfg_json)
            cfg = TensorizedNCPConfig(**cfg_dict).normalized()
            obj = cls.from_config(cfg, rng=rng)
            obj._state = np.asarray(data["state"], dtype=np.float64).reshape(obj.num_hidden)
            obj._membrane_capacitance = np.asarray(data["membrane_capacitance"], dtype=np.float64).reshape(obj.num_hidden)
            obj._bias = np.asarray(data["bias"], dtype=np.float64).reshape(obj.num_hidden)
            obj.input_weights = np.asarray(data["input_weights"], dtype=np.float64).reshape(obj.num_hidden, obj.num_inputs)
            obj.output_weights = np.asarray(data["output_weights"], dtype=np.float64).reshape(obj.num_outputs, obj.num_hidden)
            obj.tc_W_h = np.asarray(data["tc_W_h"], dtype=np.float64).reshape(obj.num_hidden, obj.num_hidden)
            obj.tc_W_x = np.asarray(data["tc_W_x"], dtype=np.float64).reshape(obj.num_hidden, obj.num_inputs)
            obj.tc_b = np.asarray(data["tc_b"], dtype=np.float64).reshape(obj.num_hidden)
            obj._tau_min_arr = np.asarray(data["tau_min_arr"], dtype=np.float64).reshape(obj.num_hidden)
            obj._tau_max_arr = np.asarray(data["tau_max_arr"], dtype=np.float64).reshape(obj.num_hidden)
            obj._step_count = int(np.asarray(data["step_count"]).item()) if "step_count" in data.files else 0
            obj._created_at = str(data["created_at"].item()) if "created_at" in data.files else _utc_ts()
            if "scale_profile_json" in data.files:
                obj.scale_profile = LobeScaleProfile(**json.loads(str(data["scale_profile_json"].item()))).normalized()
            if "freq_gate_weights" in data.files:
                obj.freq_gate_weights = np.asarray(data["freq_gate_weights"], dtype=np.float64).reshape(obj.num_freq_gates, obj.num_hidden)
            if "mvm_weights" in data.files:
                obj.mvm_weights = np.asarray(data["mvm_weights"], dtype=np.float64).reshape(obj.num_mvm_outputs, obj.num_hidden)

            mode = str(data["recurrent_mode"].item()) if "recurrent_mode" in data.files else ("tt" if obj._use_tt else "dense")
            obj._tt_in_shape = tuple(int(x) for x in np.asarray(data["tt_in_shape"], dtype=np.int64).tolist()) if "tt_in_shape" in data.files else obj._tt_in_shape
            obj._tt_out_shape = tuple(int(x) for x in np.asarray(data["tt_out_shape"], dtype=np.int64).tolist()) if "tt_out_shape" in data.files else obj._tt_out_shape
            obj._tt_bond_dims_actual = [int(x) for x in np.asarray(data["tt_bond_dims_actual"], dtype=np.int64).tolist()] if "tt_bond_dims_actual" in data.files else obj._tt_bond_dims_actual

            if mode == "dense" and "recurrent_dense" in data.files:
                obj._use_tt = False
                obj.recurrent_dense = np.asarray(data["recurrent_dense"], dtype=np.float64).reshape(obj.num_hidden, obj.num_hidden)
                obj.recurrent_weights = None
                obj._recurrent_param_count = int(obj.recurrent_dense.size)
            elif mode == "tt" and "tt_num_cores" in data.files:
                if not TENSORTRAIN_AVAILABLE:
                    raise RuntimeError("Cannot load TT checkpoint because TensorTrain is unavailable")
                ncores = int(np.asarray(data["tt_num_cores"]).item())
                cores = [np.asarray(data[f"tt_core_{i}"], dtype=np.float32) for i in range(ncores)]
                try:
                    obj.recurrent_weights = TensorTrain(
                        output_dims=list(obj._tt_out_shape),
                        input_dims=list(obj._tt_in_shape),
                        cores_data=cores,
                        config=obj._make_tt_config(),
                        metadata={"source": "TensorizedNCP.load_checkpoint"},
                        commit_metadata={"lobe_id": obj.lobe_id, "operator": "ncp_recurrent"},
                    )
                except TypeError:
                    obj.recurrent_weights = TensorTrain(
                        output_dims=list(obj._tt_out_shape),
                        input_dims=list(obj._tt_in_shape),
                        cores_data=cores,
                        config=obj._make_tt_config(),
                    )
                obj.recurrent_dense = None
                obj._use_tt = True
                obj._recurrent_param_count = int(obj.recurrent_weights.parameter_count()) if hasattr(obj.recurrent_weights, "parameter_count") else None
            elif "recurrent_dense" in data.files:
                obj._use_tt = False
                obj.recurrent_dense = np.asarray(data["recurrent_dense"], dtype=np.float64).reshape(obj.num_hidden, obj.num_hidden)
                obj.recurrent_weights = None
                obj._recurrent_param_count = int(obj.recurrent_dense.size)
            obj.sync_neuron_mirror_from_vectors()
            if obj.consensus_safe:
                obj.validate_consensus_safe(raise_on_error=True)
            return obj

    def __repr__(self) -> str:
        mode = "tt" if self._use_tt else "dense"
        return (
            f"TensorizedNCP(id='{self.policy_id}', lobe_id='{self.lobe_id}', mode='{mode}', "
            f"hidden={self.num_hidden}, active_params={self.total_parameter_count():,}, "
            f"dense_equiv={self.equivalent_dense_parameter_count():,})"
        )


# =============================================================================
# Minimal self-test
# =============================================================================

def _self_test() -> None:
    print("=== TensorizedNCP Liquid Reflex Lobe self-test ===")
    ncp = TensorizedNCP(
        policy_id="selftest_ncp",
        lobe_id="liquid_reflex_selftest",
        num_inputs=8,
        num_hidden=64,
        num_outputs=12,
        bond_dim=4,
        rng_seed=42,
        deterministic=True,
        verbose=True,
        output_activation="tanh",
        leaky_lambda=0.5,
        tau_min=2.0,
        tau_max=18.0,
        num_freq_gates=2,
        num_mvm_outputs=3,
        tt_strict=False,
        mirror_neurons_each_step=False,
    )
    rng = np.random.default_rng(7)
    x = rng.normal(size=(8,))
    y, aux = ncp.step(x, return_state=True)
    assert y.shape == (12,)
    assert aux["hidden_state"].shape == (64,)
    assert np.isfinite(y).all()
    assert aux["health"]["is_stable"]
    receipt = ncp.lobe_health_receipt().to_dict()
    assert receipt["receipt_hash"]
    manifest = ncp.lobe_commitment_manifest().to_dict()
    assert manifest["manifest_hash"]
    assert ncp.verify_lobe_commitment(manifest)

    payload = ncp.serialize_state()
    old_state = ncp.get_state().copy()
    ncp.reset_state()
    ncp.load_state(payload)
    assert np.allclose(old_state, ncp.get_state())

    tmp = Path(os.getenv("NCP_SELFTEST_CHECKPOINT", "./_ncp_selftest_checkpoint.npz"))
    ncp.save_checkpoint(tmp)
    loaded = TensorizedNCP.load_checkpoint(tmp, verify=True)
    assert loaded.num_hidden == ncp.num_hidden
    assert np.allclose(loaded.get_state(), ncp.get_state())
    assert loaded.weights_hash() == ncp.weights_hash()
    try:
        tmp.unlink()
    except Exception:
        pass

    # Determinism probe.
    a = TensorizedNCP("det", 4, 32, 3, bond_dim=4, rng_seed=2025, deterministic=True, verbose=False, tt_strict=False)
    b = TensorizedNCP("det", 4, 32, 3, bond_dim=4, rng_seed=2025, deterministic=True, verbose=False, tt_strict=False)
    z = np.random.default_rng(3).normal(size=4)
    ya = a.step(z)
    yb = b.step(z)
    assert np.allclose(ya, yb, atol=1e-6, rtol=1e-6), "determinism probe failed"

    print("Snapshot:", json.dumps(_json_safe(ncp.snapshot()), indent=2)[:1200] + "...")
    print("=== TensorizedNCP self-test passed ===")


if __name__ == "__main__":
    _self_test()
