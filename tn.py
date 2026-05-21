#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Project Chimera / Akkurat - Tensor Network Runtime
==================================================

Production CPU-first TensorTrain / MPO + Tucker tensor-network substrate for
Tensorized Blockchain, governed digital twins, recurrent lobes, compression,
and deterministic operator experiments.

Design boundaries
-----------------
This module owns numerical tensor algebra only. It does not own blockchain
consensus, digital-twin governance, orchestration, tool routing, or lobe policy.
Higher layers consume this module through deterministic TensorTrain operators,
commitments, compression reports, and verifiable NPZ artifacts.

Public API
----------
- TN_VERSION
- TTConfig
- TensorCommitment
- TTRoundingReport
- TensorTrain
- TuckerTensor
- TuckerFusionLayer
- TreeTensorFusion
- DendriticProcessor
- tensor_train_svd
- find_optimal_permutations
- get_factors
- factorize_into_modes

Production upgrades
-------------------
- Deterministic TTConfig.seed support.
- Device-independent TensorCommitment objects.
- TensorTrain.commitment() and TensorTrain.verify_commitment(...).
- round_with_report(...) with compression/fidelity diagnostics.
- NPZ manifests with core hashes and manifest hash verification.
- Dense materialization guards by default.
- Consensus-safe CPU/dtype checks.
- Stable float64 TT-SVD working precision before stored-core casting.

Dependencies
------------
Required: NumPy.
Optional: SciPy for CPU SVD speed/stability, CuPy for non-consensus GPU use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import hashlib
import json
import math
import os
import tempfile
import warnings

import numpy as np

TN_VERSION = "tensor-network-runtime-v1.0-production"
TN_COMMITMENT_SCHEME = "ProjectChimera.TensorTrainCommitment.v1"
TN_NPZ_FORMAT = "ProjectChimera.TensorTrainNPZ.v2"
DEFAULT_MAX_DENSE_ENTRIES = 1_000_000

try:  # Optional CPU SVD backend.
    import scipy.linalg as sla  # type: ignore

    _SCIPY_OK = True
except Exception:  # pragma: no cover - optional dependency
    sla = None  # type: ignore
    _SCIPY_OK = False

try:  # Optional GPU backend. Not consensus-safe by default.
    import cupy as cp  # type: ignore

    _CUPY_OK = True
except Exception:  # pragma: no cover - optional dependency
    cp = None  # type: ignore
    _CUPY_OK = False

ArrayLike = Union[np.ndarray, "cp.ndarray"] if _CUPY_OK else np.ndarray


# =============================================================================
# Canonical JSON / hashing helpers
# =============================================================================


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int)):
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
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return _json_safe(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "__dataclass_fields__"):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in sorted(dict(obj).items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    return str(obj)


def _json_dumps_stable(obj: Any) -> str:
    return json.dumps(_json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(obj: Any) -> str:
    return _sha256_bytes(_json_dumps_stable(obj).encode("utf-8"))


def _domain_hash(domain: str, *parts: bytes) -> str:
    h = hashlib.sha256()
    h.update(str(domain).encode("utf-8"))
    h.update(b"\x00")
    for p in parts:
        h.update(len(p).to_bytes(8, "big", signed=False))
        h.update(p)
    return h.hexdigest()


def _atomic_save_npz(path: Union[str, os.PathLike], arrays: Mapping[str, Any], *, compressed: bool = True) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        if compressed:
            np.savez_compressed(tmp, **dict(arrays))
        else:
            np.savez(tmp, **dict(arrays))
        # numpy appends .npz if suffix is not .npz
        actual_npz = Path(str(tmp) + ".npz")
        actual = actual_npz if actual_npz.exists() else tmp
        os.replace(actual, p)
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
    except Exception:
        for candidate in (tmp, Path(str(tmp) + ".npz")):
            try:
                if candidate.exists():
                    candidate.unlink()
            except Exception:
                pass
        raise


# =============================================================================
# Backend / numerical helpers
# =============================================================================


def get_factors(n: int) -> Tuple[int, int]:
    """Return near-square factors (a, b) with a*b == n and a <= sqrt(n)."""
    n = int(n)
    if n <= 0:
        return (1, 1)
    a = int(math.isqrt(n))
    while a > 0:
        if n % a == 0:
            return (a, n // a)
        a -= 1
    return (1, n)


def factorize_into_modes(n: int, num_modes: int = 2) -> List[int]:
    """Deterministically factor n into num_modes positive mode sizes."""
    n = int(n)
    num_modes = int(num_modes)
    if n <= 0:
        raise ValueError("n must be positive.")
    if num_modes <= 0:
        raise ValueError("num_modes must be positive.")
    if num_modes == 1:
        return [n]

    modes: List[int] = []
    remaining = n
    slots = num_modes
    while slots > 1:
        a, b = get_factors(remaining)
        modes.append(int(a))
        remaining = int(b)
        slots -= 1
    modes.append(int(remaining))
    if _prod(modes) != n:
        raise RuntimeError("factorize_into_modes produced invalid factorization.")
    return modes


def _prod(ints: Sequence[int]) -> int:
    p = 1
    for x in ints:
        p *= int(x)
    return int(p)


def _is_cupy_array(x: Any) -> bool:
    return bool(_CUPY_OK and hasattr(x, "__cuda_array_interface__"))


def _get_xp_from_arrays(*arrays: Any, prefer_gpu: bool = False):
    if _CUPY_OK:
        if prefer_gpu:
            return cp
        for a in arrays:
            if a is not None and _is_cupy_array(a):
                return cp
    return np


def _to_cpu(x: ArrayLike) -> np.ndarray:
    if _is_cupy_array(x):
        return cp.asnumpy(x)  # type: ignore[union-attr]
    return np.asarray(x)


def _to_device_array(x: ArrayLike, xp, dtype: np.dtype) -> ArrayLike:
    dt = np.dtype(dtype)
    if xp is np:
        return np.asarray(_to_cpu(x), dtype=dt)
    if not _CUPY_OK:
        raise RuntimeError("CuPy is not available.")
    if _is_cupy_array(x):
        return x.astype(dt, copy=False)  # type: ignore[attr-defined]
    return cp.asarray(x, dtype=dt)  # type: ignore[union-attr]


def _safe_norm(x: Optional[ArrayLike]) -> float:
    if x is None or not hasattr(x, "size") or int(x.size) == 0:
        return 0.0
    try:
        if _is_cupy_array(x):
            val = float(cp.linalg.norm(x).item())  # type: ignore[union-attr]
        else:
            val = float(np.linalg.norm(np.asarray(x)))
        return val if math.isfinite(val) else 0.0
    except Exception:
        return 0.0


def _is_all_finite(*arrays: Optional[ArrayLike]) -> bool:
    for arr in arrays:
        if arr is None or not hasattr(arr, "size") or int(arr.size) == 0:
            continue
        if _is_cupy_array(arr):
            ok = bool(cp.isfinite(arr).all().item())  # type: ignore[union-attr]
        else:
            ok = bool(np.isfinite(np.asarray(arr)).all())
        if not ok:
            return False
    return True


def _svd_stable(
    mat: ArrayLike,
    *,
    xp,
    use_scipy_gesdd: bool = True,
    jitter: float = 1e-12,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Stable SVD with one jittered retry."""
    if xp is np:
        M = np.asarray(mat)
        try:
            if _SCIPY_OK and use_scipy_gesdd:
                return sla.svd(M, full_matrices=False, lapack_driver="gesdd")  # type: ignore[union-attr]
            return np.linalg.svd(M, full_matrices=False)
        except Exception:
            M2 = M.copy()
            k = min(M2.shape[0], M2.shape[1])
            if k > 0 and jitter > 0:
                idx = np.arange(k)
                M2[idx, idx] = M2[idx, idx] + float(jitter)
            if _SCIPY_OK and use_scipy_gesdd:
                return sla.svd(M2, full_matrices=False, lapack_driver="gesdd")  # type: ignore[union-attr]
            return np.linalg.svd(M2, full_matrices=False)

    if not _CUPY_OK:
        raise RuntimeError("GPU SVD requested but CuPy is not available.")
    M = mat if _is_cupy_array(mat) else cp.asarray(mat)  # type: ignore[union-attr]
    try:
        return cp.linalg.svd(M, full_matrices=False)  # type: ignore[union-attr]
    except Exception:
        M2 = M.copy()
        k = min(int(M2.shape[0]), int(M2.shape[1]))
        if k > 0 and jitter > 0:
            idx = cp.arange(k)  # type: ignore[union-attr]
            M2[idx, idx] = M2[idx, idx] + float(jitter)
        return cp.linalg.svd(M2, full_matrices=False)  # type: ignore[union-attr]


def _qr_stable(mat: ArrayLike, *, xp) -> Tuple[ArrayLike, ArrayLike]:
    if xp is np:
        return np.linalg.qr(np.asarray(mat), mode="reduced")
    if not _CUPY_OK:
        raise RuntimeError("GPU QR requested but CuPy is not available.")
    return cp.linalg.qr(mat if _is_cupy_array(mat) else cp.asarray(mat), mode="reduced")  # type: ignore[union-attr]


def _rq_stable(mat: ArrayLike, *, xp) -> Tuple[ArrayLike, ArrayLike]:
    """RQ via QR on transpose: RQ(A) = (R.T, Q.T) from QR(A.T)."""
    if xp is np:
        Qt, Rt = np.linalg.qr(np.asarray(mat).T, mode="reduced")
        return Rt.T, Qt.T
    if not _CUPY_OK:
        raise RuntimeError("GPU RQ requested but CuPy is not available.")
    M = mat if _is_cupy_array(mat) else cp.asarray(mat)  # type: ignore[union-attr]
    Qt, Rt = cp.linalg.qr(M.T, mode="reduced")  # type: ignore[union-attr]
    return Rt.T, Qt.T


def _clip_rank_by_energy(S: ArrayLike, energy_tol: Optional[float]) -> int:
    if energy_tol is None:
        return int(S.shape[0])
    et = float(energy_tol)
    if not (0.0 < et < 1.0):
        return int(S.shape[0])

    if _is_cupy_array(S):
        s2 = S * S  # type: ignore[operator]
        cum = cp.cumsum(s2)  # type: ignore[union-attr]
        if int(cum.size) == 0:
            return 0
        total = float(cum[-1].item())
        if not math.isfinite(total) or total <= 0:
            return int(S.shape[0])
        r = int(cp.searchsorted(cum, et * total).item()) + 1  # type: ignore[union-attr]
        return max(1, min(r, int(S.shape[0])))

    s = np.asarray(S)
    if s.size == 0:
        return 0
    s2 = s * s
    cum = np.cumsum(s2)
    total = float(cum[-1])
    if not math.isfinite(total) or total <= 0:
        return int(s.shape[0])
    r = int(np.searchsorted(cum, et * total) + 1)
    return max(1, min(r, int(s.shape[0])))


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass(frozen=True)
class TTConfig:
    dtype: np.dtype = np.float32
    device: str = "cpu"  # "cpu" | "gpu"
    svd_use_scipy_gesdd: bool = True
    svd_jitter: float = 1e-12
    stable_factor_dtype: np.dtype = np.float64
    check_finite: bool = True
    seed: Optional[int] = None
    deterministic: bool = False
    max_dense_entries: int = DEFAULT_MAX_DENSE_ENTRIES
    consensus_safe: bool = False

    def normalized(self) -> "TTConfig":
        dev = str(self.device).lower().strip()
        if dev not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'.")
        if bool(self.consensus_safe) and dev != "cpu":
            raise ValueError("consensus_safe TensorTrain configuration must use device='cpu'.")
        if bool(self.deterministic) and self.seed is None:
            raise ValueError("deterministic=True requires TTConfig.seed.")
        dense_guard = int(self.max_dense_entries)
        if dense_guard <= 0:
            dense_guard = DEFAULT_MAX_DENSE_ENTRIES
        return TTConfig(
            dtype=np.dtype(self.dtype),
            device=dev,
            svd_use_scipy_gesdd=bool(self.svd_use_scipy_gesdd),
            svd_jitter=float(self.svd_jitter),
            stable_factor_dtype=np.dtype(self.stable_factor_dtype),
            check_finite=bool(self.check_finite),
            seed=None if self.seed is None else int(self.seed),
            deterministic=bool(self.deterministic),
            max_dense_entries=dense_guard,
            consensus_safe=bool(self.consensus_safe),
        )

    def to_public_dict(self) -> Dict[str, Any]:
        cfg = self.normalized()
        return {
            "dtype": str(np.dtype(cfg.dtype)),
            "device": cfg.device,
            "svd_use_scipy_gesdd": bool(cfg.svd_use_scipy_gesdd),
            "svd_jitter": float(cfg.svd_jitter),
            "stable_factor_dtype": str(np.dtype(cfg.stable_factor_dtype)),
            "check_finite": bool(cfg.check_finite),
            "seed": cfg.seed,
            "deterministic": bool(cfg.deterministic),
            "max_dense_entries": int(cfg.max_dense_entries),
            "consensus_safe": bool(cfg.consensus_safe),
        }


@dataclass(frozen=True)
class TensorCommitment:
    scheme: str
    version: str
    output_dims: List[int]
    input_dims: List[int]
    bond_ranks: List[int]
    dtype: str
    device_independent: bool
    consensus_safe: bool
    core_hashes: List[str]
    metadata_hash: str
    config_hash: str
    commitment_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TensorCommitment":
        p = dict(payload)
        return cls(
            scheme=str(p.get("scheme", TN_COMMITMENT_SCHEME)),
            version=str(p.get("version", TN_VERSION)),
            output_dims=[int(x) for x in p.get("output_dims", [])],
            input_dims=[int(x) for x in p.get("input_dims", [])],
            bond_ranks=[int(x) for x in p.get("bond_ranks", [])],
            dtype=str(p.get("dtype", "float32")),
            device_independent=bool(p.get("device_independent", True)),
            consensus_safe=bool(p.get("consensus_safe", False)),
            core_hashes=[str(x) for x in p.get("core_hashes", [])],
            metadata_hash=str(p.get("metadata_hash", "")),
            config_hash=str(p.get("config_hash", "")),
            commitment_hash=str(p.get("commitment_hash", "")),
        )


@dataclass(frozen=True)
class TTRoundingReport:
    status: str
    old_parameter_count: int
    new_parameter_count: int
    old_bond_ranks: List[int]
    new_bond_ranks: List[int]
    max_rank: Optional[int]
    energy_tol: Optional[float]
    compression_ratio: float
    space_saved_percent: float
    exact_relative_fro_error: Optional[float] = None
    estimated_relative_output_error: Optional[float] = None
    operator_norm_before: Optional[float] = None
    operator_norm_after: Optional[float] = None
    deterministic: bool = False
    old_commitment_hash: str = ""
    new_commitment_hash: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


# =============================================================================
# TensorTrain / MPO
# =============================================================================


class TensorTrain:
    """
    Tensor Train container with MPO-like cores.

    Core convention:
        core_i.shape == (r_i, out_i, in_i, r_{i+1})
        r_0 = r_N = 1

    Operator action:
        y = W x
        x.shape == (prod(input_dims),)
        y.shape == (prod(output_dims),)
    """

    def __init__(
        self,
        output_dims: List[int],
        input_dims: List[int],
        bond_dims: Optional[List[int]] = None,
        cores_data: Optional[List[ArrayLike]] = None,
        *,
        config: Optional[TTConfig] = None,
        dtype: Optional[np.dtype] = None,
        device: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
        init_scale: float = 1e-3,
        metadata: Optional[Dict[str, Any]] = None,
        commit_metadata: Optional[Dict[str, Any]] = None,
    ):
        base_cfg = (config or TTConfig()).normalized()
        dt = np.dtype(dtype) if dtype is not None else np.dtype(base_cfg.dtype)
        dev = (device or base_cfg.device).lower().strip()
        if dev not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'.")
        if dev == "gpu" and not _CUPY_OK:
            raise RuntimeError("device='gpu' requested but CuPy is not installed.")
        if base_cfg.consensus_safe and dev != "cpu":
            raise ValueError("consensus_safe TensorTrain must use CPU device.")

        self.config = TTConfig(
            dtype=dt,
            device=dev,
            svd_use_scipy_gesdd=base_cfg.svd_use_scipy_gesdd,
            svd_jitter=base_cfg.svd_jitter,
            stable_factor_dtype=base_cfg.stable_factor_dtype,
            check_finite=base_cfg.check_finite,
            seed=base_cfg.seed,
            deterministic=base_cfg.deterministic,
            max_dense_entries=base_cfg.max_dense_entries,
            consensus_safe=base_cfg.consensus_safe,
        ).normalized()
        self._dtype = dt
        self._xp = cp if (dev == "gpu" and _CUPY_OK) else np
        if rng is not None:
            self._rng = rng
        elif self.config.seed is not None:
            self._rng = np.random.default_rng(int(self.config.seed))
        elif self.config.deterministic:
            # normalized() already rejects this, but keep defensive clarity.
            raise ValueError("deterministic=True requires seed or rng.")
        else:
            self._rng = np.random.default_rng()

        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.commit_metadata: Dict[str, Any] = dict(commit_metadata or {})

        if cores_data is not None:
            if not isinstance(cores_data, list) or len(cores_data) == 0:
                raise TypeError("cores_data must be a non-empty list of arrays.")
            if any(c is None for c in cores_data):
                raise TypeError("cores_data cannot contain None.")
            self.cores_data = [self._as_device_array(c) for c in cores_data]
            self.num_cores = len(self.cores_data)
            self.output_dims = [int(core.shape[1]) for core in self.cores_data]
            self.input_dims = [int(core.shape[2]) for core in self.cores_data]
            self.bond_dims = [] if self.num_cores == 1 else [int(core.shape[3]) for core in self.cores_data[:-1]]
            if int(self.cores_data[0].shape[0]) != 1:
                raise ValueError("First TT core must have left bond rank r0=1.")
            if int(self.cores_data[-1].shape[3]) != 1:
                raise ValueError("Last TT core must have right bond rank rN=1.")
        else:
            if len(output_dims) != len(input_dims):
                raise ValueError("output_dims and input_dims must have the same length.")
            if len(output_dims) == 0:
                raise ValueError("TensorTrain must have at least one core.")
            if len(output_dims) > 1:
                if bond_dims is None or len(bond_dims) != len(output_dims) - 1:
                    raise ValueError("bond_dims length must be num_cores-1 for multi-core TT.")
            elif bond_dims not in (None, [], ()):  # type: ignore[comparison-overlap]
                raise ValueError("bond_dims must be empty/None for single-core TT.")
            self.output_dims = [int(d) for d in output_dims]
            self.input_dims = [int(d) for d in input_dims]
            self.bond_dims = [int(d) for d in (bond_dims or [])]
            self.num_cores = len(self.output_dims)
            self.cores_data = self._create_random_core_data(init_scale=float(init_scale))

        if any(int(d) <= 0 for d in self.output_dims + self.input_dims):
            raise ValueError("All input/output dimensions must be positive.")
        if any(int(d) <= 0 for d in self.bond_dims):
            raise ValueError("All bond dimensions must be positive.")
        self._validate_core_shapes()
        if self.config.check_finite and not _is_all_finite(*self.cores_data):
            raise FloatingPointError("Non-finite values detected in TT cores.")

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_dense(
        cls,
        matrix: ArrayLike,
        output_dims: List[int],
        input_dims: List[int],
        max_bond_dim: int,
        *,
        dtype: Optional[np.dtype] = np.float32,
        energy_tol: Optional[float] = 0.999,
        device: str = "cpu",
        config: Optional[TTConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
        commit_metadata: Optional[Dict[str, Any]] = None,
    ) -> "TensorTrain":
        cfg = (config or TTConfig(dtype=np.dtype(dtype or np.float32), device=device)).normalized()
        tt = tensor_train_svd(
            matrix,
            output_dims=output_dims,
            input_dims=input_dims,
            max_bond_dim=max_bond_dim,
            dtype=dtype,
            energy_tol=energy_tol,
            device=device,
            use_scipy_gesdd=cfg.svd_use_scipy_gesdd,
            svd_jitter=cfg.svd_jitter,
            stable_factor_dtype=cfg.stable_factor_dtype,
            check_finite=cfg.check_finite,
            seed=cfg.seed,
            deterministic=cfg.deterministic,
            consensus_safe=cfg.consensus_safe,
            metadata=metadata,
            commit_metadata=commit_metadata,
        )
        if tt is None:
            raise RuntimeError("tensor_train_svd failed to produce a TensorTrain.")
        return tt

    @classmethod
    def identity(
        cls,
        dims: List[int],
        *,
        dtype: np.dtype = np.float32,
        device: str = "cpu",
        config: Optional[TTConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
        commit_metadata: Optional[Dict[str, Any]] = None,
    ) -> "TensorTrain":
        dims = [int(d) for d in dims]
        if not dims or any(d <= 0 for d in dims):
            raise ValueError("dims must be a non-empty list of positive integers.")
        dt = np.dtype(dtype)
        cores: List[np.ndarray] = []
        for d in dims:
            core = np.zeros((1, d, d, 1), dtype=dt)
            idx = np.arange(d)
            core[0, idx, idx, 0] = 1.0
            cores.append(core)
        cfg = config or TTConfig(dtype=dt, device=device)
        return cls(dims, dims, cores_data=cores, config=cfg, metadata=metadata, commit_metadata=commit_metadata)

    # ------------------------------------------------------------------
    # Properties / utilities
    # ------------------------------------------------------------------
    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def device(self) -> str:
        return self.config.device

    @property
    def xp(self):
        return self._xp

    @property
    def input_dim(self) -> int:
        return _prod(self.input_dims)

    @property
    def output_dim(self) -> int:
        return _prod(self.output_dims)

    def is_consensus_safe_backend(self) -> bool:
        return bool(self.device == "cpu" and np.dtype(self.dtype) in (np.dtype(np.float32), np.dtype(np.float64)))

    def assert_consensus_safe(self) -> None:
        if not self.is_consensus_safe_backend():
            raise RuntimeError("TensorTrain is not consensus-safe: requires CPU and float32/float64 dtype.")
        if not self.config.deterministic:
            raise RuntimeError("TensorTrain config is not deterministic.")

    def _as_device_array(self, x: ArrayLike) -> ArrayLike:
        return _to_device_array(x, self._xp, self._dtype)

    def copy(self) -> "TensorTrain":
        return TensorTrain(
            self.output_dims,
            self.input_dims,
            cores_data=[c.copy() for c in self.cores_data],
            config=self.config,
            rng=self._rng,
            metadata=dict(self.metadata),
            commit_metadata=dict(self.commit_metadata),
        )

    def astype(self, dtype: np.dtype) -> "TensorTrain":
        dt = np.dtype(dtype)
        return TensorTrain(
            self.output_dims,
            self.input_dims,
            cores_data=[c.astype(dt, copy=False) for c in self.cores_data],
            config=TTConfig(
                dtype=dt,
                device=self.device,
                svd_use_scipy_gesdd=self.config.svd_use_scipy_gesdd,
                svd_jitter=self.config.svd_jitter,
                stable_factor_dtype=self.config.stable_factor_dtype,
                check_finite=self.config.check_finite,
                seed=self.config.seed,
                deterministic=self.config.deterministic,
                max_dense_entries=self.config.max_dense_entries,
                consensus_safe=self.config.consensus_safe,
            ),
            rng=self._rng,
            metadata=dict(self.metadata),
            commit_metadata=dict(self.commit_metadata),
        )

    def to_device(self, device: str) -> "TensorTrain":
        device = str(device).lower().strip()
        if device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'.")
        if device == "gpu" and not _CUPY_OK:
            raise RuntimeError("GPU requested but CuPy is not installed.")
        if self.config.consensus_safe and device != "cpu":
            raise ValueError("consensus_safe TensorTrain cannot be moved to GPU.")
        if device == self.device:
            return self.copy()
        new_xp = cp if device == "gpu" else np
        new_cores = [_to_device_array(c, new_xp, self._dtype) for c in self.cores_data]
        new_cfg = TTConfig(
            dtype=self._dtype,
            device=device,
            svd_use_scipy_gesdd=self.config.svd_use_scipy_gesdd,
            svd_jitter=self.config.svd_jitter,
            stable_factor_dtype=self.config.stable_factor_dtype,
            check_finite=self.config.check_finite,
            seed=self.config.seed,
            deterministic=self.config.deterministic,
            max_dense_entries=self.config.max_dense_entries,
            consensus_safe=self.config.consensus_safe,
        )
        return TensorTrain(
            self.output_dims,
            self.input_dims,
            cores_data=new_cores,
            config=new_cfg,
            rng=self._rng,
            metadata=dict(self.metadata),
            commit_metadata=dict(self.commit_metadata),
        )

    def parameter_count(self) -> int:
        return int(sum(int(core.size) for core in self.cores_data))

    def bond_ranks(self) -> List[int]:
        return [] if self.num_cores <= 1 else [int(c.shape[3]) for c in self.cores_data[:-1]]

    def core_shapes(self) -> List[Tuple[int, int, int, int]]:
        return [tuple(map(int, c.shape)) for c in self.cores_data]

    def describe(self) -> Dict[str, Any]:
        return {
            "num_cores": int(self.num_cores),
            "output_dims": list(self.output_dims),
            "input_dims": list(self.input_dims),
            "output_dim": int(self.output_dim),
            "input_dim": int(self.input_dim),
            "bond_dims": self.bond_ranks(),
            "core_shapes": self.core_shapes(),
            "dtype": str(self._dtype),
            "device": str(self.device),
            "param_count": int(self.parameter_count()),
            "metadata": dict(self.metadata),
            "commit_metadata": dict(self.commit_metadata),
            "config": self.config.to_public_dict(),
            "consensus_safe_backend": self.is_consensus_safe_backend(),
        }

    def _create_random_core_data(self, init_scale: float = 1e-3) -> List[ArrayLike]:
        ranks = [1] + self.bond_dims + [1]
        cores: List[ArrayLike] = []
        for i in range(self.num_cores):
            rL, rR = int(ranks[i]), int(ranks[i + 1])
            out_i, in_i = int(self.output_dims[i]), int(self.input_dims[i])
            core_np = self._rng.normal(0.0, float(init_scale), size=(rL, out_i, in_i, rR)).astype(self._dtype, copy=False)
            cores.append(self._as_device_array(core_np))
        return cores

    def _validate_core_shapes(self) -> None:
        if len(self.cores_data) != self.num_cores:
            raise ValueError("cores_data length mismatch.")
        ranks = [1] + self.bond_dims + [1]
        for i, core in enumerate(self.cores_data):
            if int(core.ndim) != 4:
                raise ValueError(f"Core {i} must be rank-4, got shape {tuple(core.shape)}.")
            rL, out_i, in_i, rR = map(int, core.shape)
            if out_i != int(self.output_dims[i]) or in_i != int(self.input_dims[i]):
                raise ValueError(f"Core {i} dims mismatch.")
            if rL != int(ranks[i]) or rR != int(ranks[i + 1]):
                raise ValueError(f"Core {i} bond mismatch. Expected ({ranks[i]},{ranks[i + 1]}), got ({rL},{rR}).")

    # ------------------------------------------------------------------
    # Commitments
    # ------------------------------------------------------------------
    def _canonical_core_bytes(self, core: ArrayLike, *, canonical_dtype: Optional[np.dtype] = None) -> bytes:
        dt = np.dtype(canonical_dtype or self.dtype)
        arr = np.asarray(_to_cpu(core), dtype=dt)
        # Enforce C-contiguous little-endian canonical representation.
        if arr.dtype.byteorder == ">":
            arr = arr.byteswap().newbyteorder("<")
        arr = np.ascontiguousarray(arr)
        return arr.tobytes(order="C")

    def core_hashes(self, *, canonical_dtype: Optional[np.dtype] = None) -> List[str]:
        dt = np.dtype(canonical_dtype or self.dtype)
        out: List[str] = []
        for i, c in enumerate(self.cores_data):
            shape_payload = _json_dumps_stable({"index": i, "shape": list(map(int, c.shape)), "dtype": str(dt)}).encode("utf-8")
            out.append(_domain_hash("CHIMERA_TT_CORE_V1", shape_payload, self._canonical_core_bytes(c, canonical_dtype=dt)))
        return out

    def commitment(self, *, include_runtime_metadata: bool = False, canonical_dtype: Optional[np.dtype] = None) -> TensorCommitment:
        dt = np.dtype(canonical_dtype or self.dtype)
        core_hashes = self.core_hashes(canonical_dtype=dt)
        metadata_payload = dict(self.commit_metadata)
        if include_runtime_metadata:
            metadata_payload["runtime_metadata"] = dict(self.metadata)
        metadata_hash = _sha256_json(metadata_payload)
        cfg_payload = self.config.to_public_dict()
        # Device is intentionally excluded from config hash for device-independent commitments.
        cfg_payload["device"] = "canonical"
        cfg_payload["dtype"] = str(dt)
        config_hash = _sha256_json(cfg_payload)
        base = {
            "scheme": TN_COMMITMENT_SCHEME,
            "version": TN_VERSION,
            "output_dims": list(map(int, self.output_dims)),
            "input_dims": list(map(int, self.input_dims)),
            "bond_ranks": self.bond_ranks(),
            "dtype": str(dt),
            "device_independent": True,
            "consensus_safe": bool(self.config.consensus_safe),
            "core_hashes": core_hashes,
            "metadata_hash": metadata_hash,
            "config_hash": config_hash,
        }
        commitment_hash = _domain_hash("CHIMERA_TT_COMMITMENT_V1", _json_dumps_stable(base).encode("utf-8"))
        return TensorCommitment(commitment_hash=commitment_hash, **base)  # type: ignore[arg-type]

    def verify_commitment(self, commitment: Union[TensorCommitment, Mapping[str, Any]], *, canonical_dtype: Optional[np.dtype] = None) -> bool:
        expected = commitment if isinstance(commitment, TensorCommitment) else TensorCommitment.from_dict(commitment)
        actual = self.commitment(canonical_dtype=canonical_dtype or np.dtype(expected.dtype))
        return actual.to_dict() == expected.to_dict()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def _npz_manifest(self, arrays: Mapping[str, Any], *, commit: TensorCommitment) -> Dict[str, Any]:
        core_hashes = []
        for i in range(self.num_cores):
            key = f"core_{i}"
            arr = np.asarray(arrays[key])
            shape_payload = _json_dumps_stable({"index": i, "shape": list(map(int, arr.shape)), "dtype": str(arr.dtype)}).encode("utf-8")
            core_hashes.append(_domain_hash("CHIMERA_TT_CORE_V1", shape_payload, np.ascontiguousarray(arr).tobytes(order="C")))
        manifest = {
            "format": TN_NPZ_FORMAT,
            "version": TN_VERSION,
            "num_cores": int(self.num_cores),
            "output_dims": list(map(int, self.output_dims)),
            "input_dims": list(map(int, self.input_dims)),
            "bond_dims": self.bond_ranks(),
            "dtype": str(self.dtype),
            "config": self.config.to_public_dict(),
            "metadata_hash": _sha256_json(self.metadata),
            "commit_metadata_hash": _sha256_json(self.commit_metadata),
            "core_hashes": core_hashes,
            "tensor_commitment": commit.to_dict(),
        }
        manifest["manifest_hash"] = _domain_hash("CHIMERA_TT_NPZ_MANIFEST_V1", _json_dumps_stable({k: v for k, v in manifest.items() if k != "manifest_hash"}).encode("utf-8"))
        return manifest

    def save_npz(self, path: Union[str, Path], *, compressed: bool = True, include_manifest: bool = True) -> None:
        p = Path(path)
        commit = self.commitment()
        arrays: Dict[str, Any] = {
            "__metadata_json__": np.array(_json_dumps_stable(dict(self.metadata))),
            "__commit_metadata_json__": np.array(_json_dumps_stable(dict(self.commit_metadata))),
            "num_cores": np.array([self.num_cores], dtype=np.int64),
        }
        for i, c in enumerate(self.cores_data):
            arrays[f"core_{i}"] = _to_cpu(c).astype(self.dtype, copy=False)
        if include_manifest:
            manifest = self._npz_manifest(arrays, commit=commit)
            arrays["__manifest_json__"] = np.array(_json_dumps_stable(manifest))
            arrays["__manifest_hash__"] = np.array(str(manifest["manifest_hash"]))
        _atomic_save_npz(p, arrays, compressed=compressed)

    @staticmethod
    def _verify_loaded_manifest(data: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
        expected_hash = str(manifest.get("manifest_hash", ""))
        actual_hash = _domain_hash("CHIMERA_TT_NPZ_MANIFEST_V1", _json_dumps_stable({k: v for k, v in dict(manifest).items() if k != "manifest_hash"}).encode("utf-8"))
        if not expected_hash or expected_hash != actual_hash:
            raise ValueError("TensorTrain NPZ manifest hash verification failed.")
        if "__manifest_hash__" in data:
            stored = str(data["__manifest_hash__"].item())
            if stored != expected_hash:
                raise ValueError("TensorTrain NPZ stored manifest hash mismatch.")
        core_hashes = []
        for i in range(int(manifest.get("num_cores", 0))):
            key = f"core_{i}"
            if key not in data:
                raise ValueError(f"TensorTrain NPZ missing {key}.")
            arr = np.asarray(data[key])
            shape_payload = _json_dumps_stable({"index": i, "shape": list(map(int, arr.shape)), "dtype": str(arr.dtype)}).encode("utf-8")
            core_hashes.append(_domain_hash("CHIMERA_TT_CORE_V1", shape_payload, np.ascontiguousarray(arr).tobytes(order="C")))
        if core_hashes != list(manifest.get("core_hashes", [])):
            raise ValueError("TensorTrain NPZ core hash verification failed.")

    @staticmethod
    def load_npz(
        path: Union[str, Path],
        *,
        device: str = "cpu",
        dtype: Optional[np.dtype] = None,
        check_finite: Optional[bool] = None,
        verify: bool = True,
    ) -> "TensorTrain":
        p = Path(path)
        with np.load(p, allow_pickle=False) as data:
            manifest: Optional[Dict[str, Any]] = None
            if "__manifest_json__" in data.files:
                manifest = json.loads(str(data["__manifest_json__"].item()))
                if verify:
                    TensorTrain._verify_loaded_manifest(data, manifest)

            if manifest is not None:
                ncores = int(manifest["num_cores"])
                cores = [np.asarray(data[f"core_{i}"]) for i in range(ncores)]
                stored_dtype = np.dtype(manifest.get("dtype", str(cores[0].dtype)))
                dt = np.dtype(dtype) if dtype is not None else stored_dtype
                cfg_meta = dict(manifest.get("config", {}))
                cfg = TTConfig(
                    dtype=dt,
                    device=device,
                    svd_use_scipy_gesdd=bool(cfg_meta.get("svd_use_scipy_gesdd", True)),
                    svd_jitter=float(cfg_meta.get("svd_jitter", 1e-12)),
                    stable_factor_dtype=np.dtype(cfg_meta.get("stable_factor_dtype", "float64")),
                    check_finite=bool(cfg_meta.get("check_finite", True) if check_finite is None else check_finite),
                    seed=cfg_meta.get("seed", None),
                    deterministic=bool(cfg_meta.get("deterministic", False)),
                    max_dense_entries=int(cfg_meta.get("max_dense_entries", DEFAULT_MAX_DENSE_ENTRIES)),
                    consensus_safe=bool(cfg_meta.get("consensus_safe", False)),
                )
                metadata = {}
                commit_metadata = {}
                if "__metadata_json__" in data.files:
                    metadata = json.loads(str(data["__metadata_json__"].item()))
                if "__commit_metadata_json__" in data.files:
                    commit_metadata = json.loads(str(data["__commit_metadata_json__"].item()))
                obj = TensorTrain(
                    output_dims=list(map(int, manifest.get("output_dims", []))),
                    input_dims=list(map(int, manifest.get("input_dims", []))),
                    cores_data=[c.astype(dt, copy=False) for c in cores],
                    config=cfg,
                    metadata=metadata,
                    commit_metadata=commit_metadata,
                )
                if verify:
                    expected = TensorCommitment.from_dict(manifest["tensor_commitment"])
                    if not obj.verify_commitment(expected, canonical_dtype=np.dtype(expected.dtype)):
                        raise ValueError("TensorTrain NPZ tensor commitment verification failed.")
                return obj

            # Legacy v1 loader fallback.
            if "__metadata_json__" in data.files:
                try:
                    meta = json.loads(str(data["__metadata_json__"].item()))
                except Exception:
                    meta = {}
                ncores = int(data["num_cores"][0]) if "num_cores" in data.files else int(len([k for k in data.files if k.startswith("core_")]))
                cores = [np.asarray(data[f"core_{i}"]) for i in range(ncores)]
                dt = np.dtype(dtype) if dtype is not None else np.dtype(cores[0].dtype)
                cfg = TTConfig(dtype=dt, device=device, check_finite=True if check_finite is None else bool(check_finite))
                return TensorTrain([], [], cores_data=[c.astype(dt, copy=False) for c in cores], config=cfg, metadata=meta)

            core_keys = sorted([k for k in data.files if k.startswith("core_")], key=lambda s: int(s.split("_")[1]))
            if not core_keys:
                raise ValueError("NPZ file does not contain TensorTrain core arrays.")
            cores = [np.asarray(data[k]) for k in core_keys]
            dt = np.dtype(dtype) if dtype is not None else np.dtype(cores[0].dtype)
            cfg = TTConfig(dtype=dt, device=device, check_finite=True if check_finite is None else bool(check_finite))
            return TensorTrain([], [], cores_data=[c.astype(dt, copy=False) for c in cores], config=cfg)

    # ------------------------------------------------------------------
    # TT/MPO action
    # ------------------------------------------------------------------
    def contract_with_vector(self, vector: ArrayLike) -> ArrayLike:
        xp = self._xp
        vec = _to_device_array(vector, xp, self._dtype).reshape(-1)
        if int(vec.size) != int(self.input_dim):
            raise ValueError(f"Input vector size {int(vec.size)} != TT input dim {self.input_dim}.")

        x_tensor = vec.reshape(tuple(self.input_dims))
        core0 = self.cores_data[0]
        state = xp.tensordot(core0, x_tensor, axes=([2], [0]))
        state = xp.squeeze(state, axis=0)

        for i in range(1, self.num_cores):
            core_i = self.cores_data[i]
            bond_axis = i
            in_axis = i + 1
            tmp = xp.tensordot(core_i, state, axes=([0, 2], [bond_axis, in_axis]))
            outs_prev_axes = list(range(2, 2 + i))
            rem_in_axes = list(range(2 + i, int(tmp.ndim)))
            perm = outs_prev_axes + [0, 1] + rem_in_axes
            state = tmp.transpose(perm)

        if int(state.ndim) >= 1 and int(state.shape[-1]) == 1:
            state = state[..., 0]
        y = state.reshape(-1).astype(self._dtype, copy=False)
        if self.config.check_finite and not _is_all_finite(y):
            raise FloatingPointError("Non-finite values encountered during TT contraction.")
        return y

    def contract_batch(self, X: ArrayLike, *, vectorized: bool = True, chunk_size: Optional[int] = None) -> ArrayLike:
        xp = self._xp
        Xd = _to_device_array(X, xp, self._dtype)
        if int(Xd.ndim) != 2:
            raise ValueError(f"contract_batch expects 2D array, got {tuple(Xd.shape)}.")
        B, Din = map(int, Xd.shape)
        if Din != self.input_dim:
            raise ValueError(f"Input features {Din} != TT input dim {self.input_dim}.")
        Dout = self.output_dim
        if B == 0:
            return xp.zeros((0, Dout), dtype=self._dtype)
        if not vectorized:
            Y = xp.empty((B, Dout), dtype=self._dtype)
            for b in range(B):
                Y[b] = self.contract_with_vector(Xd[b])
            if self.config.check_finite and not _is_all_finite(Y):
                raise FloatingPointError("Non-finite values encountered during TT batch contraction.")
            return Y
        if chunk_size is not None and 0 < int(chunk_size) < B:
            cs = int(chunk_size)
            Y = xp.empty((B, Dout), dtype=self._dtype)
            for start in range(0, B, cs):
                end = min(B, start + cs)
                Y[start:end] = self.contract_batch(Xd[start:end], vectorized=True, chunk_size=None)
            return Y

        x_tensor = Xd.reshape((B,) + tuple(self.input_dims))
        core0 = self.cores_data[0]
        state = xp.tensordot(core0, x_tensor, axes=([2], [1]))
        state = xp.squeeze(state, axis=0)
        state = state.transpose((2, 0, 1) + tuple(range(3, int(state.ndim))))

        for i in range(1, self.num_cores):
            core_i = self.cores_data[i]
            bond_axis = 1 + i
            in_axis = 2 + i
            tmp = xp.tensordot(core_i, state, axes=([0, 2], [bond_axis, in_axis]))
            outs_prev = list(range(3, 3 + i))
            rem_in = list(range(3 + i, int(tmp.ndim)))
            perm = [2] + outs_prev + [0, 1] + rem_in
            state = tmp.transpose(perm)

        if int(state.shape[-1]) == 1:
            state = state[..., 0]
        Y = state.reshape(B, Dout).astype(self._dtype, copy=False)
        if self.config.check_finite and not _is_all_finite(Y):
            raise FloatingPointError("Non-finite values encountered during TT batch contraction.")
        return Y

    def apply(self, x: ArrayLike, *, vectorized: bool = True, chunk_size: Optional[int] = None) -> ArrayLike:
        x_arr = _to_device_array(x, self._xp, self._dtype)
        if int(x_arr.ndim) == 1:
            return self.contract_with_vector(x_arr)
        if int(x_arr.ndim) == 2:
            return self.contract_batch(x_arr, vectorized=vectorized, chunk_size=chunk_size)
        raise ValueError(f"apply expects 1D or 2D input, got {tuple(x_arr.shape)}.")

    # ------------------------------------------------------------------
    # Dense materialization / diagnostics
    # ------------------------------------------------------------------
    def to_matrix(self, *, force_cpu: bool = False, max_entries: Optional[int] = None) -> np.ndarray:
        out_dim, in_dim = self.output_dim, self.input_dim
        entries = int(out_dim) * int(in_dim)
        guard = self.config.max_dense_entries if max_entries is None else int(max_entries)
        if guard is not None and guard > 0 and entries > int(guard):
            raise MemoryError(f"Dense materialization would create {entries:,} entries, exceeding max_entries={guard:,}.")

        xp = self._xp
        tensor = self.cores_data[0]
        tensor = xp.squeeze(tensor, axis=0)
        for i in range(1, self.num_cores):
            tensor = xp.tensordot(tensor, self.cores_data[i], axes=([-1], [0]))
        if int(tensor.shape[-1]) == 1:
            tensor = tensor[..., 0]
        num = self.num_cores
        out_axes = [2 * i for i in range(num)]
        in_axes = [2 * i + 1 for i in range(num)]
        tensor = tensor.transpose(out_axes + in_axes)
        M = tensor.reshape(out_dim, in_dim).astype(self._dtype, copy=False)
        if self.config.check_finite and not _is_all_finite(M):
            raise FloatingPointError("Non-finite values encountered while materializing TT.")
        return _to_cpu(M) if (force_cpu or xp is not np) else np.asarray(M)

    def estimate_operator_norm_power(self, *, iters: int = 16, seed: Optional[int] = None) -> float:
        iters = int(max(1, iters))
        rng = np.random.default_rng(int(seed if seed is not None else (self.config.seed or 0)))
        v = rng.normal(size=(self.input_dim,)).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-12)
        Wt = self.adjoint()
        val = 0.0
        for _ in range(iters):
            w = _to_cpu(self.apply(v)).astype(np.float64, copy=False)
            val = float(np.linalg.norm(w))
            if not math.isfinite(val) or val <= 1e-12:
                return 0.0
            z = _to_cpu(Wt.apply(w.astype(self.dtype))).astype(np.float64, copy=False)
            nz = float(np.linalg.norm(z))
            if not math.isfinite(nz) or nz <= 1e-12:
                return val
            v = (z / nz).astype(np.float32, copy=False)
        return float(val)

    # Backward-compatible alias some older scripts may probe for.
    def estimate_operator_norm(self, num_iters: int = 16, seed: Optional[int] = None) -> float:
        return self.estimate_operator_norm_power(iters=num_iters, seed=seed)

    # ------------------------------------------------------------------
    # Canonicalization / rounding
    # ------------------------------------------------------------------
    def left_orthonormalize(self) -> "TensorTrain":
        xp = self._xp
        cores = [c.copy() for c in self.cores_data]
        for i in range(self.num_cores - 1):
            c = cores[i]
            rL, out_i, in_i, rR = map(int, c.shape)
            mat = c.reshape(rL * out_i * in_i, rR)
            if xp is np and self.config.stable_factor_dtype != self._dtype:
                Q, R = _qr_stable(mat.astype(self.config.stable_factor_dtype, copy=False), xp=np)
                Q = Q.astype(self._dtype, copy=False)
                R = R.astype(self._dtype, copy=False)
            else:
                Q, R = _qr_stable(mat, xp=xp)
            new_rR = int(Q.shape[1])
            cores[i] = Q.reshape(rL, out_i, in_i, new_rR)
            cnext = cores[i + 1]
            rL2, out2, in2, rR2 = map(int, cnext.shape)
            if rL2 != rR:
                raise RuntimeError("Bond mismatch during left_orthonormalize.")
            cnext_mat = cnext.reshape(rL2, out2 * in2 * rR2)
            cores[i + 1] = (R @ cnext_mat).reshape(new_rR, out2, in2, rR2)
        return TensorTrain(self.output_dims, self.input_dims, cores_data=cores, config=self.config, rng=self._rng, metadata=dict(self.metadata), commit_metadata=dict(self.commit_metadata))

    def right_orthonormalize(self) -> "TensorTrain":
        xp = self._xp
        cores = [c.copy() for c in self.cores_data]
        for i in range(self.num_cores - 1, 0, -1):
            c = cores[i]
            rL, out_i, in_i, rR = map(int, c.shape)
            mat = c.reshape(rL, out_i * in_i * rR)
            if xp is np and self.config.stable_factor_dtype != self._dtype:
                R, Q = _rq_stable(mat.astype(self.config.stable_factor_dtype, copy=False), xp=np)
                R = R.astype(self._dtype, copy=False)
                Q = Q.astype(self._dtype, copy=False)
            else:
                R, Q = _rq_stable(mat, xp=xp)
            new_rL = int(Q.shape[0])
            cores[i] = Q.reshape(new_rL, out_i, in_i, rR)
            cprev = cores[i - 1]
            rL0, out0, in0, rR0 = map(int, cprev.shape)
            if rR0 != rL:
                raise RuntimeError("Bond mismatch during right_orthonormalize.")
            cprev_mat = cprev.reshape(rL0 * out0 * in0, rR0)
            cores[i - 1] = (cprev_mat @ R).reshape(rL0, out0, in0, new_rL)
        return TensorTrain(self.output_dims, self.input_dims, cores_data=cores, config=self.config, rng=self._rng, metadata=dict(self.metadata), commit_metadata=dict(self.commit_metadata))

    def round(self, *, max_rank: Optional[int] = None, energy_tol: Optional[float] = 0.999, min_rank: int = 1) -> "TensorTrain":
        if self.num_cores <= 1:
            return self.copy()
        xp = self._xp
        tt = self.left_orthonormalize()
        cores = [c.copy() for c in tt.cores_data]
        Rcap = int(max_rank) if max_rank is not None else None
        min_rank = int(max(1, min_rank))
        for i in range(self.num_cores - 1):
            c1, c2 = cores[i], cores[i + 1]
            rL, out1, in1, rM = map(int, c1.shape)
            rM2, out2, in2, rR = map(int, c2.shape)
            if rM2 != rM:
                raise RuntimeError("Bond mismatch during round().")
            merged = xp.tensordot(c1, c2, axes=([3], [0]))
            mat = merged.reshape(rL * out1 * in1, out2 * in2 * rR)
            if xp is np and self.config.stable_factor_dtype != self._dtype:
                U, S, Vh = _svd_stable(mat.astype(self.config.stable_factor_dtype, copy=False), xp=np, use_scipy_gesdd=self.config.svd_use_scipy_gesdd, jitter=self.config.svd_jitter)
                U = U.astype(self._dtype, copy=False)
                S = S.astype(self._dtype, copy=False)
                Vh = Vh.astype(self._dtype, copy=False)
            else:
                U, S, Vh = _svd_stable(mat, xp=xp, use_scipy_gesdd=self.config.svd_use_scipy_gesdd, jitter=self.config.svd_jitter)
            if int(S.size) == 0:
                raise RuntimeError("SVD produced empty spectrum during round().")
            r_keep = _clip_rank_by_energy(S, energy_tol)
            if Rcap is not None:
                r_keep = min(r_keep, Rcap)
            r_keep = max(min_rank, min(r_keep, int(S.shape[0])))
            U = U[:, :r_keep]
            S = S[:r_keep]
            Vh = Vh[:r_keep, :]
            cores[i] = U.reshape(rL, out1, in1, r_keep).astype(self._dtype, copy=False)
            cores[i + 1] = (S[:, None] * Vh).reshape(r_keep, out2, in2, rR).astype(self._dtype, copy=False)
        return TensorTrain(self.output_dims, self.input_dims, cores_data=cores, config=self.config, rng=self._rng, metadata=dict(self.metadata), commit_metadata=dict(self.commit_metadata))

    def round_with_report(
        self,
        *,
        max_rank: Optional[int] = None,
        energy_tol: Optional[float] = 0.999,
        min_rank: int = 1,
        exact_error_max_entries: Optional[int] = None,
        sample_vectors: int = 16,
        seed: Optional[int] = None,
    ) -> Tuple["TensorTrain", TTRoundingReport]:
        old_params = self.parameter_count()
        old_ranks = self.bond_ranks()
        old_commit = self.commitment().commitment_hash
        notes: List[str] = []
        old_norm: Optional[float] = None
        new_norm: Optional[float] = None
        exact_rel: Optional[float] = None
        est_rel: Optional[float] = None

        rounded = self.round(max_rank=max_rank, energy_tol=energy_tol, min_rank=min_rank)
        new_params = rounded.parameter_count()
        new_ranks = rounded.bond_ranks()
        new_commit = rounded.commitment().commitment_hash

        dense_guard = self.config.max_dense_entries if exact_error_max_entries is None else int(exact_error_max_entries)
        entries = int(self.input_dim) * int(self.output_dim)
        if entries <= int(dense_guard):
            try:
                W = self.to_matrix(force_cpu=True, max_entries=dense_guard)
                Wr = rounded.to_matrix(force_cpu=True, max_entries=dense_guard)
                denom = float(np.linalg.norm(W))
                exact_rel = float(np.linalg.norm(W - Wr) / max(denom, 1e-12))
                old_norm = float(np.linalg.norm(W, ord=2)) if min(W.shape) <= 2048 else None
                new_norm = float(np.linalg.norm(Wr, ord=2)) if min(Wr.shape) <= 2048 else None
            except Exception as exc:
                notes.append(f"exact_error_skipped:{exc!r}")
        else:
            notes.append(f"exact_error_skipped:dense_entries_{entries}_exceeds_{dense_guard}")

        if exact_rel is None:
            try:
                rng = np.random.default_rng(int(seed if seed is not None else (self.config.seed or 0)))
                errs = []
                for _ in range(max(1, int(sample_vectors))):
                    x = rng.normal(size=(self.input_dim,)).astype(np.float32)
                    x /= np.linalg.norm(x) + 1e-12
                    y = _to_cpu(self.apply(x)).astype(np.float64)
                    yr = _to_cpu(rounded.apply(x)).astype(np.float64)
                    errs.append(float(np.linalg.norm(y - yr) / max(np.linalg.norm(y), 1e-12)))
                est_rel = float(np.mean(errs)) if errs else None
            except Exception as exc:
                notes.append(f"sample_error_skipped:{exc!r}")

        if old_norm is None:
            try:
                old_norm = self.estimate_operator_norm_power(iters=8, seed=seed)
            except Exception:
                old_norm = None
        if new_norm is None:
            try:
                new_norm = rounded.estimate_operator_norm_power(iters=8, seed=seed)
            except Exception:
                new_norm = None

        compression_ratio = 1.0 - (float(new_params) / float(max(1, old_params)))
        report = TTRoundingReport(
            status="rounded",
            old_parameter_count=int(old_params),
            new_parameter_count=int(new_params),
            old_bond_ranks=list(old_ranks),
            new_bond_ranks=list(new_ranks),
            max_rank=max_rank,
            energy_tol=energy_tol,
            compression_ratio=float(compression_ratio),
            space_saved_percent=float(100.0 * compression_ratio),
            exact_relative_fro_error=exact_rel,
            estimated_relative_output_error=est_rel,
            operator_norm_before=old_norm,
            operator_norm_after=new_norm,
            deterministic=bool(self.config.deterministic),
            old_commitment_hash=old_commit,
            new_commitment_hash=new_commit,
            notes=notes,
        )
        return rounded, report

    # ------------------------------------------------------------------
    # Arithmetic / transforms
    # ------------------------------------------------------------------
    def scale(self, alpha: float) -> "TensorTrain":
        cores = [c.copy() for c in self.cores_data]
        cores[0] = (cores[0] * float(alpha)).astype(self._dtype, copy=False)
        meta = dict(self.metadata)
        meta["last_scale"] = float(alpha)
        return TensorTrain(self.output_dims, self.input_dims, cores_data=cores, config=self.config, rng=self._rng, metadata=meta, commit_metadata=dict(self.commit_metadata))

    def __mul__(self, alpha: float) -> "TensorTrain":
        return self.scale(float(alpha))

    def __rmul__(self, alpha: float) -> "TensorTrain":
        return self.scale(float(alpha))

    def __add__(self, other: "TensorTrain") -> "TensorTrain":
        if not isinstance(other, TensorTrain):
            return NotImplemented
        if self.output_dims != other.output_dims or self.input_dims != other.input_dims:
            raise ValueError("Cannot add TensorTrains with different dimensions.")
        if self.device != other.device:
            other = other.to_device(self.device)
        if self.dtype != other.dtype:
            other = other.astype(self.dtype)

        xp = self._xp
        A, B = self.cores_data, other.cores_data
        N = self.num_cores
        if N != other.num_cores:
            raise ValueError("Cannot add TensorTrains with different num_cores.")
        if N == 1:
            return TensorTrain(self.output_dims, self.input_dims, cores_data=[(A[0] + B[0]).astype(self._dtype, copy=False)], config=self.config, rng=self._rng)

        cores: List[ArrayLike] = [xp.concatenate([A[0], B[0]], axis=3).astype(self._dtype, copy=False)]
        for i in range(1, N - 1):
            cA, cB = A[i], B[i]
            rLa, outi, ini, rRa = map(int, cA.shape)
            rLb, outi2, ini2, rRb = map(int, cB.shape)
            if outi != outi2 or ini != ini2:
                raise ValueError("Dimension mismatch during TT addition.")
            top = xp.concatenate([cA, xp.zeros((rLa, outi, ini, rRb), dtype=self._dtype)], axis=3)
            bot = xp.concatenate([xp.zeros((rLb, outi, ini, rRa), dtype=self._dtype), cB], axis=3)
            cores.append(xp.concatenate([top, bot], axis=0).astype(self._dtype, copy=False))
        cores.append(xp.concatenate([A[-1], B[-1]], axis=0).astype(self._dtype, copy=False))
        return TensorTrain(self.output_dims, self.input_dims, cores_data=cores, config=self.config, rng=self._rng)

    def transpose(self) -> "TensorTrain":
        cores = [c.transpose(0, 2, 1, 3).copy() for c in self.cores_data]
        return TensorTrain(self.input_dims, self.output_dims, cores_data=cores, config=self.config, rng=self._rng, metadata=dict(self.metadata), commit_metadata=dict(self.commit_metadata))

    def adjoint(self) -> "TensorTrain":
        cores = [c.transpose(0, 2, 1, 3).conj().copy() for c in self.cores_data]
        return TensorTrain(self.input_dims, self.output_dims, cores_data=cores, config=self.config, rng=self._rng, metadata=dict(self.metadata), commit_metadata=dict(self.commit_metadata))

    def __repr__(self) -> str:
        return f"TensorTrain(num_cores={self.num_cores}, input_dim={self.input_dim}, output_dim={self.output_dim}, bond_dims={self.bond_ranks()}, dtype={self.dtype}, device='{self.device}')"


# =============================================================================
# Tucker blocks
# =============================================================================


class TuckerTensor:
    """Simple multiplicative Hadamard Tucker fusion block."""

    def __init__(self, input_dims: List[int], output_rank: int, rng: Optional[np.random.Generator] = None, dtype: np.dtype = np.float32):
        if not isinstance(input_dims, list) or any(int(d) <= 0 for d in input_dims):
            raise ValueError("input_dims must be a list of positive integers.")
        if int(output_rank) <= 0:
            raise ValueError("output_rank must be positive.")
        self.input_dims = [int(d) for d in input_dims]
        self.output_rank = int(output_rank)
        self.num_modalities = len(self.input_dims)
        self.dtype = np.dtype(dtype)
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()
        self.factor_matrices = [(self.rng.standard_normal((dim, self.output_rank)).astype(self.dtype) * np.asarray(1e-2, dtype=self.dtype)) for dim in self.input_dims]

    def contract(self, vectors: List[np.ndarray]) -> np.ndarray:
        if len(vectors) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} vectors, got {len(vectors)}.")
        fused = np.ones(self.output_rank, dtype=self.dtype)
        for i, (v, d) in enumerate(zip(vectors, self.input_dims)):
            arr = np.asarray(v, dtype=self.dtype).reshape(-1)
            if arr.size != d:
                raise ValueError(f"Vector #{i} invalid shape; expected ({d},), got {arr.shape}.")
            fused *= arr @ self.factor_matrices[i]
        if not _is_all_finite(fused):
            raise FloatingPointError("Non-finite values encountered during TuckerTensor.contract.")
        return fused.astype(self.dtype, copy=False)

    @property
    def parameter_count(self) -> int:
        return int(sum(m.size for m in self.factor_matrices))


class TuckerFusionLayer:
    """CPU/NumPy Tucker fusion layer using Hadamard product of projections."""

    def __init__(self, input_dims: List[int], output_rank: int, rng: Optional[np.random.Generator] = None, dtype: np.dtype = np.float32):
        if not isinstance(input_dims, list) or any(int(d) <= 0 for d in input_dims):
            raise ValueError("input_dims must be a list of positive integers.")
        if int(output_rank) <= 0:
            raise ValueError("output_rank must be positive.")
        self.input_dims = [int(d) for d in input_dims]
        self.output_rank = int(output_rank)
        self.num_modalities = len(self.input_dims)
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()
        self.dtype = np.dtype(dtype)
        self.factor_matrices = [(self.rng.standard_normal((dim, self.output_rank)).astype(self.dtype) * np.asarray(1e-2, dtype=self.dtype)) for dim in self.input_dims]

    def contract(self, vectors: List[np.ndarray]) -> np.ndarray:
        if len(vectors) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} input vectors, got {len(vectors)}.")
        projected = []
        for i, (vec, dim) in enumerate(zip(vectors, self.input_dims)):
            arr = np.asarray(vec, dtype=self.dtype).reshape(-1)
            if arr.size != dim:
                raise ValueError(f"Input #{i} invalid size: expected {dim}, got {arr.size}.")
            projected.append(arr @ self.factor_matrices[i])
        fused = projected[0].astype(self.dtype, copy=False)
        for p in projected[1:]:
            fused *= p
        if not _is_all_finite(fused):
            raise FloatingPointError("Non-finite values encountered during TuckerFusionLayer.contract.")
        return fused.astype(self.dtype, copy=False)

    def train(self, error_gradient: np.ndarray, last_inputs: List[np.ndarray], learning_rate: float, clip: float = 1.0) -> List[np.ndarray]:
        if len(last_inputs) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} inputs for training, got {len(last_inputs)}.")
        error_gradient = np.asarray(error_gradient, dtype=self.dtype).reshape(-1)
        if error_gradient.size != self.output_rank:
            raise ValueError(f"error_gradient size must be {self.output_rank}, got {error_gradient.size}.")
        inputs = [np.asarray(v, dtype=self.dtype).reshape(-1) for v in last_inputs]
        for i, (arr, dim) in enumerate(zip(inputs, self.input_dims)):
            if arr.size != dim:
                raise ValueError(f"last_inputs[{i}] size must be {dim}, got {arr.size}.")
        projected = [inputs[i] @ self.factor_matrices[i] for i in range(self.num_modalities)]
        back_grads: List[np.ndarray] = []
        for i in range(self.num_modalities):
            dy_dpi = np.ones_like(error_gradient, dtype=self.dtype)
            for j in range(self.num_modalities):
                if i != j:
                    dy_dpi *= projected[j]
            dL_dpi = error_gradient * dy_dpi
            grad_Fi = np.outer(inputs[i], dL_dpi).astype(self.dtype, copy=False)
            if clip is not None and clip > 0:
                np.clip(grad_Fi, -float(clip), float(clip), out=grad_Fi)
            self.factor_matrices[i] -= float(learning_rate) * grad_Fi
            dL_dvi = dL_dpi @ self.factor_matrices[i].T
            back_grads.append(dL_dvi.astype(self.dtype, copy=False))
        return back_grads

    @property
    def parameter_count(self) -> int:
        return int(sum(m.size for m in self.factor_matrices))


# =============================================================================
# Experimental compatibility blocks retained for existing lobe wrappers
# =============================================================================


class TreeTensorFusion:
    """Hierarchical compatibility fusion block."""

    def __init__(self, physical_dims: List[int], abstract_dims: List[int], final_rank: int, rng: Optional[np.random.Generator] = None, dtype: np.dtype = np.float32):
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()
        self.dtype = np.dtype(dtype)
        physical_fusion_rank = max(1, int(sum(int(d) for d in physical_dims) // max(1, len(physical_dims))))
        self.physical_fusion_layer = TuckerFusionLayer(physical_dims, physical_fusion_rank, rng=self.rng, dtype=self.dtype)
        self.root_fusion_layer = TuckerFusionLayer([physical_fusion_rank] + [int(d) for d in abstract_dims], int(final_rank), rng=self.rng, dtype=self.dtype)

    def contract(self, modalities: Dict[str, np.ndarray]) -> np.ndarray:
        physical_inputs = [modalities["memory"], modalities["sensory"], modalities["tcfc_pred"]]
        abstract_inputs = [modalities["vlm_strategy"]]
        physical_summary = self.physical_fusion_layer.contract(physical_inputs)
        return self.root_fusion_layer.contract([physical_summary] + abstract_inputs)

    def train(self, final_error_gradient: np.ndarray, modalities: Dict[str, np.ndarray], learning_rate: float) -> None:
        physical_inputs = [modalities["memory"], modalities["sensory"], modalities["tcfc_pred"]]
        abstract_inputs = [modalities["vlm_strategy"]]
        physical_summary = self.physical_fusion_layer.contract(physical_inputs)
        root_grads = self.root_fusion_layer.train(np.asarray(final_error_gradient, dtype=self.dtype), [physical_summary] + abstract_inputs, learning_rate)
        self.physical_fusion_layer.train(root_grads[0], physical_inputs, learning_rate)


class DendriticProcessor:
    """Small pure-NumPy 3D conv-like extractor over tiny voxel patches."""

    def __init__(self, input_channels: int, output_channels: int, patch_size: int = 7, rng: Optional[np.random.Generator] = None, dtype: np.dtype = np.float32):
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(seed=42)
        self.dtype = np.dtype(dtype)
        self.output_channels = int(output_channels)
        self.patch_size = int(patch_size)
        if self.patch_size < 3:
            raise ValueError("patch_size must be >= 3.")
        if int(input_channels) <= 0 or int(output_channels) <= 0:
            raise ValueError("input_channels and output_channels must be positive.")
        self.input_channels = int(input_channels)
        self.conv_kernel = self.rng.standard_normal((self.output_channels, self.input_channels, 3, 3, 3)).astype(self.dtype) * np.asarray(0.1, dtype=self.dtype)
        self.bias = np.zeros(self.output_channels, dtype=self.dtype)
        self.final_feature_size = self.output_channels * (self.patch_size - 2) ** 3

    def process(self, memory_patch: np.ndarray) -> np.ndarray:
        patch = np.asarray(memory_patch, dtype=self.dtype)
        if patch.ndim != 4:
            raise ValueError(f"memory_patch must be 4D (P,P,P,C), got {patch.shape}.")
        if patch.shape[:3] != (self.patch_size, self.patch_size, self.patch_size):
            raise ValueError(f"memory_patch spatial dims must be ({self.patch_size},{self.patch_size},{self.patch_size}).")
        C = int(patch.shape[3])
        if C != self.input_channels:
            raise ValueError(f"memory_patch channels must be {self.input_channels}, got {C}.")
        patch_chwd = np.transpose(patch, (3, 0, 1, 2))
        out_depth = self.patch_size - 2
        out = np.zeros((self.output_channels, out_depth, out_depth, out_depth), dtype=self.dtype)
        for c_out in range(self.output_channels):
            for z in range(out_depth):
                for y in range(out_depth):
                    for x in range(out_depth):
                        acc = np.asarray(0.0, dtype=self.dtype)
                        for c_in in range(C):
                            window = patch_chwd[c_in, z : z + 3, y : y + 3, x : x + 3]
                            acc = acc + np.sum(window * self.conv_kernel[c_out, c_in])
                        out[c_out, z, y, x] = acc + self.bias[c_out]
        feat = np.tanh(out).reshape(-1).astype(self.dtype, copy=False)
        if not _is_all_finite(feat):
            raise FloatingPointError("Non-finite values encountered during DendriticProcessor.process.")
        return feat


# =============================================================================
# Decomposition and optimization
# =============================================================================


def tensor_train_svd(
    matrix: ArrayLike,
    output_dims: List[int],
    input_dims: List[int],
    max_bond_dim: int,
    *,
    dtype: Optional[np.dtype] = np.float32,
    energy_tol: Optional[float] = 0.999,
    device: str = "cpu",
    use_scipy_gesdd: bool = True,
    svd_jitter: float = 1e-12,
    stable_factor_dtype: np.dtype = np.float64,
    check_finite: bool = True,
    seed: Optional[int] = None,
    deterministic: bool = False,
    consensus_safe: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    commit_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[TensorTrain]:
    output_dims = [int(d) for d in output_dims]
    input_dims = [int(d) for d in input_dims]
    if len(output_dims) != len(input_dims) or len(output_dims) == 0:
        raise ValueError("output_dims and input_dims must have equal non-zero length.")
    if any(d <= 0 for d in output_dims + input_dims):
        raise ValueError("All output_dims and input_dims must be positive.")
    max_bond_dim = int(max(1, max_bond_dim))
    total_out, total_in = _prod(output_dims), _prod(input_dims)
    W_cpu = _to_cpu(matrix)
    if W_cpu.shape != (total_out, total_in):
        raise ValueError(f"Matrix shape {W_cpu.shape} != ({total_out}, {total_in}).")
    if check_finite and not np.isfinite(W_cpu).all():
        return None

    num_cores = len(output_dims)
    dt = np.dtype(dtype) if dtype is not None else np.dtype(np.float32)
    work_dt = np.dtype(stable_factor_dtype)
    tensor = np.asarray(W_cpu, dtype=work_dt).reshape(tuple(output_dims + input_dims))
    perm = [j for i in range(num_cores) for j in (i, i + num_cores)]
    tensor = tensor.transpose(perm)

    cores: List[np.ndarray] = []
    left_rank = 1
    current = tensor
    for i in range(num_cores - 1):
        left_dim = left_rank * output_dims[i] * input_dims[i]
        mat = current.reshape(left_dim, -1)
        try:
            U, S, Vh = _svd_stable(mat, xp=np, use_scipy_gesdd=use_scipy_gesdd, jitter=svd_jitter)
        except Exception:
            return None
        if not _is_all_finite(U, S, Vh) or int(getattr(S, "size", 0)) == 0:
            return None
        r_energy = _clip_rank_by_energy(S, energy_tol)
        rank = max(1, int(min(max_bond_dim, r_energy, int(S.shape[0]))))
        U_r = np.asarray(U[:, :rank], dtype=dt)
        S_r = np.asarray(S[:rank], dtype=work_dt)
        Vh_r = np.asarray(Vh[:rank, :], dtype=work_dt)
        core_i = U_r.reshape(left_rank, output_dims[i], input_dims[i], rank).astype(dt, copy=False)
        if check_finite and not _is_all_finite(core_i):
            return None
        cores.append(core_i)
        current = (S_r[:, None] * Vh_r).astype(work_dt, copy=False)
        left_rank = rank

    last = current.reshape(left_rank, output_dims[-1], input_dims[-1], 1).astype(dt, copy=False)
    if check_finite and not _is_all_finite(last):
        return None
    cores.append(last)
    cfg = TTConfig(
        dtype=dt,
        device=device.lower().strip(),
        svd_use_scipy_gesdd=use_scipy_gesdd,
        svd_jitter=svd_jitter,
        stable_factor_dtype=work_dt,
        check_finite=check_finite,
        seed=seed,
        deterministic=deterministic,
        consensus_safe=consensus_safe,
    )
    try:
        tt = TensorTrain(output_dims, input_dims, cores_data=cores, config=cfg, metadata=metadata, commit_metadata=commit_metadata)
        if device.lower().strip() == "gpu":
            tt = tt.to_device("gpu")
        return tt
    except Exception:
        return None


def find_optimal_permutations(W: np.ndarray, random_restarts: int = 0, *, rng: Optional[np.random.Generator] = None, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Lightweight deterministic row/column reordering heuristic for TT compressibility."""
    W = np.asarray(W)
    if W.ndim != 2:
        raise ValueError("W must be a 2D matrix.")
    m, n = W.shape
    if m <= 1 or n <= 1:
        return np.arange(m, dtype=np.int32), np.arange(n, dtype=np.int32)
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(0 if seed is None else int(seed))

    def _one_pass(W_local: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        SAMPLE_SIZE = 4096
        sample_m = min(m, max(1, int(math.sqrt(SAMPLE_SIZE))))
        sample_n = min(n, max(1, int(math.sqrt(SAMPLE_SIZE))))
        row_perm = np.arange(m, dtype=np.int32)
        if m > 1:
            row_idx = rng.choice(m, size=sample_m, replace=False).astype(np.int32) if m > SAMPLE_SIZE else np.arange(m, dtype=np.int32)
            rows = W_local[row_idx, :].astype(np.float32, copy=False)
            rows = rows / (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-12)
            sim = rows @ rows.T
            start = int(np.argmax(np.linalg.norm(W_local[row_idx, :], axis=1)))
            path, seen, cur = [start], {start}, start
            while len(path) < sample_m:
                nbrs = sim[cur]
                best, best_val = None, -np.inf
                for j in range(sample_m):
                    if j in seen:
                        continue
                    val = float(nbrs[j])
                    if val > best_val:
                        best_val, best = val, j
                if best is None:
                    best = next((j for j in range(sample_m) if j not in seen), None)
                if best is None:
                    break
                path.append(int(best))
                seen.add(int(best))
                cur = int(best)
            row_perm = row_idx[np.array(path, dtype=np.int32)]
            if m > row_perm.size:
                row_perm = np.concatenate([row_perm, np.setdiff1d(np.arange(m, dtype=np.int32), row_perm)])
        col_perm = np.arange(n, dtype=np.int32)
        if n > 1:
            col_idx = rng.choice(n, size=sample_n, replace=False).astype(np.int32) if n > SAMPLE_SIZE else np.arange(n, dtype=np.int32)
            cols = W_local[:, col_idx].T.astype(np.float32, copy=False)
            cols = cols / (np.linalg.norm(cols, axis=1, keepdims=True) + 1e-12)
            sim = cols @ cols.T
            start = int(np.argmax(np.linalg.norm(W_local[:, col_idx], axis=0)))
            path, seen, cur = [start], {start}, start
            while len(path) < sample_n:
                nbrs = sim[cur]
                best, best_val = None, -np.inf
                for j in range(sample_n):
                    if j in seen:
                        continue
                    val = float(nbrs[j])
                    if val > best_val:
                        best_val, best = val, j
                if best is None:
                    best = next((j for j in range(sample_n) if j not in seen), None)
                if best is None:
                    break
                path.append(int(best))
                seen.add(int(best))
                cur = int(best)
            col_perm = col_idx[np.array(path, dtype=np.int32)]
            if n > col_perm.size:
                col_perm = np.concatenate([col_perm, np.setdiff1d(np.arange(n, dtype=np.int32), col_perm)])
        return row_perm.astype(np.int32), col_perm.astype(np.int32)

    def score(rr: np.ndarray, cc: np.ndarray) -> float:
        Wp = W[rr][:, cc]
        k = min(Wp.shape)
        return float(np.sum(np.abs(np.diag(Wp[:k, :k]))))

    best_r, best_c = _one_pass(W)
    best_score = score(best_r, best_c)
    for _ in range(int(max(0, random_restarts))):
        r, c = _one_pass(W)
        s = score(r, c)
        if s > best_score:
            best_score, best_r, best_c = s, r, c
    return best_r, best_c


# =============================================================================
# Self-tests
# =============================================================================


def _unit_test_tt_contract(rng: np.random.Generator, out_dims: List[int], in_dims: List[int], bond_dims: List[int]) -> None:
    tt = TensorTrain(out_dims, in_dims, bond_dims=bond_dims, rng=rng, config=TTConfig(dtype=np.float32, device="cpu", seed=0, deterministic=True))
    x = rng.normal(size=_prod(in_dims)).astype(np.float32)
    y1 = tt.contract_with_vector(x)
    W = tt.to_matrix(force_cpu=True)
    y2 = W @ x
    if not np.allclose(y1, y2, atol=1e-5, rtol=1e-5):
        raise AssertionError(f"TT vector contract failed. max_err={float(np.max(np.abs(y1 - y2)))}")
    X = rng.normal(size=(7, _prod(in_dims))).astype(np.float32)
    Y1 = tt.contract_batch(X, vectorized=True)
    Y2 = (W @ X.T).T
    if not np.allclose(Y1, Y2, atol=1e-5, rtol=1e-5):
        raise AssertionError(f"TT batch contract failed. max_err={float(np.max(np.abs(Y1 - Y2)))}")
    Y3 = tt.contract_batch(X, vectorized=False)
    if not np.allclose(Y3, Y2, atol=1e-5, rtol=1e-5):
        raise AssertionError("TT batch loop contract failed.")
    tt2, report = (tt + 0.1 * tt).round_with_report(max_rank=max(1, max(bond_dims) if bond_dims else 4), energy_tol=0.999)
    y3 = tt2.contract_with_vector(x)
    W2 = tt2.to_matrix(force_cpu=True)
    y4 = W2 @ x
    if not np.allclose(y3, y4, atol=2e-4, rtol=2e-4):
        raise AssertionError("TT add/round contract failed.")
    if report.new_parameter_count <= 0:
        raise AssertionError("round_with_report produced invalid report.")


def _unit_test_ttsvd(rng: np.random.Generator) -> None:
    out_dims = [2, 3]
    in_dims = [2, 5]
    W = rng.normal(size=(_prod(out_dims), _prod(in_dims))).astype(np.float64)
    tt = tensor_train_svd(W, out_dims, in_dims, max_bond_dim=8, dtype=np.float32, energy_tol=None, seed=42, deterministic=True)
    if tt is None:
        raise AssertionError("tensor_train_svd returned None.")
    W2 = tt.to_matrix(force_cpu=True)
    if not np.allclose(W, W2, atol=1e-4, rtol=1e-4):
        raise AssertionError(f"TT-SVD reconstruction failed. max_err={float(np.max(np.abs(W - W2)))}")


def _unit_test_serialization_and_commitment(rng: np.random.Generator) -> None:
    tt = TensorTrain(
        [2, 2],
        [2, 2],
        bond_dims=[3],
        rng=rng,
        config=TTConfig(dtype=np.float32, device="cpu", seed=123, deterministic=True, consensus_safe=True),
        metadata={"test": True},
        commit_metadata={"purpose": "self_test"},
    )
    c = tt.commitment()
    if not tt.verify_commitment(c):
        raise AssertionError("TensorTrain commitment self-verification failed.")
    x = rng.normal(size=4).astype(np.float32)
    y = tt.apply(x)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tt_test.npz"
        tt.save_npz(path)
        tt2 = TensorTrain.load_npz(path, verify=True)
    y2 = tt2.apply(x)
    if not np.allclose(y, y2, atol=1e-6, rtol=1e-6):
        raise AssertionError("TensorTrain serialization round-trip failed.")
    if not tt2.verify_commitment(c):
        raise AssertionError("Loaded TensorTrain commitment mismatch.")


def run_self_tests() -> Dict[str, Any]:
    rng = np.random.default_rng(0)
    _unit_test_tt_contract(rng, [6], [10], [])
    cases = [
        ([2, 3], [2, 5], [4]),
        ([2, 2, 2], [2, 2, 2], [3, 3]),
        ([3, 2], [4, 3], [5]),
        ([2, 3, 2], [3, 2, 2], [4, 4]),
        ([4, 2, 2], [2, 2, 3], [6, 5]),
    ]
    for out_dims, in_dims, bond_dims in cases:
        _unit_test_tt_contract(rng, out_dims, in_dims, bond_dims)
    _unit_test_ttsvd(rng)
    _unit_test_serialization_and_commitment(rng)
    return {
        "ok": True,
        "version": TN_VERSION,
        "scipy": bool(_SCIPY_OK),
        "cupy": bool(_CUPY_OK),
        "tests": 3 + len(cases),
    }


if __name__ == "__main__":
    print(json.dumps(run_self_tests(), indent=2, ensure_ascii=False))
