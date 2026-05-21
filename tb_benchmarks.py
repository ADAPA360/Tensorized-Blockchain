#!/usr/bin/env python3
# tb_benchmarks.py
r"""
Tensorized Blockchain Benchmark Harness
=======================================

Purpose
-------
Production-grade, dependency-light performance and compression benchmark runner
for the Tensorized Blockchain stack.

Responsibilities
----------------
- TensorTrain / TT-SVD compression scaling.
- Dense-vs-TT matvec comparison.
- Ledger append / validation speed where the ledger module is available.
- Digital-twin state-root / commitment speed where the twin module is available.
- NCP and CfC lobe inference speed where lobe modules are available.
- Memory-footprint reporting using tracemalloc and process RSS where available.
- Deterministic benchmark profiles: quick, default, deep.
- JSON and CSV export suitable for demo dashboards and CI artifacts.

Design notes
------------
This module is intentionally a benchmark boundary. It imports heavy project
modules lazily and records degraded/unavailable modules as structured benchmark
results instead of failing the entire suite.

It does not mutate production ledgers or checkpoints unless explicitly given an
output directory. All generated benchmark artifacts are deterministic fixtures.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import gc
import importlib
import json
import math
import os
import platform
import resource
import statistics
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


BENCHMARK_VERSION = "tb-benchmarks-v1.0-production"
_EPS = 1e-12


# =============================================================================
# Import / serialization helpers
# =============================================================================


def _module_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()


def _ensure_local_import_path() -> None:
    here = _module_dir()
    for p in (here, Path.cwd()):
        s = str(p.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)


def _import_optional(module_name: str) -> Tuple[Optional[Any], Optional[str]]:
    _ensure_local_import_path()
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:
        return None, repr(exc)


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
    return str(obj)


def _canonical_json(obj: Any, *, pretty: bool = False) -> str:
    return json.dumps(
        _json_safe(obj),
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        ensure_ascii=False,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# =============================================================================
# Timing / memory helpers
# =============================================================================


def _perf_counter() -> float:
    return time.perf_counter()


def _rss_mb() -> Optional[float]:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux reports kilobytes; macOS reports bytes. Infer conservatively.
        ru = float(usage.ru_maxrss)
        if sys.platform == "darwin":
            return ru / (1024.0 * 1024.0)
        return ru / 1024.0
    except Exception:
        return None


@dataclass
class TimedSample:
    seconds: float
    peak_tracemalloc_mb: float = 0.0
    rss_mb: Optional[float] = None


@contextlib.contextmanager
def _timed_sample() -> Iterable[TimedSample]:
    gc.collect()
    import tracemalloc

    tracemalloc.start()
    start = _perf_counter()
    sample = TimedSample(seconds=0.0, peak_tracemalloc_mb=0.0, rss_mb=_rss_mb())
    try:
        yield sample
    finally:
        elapsed = _perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        sample.seconds = float(elapsed)
        sample.peak_tracemalloc_mb = float(peak / (1024.0 * 1024.0))
        sample.rss_mb = _rss_mb()


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return {
            "count": 0,
            "mean_s": None,
            "median_s": None,
            "min_s": None,
            "max_s": None,
            "stdev_s": None,
            "ops_per_s_mean": None,
        }
    mean = float(statistics.mean(vals))
    return {
        "count": len(vals),
        "mean_s": mean,
        "median_s": float(statistics.median(vals)),
        "min_s": float(min(vals)),
        "max_s": float(max(vals)),
        "stdev_s": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        "ops_per_s_mean": float(1.0 / mean) if mean > 0 else None,
    }


def _repeat(fn: Callable[[], Any], *, iterations: int, warmup: int = 1) -> Tuple[List[float], Any, List[Dict[str, Any]]]:
    last: Any = None
    memory: List[Dict[str, Any]] = []
    for _ in range(max(0, int(warmup))):
        last = fn()
    times: List[float] = []
    for _ in range(max(1, int(iterations))):
        with _timed_sample() as sample:
            last = fn()
        times.append(float(sample.seconds))
        memory.append({
            "peak_tracemalloc_mb": float(sample.peak_tracemalloc_mb),
            "rss_mb": sample.rss_mb,
        })
    return times, last, memory


# =============================================================================
# Benchmark dataclasses
# =============================================================================


@dataclass
class BenchmarkConfig:
    profile: str = "default"
    seed: int = 2027
    iterations: int = 5
    warmup: int = 1
    output_dir: str = ""
    max_dense_entries: int = 1_000_000
    include_expensive: bool = False
    force_cpu: bool = True

    def normalized(self) -> "BenchmarkConfig":
        cfg = dataclasses.replace(self)
        cfg.profile = str(cfg.profile or "default").lower().strip()
        if cfg.profile not in {"quick", "default", "deep"}:
            cfg.profile = "default"
        cfg.seed = int(cfg.seed)
        cfg.iterations = int(max(1, cfg.iterations))
        cfg.warmup = int(max(0, cfg.warmup))
        cfg.max_dense_entries = int(max(1, cfg.max_dense_entries))
        cfg.include_expensive = bool(cfg.include_expensive)
        cfg.force_cpu = bool(cfg.force_cpu)
        return cfg


@dataclass
class BenchmarkResult:
    name: str
    ok: bool
    status: str = "ok"
    category: str = "general"
    summary: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    memory: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    traceback: str = ""

    @classmethod
    def unavailable(cls, name: str, *, category: str, reason: str) -> "BenchmarkResult":
        return cls(name=name, category=category, ok=False, status="unavailable", error=str(reason))

    @classmethod
    def failed(cls, name: str, *, category: str, exc: BaseException) -> "BenchmarkResult":
        return cls(
            name=name,
            category=category,
            ok=False,
            status="failed",
            error=repr(exc),
            traceback=traceback.format_exc(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class BenchmarkSuiteReport:
    ok: bool
    version: str
    generated_at: str
    config: BenchmarkConfig
    environment: Dict[str, Any]
    dependency_status: Dict[str, Any]
    results: List[BenchmarkResult]

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


# =============================================================================
# Dependency status
# =============================================================================


def dependency_status() -> Dict[str, Any]:
    modules = [
        "tn",
        "digital_twin_kernel",
        "tensorized_blockchain_kernel",
        "twin_anything",
        "ncp",
        "cfc",
        "cognitive_lobe_runtime",
        "tb_schemas",
        "tb_hashing",
        "tb_tensor_commitments",
    ]
    out: Dict[str, Any] = {}
    for name in modules:
        mod, err = _import_optional(name)
        out[name] = {
            "available": mod is not None,
            "module": getattr(mod, "__file__", None) if mod is not None else None,
            "error": err,
        }
    return out


def environment_report() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "module_dir": str(_module_dir()),
        "rss_mb": _rss_mb(),
    }


# =============================================================================
# Benchmark profile sizing
# =============================================================================


def _profile_sizes(profile: str) -> Dict[str, Any]:
    profile = str(profile or "default").lower()
    if profile == "quick":
        return {
            "tt_shapes": [([4, 4], [4, 4], 4), ([8, 8], [8, 8], 6)],
            "dense_vs_tt_shape": ([8, 8], [8, 8], 6),
            "twin_vector_dim": 64,
            "lobe": {"ncp_inputs": 16, "ncp_hidden": 64, "ncp_outputs": 16, "cfc_inputs": 16, "cfc_hidden": 32, "cfc_cells": 4},
            "ledger_blocks": 3,
        }
    if profile == "deep":
        return {
            "tt_shapes": [([8, 8], [8, 8], 8), ([16, 16], [16, 16], 12), ([8, 8, 4], [8, 8, 4], 12)],
            "dense_vs_tt_shape": ([16, 16], [16, 16], 12),
            "twin_vector_dim": 256,
            "lobe": {"ncp_inputs": 64, "ncp_hidden": 512, "ncp_outputs": 64, "cfc_inputs": 64, "cfc_hidden": 128, "cfc_cells": 8},
            "ledger_blocks": 12,
        }
    return {
        "tt_shapes": [([4, 8], [4, 8], 6), ([8, 8], [8, 8], 8), ([4, 4, 4], [4, 4, 4], 8)],
        "dense_vs_tt_shape": ([8, 8], [8, 8], 8),
        "twin_vector_dim": 128,
        "lobe": {"ncp_inputs": 32, "ncp_hidden": 128, "ncp_outputs": 32, "cfc_inputs": 32, "cfc_hidden": 64, "cfc_cells": 4},
        "ledger_blocks": 6,
    }


def _prod(xs: Sequence[int]) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return int(p)


# =============================================================================
# Tensor benchmarks
# =============================================================================


def benchmark_tt_compression(cfg: BenchmarkConfig) -> BenchmarkResult:
    name = "tt_compression_scaling"
    tn, err = _import_optional("tn")
    if tn is None:
        return BenchmarkResult.unavailable(name, category="tensor", reason=err or "tn unavailable")

    try:
        rng = np.random.default_rng(cfg.seed)
        sizes = _profile_sizes(cfg.profile)["tt_shapes"]
        cases: List[Dict[str, Any]] = []
        all_times: List[float] = []
        all_memory: List[Dict[str, Any]] = []

        for output_dims, input_dims, max_rank in sizes:
            out_dim = _prod(output_dims)
            in_dim = _prod(input_dims)
            entries = out_dim * in_dim
            if entries > cfg.max_dense_entries:
                cases.append({
                    "output_dims": output_dims,
                    "input_dims": input_dims,
                    "skipped": True,
                    "reason": f"dense entries {entries} exceeds max_dense_entries={cfg.max_dense_entries}",
                })
                continue

            # Structured low-rank-ish dense matrix so TT-SVD compression is meaningful.
            A = rng.normal(size=(out_dim, min(out_dim, in_dim, max_rank * 2))).astype(np.float64)
            B = rng.normal(size=(min(out_dim, in_dim, max_rank * 2), in_dim)).astype(np.float64)
            W = (A @ B / math.sqrt(max(1, A.shape[1]))).astype(np.float64)

            def run_once() -> Any:
                if hasattr(tn, "TensorTrain") and hasattr(tn.TensorTrain, "from_dense"):
                    return tn.TensorTrain.from_dense(
                        W,
                        output_dims=list(output_dims),
                        input_dims=list(input_dims),
                        max_bond_dim=int(max_rank),
                        dtype=np.float32,
                        energy_tol=0.999,
                        device="cpu",
                    )
                return tn.tensor_train_svd(
                    W,
                    output_dims=list(output_dims),
                    input_dims=list(input_dims),
                    max_bond_dim=int(max_rank),
                    dtype=np.float32,
                    energy_tol=0.999,
                    device="cpu",
                )

            times, tt, mem = _repeat(run_once, iterations=cfg.iterations, warmup=cfg.warmup)
            all_times.extend(times)
            all_memory.extend(mem)
            if tt is None:
                cases.append({"output_dims": output_dims, "input_dims": input_dims, "ok": False, "error": "TT construction returned None"})
                continue

            params = int(tt.parameter_count()) if hasattr(tt, "parameter_count") else int(sum(np.asarray(c).size for c in getattr(tt, "cores_data", [])))
            dense_params = int(entries)
            compression_factor = float(dense_params / max(1, params))
            rel_error: Optional[float] = None
            if entries <= cfg.max_dense_entries and hasattr(tt, "to_matrix"):
                try:
                    W2 = np.asarray(tt.to_matrix(force_cpu=True, max_entries=cfg.max_dense_entries), dtype=np.float64)
                    rel_error = float(np.linalg.norm(W - W2) / (np.linalg.norm(W) + _EPS))
                except Exception:
                    rel_error = None

            commit_hash = None
            if hasattr(tt, "commitment"):
                try:
                    c = tt.commitment()
                    commit_hash = getattr(c, "commitment_hash", None) or (c.get("commitment_hash") if isinstance(c, dict) else None)
                except Exception:
                    commit_hash = None

            cases.append({
                "output_dims": list(output_dims),
                "input_dims": list(input_dims),
                "max_rank": int(max_rank),
                "dense_shape": [int(out_dim), int(in_dim)],
                "dense_parameters": dense_params,
                "tt_parameters": params,
                "compression_factor": compression_factor,
                "relative_reconstruction_error": rel_error,
                "bond_ranks": tt.bond_ranks() if hasattr(tt, "bond_ranks") else [],
                "commitment_hash": commit_hash,
                "timing": _summary(times),
            })

        return BenchmarkResult(
            name=name,
            ok=True,
            category="tensor",
            summary=_summary(all_times),
            metrics={"cases": cases},
            parameters={"profile": cfg.profile, "iterations": cfg.iterations, "max_dense_entries": cfg.max_dense_entries},
            memory=all_memory,
        )
    except Exception as exc:
        return BenchmarkResult.failed(name, category="tensor", exc=exc)


def benchmark_dense_vs_tt(cfg: BenchmarkConfig) -> BenchmarkResult:
    name = "dense_vs_tt_matvec"
    tn, err = _import_optional("tn")
    if tn is None:
        return BenchmarkResult.unavailable(name, category="tensor", reason=err or "tn unavailable")

    try:
        rng = np.random.default_rng(cfg.seed + 101)
        output_dims, input_dims, max_rank = _profile_sizes(cfg.profile)["dense_vs_tt_shape"]
        out_dim = _prod(output_dims)
        in_dim = _prod(input_dims)
        entries = out_dim * in_dim
        if entries > cfg.max_dense_entries:
            return BenchmarkResult.unavailable(name, category="tensor", reason="dense matrix would exceed max_dense_entries")

        W = rng.normal(scale=1.0 / math.sqrt(in_dim), size=(out_dim, in_dim)).astype(np.float64)
        x = rng.normal(size=(in_dim,)).astype(np.float32)
        if hasattr(tn, "TensorTrain") and hasattr(tn.TensorTrain, "from_dense"):
            tt = tn.TensorTrain.from_dense(W, list(output_dims), list(input_dims), int(max_rank), dtype=np.float32, energy_tol=0.999, device="cpu")
        else:
            tt = tn.tensor_train_svd(W, list(output_dims), list(input_dims), int(max_rank), dtype=np.float32, energy_tol=0.999, device="cpu")
        if tt is None:
            return BenchmarkResult.unavailable(name, category="tensor", reason="TT construction returned None")

        dense_times, y_dense, dense_mem = _repeat(lambda: W.astype(np.float32) @ x, iterations=cfg.iterations, warmup=cfg.warmup)
        tt_times, y_tt, tt_mem = _repeat(lambda: np.asarray(tt.apply(x)), iterations=cfg.iterations, warmup=cfg.warmup)
        err_rel = float(np.linalg.norm(np.asarray(y_dense, dtype=np.float64) - np.asarray(y_tt, dtype=np.float64)) / (np.linalg.norm(y_dense) + _EPS))
        dense_mean = statistics.mean(dense_times)
        tt_mean = statistics.mean(tt_times)
        return BenchmarkResult(
            name=name,
            ok=True,
            category="tensor",
            summary={
                "dense": _summary(dense_times),
                "tt": _summary(tt_times),
                "tt_speedup_vs_dense": float(dense_mean / tt_mean) if tt_mean > 0 else None,
            },
            metrics={
                "relative_output_error": err_rel,
                "dense_parameters": int(entries),
                "tt_parameters": int(tt.parameter_count()) if hasattr(tt, "parameter_count") else None,
                "compression_factor": float(entries / max(1, int(tt.parameter_count()))) if hasattr(tt, "parameter_count") else None,
                "bond_ranks": tt.bond_ranks() if hasattr(tt, "bond_ranks") else [],
            },
            parameters={"output_dims": list(output_dims), "input_dims": list(input_dims), "max_rank": int(max_rank)},
            memory=[{"dense": dense_mem, "tt": tt_mem}],
        )
    except Exception as exc:
        return BenchmarkResult.failed(name, category="tensor", exc=exc)


# =============================================================================
# Digital twin benchmarks
# =============================================================================


def benchmark_twin_state_root(cfg: BenchmarkConfig) -> BenchmarkResult:
    name = "twin_state_root_speed"
    dtk, err = _import_optional("digital_twin_kernel")
    if dtk is None:
        return BenchmarkResult.unavailable(name, category="digital_twin", reason=err or "digital_twin_kernel unavailable")

    try:
        vector_dim = int(_profile_sizes(cfg.profile)["twin_vector_dim"])
        if hasattr(dtk, "RuntimeMode"):
            # Avoid relying on exact constructor; older/newer variants may differ.
            pass
        ttn = dtk.DigitalTwinsBuilder.build_platform(
            vector_dim=vector_dim,
            seed=cfg.seed,
            sketch_dim=min(96, vector_dim),
            history_capacity=32,
            use_tn_projection=False,
        )
        rng = np.random.default_rng(cfg.seed + 202)
        update_nodes = ["2.3.2", "4.1.1", "3.1.1", "4.1.4"]
        for i, nid in enumerate(update_nodes):
            if nid in getattr(ttn, "nodes", {}):
                try:
                    ttn.update_node_data(nid, rng.normal(size=(8,)).astype(np.float32), note=f"benchmark_seed_{i}")
                except TypeError:
                    ttn.update_node_data(nid, rng.normal(size=(8,)).astype(np.float32))

        def compute_commitment() -> Any:
            if hasattr(ttn, "state_commitment"):
                return ttn.state_commitment()
            if hasattr(ttn, "merkle_root"):
                h = ttn.merkle_root()
                if hasattr(dtk, "MerkleHasher") and hasattr(dtk.MerkleHasher, "hex64"):
                    return {"sketch_root": dtk.MerkleHasher.hex64(h)}
                return {"sketch_root": str(h)}
            return ttn.to_dict(include_histories=False) if hasattr(ttn, "to_dict") else str(ttn)

        times, commitment, mem = _repeat(compute_commitment, iterations=cfg.iterations, warmup=cfg.warmup)
        validation = None
        if hasattr(ttn, "validate_topology"):
            try:
                validation = ttn.validate_topology()
            except Exception as exc:
                validation = {"ok": False, "error": repr(exc)}

        return BenchmarkResult(
            name=name,
            ok=True,
            category="digital_twin",
            summary=_summary(times),
            metrics={
                "node_count": len(getattr(ttn, "nodes", {})),
                "vector_dim": vector_dim,
                "commitment_preview": _json_safe(commitment),
                "topology_validation": _json_safe(validation),
            },
            parameters={"profile": cfg.profile, "iterations": cfg.iterations},
            memory=mem,
        )
    except Exception as exc:
        return BenchmarkResult.failed(name, category="digital_twin", exc=exc)


# =============================================================================
# Lobe benchmarks
# =============================================================================


def benchmark_lobe_inference(cfg: BenchmarkConfig) -> BenchmarkResult:
    name = "lobe_inference_speed"
    ncp_mod, ncp_err = _import_optional("ncp")
    cfc_mod, cfc_err = _import_optional("cfc")
    if ncp_mod is None and cfc_mod is None:
        return BenchmarkResult.unavailable(name, category="lobes", reason=f"ncp={ncp_err}; cfc={cfc_err}")

    try:
        rng = np.random.default_rng(cfg.seed + 303)
        spec = _profile_sizes(cfg.profile)["lobe"]
        subresults: Dict[str, Any] = {}
        memory: List[Dict[str, Any]] = []
        all_times: List[float] = []

        if ncp_mod is not None and hasattr(ncp_mod, "TensorizedNCP"):
            ncp = ncp_mod.TensorizedNCP(
                policy_id="benchmark_ncp_liquid_reflex",
                num_inputs=int(spec["ncp_inputs"]),
                num_hidden=int(spec["ncp_hidden"]),
                num_outputs=int(spec["ncp_outputs"]),
                bond_dim=4 if cfg.profile == "quick" else 8,
                rng=np.random.default_rng(cfg.seed + 304),
                verbose=False,
                tt_strict=False,
                output_activation="tanh",
                leaky_lambda=0.5,
                force_dense=False,
            )
            x = rng.normal(size=(int(spec["ncp_inputs"]),)).astype(np.float64)
            times, last, mem = _repeat(lambda: ncp.step(x, dt=1.0, return_state=True), iterations=cfg.iterations, warmup=cfg.warmup)
            all_times.extend(times)
            memory.extend(mem)
            y, aux = last
            subresults["ncp"] = {
                "timing": _summary(times),
                "output_shape": list(np.asarray(y).shape),
                "parameter_count": int(ncp.parameter_count()) if hasattr(ncp, "parameter_count") else None,
                "total_parameter_count": int(ncp.total_parameter_count()) if hasattr(ncp, "total_parameter_count") else None,
                "dense_equivalent_parameter_count": getattr(ncp, "dense_equivalent_parameter_count", lambda: None)(),
                "health": _json_safe(aux.get("health", {}) if isinstance(aux, dict) else {}),
                "snapshot": _json_safe(ncp.snapshot() if hasattr(ncp, "snapshot") else {}),
            }
        else:
            subresults["ncp"] = {"available": False, "error": ncp_err}

        if cfc_mod is not None:
            cls = getattr(cfc_mod, "TCfC_Policy", getattr(cfc_mod, "TensorizedCfCPolicy", None))
            if cls is not None:
                cfc = cls(
                    policy_id="benchmark_cfc_predictive_transition",
                    input_size=int(spec["cfc_inputs"]),
                    hidden_size=int(spec["cfc_hidden"]),
                    num_cells=int(spec["cfc_cells"]),
                    bond_dim=4 if cfg.profile == "quick" else 8,
                    rng=np.random.default_rng(cfg.seed + 305),
                    dtype=np.float32,
                    verbose=False,
                    tt_strict=False,
                    force_dense=False,
                )
                x2 = rng.normal(size=(int(spec["cfc_inputs"]),)).astype(np.float32)
                times2, last2, mem2 = _repeat(lambda: cfc.step(x2, time_delta=0.5, return_state=True), iterations=cfg.iterations, warmup=cfg.warmup)
                all_times.extend(times2)
                memory.extend(mem2)
                state, aux2 = last2
                subresults["cfc"] = {
                    "timing": _summary(times2),
                    "state_shape": list(np.asarray(state).shape),
                    "parameter_count": int(cfc.parameter_count()) if hasattr(cfc, "parameter_count") else None,
                    "equivalent_dense_parameter_count": int(cfc.equivalent_dense_parameter_count()) if hasattr(cfc, "equivalent_dense_parameter_count") else None,
                    "compression_factor": cfc.compression_factor() if hasattr(cfc, "compression_factor") else None,
                    "health": _json_safe(aux2.get("health", {}) if isinstance(aux2, dict) else {}),
                    "snapshot": _json_safe(cfc.snapshot() if hasattr(cfc, "snapshot") else {}),
                }
            else:
                subresults["cfc"] = {"available": False, "error": "TCfC_Policy not found"}
        else:
            subresults["cfc"] = {"available": False, "error": cfc_err}

        return BenchmarkResult(
            name=name,
            ok=True,
            category="lobes",
            summary=_summary(all_times),
            metrics=subresults,
            parameters={"profile": cfg.profile, **spec},
            memory=memory,
        )
    except Exception as exc:
        return BenchmarkResult.failed(name, category="lobes", exc=exc)


# =============================================================================
# Ledger benchmarks
# =============================================================================


def benchmark_ledger_append(cfg: BenchmarkConfig) -> BenchmarkResult:
    name = "ledger_append_speed"
    ledger_mod, led_err = _import_optional("tensorized_blockchain_kernel")
    dtk, dtk_err = _import_optional("digital_twin_kernel")
    ta, ta_err = _import_optional("twin_anything")
    if ledger_mod is None:
        return BenchmarkResult.unavailable(name, category="ledger", reason=led_err or "tensorized_blockchain_kernel unavailable")
    if dtk is None:
        return BenchmarkResult.unavailable(name, category="ledger", reason=dtk_err or "digital_twin_kernel unavailable")

    try:
        block_count = int(_profile_sizes(cfg.profile)["ledger_blocks"])
        # Prefer the current stack's own config and demo-twin helpers if present.
        if hasattr(ledger_mod, "LedgerConfig"):
            lcfg = ledger_mod.LedgerConfig(out_dim=4 if cfg.profile == "quick" else 8, in_dim=4 if cfg.profile == "quick" else 8, max_bond_dim=2 if cfg.profile == "quick" else 4, seed=cfg.seed)
            if hasattr(lcfg, "normalized"):
                lcfg = lcfg.normalized()
        else:
            lcfg = None

        if hasattr(ledger_mod, "LedgerMPS"):
            ledger = ledger_mod.LedgerMPS(lcfg) if lcfg is not None else ledger_mod.LedgerMPS()
        else:
            return BenchmarkResult.unavailable(name, category="ledger", reason="LedgerMPS not found")

        # Build a minimal twin with twin_anything when possible, otherwise platform.
        if ta is not None and hasattr(ta, "TwinAnythingFactory"):
            try:
                vector_dim = getattr(lcfg, "vector_dim", 64) if lcfg is not None else 64
                factory = ta.TwinAnythingFactory(vector_dim=int(vector_dim), sketch_dim=min(64, int(vector_dim)), seed=cfg.seed, use_tn_projection=False)
                data = factory.create_twin(
                    thing_name="Benchmark Robot",
                    thing_type="industrial_machine",
                    description="Deterministic benchmark fixture.",
                    observables={"temperature_c": 22.0, "battery_pct": 99.0},
                    components={"platform": "synthetic"},
                    success_criteria={"minimum_probability": 0.8},
                    tags=["benchmark"],
                )
                ttn = data.get("network")
                cp = data.get("control_plane")
            except Exception:
                vector_dim = getattr(lcfg, "vector_dim", 64) if lcfg is not None else 64
                ttn = dtk.DigitalTwinsBuilder.build_platform(vector_dim=int(vector_dim), seed=cfg.seed, sketch_dim=min(64, int(vector_dim)), use_tn_projection=False)
                cp = dtk.AkkuratInterface(ttn, strict=False) if hasattr(dtk, "AkkuratInterface") else None
        else:
            vector_dim = getattr(lcfg, "vector_dim", 64) if lcfg is not None else 64
            ttn = dtk.DigitalTwinsBuilder.build_platform(vector_dim=int(vector_dim), seed=cfg.seed, sketch_dim=min(64, int(vector_dim)), use_tn_projection=False)
            cp = dtk.AkkuratInterface(ttn, strict=False) if hasattr(dtk, "AkkuratInterface") else None

        rng = np.random.default_rng(cfg.seed + 404)
        times: List[float] = []
        memory: List[Dict[str, Any]] = []
        tx_ids: List[str] = []

        def make_tx(i: int) -> Any:
            action = dtk.Action(
                "update_node_data",
                node_id="5.4" if "5.4" in getattr(ttn, "nodes", {}) else "2.3.2",
                payload={"temperature_c": float(20.0 + i), "benchmark_noise": float(rng.normal())},
                note=f"benchmark_tx_{i}",
            )
            if hasattr(ledger_mod, "TensorTransaction"):
                if hasattr(ledger_mod.TensorTransaction, "new"):
                    return ledger_mod.TensorTransaction.new(sender_id="benchmark_sender", twin_node_id=getattr(action, "node_id", "0"), action=action, chain_id="tb-benchmark-chain-id", sender_sequence=i + 1, metadata={"benchmark_index": i})
                return ledger_mod.TensorTransaction(
                    tx_id=f"benchmark_tx_{i}",
                    sender_id="benchmark_sender",
                    twin_node_id=getattr(action, "node_id", "0"),
                    action=action,
                    timestamp=float(i),
                )
            raise RuntimeError("TensorTransaction not found")

        for i in range(block_count):
            tx = make_tx(i)
            tx_ids.append(str(getattr(tx, "tx_id", f"tx_{i}")))
            if cp is not None and hasattr(cp, "governed_actions"):
                with contextlib.suppress(Exception):
                    cp.governed_actions([tx.action], intent=f"benchmark_{i}")
            else:
                with contextlib.suppress(Exception):
                    ttn.update_node_data(getattr(tx.action, "node_id", "2.3.2"), getattr(tx.action, "payload", {}), note=getattr(tx.action, "note", ""))
            with _timed_sample() as sample:
                # New production kernels may expose append_transition/append_executed_transaction.
                if hasattr(ledger, "append_block"):
                    ledger.append_block([tx], ttn, metadata={"benchmark_index": i})
                elif hasattr(ledger, "append_transactions"):
                    ledger.append_transactions([tx], ttn)
                else:
                    raise RuntimeError("No supported ledger append method found")
            times.append(sample.seconds)
            memory.append({"peak_tracemalloc_mb": sample.peak_tracemalloc_mb, "rss_mb": sample.rss_mb})

        validate = ledger.validate_chain() if hasattr(ledger, "validate_chain") else {"ok": None, "reason": "validate_chain unavailable"}
        compression = None
        if hasattr(ledger, "compress_ledger"):
            with contextlib.suppress(Exception):
                compression = ledger.compress_ledger()

        return BenchmarkResult(
            name=name,
            ok=bool(validate.get("ok", True)) if isinstance(validate, Mapping) else True,
            category="ledger",
            summary=_summary(times),
            metrics={
                "block_count_appended": block_count,
                "chain_validation": _json_safe(validate),
                "compression": _json_safe(compression),
                "head_hash": getattr(ledger, "head_hash", None),
                "tx_ids": tx_ids,
            },
            parameters={"profile": cfg.profile, "block_count": block_count},
            memory=memory,
        )
    except Exception as exc:
        return BenchmarkResult.failed(name, category="ledger", exc=exc)


# =============================================================================
# Suite runner / exports
# =============================================================================


BENCHMARKS: Dict[str, Callable[[BenchmarkConfig], BenchmarkResult]] = {
    "tt": benchmark_tt_compression,
    "dense-vs-tt": benchmark_dense_vs_tt,
    "twin": benchmark_twin_state_root,
    "lobe": benchmark_lobe_inference,
    "ledger": benchmark_ledger_append,
}


def run_suite(cfg: BenchmarkConfig, *, include: Optional[Sequence[str]] = None) -> BenchmarkSuiteReport:
    cfg = cfg.normalized()
    selected = list(include or ["tt", "dense-vs-tt", "twin", "lobe", "ledger"])
    results: List[BenchmarkResult] = []
    for key in selected:
        fn = BENCHMARKS.get(key)
        if fn is None:
            results.append(BenchmarkResult.unavailable(key, category="unknown", reason="unknown benchmark"))
            continue
        results.append(fn(cfg))
    ok = all(r.ok or r.status == "unavailable" for r in results)
    return BenchmarkSuiteReport(
        ok=bool(ok),
        version=BENCHMARK_VERSION,
        generated_at=_now_iso_utc(),
        config=cfg,
        environment=environment_report(),
        dependency_status=dependency_status(),
        results=results,
    )


def write_report(report: BenchmarkSuiteReport, output_dir: Path, *, pretty: bool = True) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tb_benchmark_report.json"
    csv_path = output_dir / "tb_benchmark_summary.csv"
    _atomic_write_text(json_path, _canonical_json(report.to_dict(), pretty=pretty))

    rows: List[Dict[str, Any]] = []
    for r in report.results:
        flat = {
            "name": r.name,
            "category": r.category,
            "ok": r.ok,
            "status": r.status,
            "error": r.error,
        }
        s = r.summary
        if "mean_s" in s:
            flat.update({
                "mean_s": s.get("mean_s"),
                "median_s": s.get("median_s"),
                "min_s": s.get("min_s"),
                "max_s": s.get("max_s"),
                "ops_per_s_mean": s.get("ops_per_s_mean"),
            })
        elif "tt" in s or "dense" in s:
            flat.update({
                "dense_mean_s": s.get("dense", {}).get("mean_s") if isinstance(s.get("dense"), Mapping) else None,
                "tt_mean_s": s.get("tt", {}).get("mean_s") if isinstance(s.get("tt"), Mapping) else None,
                "tt_speedup_vs_dense": s.get("tt_speedup_vs_dense"),
            })
        rows.append(flat)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else ["name"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return {"json": str(json_path.resolve()), "csv": str(csv_path.resolve())}


# =============================================================================
# Self-test / CLI
# =============================================================================


def run_self_test() -> Dict[str, Any]:
    cfg = BenchmarkConfig(profile="quick", seed=123, iterations=1, warmup=0, max_dense_entries=200_000)
    report = run_suite(cfg, include=["tt", "dense-vs-tt"])
    if not report.results:
        raise AssertionError("No benchmark results produced")
    # At least one tensor benchmark should be ok when tn.py exists in this stack.
    ok_count = sum(1 for r in report.results if r.ok)
    if dependency_status().get("tn", {}).get("available") and ok_count == 0:
        raise AssertionError("tn.py is available but tensor benchmarks did not run")
    return {"ok": True, "results": [r.to_dict() for r in report.results]}


def _parse_modes(value: str) -> List[str]:
    v = str(value or "all").lower().strip()
    if v == "all":
        return ["tt", "dense-vs-tt", "twin", "lobe", "ledger"]
    if v == "self-test":
        return ["self-test"]
    aliases = {
        "tensor": "tt",
        "tt-compression": "tt",
        "dense_vs_tt": "dense-vs-tt",
        "lobes": "lobe",
        "digital-twin": "twin",
    }
    out = []
    for part in v.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(aliases.get(p, p))
    return out or ["tt"]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tensorized Blockchain benchmark harness")
    p.add_argument("--mode", default="all", help="all, self-test, or comma-separated: tt,dense-vs-tt,twin,lobe,ledger")
    p.add_argument("--profile", choices=["quick", "default", "deep"], default="default")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--max-dense-entries", type=int, default=1_000_000)
    p.add_argument("--include-expensive", action="store_true")
    p.add_argument("--output-dir", default="", help="Optional directory for JSON/CSV report export")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON to stdout and report file")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    modes = _parse_modes(args.mode)
    try:
        if modes == ["self-test"]:
            payload = run_self_test()
            print(_canonical_json(payload, pretty=True))
            return 0

        cfg = BenchmarkConfig(
            profile=args.profile,
            seed=int(args.seed),
            iterations=int(args.iterations),
            warmup=int(args.warmup),
            max_dense_entries=int(args.max_dense_entries),
            include_expensive=bool(args.include_expensive),
            output_dir=str(args.output_dir or ""),
        ).normalized()
        report = run_suite(cfg, include=modes)
        payload = report.to_dict()
        if args.output_dir:
            payload["artifacts"] = write_report(report, Path(args.output_dir), pretty=bool(args.pretty))
        print(_canonical_json(payload, pretty=bool(args.pretty)))
        return 0 if report.ok else 2
    except Exception as exc:
        err = {
            "ok": False,
            "version": BENCHMARK_VERSION,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "environment": environment_report(),
            "dependency_status": dependency_status(),
        }
        print(_canonical_json(err, pretty=True), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
