#!/usr/bin/env python3
# tb_tensor_commitments.py
r"""
Tensorized Blockchain tensor commitment bridge
==============================================

Project Chimera / Akkurat - Tensorized Blockchain tensor artifact commitment
layer.

Purpose
-------
This module bridges ``tn.py`` TensorTrain/MPO artifacts into ledger-grade
commitments without duplicating the numerical runtime. The low-level tensor
library remains responsible for contraction, TT-SVD, rounding, and NPZ storage;
this module is responsible for canonical artifact hashing, schema conversion,
compression-report hashing, artifact integrity inspection, and links between
TensorTrain commitments, block hashes, and digital-twin state roots.

Design boundaries
-----------------
- No ledger append logic.
- No digital-twin state mutation.
- No tensor numerical implementation beyond byte normalization for hashing.
- Device-independent commitments: CuPy/GPU cores are converted to CPU NumPy
  arrays before hashing.
- SHA-256 domain-separated hashing is the consensus baseline.

Public API
----------
- TensorArtifactLink
- TensorArtifactIntegrityReport
- tensor_train_commitment(...)
- verify_tensor_commitment(...)
- hash_tensor_core(...)
- tensor_train_core_hashes(...)
- compression_report_from_rounding_report(...)
- compression_report_hash(...)
- round_tensor_train_with_commitment(...)
- link_tensor_to_ledger(...)
- inspect_npz_manifest(...)
- verify_tensor_artifact(...)
- load_tensor_artifact(...)
- status_report(...)
- run_self_test(...)
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import hmac
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:
    import numpy as np
    _NP_OK = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _NP_OK = False

try:
    from tb_canonical import canonical_json_safe, canonical_dumps, canonical_bytes  # type: ignore
    _CANONICAL_OK = True
except Exception:  # pragma: no cover
    _CANONICAL_OK = False

    def canonical_json_safe(obj: Any) -> Any:  # type: ignore[no-redef]
        if obj is None or isinstance(obj, (str, int, bool)):
            return obj
        if isinstance(obj, float):
            if not math.isfinite(obj):
                return str(obj)
            return 0.0 if obj == 0.0 else float(obj)
        if isinstance(obj, (bytes, bytearray)):
            return {"__bytes_hex__": bytes(obj).hex()}
        if is_dataclass(obj):
            return canonical_json_safe(asdict(obj))
        if _NP_OK and isinstance(obj, np.generic):  # type: ignore[union-attr]
            return canonical_json_safe(obj.item())
        if _NP_OK and isinstance(obj, np.ndarray):  # type: ignore[union-attr]
            arr = np.asarray(obj)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return arr.astype(float).tolist()
        if isinstance(obj, Mapping):
            return {str(k): canonical_json_safe(v) for k, v in sorted(dict(obj).items(), key=lambda kv: str(kv[0]))}
        if isinstance(obj, (list, tuple, set)):
            return [canonical_json_safe(v) for v in list(obj)]
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            try:
                return canonical_json_safe(obj.to_dict())
            except Exception:
                pass
        return str(obj)

    def canonical_dumps(obj: Any) -> str:  # type: ignore[no-redef]
        return json.dumps(canonical_json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def canonical_bytes(obj: Any) -> bytes:  # type: ignore[no-redef]
        return canonical_dumps(obj).encode("utf-8", errors="strict")

try:
    from tb_hashing import (  # type: ignore
        DEFAULT_HASH_ALGORITHM,
        domain_hash_hex as _tb_domain_hash_hex,
        hash_file as _tb_hash_file,
        hash_hex as _tb_hash_hex,
        constant_time_hex_equal as _constant_time_hex_equal,
        normalize_hex as _normalize_hex,
    )
    _HASHING_OK = True
except Exception:  # pragma: no cover
    DEFAULT_HASH_ALGORITHM = "sha256"  # type: ignore
    _HASHING_OK = False
    _tb_domain_hash_hex = None  # type: ignore
    _tb_hash_file = None  # type: ignore
    _tb_hash_hex = None  # type: ignore
    _constant_time_hex_equal = None  # type: ignore
    _normalize_hex = None  # type: ignore

try:
    from tb_schemas import TensorCommitment as SchemaTensorCommitment  # type: ignore
    from tb_schemas import CompressionReport as SchemaCompressionReport  # type: ignore
    from tb_schemas import ValidationIssue  # type: ignore
    _SCHEMAS_OK = True
except Exception:  # pragma: no cover
    _SCHEMAS_OK = False
    SchemaTensorCommitment = None  # type: ignore
    SchemaCompressionReport = None  # type: ignore
    ValidationIssue = None  # type: ignore

try:
    import tn as _tn  # type: ignore
    _TN_OK = True
except Exception as _tn_exc:  # pragma: no cover
    _tn = None  # type: ignore
    _TN_OK = False
    _TN_IMPORT_ERROR = repr(_tn_exc)
else:
    _TN_IMPORT_ERROR = ""


# =============================================================================
# Constants
# =============================================================================

TB_TENSOR_COMMITMENTS_VERSION = "tb-tensor-commitments-v1.0-production"
TENSOR_CORE_DOMAIN = "TB:TENSOR_CORE:v1"
TENSOR_METADATA_DOMAIN = "TB:TENSOR_METADATA:v1"
TENSOR_CONFIG_DOMAIN = "TB:TENSOR_CONFIG:v1"
TENSOR_COMMITMENT_DOMAIN = "TB:TENSOR_COMMITMENT_BRIDGE:v1"
COMPRESSION_REPORT_DOMAIN = "TB:COMPRESSION_REPORT_BRIDGE:v1"
TENSOR_LEDGER_LINK_DOMAIN = "TB:TENSOR_LEDGER_LINK:v1"
TENSOR_ARTIFACT_DOMAIN = "TB:TENSOR_ARTIFACT:v1"
DEFAULT_TENSOR_SCHEME = "tensor_train_mpo"
DEFAULT_TENSOR_COMMITMENT_VERSION = "1"
CONSENSUS_SAFE_DTYPES = {"float32", "float64"}
ZERO_HASH_256 = "0" * 64


# =============================================================================
# Errors and dataclasses
# =============================================================================

class TensorCommitmentError(ValueError):
    """Raised for invalid tensor commitment or artifact operations."""


@dataclass(frozen=True)
class TensorArtifactLink:
    """Ledger link binding tensor commitments to blocks and state roots."""

    tensor_commitment_hash: str
    block_hash: str = ""
    state_root: str = ""
    pre_state_root: str = ""
    post_state_root: str = ""
    compression_report_hash: str = ""
    artifact_hash: str = ""
    artifact_uri: str = ""
    role: str = "ledger_tensor"
    algorithm: str = DEFAULT_HASH_ALGORITHM
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "tb.tensor_artifact_link.v1"
    link_hash: str = ""

    def payload_without_hash(self) -> Dict[str, Any]:
        out = canonical_json_safe(asdict(self))
        out["link_hash"] = ""
        return out

    def compute_hash(self) -> str:
        return _domain_hash_hex(self.payload_without_hash(), domain=TENSOR_LEDGER_LINK_DOMAIN, tag="link", algorithm=self.algorithm)

    def normalized(self) -> "TensorArtifactLink":
        return TensorArtifactLink(
            tensor_commitment_hash=str(self.tensor_commitment_hash),
            block_hash=str(self.block_hash or ""),
            state_root=str(self.state_root or ""),
            pre_state_root=str(self.pre_state_root or ""),
            post_state_root=str(self.post_state_root or ""),
            compression_report_hash=str(self.compression_report_hash or ""),
            artifact_hash=str(self.artifact_hash or ""),
            artifact_uri=str(self.artifact_uri or ""),
            role=str(self.role or "ledger_tensor"),
            algorithm=str(self.algorithm or DEFAULT_HASH_ALGORITHM),
            metadata=dict(self.metadata or {}),
            schema_version=str(self.schema_version or "tb.tensor_artifact_link.v1"),
            link_hash=str(self.link_hash or self.compute_hash()),
        )

    def verify_self_hash(self) -> bool:
        return bool(self.link_hash) and constant_time_hex_equal(self.link_hash, self.compute_hash())

    def to_dict(self) -> Dict[str, Any]:
        return canonical_json_safe(asdict(self.normalized()))


@dataclass(frozen=True)
class TensorArtifactIntegrityReport:
    """Result of checking a saved tensor artifact such as a TT NPZ file."""

    ok: bool
    path: str
    artifact_hash: str = ""
    manifest_hash: str = ""
    tensor_commitment_hash: str = ""
    expected_commitment_hash: str = ""
    loaded: bool = False
    verified_npz_manifest: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "tb.tensor_artifact_integrity_report.v1"

    def to_dict(self) -> Dict[str, Any]:
        return canonical_json_safe(asdict(self))


# =============================================================================
# Hashing helpers
# =============================================================================

def _domain_hash_hex(payload: Any, *, domain: str, tag: str = "", algorithm: str = DEFAULT_HASH_ALGORITHM) -> str:
    if _HASHING_OK and _tb_domain_hash_hex is not None:
        try:
            return str(_tb_domain_hash_hex(payload, domain=domain, tag=tag, algorithm=algorithm))
        except TypeError:
            # Defensive fallback for older generated helper signatures.
            try:
                return str(_tb_domain_hash_hex(payload, domain=domain, algorithm=algorithm))
            except Exception:
                pass
        except Exception:
            pass
    framed = (
        b"TBCOMMIT-DOMAIN\x00"
        + str(algorithm or "sha256").encode("utf-8")
        + b"\x00"
        + str(domain or "").encode("utf-8")
        + b"\x00"
        + str(tag or "").encode("utf-8")
        + b"\x00"
        + canonical_bytes(payload)
    )
    return hashlib.sha256(framed).hexdigest()


def _hash_bytes_hex(data: bytes, *, algorithm: str = DEFAULT_HASH_ALGORITHM) -> str:
    if _HASHING_OK and _tb_hash_hex is not None:
        try:
            return str(_tb_hash_hex(data, algorithm=algorithm))
        except Exception:
            pass
    if str(algorithm).lower().strip() not in {"sha256", "sha-256"}:
        raise TensorCommitmentError("Fallback hashing supports SHA-256 only.")
    return hashlib.sha256(data).hexdigest()


def normalize_hex_or_empty(value: Any) -> str:
    if value is None or str(value) == "":
        return ""
    if _HASHING_OK and _normalize_hex is not None:
        return str(_normalize_hex(value))
    s = str(value).lower().strip()
    if s.startswith("0x"):
        s = s[2:]
    bytes.fromhex(s)
    return s


def constant_time_hex_equal(a: Any, b: Any) -> bool:
    if _HASHING_OK and _constant_time_hex_equal is not None:
        try:
            return bool(_constant_time_hex_equal(a, b))
        except Exception:
            return False
    try:
        return hmac.compare_digest(bytes.fromhex(str(a)), bytes.fromhex(str(b)))
    except Exception:
        return False


# =============================================================================
# TensorTrain normalization and commitment helpers
# =============================================================================

def _to_numpy_array(x: Any, *, dtype: Optional[Any] = None) -> "np.ndarray":
    if not _NP_OK:
        raise TensorCommitmentError("NumPy is required for tensor artifact commitments.")
    if hasattr(x, "__cuda_array_interface__"):
        try:
            import cupy as cp  # type: ignore
            arr = cp.asnumpy(x)
        except Exception:
            arr = np.asarray(x)  # type: ignore[union-attr]
    else:
        arr = np.asarray(x)  # type: ignore[union-attr]
    if dtype is not None:
        arr = arr.astype(np.dtype(dtype), copy=False)  # type: ignore[union-attr]
    if not np.all(np.isfinite(arr)):  # type: ignore[union-attr]
        raise TensorCommitmentError("Tensor core contains non-finite values.")
    # Canonicalize endian and contiguity without changing semantic dtype.
    if arr.dtype.byteorder == ">":
        arr = arr.byteswap().newbyteorder("<")
    return np.ascontiguousarray(arr)  # type: ignore[union-attr]


def _tensor_train_cores(tt: Any) -> List[Any]:
    cores = getattr(tt, "cores_data", None)
    if cores is None:
        cores = getattr(tt, "cores", None)
    if not isinstance(cores, (list, tuple)) or len(cores) == 0:
        raise TensorCommitmentError("TensorTrain-like object does not expose non-empty cores_data/cores.")
    return list(cores)


def _tensor_train_output_dims(tt: Any) -> List[int]:
    if hasattr(tt, "output_dims"):
        return [int(x) for x in list(getattr(tt, "output_dims"))]
    cores = _tensor_train_cores(tt)
    return [int(_to_numpy_array(c).shape[1]) for c in cores]


def _tensor_train_input_dims(tt: Any) -> List[int]:
    if hasattr(tt, "input_dims"):
        return [int(x) for x in list(getattr(tt, "input_dims"))]
    cores = _tensor_train_cores(tt)
    return [int(_to_numpy_array(c).shape[2]) for c in cores]


def _tensor_train_bond_ranks(tt: Any) -> List[int]:
    if hasattr(tt, "bond_ranks") and callable(getattr(tt, "bond_ranks")):
        try:
            return [int(x) for x in list(tt.bond_ranks())]
        except Exception:
            pass
    if hasattr(tt, "bond_dims"):
        return [int(x) for x in list(getattr(tt, "bond_dims"))]
    cores = _tensor_train_cores(tt)
    if len(cores) <= 1:
        return []
    return [int(_to_numpy_array(c).shape[3]) for c in cores[:-1]]


def _tensor_train_dtype(tt: Any, canonical_dtype: Optional[Any] = None) -> str:
    if canonical_dtype is not None and _NP_OK:
        return str(np.dtype(canonical_dtype))  # type: ignore[union-attr]
    dt = getattr(tt, "dtype", None)
    if dt is not None:
        try:
            return str(np.dtype(dt))  # type: ignore[union-attr]
        except Exception:
            return str(dt)
    cores = _tensor_train_cores(tt)
    return str(_to_numpy_array(cores[0]).dtype)


def _tensor_train_config_payload(tt: Any, *, canonical_dtype: Optional[Any] = None) -> Dict[str, Any]:
    cfg = getattr(tt, "config", None)
    if cfg is None:
        payload: Dict[str, Any] = {}
    elif hasattr(cfg, "to_public_dict") and callable(cfg.to_public_dict):
        payload = dict(cfg.to_public_dict())
    elif is_dataclass(cfg):
        payload = canonical_json_safe(asdict(cfg))
    elif hasattr(cfg, "__dict__"):
        payload = canonical_json_safe(vars(cfg))
    else:
        payload = {"repr": repr(cfg)}
    payload = dict(payload or {})
    payload["device"] = "canonical"
    payload["dtype"] = _tensor_train_dtype(tt, canonical_dtype=canonical_dtype)
    return payload


def _tensor_train_metadata_payload(
    tt: Any,
    *,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    block_hash: str = "",
    state_root: str = "",
    include_runtime_metadata: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    commit_meta = getattr(tt, "commit_metadata", None)
    if isinstance(commit_meta, Mapping):
        payload.update(dict(commit_meta))
    if block_hash:
        payload["block_hash"] = str(block_hash)
    if state_root:
        payload["state_root"] = str(state_root)
    if extra_metadata:
        payload.update(dict(extra_metadata))
    if include_runtime_metadata:
        runtime_meta = getattr(tt, "metadata", None)
        if isinstance(runtime_meta, Mapping):
            payload["runtime_metadata"] = dict(runtime_meta)
    return canonical_json_safe(payload)


def hash_tensor_core(
    core: Any,
    *,
    index: int = 0,
    canonical_dtype: Optional[Any] = None,
    algorithm: str = DEFAULT_HASH_ALGORITHM,
) -> str:
    """Return a domain-separated hash for one TT/MPO core."""
    arr = _to_numpy_array(core, dtype=canonical_dtype)
    raw = arr.tobytes(order="C")
    payload = {
        "schema": "tb.tensor_core.v1",
        "index": int(index),
        "shape": list(map(int, arr.shape)),
        "dtype": str(arr.dtype),
        "order": "C",
        "size_bytes": int(len(raw)),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return _domain_hash_hex(payload, domain=TENSOR_CORE_DOMAIN, tag="core", algorithm=algorithm)


def tensor_train_core_hashes(
    tt: Any,
    *,
    canonical_dtype: Optional[Any] = None,
    algorithm: str = DEFAULT_HASH_ALGORITHM,
) -> List[str]:
    return [hash_tensor_core(c, index=i, canonical_dtype=canonical_dtype, algorithm=algorithm) for i, c in enumerate(_tensor_train_cores(tt))]


def tensor_train_metadata_hash(
    tt: Any,
    *,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    block_hash: str = "",
    state_root: str = "",
    include_runtime_metadata: bool = False,
    algorithm: str = DEFAULT_HASH_ALGORITHM,
) -> str:
    payload = _tensor_train_metadata_payload(
        tt,
        extra_metadata=extra_metadata,
        block_hash=block_hash,
        state_root=state_root,
        include_runtime_metadata=include_runtime_metadata,
    )
    return _domain_hash_hex(payload, domain=TENSOR_METADATA_DOMAIN, tag="metadata", algorithm=algorithm)


def tensor_train_config_hash(
    tt: Any,
    *,
    canonical_dtype: Optional[Any] = None,
    algorithm: str = DEFAULT_HASH_ALGORITHM,
) -> str:
    return _domain_hash_hex(_tensor_train_config_payload(tt, canonical_dtype=canonical_dtype), domain=TENSOR_CONFIG_DOMAIN, tag="config", algorithm=algorithm)


def tensor_train_mode_layout(tt: Any) -> List[int]:
    """Return a compact interleaved mode layout: out0,in0,out1,in1,..."""
    out = _tensor_train_output_dims(tt)
    inn = _tensor_train_input_dims(tt)
    layout: List[int] = []
    for a, b in zip(out, inn):
        layout.extend([int(a), int(b)])
    return layout


def _is_consensus_safe_tensor(tt: Any, *, canonical_dtype: Optional[Any] = None) -> bool:
    device = str(getattr(tt, "device", getattr(getattr(tt, "config", None), "device", "cpu"))).lower().strip()
    dtype_s = _tensor_train_dtype(tt, canonical_dtype=canonical_dtype)
    if device != "cpu":
        return False
    if dtype_s not in CONSENSUS_SAFE_DTYPES:
        return False
    cfg = getattr(tt, "config", None)
    if cfg is not None and hasattr(cfg, "consensus_safe"):
        return bool(getattr(cfg, "consensus_safe"))
    # Absence of the flag is treated as numerically usable but not explicitly consensus-safe.
    return False


def tensor_train_commitment(
    tt: Any,
    *,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    block_hash: str = "",
    state_root: str = "",
    compression_parent_hash: str = "",
    include_runtime_metadata: bool = False,
    canonical_dtype: Optional[Any] = None,
    algorithm: str = DEFAULT_HASH_ALGORITHM,
    require_consensus_safe: bool = False,
) -> Any:
    """
    Convert a TensorTrain-like object into the shared ``tb_schemas.TensorCommitment``.

    The returned commitment is schema-native, not the private ``tn.py`` dataclass,
    so ledger and validation code share one canonical language.
    """
    if SchemaTensorCommitment is None:
        raise TensorCommitmentError("tb_schemas.TensorCommitment is required for schema commitments.")
    if require_consensus_safe and not _is_consensus_safe_tensor(tt, canonical_dtype=canonical_dtype):
        raise TensorCommitmentError("TensorTrain is not consensus-safe: require CPU, allowed dtype, and explicit consensus_safe config.")

    dtype_s = _tensor_train_dtype(tt, canonical_dtype=canonical_dtype)
    metadata_payload = _tensor_train_metadata_payload(
        tt,
        extra_metadata=extra_metadata,
        block_hash=block_hash,
        state_root=state_root,
        include_runtime_metadata=include_runtime_metadata,
    )
    config_hash = tensor_train_config_hash(tt, canonical_dtype=canonical_dtype, algorithm=algorithm)
    metadata_payload = dict(metadata_payload)
    metadata_payload.setdefault("tensor_config_hash", config_hash)
    if block_hash:
        metadata_payload.setdefault("linked_block_hash", str(block_hash))
    if state_root:
        metadata_payload.setdefault("linked_state_root", str(state_root))

    obj = SchemaTensorCommitment(
        scheme=DEFAULT_TENSOR_SCHEME,
        version=DEFAULT_TENSOR_COMMITMENT_VERSION,
        output_dims=_tensor_train_output_dims(tt),
        input_dims=_tensor_train_input_dims(tt),
        bond_ranks=_tensor_train_bond_ranks(tt),
        dtype=dtype_s,
        device_independent=True,
        core_hashes=tensor_train_core_hashes(tt, canonical_dtype=canonical_dtype, algorithm=algorithm),
        metadata_hash=_domain_hash_hex(metadata_payload, domain=TENSOR_METADATA_DOMAIN, tag="metadata", algorithm=algorithm),
        commitment_hash="",
        algorithm=algorithm,
        mode_layout=tensor_train_mode_layout(tt),
        compression_parent_hash=str(compression_parent_hash or ""),
        metadata=metadata_payload,
        schema_version="tb.tensor_commitment.v1",
    )
    return obj.normalized()


def coerce_tensor_commitment(commitment: Any) -> Any:
    if SchemaTensorCommitment is None:
        raise TensorCommitmentError("tb_schemas.TensorCommitment is unavailable.")
    if isinstance(commitment, SchemaTensorCommitment):
        return commitment.normalized()
    if hasattr(commitment, "to_dict") and callable(commitment.to_dict):
        return SchemaTensorCommitment.from_dict(commitment.to_dict()).normalized()
    if isinstance(commitment, Mapping):
        return SchemaTensorCommitment.from_dict(dict(commitment)).normalized()
    raise TypeError("Unsupported tensor commitment object.")


def verify_tensor_commitment(
    tt: Any,
    commitment: Any,
    *,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    block_hash: str = "",
    state_root: str = "",
    include_runtime_metadata: bool = False,
    canonical_dtype: Optional[Any] = None,
    require_consensus_safe: bool = False,
) -> bool:
    expected = coerce_tensor_commitment(commitment)
    actual = tensor_train_commitment(
        tt,
        extra_metadata=extra_metadata,
        block_hash=block_hash,
        state_root=state_root,
        compression_parent_hash=getattr(expected, "compression_parent_hash", ""),
        include_runtime_metadata=include_runtime_metadata,
        canonical_dtype=canonical_dtype if canonical_dtype is not None else getattr(expected, "dtype", None),
        algorithm=getattr(expected, "algorithm", DEFAULT_HASH_ALGORITHM),
        require_consensus_safe=require_consensus_safe,
    )
    return constant_time_hex_equal(actual.commitment_hash, expected.commitment_hash) and actual.to_dict() == expected.to_dict()


# =============================================================================
# Compression reports
# =============================================================================

def _as_mapping(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return dict(obj.to_dict())
        except Exception:
            pass
    if is_dataclass(obj):
        return dict(asdict(obj))
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {}


def compression_report_from_rounding_report(
    report: Any,
    *,
    pre_commitment: Any,
    post_commitment: Any,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Any:
    if SchemaCompressionReport is None:
        raise TensorCommitmentError("tb_schemas.CompressionReport is required.")
    if isinstance(report, SchemaCompressionReport):
        return report.normalized()

    p = _as_mapping(report)
    pre = coerce_tensor_commitment(pre_commitment)
    post = coerce_tensor_commitment(post_commitment)

    old_params = int(p.get("old_parameters", p.get("old_parameter_count", 0)) or 0)
    new_params = int(p.get("new_parameters", p.get("new_parameter_count", 0)) or 0)
    relative_error = p.get("relative_error", p.get("exact_relative_fro_error", p.get("estimated_relative_output_error", None)))
    if relative_error is not None:
        try:
            relative_error = float(relative_error)
        except Exception:
            relative_error = None
    fidelity = p.get("fidelity", None)
    if fidelity is None and relative_error is not None and math.isfinite(float(relative_error)):
        fidelity = max(0.0, 1.0 - float(relative_error))

    obj = SchemaCompressionReport(
        algorithm=str(p.get("algorithm", "tt_round")),
        old_parameters=old_params,
        new_parameters=new_params,
        compression_ratio=float(p.get("compression_ratio", 0.0) or 0.0),
        space_saved_percent=float(p.get("space_saved_percent", 0.0) or 0.0),
        old_bond_ranks=[int(x) for x in list(p.get("old_bond_ranks", []))],
        new_bond_ranks=[int(x) for x in list(p.get("new_bond_ranks", []))],
        max_rank=(None if p.get("max_rank", None) is None else int(p.get("max_rank"))),
        energy_tol=(None if p.get("energy_tol", None) is None else float(p.get("energy_tol"))),
        relative_error=relative_error,
        fidelity=(None if fidelity is None else float(fidelity)),
        operator_norm_before=(None if p.get("operator_norm_before", None) is None else float(p.get("operator_norm_before"))),
        operator_norm_after=(None if p.get("operator_norm_after", None) is None else float(p.get("operator_norm_after"))),
        pre_tensor_commitment_hash=pre.commitment_hash,
        post_tensor_commitment_hash=post.commitment_hash,
        deterministic=bool(p.get("deterministic", True)),
        consensus_safe=bool(p.get("consensus_safe", True)),
        report_hash="",
        metadata={**dict(p.get("metadata", {}) or {}), **dict(metadata or {})},
        schema_version="tb.compression_report.v1",
    )
    return obj.normalized()


def compression_report_hash(report: Any) -> str:
    if SchemaCompressionReport is not None and isinstance(report, SchemaCompressionReport):
        return report.normalized().report_hash
    if hasattr(report, "report_hash") and getattr(report, "report_hash"):
        return str(getattr(report, "report_hash"))
    payload = _as_mapping(report)
    return _domain_hash_hex(payload, domain=COMPRESSION_REPORT_DOMAIN, tag="report")


def round_tensor_train_with_commitment(
    tt: Any,
    *,
    max_rank: Optional[int] = None,
    energy_tol: Optional[float] = 0.999,
    min_rank: int = 1,
    canonical_dtype: Optional[Any] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[Any, Any]:
    """Round a TensorTrain and return ``(rounded_tt, SchemaCompressionReport)``."""
    pre = tensor_train_commitment(tt, canonical_dtype=canonical_dtype, extra_metadata=metadata)
    if hasattr(tt, "round_with_report") and callable(getattr(tt, "round_with_report")):
        rounded, native_report = tt.round_with_report(max_rank=max_rank, energy_tol=energy_tol, min_rank=min_rank)
    elif hasattr(tt, "round") and callable(getattr(tt, "round")):
        old_params = int(tt.parameter_count()) if hasattr(tt, "parameter_count") else 0
        old_ranks = _tensor_train_bond_ranks(tt)
        rounded = tt.round(max_rank=max_rank, energy_tol=energy_tol, min_rank=min_rank)
        new_params = int(rounded.parameter_count()) if hasattr(rounded, "parameter_count") else 0
        native_report = {
            "algorithm": "tt_round",
            "old_parameter_count": old_params,
            "new_parameter_count": new_params,
            "old_bond_ranks": old_ranks,
            "new_bond_ranks": _tensor_train_bond_ranks(rounded),
            "max_rank": max_rank,
            "energy_tol": energy_tol,
            "deterministic": True,
        }
    else:
        raise TensorCommitmentError("TensorTrain-like object does not expose round/round_with_report.")
    post = tensor_train_commitment(
        rounded,
        canonical_dtype=canonical_dtype,
        extra_metadata=metadata,
        compression_parent_hash=pre.commitment_hash,
    )
    return rounded, compression_report_from_rounding_report(native_report, pre_commitment=pre, post_commitment=post, metadata=metadata)


# =============================================================================
# Ledger links and NPZ artifacts
# =============================================================================

def link_tensor_to_ledger(
    tensor_commitment: Any,
    *,
    block_hash: str = "",
    state_root: str = "",
    pre_state_root: str = "",
    post_state_root: str = "",
    compression_report: Any = None,
    artifact_hash: str = "",
    artifact_uri: str = "",
    role: str = "ledger_tensor",
    metadata: Optional[Mapping[str, Any]] = None,
) -> TensorArtifactLink:
    c = coerce_tensor_commitment(tensor_commitment)
    report_hash = compression_report_hash(compression_report) if compression_report is not None else ""
    link = TensorArtifactLink(
        tensor_commitment_hash=c.commitment_hash,
        block_hash=str(block_hash or ""),
        state_root=str(state_root or ""),
        pre_state_root=str(pre_state_root or ""),
        post_state_root=str(post_state_root or ""),
        compression_report_hash=report_hash,
        artifact_hash=str(artifact_hash or ""),
        artifact_uri=str(artifact_uri or ""),
        role=str(role or "ledger_tensor"),
        metadata=dict(metadata or {}),
    )
    return link.normalized()


def inspect_npz_manifest(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    if not _NP_OK:
        raise TensorCommitmentError("NumPy is required to inspect NPZ artifacts.")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    out: Dict[str, Any] = {
        "path": str(p),
        "exists": True,
        "size_bytes": int(p.stat().st_size),
        "has_manifest": False,
        "manifest": {},
        "core_keys": [],
    }
    with np.load(p, allow_pickle=False) as data:  # type: ignore[union-attr]
        out["files"] = list(data.files)
        out["core_keys"] = sorted([k for k in data.files if k.startswith("core_") or "tt_core" in k])
        if "__manifest_json__" in data.files:
            manifest = json.loads(str(data["__manifest_json__"].item()))
            out["has_manifest"] = True
            out["manifest"] = canonical_json_safe(manifest)
            out["manifest_hash"] = str(manifest.get("manifest_hash", ""))
            tc = manifest.get("tensor_commitment", {}) if isinstance(manifest, Mapping) else {}
            if isinstance(tc, Mapping):
                out["native_tensor_commitment_hash"] = str(tc.get("commitment_hash", ""))
    return out


def _hash_artifact_file(path: Union[str, os.PathLike], *, algorithm: str = DEFAULT_HASH_ALGORITHM) -> str:
    if _HASHING_OK and _tb_hash_file is not None:
        try:
            return str(_tb_hash_file(path, algorithm=algorithm, domain=TENSOR_ARTIFACT_DOMAIN))
        except TypeError:
            return str(_tb_hash_file(path, algorithm=algorithm))
        except Exception:
            pass
    p = Path(path)
    h = hashlib.sha256()
    h.update(b"TB:TENSOR_ARTIFACT:v1\x00")
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tensor_artifact(
    path: Union[str, os.PathLike],
    *,
    device: str = "cpu",
    dtype: Optional[Any] = None,
    verify_npz_manifest: bool = True,
) -> Any:
    if not _TN_OK or _tn is None:
        raise TensorCommitmentError(f"tn.py is unavailable: {_TN_IMPORT_ERROR}")
    if not hasattr(_tn, "TensorTrain") or not hasattr(_tn.TensorTrain, "load_npz"):
        raise TensorCommitmentError("tn.TensorTrain.load_npz is unavailable.")
    return _tn.TensorTrain.load_npz(path, device=device, dtype=dtype, verify=bool(verify_npz_manifest))


def verify_tensor_artifact(
    path: Union[str, os.PathLike],
    *,
    expected_commitment: Any = None,
    device: str = "cpu",
    dtype: Optional[Any] = None,
    canonical_dtype: Optional[Any] = None,
    verify_npz_manifest: bool = True,
) -> TensorArtifactIntegrityReport:
    p = Path(path)
    errors: List[str] = []
    warnings: List[str] = []
    metadata: Dict[str, Any] = {}
    artifact_hash = ""
    manifest_hash = ""
    tensor_commitment_hash = ""
    expected_hash = ""
    loaded = False
    npz_ok = False

    try:
        artifact_hash = _hash_artifact_file(p)
    except Exception as exc:
        errors.append(f"artifact_hash_failed:{exc!r}")

    try:
        info = inspect_npz_manifest(p)
        metadata["npz"] = info
        manifest_hash = str(info.get("manifest_hash", ""))
    except Exception as exc:
        warnings.append(f"npz_manifest_inspection_failed:{exc!r}")

    try:
        tt = load_tensor_artifact(p, device=device, dtype=dtype, verify_npz_manifest=verify_npz_manifest)
        loaded = True
        npz_ok = bool(verify_npz_manifest)
        commit = tensor_train_commitment(tt, canonical_dtype=canonical_dtype)
        tensor_commitment_hash = commit.commitment_hash
        metadata["tensor_commitment"] = commit.to_dict()
        if expected_commitment is not None:
            expected = coerce_tensor_commitment(expected_commitment)
            expected_hash = expected.commitment_hash
            if not constant_time_hex_equal(tensor_commitment_hash, expected_hash):
                errors.append("expected_tensor_commitment_mismatch")
    except Exception as exc:
        errors.append(f"tensor_artifact_load_or_commitment_failed:{exc!r}")

    return TensorArtifactIntegrityReport(
        ok=(len(errors) == 0),
        path=str(p),
        artifact_hash=artifact_hash,
        manifest_hash=manifest_hash,
        tensor_commitment_hash=tensor_commitment_hash,
        expected_commitment_hash=expected_hash,
        loaded=loaded,
        verified_npz_manifest=npz_ok,
        errors=errors,
        warnings=warnings,
        metadata=metadata,
    )


# =============================================================================
# Status / self-test / CLI
# =============================================================================

def status_report() -> Dict[str, Any]:
    return {
        "ok": True,
        "version": TB_TENSOR_COMMITMENTS_VERSION,
        "numpy_available": bool(_NP_OK),
        "tn_available": bool(_TN_OK),
        "tn_import_error": _TN_IMPORT_ERROR,
        "tb_schemas_available": bool(_SCHEMAS_OK),
        "tb_hashing_available": bool(_HASHING_OK),
        "tb_canonical_available": bool(_CANONICAL_OK),
        "domains": {
            "tensor_core": TENSOR_CORE_DOMAIN,
            "tensor_commitment": TENSOR_COMMITMENT_DOMAIN,
            "compression_report": COMPRESSION_REPORT_DOMAIN,
            "ledger_link": TENSOR_LEDGER_LINK_DOMAIN,
        },
    }


def run_self_test() -> Dict[str, Any]:
    if not _NP_OK:
        raise AssertionError("NumPy is required for self-test.")
    if not _TN_OK or _tn is None:
        raise AssertionError(f"tn.py unavailable: {_TN_IMPORT_ERROR}")
    if SchemaTensorCommitment is None or SchemaCompressionReport is None:
        raise AssertionError("tb_schemas commitments are required for self-test.")

    cfg = _tn.TTConfig(dtype=np.float32, device="cpu", seed=123, deterministic=True, consensus_safe=True)  # type: ignore[union-attr]
    tt = _tn.TensorTrain([2, 2], [2, 2], bond_dims=[2], config=cfg, metadata={"runtime": "selftest"}, commit_metadata={"purpose": "selftest"})  # type: ignore[union-attr]
    c1 = tensor_train_commitment(tt, extra_metadata={"block_hash": ZERO_HASH_256}, require_consensus_safe=True)
    if not c1.verify_self_hash():
        raise AssertionError("Schema tensor commitment self-hash failed.")
    if not verify_tensor_commitment(tt, c1, extra_metadata={"block_hash": ZERO_HASH_256}, require_consensus_safe=True):
        raise AssertionError("Tensor commitment verification failed.")

    rounded, comp = round_tensor_train_with_commitment(tt, max_rank=1, energy_tol=0.999, metadata={"round": "selftest"})
    if not comp.report_hash or not comp.validate() == []:
        # Some generated schema versions may return warnings only. Treat error-like issue counts strictly.
        issues = comp.validate()
        if issues:
            raise AssertionError(f"Compression report validation issues: {issues}")
    c2 = tensor_train_commitment(rounded, extra_metadata={"round": "selftest"}, compression_parent_hash=c1.commitment_hash)
    link = link_tensor_to_ledger(c2, block_hash="a" * 64, state_root="b" * 64, compression_report=comp, artifact_hash="c" * 64)
    if not link.verify_self_hash():
        raise AssertionError("TensorArtifactLink self-hash failed.")

    tmp = Path(os.environ.get("TB_TENSOR_COMMITMENTS_SELFTEST_NPZ", "./_tb_tensor_commitments_selftest.npz"))
    try:
        rounded.save_npz(tmp, include_manifest=True)
        report = verify_tensor_artifact(tmp)
        if not report.ok:
            raise AssertionError(f"Artifact verification failed: {report.errors}")
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass

    return {
        "ok": True,
        "status": status_report(),
        "tensor_commitment_hash": c1.commitment_hash,
        "rounded_tensor_commitment_hash": c2.commitment_hash,
        "compression_report_hash": comp.report_hash,
        "ledger_link_hash": link.link_hash,
        "artifact_report_ok": True,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tensorized Blockchain tensor commitment bridge")
    p.add_argument("--mode", choices=["status", "self-test", "commit-npz", "verify-npz"], default="self-test")
    p.add_argument("--path", default="", help="NPZ path for commit-npz or verify-npz")
    p.add_argument("--pretty", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        if args.mode == "status":
            out = status_report()
        elif args.mode == "commit-npz":
            if not args.path:
                raise ValueError("--path is required for commit-npz")
            tt = load_tensor_artifact(args.path, verify_npz_manifest=True)
            commit = tensor_train_commitment(tt)
            out = {"ok": True, "path": args.path, "tensor_commitment": commit.to_dict()}
        elif args.mode == "verify-npz":
            if not args.path:
                raise ValueError("--path is required for verify-npz")
            out = verify_tensor_artifact(args.path).to_dict()
        else:
            out = run_self_test()
        print(json.dumps(canonical_json_safe(out), indent=2 if args.pretty else None, sort_keys=True, ensure_ascii=False))
        return 0 if bool(out.get("ok", False)) else 2
    except Exception as exc:
        err = {"ok": False, "error": repr(exc), "status": status_report()}
        print(json.dumps(canonical_json_safe(err), indent=2, sort_keys=True, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
