#!/usr/bin/env python3
# tb_observability.py
r"""
Project Chimera / Akkurat - Tensorized Blockchain Observability Layer
=====================================================================

Purpose
-------
Structured diagnostics, logging, health reporting, dependency status, warning
aggregation, runtime-mode reporting, and demo-friendly trace summaries for the
Tensorized Blockchain stack.

Design boundaries
-----------------
- Dependency-light: stdlib first; optional integration with tb_config,
  tb_canonical, tb_hashing, and tb_schemas when available.
- Import-safe: this module must not import heavyweight runtime modules such as
  tensorized_blockchain_kernel, digital_twin_kernel, tn, ncp, or cfc at import
  time.
- Deterministic-friendly: diagnostic payloads are canonicalizable and hashable.
- Production demo friendly: degraded-mode reports should be explicit, compact,
  and serializable.

Primary API
-----------
- Severity
- DiagnosticEvent
- DiagnosticCollector
- StructuredLogger
- RuntimeHealthReport
- DemoTrace
- dependency_status(...)
- runtime_mode_report(...)
- collect_stack_status(...)
- capture_runtime_warnings(...)
- make_health_report(...)

CLI examples
------------
    python tb_observability.py --mode self-test
    python tb_observability.py --mode status --pretty
    python tb_observability.py --mode deps --modules tn,digital_twin_kernel,tb_crypto
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import enum
import hashlib
import importlib
import inspect
import json
import logging
import os
import platform
import sys
import tempfile
import time
import traceback
import types
import uuid
import warnings
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union


# =============================================================================
# Version / constants
# =============================================================================

TB_OBSERVABILITY_VERSION = "tb-observability-v1.0-production"
DEFAULT_EVENT_SCHEMA = "tb.diagnostic_event.v1"
DEFAULT_HEALTH_SCHEMA = "tb.runtime_health_report.v1"
DEFAULT_TRACE_SCHEMA = "tb.demo_trace.v1"


# =============================================================================
# Optional shared stack integrations
# =============================================================================

_HAS_TB_CANONICAL = False
_HAS_TB_HASHING = False
_HAS_TB_CONFIG = False
_HAS_TB_SCHEMAS = False

try:  # pragma: no cover - integration depends on package layout
    import tb_canonical as _tb_canonical  # type: ignore
    _HAS_TB_CANONICAL = True
except Exception:  # pragma: no cover
    _tb_canonical = None  # type: ignore

try:  # pragma: no cover - integration depends on package layout
    import tb_hashing as _tb_hashing  # type: ignore
    _HAS_TB_HASHING = True
except Exception:  # pragma: no cover
    _tb_hashing = None  # type: ignore

try:  # pragma: no cover - integration depends on package layout
    import tb_config as _tb_config  # type: ignore
    _HAS_TB_CONFIG = True
except Exception:  # pragma: no cover
    _tb_config = None  # type: ignore

try:  # pragma: no cover - integration depends on package layout
    import tb_schemas as _tb_schemas  # type: ignore
    _HAS_TB_SCHEMAS = True
except Exception:  # pragma: no cover
    _tb_schemas = None  # type: ignore


# =============================================================================
# Canonical JSON / hashing fallback helpers
# =============================================================================

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_safe(obj: Any) -> Any:
    """JSON-safe conversion with optional delegation to tb_canonical."""
    if _HAS_TB_CANONICAL:
        for name in ("json_safe", "to_json_safe", "canonical_json_safe"):
            fn = getattr(_tb_canonical, name, None)
            if callable(fn):
                try:
                    return fn(obj)
                except Exception:
                    break

    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else str(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, np.generic):
            return _json_safe(obj.item())
        if isinstance(obj, np.ndarray):
            arr = np.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0)
            return arr.astype(float).tolist()
    except Exception:
        pass
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, BaseException):
        return {"type": obj.__class__.__name__, "message": str(obj)}
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return _json_safe(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "describe") and callable(getattr(obj, "describe")):
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


def canonical_dumps(obj: Any) -> str:
    if _HAS_TB_CANONICAL:
        for name in ("canonical_dumps", "dumps_canonical", "canonical_json_dumps"):
            fn = getattr(_tb_canonical, name, None)
            if callable(fn):
                try:
                    return str(fn(obj))
                except Exception:
                    break
    return json.dumps(_json_safe(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_bytes(obj: Any) -> bytes:
    if _HAS_TB_CANONICAL:
        for name in ("canonical_bytes", "to_canonical_bytes"):
            fn = getattr(_tb_canonical, name, None)
            if callable(fn):
                try:
                    out = fn(obj)
                    return out if isinstance(out, bytes) else bytes(out)
                except Exception:
                    break
    return canonical_dumps(obj).encode("utf-8")


def hash_payload(obj: Any, *, domain: str = "TB_OBSERVABILITY") -> str:
    if _HAS_TB_HASHING:
        for name in ("hash_payload", "sha256_hex", "domain_hash_hex", "hash_object"):
            fn = getattr(_tb_hashing, name, None)
            if callable(fn):
                try:
                    try:
                        return str(fn(obj, domain=domain))
                    except TypeError:
                        return str(fn(obj))
                except Exception:
                    break
    h = hashlib.sha256()
    h.update(str(domain).encode("utf-8"))
    h.update(b"\x00")
    h.update(canonical_bytes(obj))
    return h.hexdigest()


# =============================================================================
# Severity and statuses
# =============================================================================

class Severity(str, enum.Enum):
    DEBUG = "debug"
    INFO = "info"
    NOTICE = "notice"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @classmethod
    def coerce(cls, value: Any) -> "Severity":
        if isinstance(value, Severity):
            return value
        s = str(value or "info").lower().strip()
        aliases = {
            "warn": "warning",
            "err": "error",
            "fatal": "critical",
            "ok": "info",
        }
        s = aliases.get(s, s)
        try:
            return Severity(s)
        except Exception:
            return Severity.INFO

    @property
    def rank(self) -> int:
        return {
            Severity.DEBUG: 10,
            Severity.INFO: 20,
            Severity.NOTICE: 25,
            Severity.WARNING: 30,
            Severity.ERROR: 40,
            Severity.CRITICAL: 50,
        }[self]


class HealthStatus(str, enum.Enum):
    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"

    @classmethod
    def from_events(cls, events: Sequence["DiagnosticEvent"]) -> "HealthStatus":
        if any(e.severity == Severity.CRITICAL for e in events):
            return HealthStatus.FAILED
        if any(e.severity == Severity.ERROR and not bool(e.recoverable) for e in events):
            return HealthStatus.FAILED
        if any(e.severity.rank >= Severity.WARNING.rank for e in events):
            return HealthStatus.DEGRADED
        return HealthStatus.OK


# =============================================================================
# Diagnostic event model
# =============================================================================

@dataclass
class DiagnosticEvent:
    code: str
    severity: Union[Severity, str]
    component: str
    message: str
    recoverable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=_now_iso)
    schema: str = DEFAULT_EVENT_SCHEMA
    event_id: str = ""

    def __post_init__(self) -> None:
        self.code = str(self.code or "diagnostic_event")
        self.severity = Severity.coerce(self.severity)
        self.component = str(self.component or "unknown")
        self.message = str(self.message or "")
        self.recoverable = bool(self.recoverable)
        self.metadata = dict(self.metadata or {})
        self.ts = str(self.ts or _now_iso())
        self.schema = str(self.schema or DEFAULT_EVENT_SCHEMA)
        if not self.event_id:
            base = {
                "schema": self.schema,
                "ts": self.ts,
                "code": self.code,
                "severity": self.severity.value,
                "component": self.component,
                "message": self.message,
                "metadata": _json_safe(self.metadata),
            }
            self.event_id = hash_payload(base, domain="TB_DIAGNOSTIC_EVENT")[:24]

    @classmethod
    def info(cls, component: str, code: str, message: str, **metadata: Any) -> "DiagnosticEvent":
        return cls(code=code, severity=Severity.INFO, component=component, message=message, metadata=metadata)

    @classmethod
    def warning(cls, component: str, code: str, message: str, recoverable: bool = True, **metadata: Any) -> "DiagnosticEvent":
        return cls(code=code, severity=Severity.WARNING, component=component, message=message, recoverable=recoverable, metadata=metadata)

    @classmethod
    def error(cls, component: str, code: str, message: str, recoverable: bool = False, **metadata: Any) -> "DiagnosticEvent":
        return cls(code=code, severity=Severity.ERROR, component=component, message=message, recoverable=recoverable, metadata=metadata)

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        component: str,
        code: str = "exception",
        severity: Union[Severity, str] = Severity.ERROR,
        recoverable: bool = False,
        include_traceback: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "DiagnosticEvent":
        meta = dict(metadata or {})
        meta.update({
            "exception_type": exc.__class__.__name__,
            "exception_message": str(exc),
        })
        if include_traceback:
            meta["traceback"] = traceback.format_exc()
        return cls(
            code=code,
            severity=severity,
            component=component,
            message=f"{exc.__class__.__name__}: {exc}",
            recoverable=recoverable,
            metadata=meta,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "event_id": self.event_id,
            "ts": self.ts,
            "code": self.code,
            "severity": Severity.coerce(self.severity).value,
            "component": self.component,
            "message": self.message,
            "recoverable": bool(self.recoverable),
            "metadata": _json_safe(self.metadata),
        }

    def canonical_hash(self) -> str:
        return hash_payload(self.to_dict(), domain="TB_DIAGNOSTIC_EVENT_HASH")


# =============================================================================
# Diagnostic collector
# =============================================================================

@dataclass
class DiagnosticSummary:
    total: int
    by_severity: Dict[str, int]
    by_component: Dict[str, int]
    status: str
    highest_severity: str
    recoverable_errors: int
    nonrecoverable_errors: int

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


class DiagnosticCollector:
    """In-memory diagnostic event collector with report helpers."""

    def __init__(self, *, max_events: int = 1000, component: str = "runtime"):
        self.max_events = int(max(1, max_events))
        self.component = str(component or "runtime")
        self.events: List[DiagnosticEvent] = []

    def add(self, event: Union[DiagnosticEvent, Mapping[str, Any]], **override_metadata: Any) -> DiagnosticEvent:
        if isinstance(event, DiagnosticEvent):
            ev = event
            if override_metadata:
                ev.metadata.update(override_metadata)
        elif isinstance(event, Mapping):
            ev = DiagnosticEvent(
                code=str(event.get("code", "diagnostic_event")),
                severity=event.get("severity", Severity.INFO),
                component=str(event.get("component", self.component)),
                message=str(event.get("message", "")),
                recoverable=bool(event.get("recoverable", True)),
                metadata=dict(event.get("metadata", {})),
                ts=str(event.get("ts", _now_iso())),
                schema=str(event.get("schema", DEFAULT_EVENT_SCHEMA)),
                event_id=str(event.get("event_id", "")),
            )
            if override_metadata:
                ev.metadata.update(override_metadata)
        else:
            ev = DiagnosticEvent.info(self.component, "message", str(event), **override_metadata)

        self.events.append(ev)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]
        return ev

    def debug(self, code: str, message: str, *, component: Optional[str] = None, **metadata: Any) -> DiagnosticEvent:
        return self.add(DiagnosticEvent(code, Severity.DEBUG, component or self.component, message, True, metadata))

    def info(self, code: str, message: str, *, component: Optional[str] = None, **metadata: Any) -> DiagnosticEvent:
        return self.add(DiagnosticEvent(code, Severity.INFO, component or self.component, message, True, metadata))

    def notice(self, code: str, message: str, *, component: Optional[str] = None, **metadata: Any) -> DiagnosticEvent:
        return self.add(DiagnosticEvent(code, Severity.NOTICE, component or self.component, message, True, metadata))

    def warning(self, code: str, message: str, *, component: Optional[str] = None, recoverable: bool = True, **metadata: Any) -> DiagnosticEvent:
        return self.add(DiagnosticEvent(code, Severity.WARNING, component or self.component, message, recoverable, metadata))

    def error(self, code: str, message: str, *, component: Optional[str] = None, recoverable: bool = False, **metadata: Any) -> DiagnosticEvent:
        return self.add(DiagnosticEvent(code, Severity.ERROR, component or self.component, message, recoverable, metadata))

    def critical(self, code: str, message: str, *, component: Optional[str] = None, recoverable: bool = False, **metadata: Any) -> DiagnosticEvent:
        return self.add(DiagnosticEvent(code, Severity.CRITICAL, component or self.component, message, recoverable, metadata))

    def exception(self, exc: BaseException, *, component: Optional[str] = None, code: str = "exception", recoverable: bool = False, **metadata: Any) -> DiagnosticEvent:
        return self.add(DiagnosticEvent.from_exception(exc, component=component or self.component, code=code, recoverable=recoverable, metadata=metadata))

    def extend(self, events: Iterable[Union[DiagnosticEvent, Mapping[str, Any]]]) -> None:
        for ev in events:
            self.add(ev)

    def filter(
        self,
        *,
        min_severity: Union[Severity, str, None] = None,
        component: Optional[str] = None,
        code: Optional[str] = None,
    ) -> List[DiagnosticEvent]:
        out = list(self.events)
        if min_severity is not None:
            rank = Severity.coerce(min_severity).rank
            out = [e for e in out if Severity.coerce(e.severity).rank >= rank]
        if component is not None:
            out = [e for e in out if e.component == component]
        if code is not None:
            out = [e for e in out if e.code == code]
        return out

    def has_errors(self, *, include_recoverable: bool = True) -> bool:
        if include_recoverable:
            return any(e.severity.rank >= Severity.ERROR.rank for e in self.events)
        return any(e.severity.rank >= Severity.ERROR.rank and not e.recoverable for e in self.events)

    def status(self) -> HealthStatus:
        return HealthStatus.from_events(self.events)

    def summary(self) -> DiagnosticSummary:
        by_sev: Dict[str, int] = {}
        by_comp: Dict[str, int] = {}
        highest = Severity.DEBUG
        rec_errors = 0
        nonrec_errors = 0
        for ev in self.events:
            sev = Severity.coerce(ev.severity)
            by_sev[sev.value] = by_sev.get(sev.value, 0) + 1
            by_comp[ev.component] = by_comp.get(ev.component, 0) + 1
            if sev.rank > highest.rank:
                highest = sev
            if sev.rank >= Severity.ERROR.rank:
                if ev.recoverable:
                    rec_errors += 1
                else:
                    nonrec_errors += 1
        return DiagnosticSummary(
            total=len(self.events),
            by_severity=by_sev,
            by_component=by_comp,
            status=self.status().value,
            highest_severity=highest.value,
            recoverable_errors=rec_errors,
            nonrecoverable_errors=nonrec_errors,
        )

    def to_dict(self, *, include_events: bool = True) -> Dict[str, Any]:
        out = {"summary": self.summary().to_dict()}
        if include_events:
            out["events"] = [e.to_dict() for e in self.events]
        out["events_hash"] = hash_payload([e.to_dict() for e in self.events], domain="TB_DIAGNOSTIC_EVENTS")
        return out

    def raise_if_errors(self, *, include_recoverable: bool = False) -> None:
        bad = self.filter(min_severity=Severity.ERROR)
        if not include_recoverable:
            bad = [e for e in bad if not e.recoverable]
        if bad:
            first = bad[0]
            raise RuntimeError(f"Diagnostic errors present: {len(bad)}; first={first.code}:{first.message}")


# =============================================================================
# Structured logging
# =============================================================================

class JSONLogFormatter(logging.Formatter):
    """Minimal JSON log formatter for Python logging records."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - thin logging adapter
        payload = {
            "ts": _now_iso(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": int(record.lineno),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extra = getattr(record, "structured", None)
        if isinstance(extra, Mapping):
            payload["metadata"] = _json_safe(extra)
        return canonical_dumps(payload)


class StructuredLogger:
    """JSON-lines structured logger plus in-memory diagnostics."""

    def __init__(
        self,
        name: str = "tensorized_blockchain",
        *,
        component: Optional[str] = None,
        log_path: Optional[Union[str, os.PathLike[str]]] = None,
        echo: bool = False,
        max_events: int = 1000,
    ):
        self.name = str(name or "tensorized_blockchain")
        self.component = str(component or self.name)
        self.collector = DiagnosticCollector(component=self.component, max_events=max_events)
        self.log_path = Path(log_path).expanduser().resolve() if log_path else None
        self.echo = bool(echo)
        self._logger = logging.getLogger(self.name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        self._ensure_handlers()

    def _ensure_handlers(self) -> None:
        # Avoid adding duplicate handlers for same logger/path.
        if self.echo and not any(getattr(h, "_tb_echo", False) for h in self._logger.handlers):
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(JSONLogFormatter())
            setattr(h, "_tb_echo", True)
            self._logger.addHandler(h)
        if self.log_path is not None:
            target = str(self.log_path)
            if not any(getattr(h, "_tb_log_path", None) == target for h in self._logger.handlers):
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                h = logging.FileHandler(target, encoding="utf-8")
                h.setFormatter(JSONLogFormatter())
                setattr(h, "_tb_log_path", target)
                self._logger.addHandler(h)

    def event(
        self,
        code: str,
        message: str,
        *,
        severity: Union[Severity, str] = Severity.INFO,
        component: Optional[str] = None,
        recoverable: bool = True,
        **metadata: Any,
    ) -> DiagnosticEvent:
        ev = DiagnosticEvent(
            code=code,
            severity=severity,
            component=component or self.component,
            message=message,
            recoverable=recoverable,
            metadata=metadata,
        )
        self.collector.add(ev)
        level = {
            Severity.DEBUG: logging.DEBUG,
            Severity.INFO: logging.INFO,
            Severity.NOTICE: logging.INFO,
            Severity.WARNING: logging.WARNING,
            Severity.ERROR: logging.ERROR,
            Severity.CRITICAL: logging.CRITICAL,
        }[Severity.coerce(severity)]
        self._logger.log(level, message, extra={"structured": ev.to_dict()})
        return ev

    def debug(self, code: str, message: str, **metadata: Any) -> DiagnosticEvent:
        return self.event(code, message, severity=Severity.DEBUG, **metadata)

    def info(self, code: str, message: str, **metadata: Any) -> DiagnosticEvent:
        return self.event(code, message, severity=Severity.INFO, **metadata)

    def warning(self, code: str, message: str, *, recoverable: bool = True, **metadata: Any) -> DiagnosticEvent:
        return self.event(code, message, severity=Severity.WARNING, recoverable=recoverable, **metadata)

    def error(self, code: str, message: str, *, recoverable: bool = False, **metadata: Any) -> DiagnosticEvent:
        return self.event(code, message, severity=Severity.ERROR, recoverable=recoverable, **metadata)

    def exception(self, exc: BaseException, *, code: str = "exception", recoverable: bool = False, **metadata: Any) -> DiagnosticEvent:
        ev = DiagnosticEvent.from_exception(exc, component=self.component, code=code, recoverable=recoverable, metadata=metadata)
        self.collector.add(ev)
        self._logger.error(ev.message, exc_info=True, extra={"structured": ev.to_dict()})
        return ev

    def report(self, *, include_events: bool = True) -> Dict[str, Any]:
        return self.collector.to_dict(include_events=include_events)


# =============================================================================
# Runtime health reports
# =============================================================================

@dataclass
class DependencyReport:
    name: str
    available: bool
    module_file: Optional[str] = None
    version: Optional[str] = None
    error: Optional[str] = None
    attributes: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class RuntimeHealthReport:
    schema: str = DEFAULT_HEALTH_SCHEMA
    status: str = HealthStatus.UNKNOWN.value
    ts: str = field(default_factory=_now_iso)
    component: str = "runtime"
    version: str = TB_OBSERVABILITY_VERSION
    runtime_mode: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    report_hash: str = ""

    def finalize(self) -> "RuntimeHealthReport":
        payload = self.to_dict(include_hash=False)
        self.report_hash = hash_payload(payload, domain="TB_RUNTIME_HEALTH_REPORT")
        return self

    def to_dict(self, *, include_hash: bool = True) -> Dict[str, Any]:
        out = {
            "schema": self.schema,
            "status": self.status,
            "ts": self.ts,
            "component": self.component,
            "version": self.version,
            "runtime_mode": _json_safe(self.runtime_mode),
            "dependencies": _json_safe(self.dependencies),
            "diagnostics": _json_safe(self.diagnostics),
            "metrics": _json_safe(self.metrics),
        }
        if include_hash:
            out["report_hash"] = self.report_hash or hash_payload(out, domain="TB_RUNTIME_HEALTH_REPORT")
        return out


# =============================================================================
# Dependency status and runtime mode
# =============================================================================

def _module_version(mod: types.ModuleType) -> Optional[str]:
    for attr in ("__version__", "VERSION", "version"):
        v = getattr(mod, attr, None)
        if v is not None:
            return str(v)
    return None


def dependency_status(
    modules: Sequence[str],
    *,
    required: Optional[Sequence[str]] = None,
    attrs: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, DependencyReport]:
    """Import-light dependency availability report."""
    required_set = {str(x) for x in (required or [])}
    attr_map = {str(k): list(v) for k, v in dict(attrs or {}).items()}
    out: Dict[str, DependencyReport] = {}
    for name in modules:
        name = str(name).strip()
        if not name:
            continue
        try:
            mod = importlib.import_module(name)
            attr_status: Dict[str, bool] = {}
            for a in attr_map.get(name, []):
                attr_status[str(a)] = hasattr(mod, str(a))
            out[name] = DependencyReport(
                name=name,
                available=True,
                module_file=getattr(mod, "__file__", None),
                version=_module_version(mod),
                error=None,
                attributes=attr_status,
            )
        except Exception as exc:
            out[name] = DependencyReport(
                name=name,
                available=False,
                module_file=None,
                version=None,
                error=repr(exc),
                attributes={a: False for a in attr_map.get(name, [])},
            )
            # Mark required status in attributes for downstream summaries.
            if name in required_set:
                out[name].attributes["required"] = True
    return out


def runtime_mode_report(*, include_env: bool = False) -> Dict[str, Any]:
    mode = os.getenv("TB_RUNTIME_MODE") or os.getenv("TENSORIZED_BLOCKCHAIN_MODE") or "unknown"
    consensus_safe = str(os.getenv("TB_CONSENSUS_SAFE", "")).lower() in {"1", "true", "yes", "on"}
    deterministic = str(os.getenv("TB_DETERMINISTIC", "")).lower() in {"1", "true", "yes", "on"}
    device_policy = os.getenv("TB_DEVICE_POLICY", "cpu-first")
    report = {
        "mode": mode,
        "consensus_safe": bool(consensus_safe),
        "deterministic": bool(deterministic),
        "device_policy": device_policy,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "process_id": os.getpid(),
        "cwd": str(Path.cwd()),
        "time_utc": _now_iso(),
        "observability_version": TB_OBSERVABILITY_VERSION,
    }
    if _HAS_TB_CONFIG:
        for name in ("status_report", "runtime_mode_report", "get_runtime_config"):
            fn = getattr(_tb_config, name, None)
            if callable(fn):
                try:
                    report["tb_config"] = _json_safe(fn())
                    break
                except Exception as exc:
                    report["tb_config_error"] = repr(exc)
                    break
    if include_env:
        allow_prefixes = ("TB_", "TENSORIZED_", "AKKURAT_", "LK20_")
        report["environment"] = {k: v for k, v in sorted(os.environ.items()) if k.startswith(allow_prefixes)}
    return report


def collect_stack_status(
    *,
    modules: Optional[Sequence[str]] = None,
    include_env: bool = False,
) -> Dict[str, Any]:
    default_modules = [
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
        "tb_api",
        "tb_cli",
    ]
    mods = list(modules or default_modules)
    attrs = {
        "tn": ["TensorTrain", "TTConfig"],
        "digital_twin_kernel": ["TreeTensorNetwork", "AkkuratInterface", "StateCommitment"],
        "twin_anything": ["TwinAnythingFactory", "TwinCreationReceipt"],
        "tensorized_blockchain_kernel": ["LedgerMPS", "TensorTransaction"],
        "ncp": ["TensorizedNCP"],
        "cfc": ["TCfC_Policy", "TensorizedCfCPolicy"],
    }
    deps = dependency_status(mods, attrs=attrs)
    collector = DiagnosticCollector(component="stack_status")
    for name, dep in deps.items():
        if not dep.available:
            collector.warning("dependency_unavailable", f"Module unavailable: {name}", component="dependency", module=name, error=dep.error)
        elif dep.attributes and not all(dep.attributes.values()):
            missing = [k for k, ok in dep.attributes.items() if not ok]
            collector.warning("dependency_missing_attributes", f"Module {name} missing expected attributes", component="dependency", module=name, missing=missing)
    status = collector.status().value
    return {
        "schema": "tb.stack_status.v1",
        "status": status,
        "ts": _now_iso(),
        "runtime_mode": runtime_mode_report(include_env=include_env),
        "dependencies": {k: v.to_dict() for k, v in deps.items()},
        "diagnostics": collector.to_dict(),
        "status_hash": hash_payload({"deps": {k: v.to_dict() for k, v in deps.items()}}, domain="TB_STACK_STATUS"),
    }


# =============================================================================
# Warning capture
# =============================================================================

@contextlib.contextmanager
def capture_runtime_warnings(
    collector: Optional[DiagnosticCollector] = None,
    *,
    component: str = "warnings",
) -> Iterator[List[warnings.WarningMessage]]:
    """Capture Python warnings and mirror them into a diagnostic collector."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        yield captured
        if collector is not None:
            for w in captured:
                collector.warning(
                    "python_warning",
                    str(w.message),
                    component=component,
                    category=getattr(w.category, "__name__", str(w.category)),
                    filename=str(w.filename),
                    lineno=int(w.lineno),
                )


# =============================================================================
# Demo trace summaries
# =============================================================================

@dataclass
class TraceStep:
    name: str
    status: str = HealthStatus.OK.value
    started_at: str = field(default_factory=_now_iso)
    finished_at: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    step_hash: str = ""

    def finish(self, *, status: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None) -> "TraceStep":
        self.finished_at = _now_iso()
        if status is not None:
            self.status = str(status)
        if metadata:
            self.metadata.update(dict(metadata))
        try:
            t0 = time.strptime(self.started_at[:19], "%Y-%m-%dT%H:%M:%S")
            t1 = time.strptime(self.finished_at[:19], "%Y-%m-%dT%H:%M:%S") if self.finished_at else t0
            self.duration_ms = max(0.0, (time.mktime(t1) - time.mktime(t0)) * 1000.0)
        except Exception:
            self.duration_ms = None
        self.step_hash = hash_payload(self.to_dict(include_hash=False), domain="TB_TRACE_STEP")
        return self

    def to_dict(self, *, include_hash: bool = True) -> Dict[str, Any]:
        out = {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "metadata": _json_safe(self.metadata),
            "diagnostics": _json_safe(self.diagnostics),
        }
        if include_hash:
            out["step_hash"] = self.step_hash or hash_payload(out, domain="TB_TRACE_STEP")
        return out


class DemoTrace:
    """Demo-friendly sequential trace with compact final summary."""

    def __init__(self, trace_id: Optional[str] = None, *, scenario: str = "demo"):
        self.trace_id = trace_id or uuid.uuid4().hex
        self.scenario = str(scenario)
        self.started_at = _now_iso()
        self.finished_at: Optional[str] = None
        self.steps: List[TraceStep] = []
        self.collector = DiagnosticCollector(component="demo_trace")

    @contextlib.contextmanager
    def step(self, name: str, **metadata: Any) -> Iterator[TraceStep]:
        st = TraceStep(name=name, metadata=dict(metadata))
        self.steps.append(st)
        try:
            yield st
            st.finish(status=HealthStatus.OK.value)
        except Exception as exc:
            ev = self.collector.exception(exc, component="demo_trace", code="trace_step_exception", step=name)
            st.diagnostics.append(ev.to_dict())
            st.finish(status=HealthStatus.FAILED.value)
            raise

    def add_step(self, name: str, *, status: str = HealthStatus.OK.value, metadata: Optional[Mapping[str, Any]] = None, diagnostics: Optional[Sequence[Union[DiagnosticEvent, Mapping[str, Any]]]] = None) -> TraceStep:
        st = TraceStep(name=name, status=status, metadata=dict(metadata or {}))
        if diagnostics:
            st.diagnostics = [d.to_dict() if isinstance(d, DiagnosticEvent) else _json_safe(d) for d in diagnostics]
        st.finish(status=status)
        self.steps.append(st)
        return st

    def finish(self) -> Dict[str, Any]:
        self.finished_at = _now_iso()
        return self.to_dict()

    def status(self) -> str:
        if any(s.status == HealthStatus.FAILED.value for s in self.steps):
            return HealthStatus.FAILED.value
        if any(s.status == HealthStatus.DEGRADED.value for s in self.steps) or self.collector.status() == HealthStatus.DEGRADED:
            return HealthStatus.DEGRADED.value
        return HealthStatus.OK.value

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "schema": DEFAULT_TRACE_SCHEMA,
            "trace_id": self.trace_id,
            "scenario": self.scenario,
            "status": self.status(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "steps": [s.to_dict() for s in self.steps],
            "diagnostics": self.collector.to_dict(),
        }
        payload["trace_hash"] = hash_payload(payload, domain="TB_DEMO_TRACE")
        return payload

    def summary(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "scenario": self.scenario,
            "status": self.status(),
            "step_count": len(self.steps),
            "failed_steps": [s.name for s in self.steps if s.status == HealthStatus.FAILED.value],
            "degraded_steps": [s.name for s in self.steps if s.status == HealthStatus.DEGRADED.value],
            "trace_hash": self.to_dict().get("trace_hash"),
        }


# =============================================================================
# Health report factory
# =============================================================================

def make_health_report(
    *,
    component: str = "runtime",
    collector: Optional[DiagnosticCollector] = None,
    dependencies: Optional[Mapping[str, DependencyReport]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    include_env: bool = False,
) -> RuntimeHealthReport:
    col = collector or DiagnosticCollector(component=component)
    deps = {k: v.to_dict() if isinstance(v, DependencyReport) else _json_safe(v) for k, v in dict(dependencies or {}).items()}
    dep_events = []
    for name, dep in deps.items():
        if not bool(dep.get("available", False)):
            dep_events.append(DiagnosticEvent.warning("dependency", "dependency_unavailable", f"Module unavailable: {name}", module=name, error=dep.get("error")))
    tmp_col = DiagnosticCollector(component=component)
    tmp_col.extend(col.events)
    tmp_col.extend(dep_events)
    report = RuntimeHealthReport(
        status=tmp_col.status().value,
        component=component,
        runtime_mode=runtime_mode_report(include_env=include_env),
        dependencies=deps,
        diagnostics=tmp_col.to_dict(),
        metrics=_json_safe(dict(metrics or {})),
    )
    return report.finalize()


def safe_write_json(path: Union[str, os.PathLike[str]], payload: Any) -> None:
    """Atomic JSON write helper for observability reports."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".tmp-{uuid.uuid4().hex}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


# =============================================================================
# Self-test
# =============================================================================

def run_self_test() -> Dict[str, Any]:
    collector = DiagnosticCollector(component="self_test")
    collector.info("self_test_started", "Observability self-test started")

    # Diagnostic event stability
    ev = DiagnosticEvent.info("self_test", "event_ok", "event constructed", value=1)
    assert ev.to_dict()["severity"] == "info"
    assert len(ev.canonical_hash()) == 64
    collector.add(ev)

    # Warning capture
    with capture_runtime_warnings(collector, component="self_test_warning"):
        warnings.warn("synthetic warning", RuntimeWarning)
    assert any(e.code == "python_warning" for e in collector.events)

    # Structured logger file path
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "events.jsonl"
        slog = StructuredLogger("tb_observability_self_test", component="self_test_logger", log_path=log_path, echo=False)
        slog.info("logger_ok", "logger emitted", answer=42)
        slog.warning("logger_warning", "warning emitted", recoverable=True)
        assert log_path.exists() and log_path.read_text(encoding="utf-8").strip()

        report_path = Path(td) / "health.json"
        deps = dependency_status(["json", "definitely_missing_tb_module_zzz"])
        report = make_health_report(component="self_test", collector=collector, dependencies=deps, metrics={"x": 1})
        safe_write_json(report_path, report.to_dict())
        loaded = json.loads(report_path.read_text(encoding="utf-8"))
        assert "report_hash" in loaded

    # Demo trace
    trace = DemoTrace("selftest-trace", scenario="observability")
    with trace.step("one"):
        pass
    trace.add_step("two", status=HealthStatus.DEGRADED.value, metadata={"reason": "synthetic"})
    summary = trace.summary()
    assert summary["status"] == HealthStatus.DEGRADED.value

    collector.info("self_test_finished", "Observability self-test finished")
    deps = dependency_status(["json", "logging"])
    final_report = make_health_report(component="tb_observability", collector=collector, dependencies=deps)
    return {
        "ok": True,
        "version": TB_OBSERVABILITY_VERSION,
        "collector": collector.to_dict(include_events=True),
        "trace": trace.to_dict(),
        "health": final_report.to_dict(),
    }


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tensorized Blockchain observability utilities")
    p.add_argument("--mode", choices=["self-test", "status", "deps"], default="status")
    p.add_argument("--modules", default="", help="Comma-separated module names for --mode deps")
    p.add_argument("--output", default="", help="Optional JSON output path")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    p.add_argument("--include-env", action="store_true", help="Include safe TB_/AKKURAT_/LK20_ environment variables")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        if args.mode == "self-test":
            payload = run_self_test()
        elif args.mode == "deps":
            mods = [m.strip() for m in args.modules.split(",") if m.strip()] or ["tb_config", "tb_canonical", "tb_hashing", "tb_crypto", "tb_schemas"]
            deps = dependency_status(mods)
            payload = {
                "ok": all(d.available for d in deps.values()),
                "ts": _now_iso(),
                "dependencies": {k: v.to_dict() for k, v in deps.items()},
            }
        else:
            payload = collect_stack_status(include_env=bool(args.include_env))
            payload["ok"] = payload.get("status") in {HealthStatus.OK.value, HealthStatus.DEGRADED.value}

        if args.output:
            safe_write_json(args.output, payload)
            payload["saved_to"] = str(Path(args.output).expanduser().resolve())

        print(json.dumps(_json_safe(payload), indent=2 if args.pretty else None, ensure_ascii=False, sort_keys=True))
        return 0 if bool(payload.get("ok", True)) else 2
    except Exception as exc:
        err = {
            "ok": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "version": TB_OBSERVABILITY_VERSION,
        }
        print(json.dumps(_json_safe(err), indent=2, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
