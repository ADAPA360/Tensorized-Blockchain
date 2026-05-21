#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# tb_open_source_audit.py
r"""
Tensorized Blockchain Open-Source Release Audit
===============================================

Purpose
-------
Release-readiness audit script for the Tensorized Blockchain stack.

Responsibilities
----------------
- Check license headers.
- Check for absolute local path leakage.
- Check for likely secrets in repository files.
- Check dependency versions and import health.
- Optionally run tests.
- Optionally verify reproducible demo output.
- Emit machine-readable JSON and human-readable Markdown reports.

Design boundaries
-----------------
This script is deliberately standalone and dependency-light. It does not import
the Tensorized Blockchain runtime modules by default, so it can be run before the
stack is fully installable.

Typical usage
-------------
    python tb_open_source_audit.py --root . --output audit_report.json --markdown audit_report.md

Strict release gate:
    python tb_open_source_audit.py --root . --strict --run-tests --test-command "python -m pytest -q"

Reproducible demo gate:
    python tb_open_source_audit.py --root . \
      --demo-command "python tb_cli.py run-demo --deterministic --json" \
      --demo-runs 2

Exit codes
----------
0: release-ready according to selected gates
1: audit completed with blocking issues
2: internal audit error
"""

from __future__ import annotations

import argparse
import ast
import base64
import dataclasses
import fnmatch
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple


TB_OPEN_SOURCE_AUDIT_VERSION = "tb-open-source-audit-v1.0-production"


# =============================================================================
# Small canonical helpers
# =============================================================================

def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_safe(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("ascii")
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _canonical_json(obj: Any) -> str:
    return json.dumps(_json_safe(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_hex_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_hex_text(text: str) -> str:
    return sha256_hex_bytes(text.encode("utf-8", errors="replace"))


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n")


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class AuditIssue:
    code: str
    severity: str  # "error" | "warning" | "info"
    path: str
    line: int = 0
    message: str = ""
    evidence: str = ""
    recommendation: str = ""

    def is_blocking(self, *, strict: bool = False) -> bool:
        if self.severity == "error":
            return True
        return bool(strict and self.severity == "warning")


@dataclass
class CommandResult:
    command: str
    ok: bool
    returncode: int
    duration_s: float
    stdout_sha256: str
    stderr_sha256: str
    combined_sha256: str
    stdout_tail: str = ""
    stderr_tail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class ReproducibilityResult:
    command: str
    runs: int
    ok: bool
    output_hashes: List[str]
    command_results: List[CommandResult] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class DependencyReport:
    required_files_present: Dict[str, bool] = field(default_factory=dict)
    python_version: str = ""
    platform: str = ""
    import_versions: Dict[str, str] = field(default_factory=dict)
    import_errors: Dict[str, str] = field(default_factory=dict)
    discovered_imports: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class AuditConfig:
    root: Path
    strict: bool = False
    max_file_bytes: int = 2_000_000
    scan_binary: bool = False
    run_tests: bool = False
    test_command: str = ""
    demo_command: str = ""
    demo_runs: int = 2
    command_timeout_s: int = 120
    allow_absolute_path_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    required_files: List[str] = field(default_factory=lambda: [
        "tn.py",
        "digital_twin_kernel.py",
        "twin_anything.py",
        "tensorized_blockchain_kernel.py",
        "ncp.py",
        "cfc.py",
        "tb_config.py",
        "tb_canonical.py",
        "tb_hashing.py",
        "tb_crypto.py",
        "tb_schemas.py",
        "tb_validation.py",
        "tb_storage.py",
        "tb_mempool.py",
        "tb_consensus.py",
        "tb_execution.py",
        "cognitive_lobe_runtime.py",
        "lobe_scaler.py",
        "tb_tensor_commitments.py",
        "tb_demo_scenarios.py",
        "tb_api.py",
        "tb_cli.py",
        "tb_observability.py",
        "tb_security.py",
        "tb_benchmarks.py",
        "tb_replay.py",
        "tb_open_source_audit.py",
    ])

    def normalized(self) -> "AuditConfig":
        root = Path(self.root).expanduser().resolve()
        cfg = dataclasses.replace(self, root=root)
        default_excludes = [
            ".git/**",
            ".hg/**",
            ".svn/**",
            "__pycache__/**",
            "*.pyc",
            "*.pyo",
            ".pytest_cache/**",
            ".mypy_cache/**",
            ".ruff_cache/**",
            ".venv/**",
            "venv/**",
            "env/**",
            "node_modules/**",
            "dist/**",
            "build/**",
            "*.egg-info/**",
            ".DS_Store",
            "*.npz",
            "*.npy",
            "*.parquet",
            "*.sqlite",
            "*.sqlite3",
            "*.db",
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.gif",
            "*.webp",
            "*.pdf",
            "*.zip",
            "*.tar",
            "*.gz",
            "*.7z",
        ]
        cfg.exclude_patterns = list(dict.fromkeys(default_excludes + list(cfg.exclude_patterns or [])))
        cfg.allow_absolute_path_patterns = list(cfg.allow_absolute_path_patterns or [])
        cfg.demo_runs = int(max(1, cfg.demo_runs))
        cfg.max_file_bytes = int(max(1_024, cfg.max_file_bytes))
        cfg.command_timeout_s = int(max(1, cfg.command_timeout_s))
        return cfg


@dataclass
class AuditReport:
    version: str
    root: str
    generated_at: str
    release_ready: bool
    strict: bool
    counts: Dict[str, int]
    issues: List[AuditIssue]
    dependency_report: DependencyReport
    test_result: Optional[CommandResult] = None
    demo_reproducibility: Optional[ReproducibilityResult] = None
    scanned_file_count: int = 0
    scanned_bytes: int = 0
    report_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "version": self.version,
            "root": self.root,
            "generated_at": self.generated_at,
            "release_ready": bool(self.release_ready),
            "strict": bool(self.strict),
            "counts": dict(self.counts),
            "issues": [_json_safe(i) for i in self.issues],
            "dependency_report": self.dependency_report.to_dict(),
            "test_result": None if self.test_result is None else self.test_result.to_dict(),
            "demo_reproducibility": None if self.demo_reproducibility is None else self.demo_reproducibility.to_dict(),
            "scanned_file_count": int(self.scanned_file_count),
            "scanned_bytes": int(self.scanned_bytes),
            "report_hash": self.report_hash,
        }
        return out


# =============================================================================
# File scanning
# =============================================================================

TEXT_SUFFIXES = {
    ".py", ".pyi", ".md", ".rst", ".txt", ".json", ".jsonl", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".env", ".example", ".sh", ".bat", ".ps1",
    ".dockerfile", ".gitignore", ".gitattributes", ".csv", ".tsv", ".html",
    ".css", ".js", ".ts", ".tsx", ".jsx", ".vue", ".sql",
}

PROJECT_IMPORT_CANDIDATES = {
    "tn",
    "digital_twin_kernel",
    "twin_anything",
    "tensorized_blockchain_kernel",
    "ncp",
    "cfc",
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
    "cognitive_lobe_runtime",
    "lobe_scaler",
    "tb_tensor_commitments",
    "tb_demo_scenarios",
    "tb_api",
    "tb_cli",
    "tb_observability",
    "tb_security",
    "tb_benchmarks",
    "tb_replay",
    "tb_open_source_audit",
}

COMMON_STDLIB = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else {
    "argparse", "ast", "base64", "collections", "copy", "csv", "dataclasses",
    "datetime", "enum", "functools", "hashlib", "hmac", "importlib", "inspect",
    "io", "itertools", "json", "logging", "math", "os", "pathlib", "platform",
    "random", "re", "secrets", "shlex", "shutil", "statistics", "subprocess",
    "sys", "tempfile", "time", "traceback", "typing", "uuid", "warnings",
}


def _is_excluded(rel: str, patterns: Sequence[str]) -> bool:
    rel_norm = rel.replace(os.sep, "/")
    for pat in patterns:
        p = str(pat).replace(os.sep, "/")
        if fnmatch.fnmatch(rel_norm, p) or fnmatch.fnmatch(Path(rel_norm).name, p):
            return True
        if p.endswith("/**") and rel_norm.startswith(p[:-3]):
            return True
    return False


def iter_candidate_files(cfg: AuditConfig) -> Iterator[Path]:
    root = cfg.root
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root)).replace(os.sep, "/")
        if _is_excluded(rel, cfg.exclude_patterns):
            continue
        if p.stat().st_size > cfg.max_file_bytes:
            yield p
            continue
        if cfg.scan_binary:
            yield p
            continue
        suffix = p.suffix.lower()
        if suffix in TEXT_SUFFIXES or p.name in {"Dockerfile", "LICENSE", "NOTICE", "README", "Makefile"}:
            yield p


def _read_text_for_scan(path: Path, *, max_file_bytes: int) -> Tuple[Optional[str], Optional[str]]:
    try:
        size = path.stat().st_size
        if size > max_file_bytes:
            return None, f"file_too_large:{size}"
        data = path.read_bytes()
        if b"\x00" in data[:4096]:
            return None, "binary_file"
        return data.decode("utf-8", errors="replace"), None
    except Exception as exc:
        return None, f"read_error:{exc!r}"


def _line_number_at(text: str, idx: int) -> int:
    return text.count("\n", 0, max(0, idx)) + 1


def _line_at(text: str, line: int) -> str:
    if line <= 0:
        return ""
    lines = text.splitlines()
    if line > len(lines):
        return ""
    return lines[line - 1]


def _mask_secret(s: str) -> str:
    s = str(s)
    if len(s) <= 8:
        return "*" * len(s)
    return s[:4] + "*" * max(4, min(32, len(s) - 8)) + s[-4:]


def _evidence_snippet(line_text: str, match_text: str = "") -> str:
    line = re.sub(r"\s+", " ", line_text.strip())
    if match_text:
        line = line.replace(match_text, _mask_secret(match_text))
    return line[:240]


# =============================================================================
# Checks
# =============================================================================

LICENSE_PATTERNS = [
    re.compile(r"SPDX-License-Identifier\s*:\s*[A-Za-z0-9.\-+]+", re.IGNORECASE),
    re.compile(r"\bMIT License\b", re.IGNORECASE),
    re.compile(r"\bApache License\b", re.IGNORECASE),
    re.compile(r"\bBSD[- ]\d?-?Clause\b", re.IGNORECASE),
    re.compile(r"\bLicense:\s*permissive\b", re.IGNORECASE),
    re.compile(r"\bLicensed under\b", re.IGNORECASE),
]

ABSOLUTE_PATH_PATTERNS = [
    # Match both literal Windows paths (C:\\Users\\...) and source-escaped
    # examples (C:\\\\Users\\\\...). The repeated backslash group is deliberate.
    re.compile(r"\b[A-Za-z]:(?:\\\\|\\)+(?:Users|Documents and Settings)(?:\\\\|\\)+[^\"'\s<>]+", re.IGNORECASE),
    re.compile(r"\b[A-Za-z]:(?:\\\\|\\)+[^\"'\s<>]+(?:\\\\|\\)+Tensorized_Blockchain(?:\\\\|\\)*[^\"'\s<>]*", re.IGNORECASE),
    re.compile(r"(?<![\w])/(?:Users|home)/[^\"'\s<>]+", re.IGNORECASE),
    re.compile(r"(?<![\w])/" + r"mnt/data/[^\"'\s<>]*", re.IGNORECASE),
    re.compile(r"(?<![\w])/private/var/[^\"'\s<>]+", re.IGNORECASE),
]

SECRET_PATTERNS = [
    ("PRIVATE_KEY_BLOCK", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ED25519 )?PRIVATE KEY-----")),
    ("AWS_ACCESS_KEY_ID", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("AWS_SECRET_ACCESS_KEY_ASSIGNMENT", re.compile(r"(?i)\baws_secret_access_key\b\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{32,})")),
    ("OPENAI_API_KEY", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b")),
    ("GITHUB_TOKEN", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b")),
    ("SLACK_TOKEN", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{20,}\b")),
    ("GOOGLE_API_KEY", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b")),
    ("GENERIC_SECRET_ASSIGNMENT", re.compile(r"(?i)\b(secret|api[_-]?key|access[_-]?token|auth[_-]?token|private[_-]?key|password)\b\s*[:=]\s*['\"]([^'\"\n]{12,})['\"]")),
]

LOW_CONFIDENCE_SECRET_WORDS = {"example", "dummy", "placeholder", "changeme", "test", "fake", "redacted", "your_", "<", ">"}


def check_license_headers(root: Path, files: Sequence[Path]) -> List[AuditIssue]:
    issues: List[AuditIssue] = []
    for p in files:
        if p.suffix.lower() not in {".py", ".sh", ".ps1"}:
            continue
        text, err = _read_text_for_scan(p, max_file_bytes=128_000)
        if err or text is None:
            continue
        head = "\n".join(text.splitlines()[:40])
        if not any(rx.search(head) for rx in LICENSE_PATTERNS):
            issues.append(AuditIssue(
                code="LICENSE_HEADER_MISSING",
                severity="warning",
                path=str(p.relative_to(root)).replace(os.sep, "/"),
                line=1,
                message="Source file does not contain a recognizable license/SPDX header in the first 40 lines.",
                recommendation="Add an SPDX license line, for example: # SPDX-License-Identifier: MIT",
            ))
    return issues


def check_absolute_paths(cfg: AuditConfig, files: Sequence[Path]) -> List[AuditIssue]:
    issues: List[AuditIssue] = []
    allow = [re.compile(p) for p in cfg.allow_absolute_path_patterns]
    for p in files:
        text, err = _read_text_for_scan(p, max_file_bytes=cfg.max_file_bytes)
        if err or text is None:
            continue
        rel = str(p.relative_to(cfg.root)).replace(os.sep, "/")
        for rx in ABSOLUTE_PATH_PATTERNS:
            for m in rx.finditer(text):
                hit = m.group(0)
                if any(a.search(hit) for a in allow):
                    continue
                line = _line_number_at(text, m.start())
                issues.append(AuditIssue(
                    code="ABSOLUTE_LOCAL_PATH_LEAK",
                    severity="warning",
                    path=rel,
                    line=line,
                    message="Potential local machine path leaked into source or docs.",
                    evidence=_evidence_snippet(_line_at(text, line), hit),
                    recommendation="Replace absolute local paths with relative paths, environment variables, or sanitized examples.",
                ))
    return issues


def _looks_like_placeholder(line: str) -> bool:
    l = line.lower()
    return any(word in l for word in LOW_CONFIDENCE_SECRET_WORDS)


def check_secrets(cfg: AuditConfig, files: Sequence[Path]) -> List[AuditIssue]:
    issues: List[AuditIssue] = []
    for p in files:
        text, err = _read_text_for_scan(p, max_file_bytes=cfg.max_file_bytes)
        if err or text is None:
            continue
        rel = str(p.relative_to(cfg.root)).replace(os.sep, "/")
        for code, rx in SECRET_PATTERNS:
            for m in rx.finditer(text):
                line_no = _line_number_at(text, m.start())
                line = _line_at(text, line_no)
                severity = "warning" if _looks_like_placeholder(line) else "error"
                secret_value = m.group(1) if m.lastindex and code != "GENERIC_SECRET_ASSIGNMENT" else m.group(0)
                if code == "GENERIC_SECRET_ASSIGNMENT" and m.lastindex and m.lastindex >= 2:
                    secret_value = m.group(2)
                    # Suppress obvious non-secret config values.
                    if _looks_like_placeholder(line):
                        severity = "warning"
                    if len(str(secret_value).strip()) < 20 and not any(ch.isdigit() for ch in str(secret_value)):
                        continue
                issues.append(AuditIssue(
                    code=f"SECRET_PATTERN_{code}",
                    severity=severity,
                    path=rel,
                    line=line_no,
                    message="Potential secret or private credential found.",
                    evidence=_evidence_snippet(line, str(secret_value)),
                    recommendation="Remove the secret from the repository, rotate it if real, and use environment variables or a secret manager.",
                ))
    return issues


def discover_imports(root: Path, files: Sequence[Path], *, max_file_bytes: int) -> List[str]:
    found: Set[str] = set()
    for p in files:
        if p.suffix.lower() != ".py":
            continue
        text, err = _read_text_for_scan(p, max_file_bytes=max_file_bytes)
        if err or text is None:
            continue
        try:
            tree = ast.parse(text, filename=str(p))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top and top not in COMMON_STDLIB:
                        found.add(top)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    if top and top not in COMMON_STDLIB and node.level == 0:
                        found.add(top)
    return sorted(found)


def check_dependencies(cfg: AuditConfig, files: Sequence[Path]) -> DependencyReport:
    report = DependencyReport(
        python_version=sys.version.replace("\n", " "),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
    )

    for req in cfg.required_files:
        report.required_files_present[req] = bool((cfg.root / req).exists())

    imports = discover_imports(cfg.root, files, max_file_bytes=cfg.max_file_bytes)
    report.discovered_imports = imports

    # Focus on project modules and common runtime dependencies. Avoid importing arbitrary packages with side effects.
    safe_to_probe = sorted(set(imports).intersection(PROJECT_IMPORT_CANDIDATES | {
        "numpy", "scipy", "cupy", "cryptography", "fastapi", "uvicorn", "pydantic", "pytest"
    }))

    if str(cfg.root) not in sys.path:
        sys.path.insert(0, str(cfg.root))

    for mod in safe_to_probe:
        try:
            importlib.import_module(mod)
            try:
                report.import_versions[mod] = importlib.metadata.version(mod)
            except Exception:
                # Project-local modules usually do not have package metadata.
                report.import_versions[mod] = "import-ok"
        except Exception as exc:
            report.import_errors[mod] = repr(exc)

    return report


def check_required_files(root: Path, dep: DependencyReport) -> List[AuditIssue]:
    issues: List[AuditIssue] = []
    for rel, present in sorted(dep.required_files_present.items()):
        if not present:
            issues.append(AuditIssue(
                code="REQUIRED_FILE_MISSING",
                severity="error",
                path=rel,
                line=0,
                message="Expected production stack file is missing.",
                recommendation="Create the file or remove it from the required production stack manifest.",
            ))
    return issues


def check_syntax(root: Path, files: Sequence[Path], *, max_file_bytes: int) -> List[AuditIssue]:
    issues: List[AuditIssue] = []
    for p in files:
        if p.suffix.lower() != ".py":
            continue
        text, err = _read_text_for_scan(p, max_file_bytes=max_file_bytes)
        rel = str(p.relative_to(root)).replace(os.sep, "/")
        if err:
            if err.startswith("file_too_large"):
                issues.append(AuditIssue(
                    code="PYTHON_FILE_TOO_LARGE_FOR_SYNTAX_SCAN",
                    severity="warning",
                    path=rel,
                    message=f"Skipped syntax scan: {err}",
                    recommendation="Increase --max-file-bytes or split very large source files.",
                ))
            continue
        if text is None:
            continue
        try:
            ast.parse(text, filename=rel)
        except SyntaxError as exc:
            issues.append(AuditIssue(
                code="PYTHON_SYNTAX_ERROR",
                severity="error",
                path=rel,
                line=int(exc.lineno or 0),
                message=f"Python syntax error: {exc.msg}",
                evidence=(exc.text or "").strip()[:240],
                recommendation="Fix the syntax error before release.",
            ))
    return issues


# =============================================================================
# Command execution gates
# =============================================================================

def _tail(text: str, max_chars: int = 4000) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def run_command(command: str, *, cwd: Path, timeout_s: int) -> CommandResult:
    start = time.perf_counter()
    try:
        args = shlex.split(command)
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=int(timeout_s),
            check=False,
        )
        duration = time.perf_counter() - start
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        combined = stdout + "\n--- STDERR ---\n" + stderr
        return CommandResult(
            command=command,
            ok=(proc.returncode == 0),
            returncode=int(proc.returncode),
            duration_s=float(duration),
            stdout_sha256=sha256_hex_text(stdout.replace("\r\n", "\n")),
            stderr_sha256=sha256_hex_text(stderr.replace("\r\n", "\n")),
            combined_sha256=sha256_hex_text(combined.replace("\r\n", "\n")),
            stdout_tail=_tail(stdout),
            stderr_tail=_tail(stderr),
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - start
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        combined = stdout + "\n--- STDERR ---\n" + stderr
        return CommandResult(
            command=command,
            ok=False,
            returncode=124,
            duration_s=float(duration),
            stdout_sha256=sha256_hex_text(stdout.replace("\r\n", "\n")),
            stderr_sha256=sha256_hex_text(stderr.replace("\r\n", "\n")),
            combined_sha256=sha256_hex_text(combined.replace("\r\n", "\n")),
            stdout_tail=_tail(stdout),
            stderr_tail=_tail(stderr + f"\nTIMEOUT after {timeout_s}s"),
        )
    except Exception as exc:
        duration = time.perf_counter() - start
        err = repr(exc)
        return CommandResult(
            command=command,
            ok=False,
            returncode=125,
            duration_s=float(duration),
            stdout_sha256=sha256_hex_text(""),
            stderr_sha256=sha256_hex_text(err),
            combined_sha256=sha256_hex_text(err),
            stdout_tail="",
            stderr_tail=err,
        )


def run_demo_reproducibility(command: str, *, cwd: Path, runs: int, timeout_s: int) -> ReproducibilityResult:
    results = [run_command(command, cwd=cwd, timeout_s=timeout_s) for _ in range(int(max(1, runs)))]
    hashes = [r.combined_sha256 for r in results]
    ok_commands = all(r.ok for r in results)
    same_hash = len(set(hashes)) <= 1
    ok = bool(ok_commands and same_hash)
    msg = "ok" if ok else ("command_failed" if not ok_commands else "output_not_reproducible")
    return ReproducibilityResult(
        command=command,
        runs=len(results),
        ok=ok,
        output_hashes=hashes,
        command_results=results,
        message=msg,
    )


# =============================================================================
# Reporting
# =============================================================================

def _count_issues(issues: Sequence[AuditIssue]) -> Dict[str, int]:
    counts = {"error": 0, "warning": 0, "info": 0, "total": len(issues)}
    for i in issues:
        counts[i.severity] = counts.get(i.severity, 0) + 1
    return counts


def build_markdown_report(report: AuditReport) -> str:
    status = "READY" if report.release_ready else "BLOCKED"
    lines: List[str] = []
    lines.append(f"# Tensorized Blockchain Open-Source Audit — {status}")
    lines.append("")
    lines.append(f"- Generated: `{report.generated_at}`")
    lines.append(f"- Root: `{report.root}`")
    lines.append(f"- Strict mode: `{report.strict}`")
    lines.append(f"- Scanned files: `{report.scanned_file_count}`")
    lines.append(f"- Scanned bytes: `{report.scanned_bytes}`")
    lines.append(f"- Report hash: `{report.report_hash}`")
    lines.append("")
    lines.append("## Issue counts")
    lines.append("")
    lines.append("| Severity | Count |")
    lines.append("|---|---:|")
    for k in ("error", "warning", "info", "total"):
        lines.append(f"| {k} | {report.counts.get(k, 0)} |")
    lines.append("")

    lines.append("## Required files")
    lines.append("")
    missing = [k for k, v in sorted(report.dependency_report.required_files_present.items()) if not v]
    if missing:
        lines.append("Missing:")
        for m in missing:
            lines.append(f"- `{m}`")
    else:
        lines.append("All required files are present.")
    lines.append("")

    lines.append("## Dependency/import status")
    lines.append("")
    if report.dependency_report.import_versions:
        lines.append("Imported successfully:")
        for k, v in sorted(report.dependency_report.import_versions.items()):
            lines.append(f"- `{k}`: `{v}`")
    if report.dependency_report.import_errors:
        lines.append("")
        lines.append("Import errors:")
        for k, v in sorted(report.dependency_report.import_errors.items()):
            lines.append(f"- `{k}`: `{v}`")
    if not report.dependency_report.import_versions and not report.dependency_report.import_errors:
        lines.append("No dependency probes were run or discovered.")
    lines.append("")

    if report.test_result is not None:
        lines.append("## Test command")
        lines.append("")
        lines.append(f"- Command: `{report.test_result.command}`")
        lines.append(f"- OK: `{report.test_result.ok}`")
        lines.append(f"- Return code: `{report.test_result.returncode}`")
        lines.append(f"- Duration: `{report.test_result.duration_s:.3f}s`")
        lines.append(f"- Combined hash: `{report.test_result.combined_sha256}`")
        lines.append("")

    if report.demo_reproducibility is not None:
        lines.append("## Demo reproducibility")
        lines.append("")
        lines.append(f"- Command: `{report.demo_reproducibility.command}`")
        lines.append(f"- Runs: `{report.demo_reproducibility.runs}`")
        lines.append(f"- OK: `{report.demo_reproducibility.ok}`")
        lines.append(f"- Message: `{report.demo_reproducibility.message}`")
        lines.append("")
        for idx, h in enumerate(report.demo_reproducibility.output_hashes, start=1):
            lines.append(f"- Run {idx}: `{h}`")
        lines.append("")

    lines.append("## Issues")
    lines.append("")
    if not report.issues:
        lines.append("No issues found.")
    else:
        lines.append("| Severity | Code | Path | Line | Message |")
        lines.append("|---|---|---|---:|---|")
        for i in report.issues:
            msg = (i.message or "").replace("|", "\\|")
            path = (i.path or "").replace("|", "\\|")
            lines.append(f"| {i.severity} | `{i.code}` | `{path}` | {i.line} | {msg} |")
        lines.append("")
        lines.append("### Detailed evidence")
        lines.append("")
        for i in report.issues:
            lines.append(f"#### `{i.code}` — {i.severity}")
            lines.append("")
            lines.append(f"- Path: `{i.path}`")
            if i.line:
                lines.append(f"- Line: `{i.line}`")
            if i.message:
                lines.append(f"- Message: {i.message}")
            if i.evidence:
                lines.append(f"- Evidence: `{i.evidence}`")
            if i.recommendation:
                lines.append(f"- Recommendation: {i.recommendation}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# =============================================================================
# Main audit
# =============================================================================

def run_audit(cfg: AuditConfig) -> AuditReport:
    cfg = cfg.normalized()
    if not cfg.root.exists() or not cfg.root.is_dir():
        raise FileNotFoundError(f"Audit root does not exist or is not a directory: {cfg.root}")

    files = list(iter_candidate_files(cfg))
    scanned_bytes = 0
    for p in files:
        try:
            scanned_bytes += int(p.stat().st_size)
        except Exception:
            pass

    issues: List[AuditIssue] = []
    issues.extend(check_syntax(cfg.root, files, max_file_bytes=cfg.max_file_bytes))
    issues.extend(check_license_headers(cfg.root, files))
    issues.extend(check_absolute_paths(cfg, files))
    issues.extend(check_secrets(cfg, files))

    dep_report = check_dependencies(cfg, files)
    issues.extend(check_required_files(cfg.root, dep_report))

    test_result: Optional[CommandResult] = None
    if cfg.run_tests:
        test_cmd = cfg.test_command.strip() or "python -m pytest -q"
        test_result = run_command(test_cmd, cwd=cfg.root, timeout_s=cfg.command_timeout_s)
        if not test_result.ok:
            issues.append(AuditIssue(
                code="TEST_COMMAND_FAILED",
                severity="error",
                path=".",
                line=0,
                message=f"Test command failed with exit code {test_result.returncode}.",
                evidence=test_result.stderr_tail[:240] or test_result.stdout_tail[:240],
                recommendation="Fix failing tests before release.",
            ))

    demo_result: Optional[ReproducibilityResult] = None
    if cfg.demo_command.strip():
        demo_result = run_demo_reproducibility(
            cfg.demo_command.strip(),
            cwd=cfg.root,
            runs=cfg.demo_runs,
            timeout_s=cfg.command_timeout_s,
        )
        if not demo_result.ok:
            issues.append(AuditIssue(
                code="DEMO_NOT_REPRODUCIBLE",
                severity="error",
                path=".",
                line=0,
                message=f"Demo reproducibility check failed: {demo_result.message}.",
                evidence=", ".join(demo_result.output_hashes[:4]),
                recommendation="Ensure deterministic demo seeds, fixed timestamps, and stable serialized output.",
            ))

    counts = _count_issues(issues)
    release_ready = not any(i.is_blocking(strict=cfg.strict) for i in issues)
    report = AuditReport(
        version=TB_OPEN_SOURCE_AUDIT_VERSION,
        root=str(cfg.root),
        generated_at=_now_iso_utc(),
        release_ready=bool(release_ready),
        strict=bool(cfg.strict),
        counts=counts,
        issues=issues,
        dependency_report=dep_report,
        test_result=test_result,
        demo_reproducibility=demo_result,
        scanned_file_count=len(files),
        scanned_bytes=int(scanned_bytes),
        report_hash="",
    )

    # Hash excludes the hash field itself.
    report_dict = report.to_dict()
    report_dict["report_hash"] = ""
    report.report_hash = sha256_hex_text(_canonical_json(report_dict))
    return report


# =============================================================================
# Self-test
# =============================================================================

def _self_test() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # Required files for a minimal no-missing test.
        for name in AuditConfig(root=root).required_files:
            p = root / name
            p.write_text("# SPDX-License-Identifier: MIT\nprint('ok')\n", encoding="utf-8")

        clean = root / "clean_module.py"
        clean.write_text("# SPDX-License-Identifier: MIT\nx = 1\n", encoding="utf-8")

        cfg = AuditConfig(root=root, strict=True).normalized()
        report = run_audit(cfg)
        if not report.release_ready:
            raise AssertionError(f"Expected clean temp repo to be release-ready, got {report.counts}: {report.issues[:3]}")

        bad = root / "bad_secret.py"
        bad.write_text(
            "# no license\n"
            "API_KEY = 'sk-proj-" + "abcdefghijklmnopqrstuvwxyz1234567890'\n"
            "LOCAL = r'C:\\\\Users\\\\" + "alice\\\\secret\\\\repo\\\\file.py'\n",
            encoding="utf-8",
        )
        report2 = run_audit(AuditConfig(root=root, strict=False))
        codes = {i.code for i in report2.issues}
        if "SECRET_PATTERN_OPENAI_API_KEY" not in codes:
            raise AssertionError(f"Secret pattern was not detected. Codes={codes}")
        if "ABSOLUTE_LOCAL_PATH_LEAK" not in codes:
            raise AssertionError(f"Absolute path leak was not detected. Codes={codes}")
        if report2.release_ready:
            raise AssertionError("Repo with high-confidence secret should not be release-ready.")

    print(json.dumps({"ok": True, "self_test": "passed", "version": TB_OPEN_SOURCE_AUDIT_VERSION}, indent=2))


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tensorized Blockchain open-source release-readiness audit.")
    p.add_argument("--root", default=".", help="Repository root to audit.")
    p.add_argument("--output", default="", help="Optional JSON report output path.")
    p.add_argument("--markdown", default="", help="Optional Markdown report output path.")
    p.add_argument("--strict", action="store_true", help="Treat warnings as blocking release issues.")
    p.add_argument("--max-file-bytes", type=int, default=2_000_000, help="Maximum text file size scanned.")
    p.add_argument("--scan-binary", action="store_true", help="Include binary-looking files in candidate iteration; text decode still guarded.")
    p.add_argument("--exclude", action="append", default=[], help="Additional glob exclude pattern. Repeatable.")
    p.add_argument("--allow-absolute-path-pattern", action="append", default=[], help="Regex for allowed absolute path evidence. Repeatable.")
    p.add_argument("--run-tests", action="store_true", help="Run test command as a release gate.")
    p.add_argument("--test-command", default="", help="Test command. Defaults to 'python -m pytest -q' when --run-tests is used.")
    p.add_argument("--demo-command", default="", help="Command to run multiple times for reproducible demo output.")
    p.add_argument("--demo-runs", type=int, default=2, help="Number of demo reproducibility runs.")
    p.add_argument("--command-timeout-s", type=int, default=120, help="Timeout for test/demo commands.")
    p.add_argument("--self-test", action="store_true", help="Run internal self-test and exit.")
    p.add_argument("--quiet", action="store_true", help="Suppress console summary except errors.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.self_test:
        try:
            _self_test()
            return 0
        except Exception as exc:
            print(json.dumps({"ok": False, "error": repr(exc)}, indent=2), file=sys.stderr)
            return 2

    try:
        cfg = AuditConfig(
            root=Path(args.root),
            strict=bool(args.strict),
            max_file_bytes=int(args.max_file_bytes),
            scan_binary=bool(args.scan_binary),
            run_tests=bool(args.run_tests),
            test_command=str(args.test_command or ""),
            demo_command=str(args.demo_command or ""),
            demo_runs=int(args.demo_runs),
            command_timeout_s=int(args.command_timeout_s),
            allow_absolute_path_patterns=list(args.allow_absolute_path_pattern or []),
            exclude_patterns=list(args.exclude or []),
        )
        report = run_audit(cfg)

        if args.output:
            _atomic_write_json(Path(args.output), report.to_dict())

        if args.markdown:
            _atomic_write_text(Path(args.markdown), build_markdown_report(report))

        if not args.quiet:
            summary = {
                "release_ready": report.release_ready,
                "counts": report.counts,
                "scanned_file_count": report.scanned_file_count,
                "report_hash": report.report_hash,
                "json_report": str(Path(args.output).resolve()) if args.output else "",
                "markdown_report": str(Path(args.markdown).resolve()) if args.markdown else "",
            }
            print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))

        return 0 if report.release_ready else 1

    except Exception as exc:
        print(json.dumps({"ok": False, "error": repr(exc), "version": TB_OPEN_SOURCE_AUDIT_VERSION}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
