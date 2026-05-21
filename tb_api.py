#!/usr/bin/env python3
"""Project Chimera / Akkurat - Tensorized Blockchain FastAPI boundary.

Local API server for the demo/app. It is intentionally a thin orchestration
boundary: Python internals stay behind JSON endpoints, while ledger/twin/lobe
modules remain responsible for their own invariants.
"""
from __future__ import annotations

import argparse, dataclasses, hashlib, importlib, json, os, sys, time, traceback
from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.encoders import jsonable_encoder
    from pydantic import BaseModel, Field, ConfigDict
    FASTAPI_OK, FASTAPI_ERROR = True, ""
except Exception as exc:  # pragma: no cover
    FastAPI = None  # type: ignore
    HTTPException = RuntimeError  # type: ignore
    Request = Any  # type: ignore
    CORSMiddleware = None  # type: ignore
    JSONResponse = None  # type: ignore
    jsonable_encoder = None  # type: ignore
    BaseModel = object  # type: ignore
    Field = lambda default=None, **_: default  # type: ignore
    ConfigDict = dict  # type: ignore
    FASTAPI_OK, FASTAPI_ERROR = False, repr(exc)

TB_API_VERSION = "tensorized-blockchain-api-v1.0-production"
DEFAULT_CHAIN_ID = "project-chimera-local"
DEFAULT_ARTIFACT_ROOT = "./tb_api_artifacts"


def _module_dir() -> Path:
    try: return Path(__file__).resolve().parent
    except Exception: return Path.cwd()


def _add_path(path: Any) -> Optional[Path]:
    if not path: return None
    try: p = Path(path).expanduser().resolve()
    except Exception: return None
    if not p.exists(): return None
    s = str(p)
    if s not in sys.path: sys.path.insert(0, s)
    return p


def configure_paths(project_root: Optional[Union[str, os.PathLike]] = None) -> Dict[str, str]:
    import tempfile
    here = _module_dir()
    for c in [project_root, os.getenv("TENSORIZED_BLOCKCHAIN_ROOT"), os.getenv("TB_ROOT"), here, here.parent, Path.cwd(), Path(tempfile.gettempdir())]:
        _add_path(c)
    return {"module_dir": str(here), "cwd": str(Path.cwd()), "project_root": "" if project_root is None else str(project_root)}

configure_paths()


def _imp(name: str) -> Tuple[Optional[Any], str]:
    try: return importlib.import_module(name), ""
    except Exception as exc: return None, repr(exc)

mods: Dict[str, Optional[Any]] = {}
errs: Dict[str, str] = {}
for _n in [
    "tb_config","tb_canonical","tb_hashing","tb_crypto","tb_schemas","tb_validation","tb_storage",
    "tb_mempool","tb_consensus","tb_execution","tb_tensor_commitments","tb_demo_scenarios",
    "tn","digital_twin_kernel","twin_anything","tensorized_blockchain_kernel","cognitive_lobe_runtime","lobe_scaler",
]:
    mods[_n], errs[_n] = _imp(_n)

def _now() -> str: return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def js(obj: Any) -> Any:
    c = mods.get("tb_canonical")
    if c is not None and hasattr(c, "json_safe"):
        try: return c.json_safe(obj)
        except Exception: pass
    if obj is None or isinstance(obj, (str, int, bool)): return obj
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else str(obj)
    if np is not None:
        try:
            if isinstance(obj, np.generic): return js(obj.item())
            if isinstance(obj, np.ndarray): return np.nan_to_num(obj).astype(float).tolist()
        except Exception: pass
    if is_dataclass(obj): return js(asdict(obj))
    if isinstance(obj, Mapping): return {str(k): js(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)): return [js(v) for v in obj]
    for m in ("to_dict", "payload", "summary", "snapshot"):
        fn = getattr(obj, m, None)
        if callable(fn):
            try: return js(fn())
            except Exception: pass
    if hasattr(obj, "__dict__"):
        try: return js(vars(obj))
        except Exception: pass
    return str(obj)


def canonical(obj: Any) -> str:
    c = mods.get("tb_canonical")
    if c is not None and hasattr(c, "canonical_dumps"):
        try: return str(c.canonical_dumps(obj))
        except Exception: pass
    return json.dumps(js(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hsh(obj: Any, domain: str = "TB:API:OBJECT:v1") -> str:
    h = mods.get("tb_hashing")
    if h is not None and hasattr(h, "domain_hash"):
        try: return str(h.domain_hash(domain, obj))
        except Exception: pass
    return hashlib.sha256((domain + "\0" + canonical(obj)).encode()).hexdigest()


def out(payload: Any, code: int = 200) -> Any:
    data = js(payload)
    if jsonable_encoder is not None: data = jsonable_encoder(data)
    return JSONResponse(content=data, status_code=code) if JSONResponse is not None else data


def asdictish(obj: Any) -> Dict[str, Any]:
    x = js(obj)
    return dict(x) if isinstance(x, Mapping) else {"value": x}


class ApiModel(BaseModel):  # type: ignore[misc]
    if hasattr(BaseModel, "model_config"):
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)  # type: ignore


class CreateTwinRequest(ApiModel):
    thing_name: str = "Alicante Fire Dept Robot Alpha"
    thing_type: str = "industrial_machine"
    description: str = "Governed tensorized digital twin created by tb_api."
    observables: Dict[str, Any] = Field(default_factory=dict)
    process_steps: List[str] = Field(default_factory=list)
    resources: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    components: Dict[str, Any] = Field(default_factory=dict)
    environment: Dict[str, Any] = Field(default_factory=dict)
    success_criteria: Dict[str, Any] = Field(default_factory=dict)
    risks: List[Any] = Field(default_factory=list)
    actions: List[Any] = Field(default_factory=list)
    source_files: List[Dict[str, Any]] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    deterministic: Optional[bool] = None
    consensus_safe: Optional[bool] = None
    include_network_snapshot: bool = True


class CreateTransactionRequest(ApiModel):
    sender_id: str = "demo_operator"
    sender_sequence: int = 1
    sender_public_key: str = ""
    private_key: str = ""
    twin_node_id: str = "5.4"
    action: Dict[str, Any] = Field(default_factory=lambda: {"kind":"update_node_data","node_id":"5.4","payload":{"status":"nominal"},"note":"api_transaction"})
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[float] = None
    nonce: Optional[str] = None
    sign: bool = False


class SubmitTransactionRequest(ApiModel):
    transaction: Dict[str, Any]
    strict: bool = False


class BatchRequest(ApiModel):
    max_count: int = 16
    max_bytes: int = 1_000_000
    remove: bool = False


class ExecuteRequest(ApiModel):
    transaction: Optional[Dict[str, Any]] = None
    tx_id_or_hash: str = ""
    use_mempool: bool = False
    remove_from_mempool: bool = False
    max_count: int = 1
    intent: str = "api_transaction_execution"
    run_lobes: bool = True
    strict: bool = False
    require_valid_transaction: bool = False
    allow_direct_network_fallback: bool = True
    cfc_time_delta: float = 1.0


class LobeAnalysisRequest(ApiModel):
    twin_snapshot: Optional[Dict[str, Any]] = None
    transaction: Optional[Dict[str, Any]] = None
    pre_state_commitment: Optional[Dict[str, Any]] = None
    post_state_commitment: Optional[Dict[str, Any]] = None
    policy_receipt: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = Field(default_factory=dict)
    raw_vector: Optional[List[float]] = None
    time_delta: float = 1.0
    advance_cfc: bool = True


class AppendBlockRequest(ApiModel):
    execution_hash: str = ""
    transaction: Optional[Dict[str, Any]] = None
    transactions: List[Dict[str, Any]] = Field(default_factory=list)
    pre_state_root: str = ""
    post_state_root: str = ""
    policy_receipts: List[Dict[str, Any]] = Field(default_factory=list)
    action_receipts: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CompressionRequest(ApiModel):
    run: bool = True
    use_compressed: bool = False


class ExportArtifactsRequest(ApiModel):
    export_id: str = "demo_export"
    include_ledger: bool = True
    include_twin_snapshot: bool = True
    include_mempool: bool = True
    include_lobe_snapshot: bool = True
    include_executions: bool = True


class ScenarioRequest(ApiModel):
    scenario: str = "alicante_fire_department"
    seed: int = 2027
    materialize_twin: bool = False


@dataclass
class ApiSettings:
    chain_id: str = DEFAULT_CHAIN_ID
    deterministic: bool = True
    consensus_safe: bool = True
    project_root: str = ""
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    enable_lobes: bool = True
    require_signatures: bool = False
    verify_mempool_signatures: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    @classmethod
    def from_env(cls) -> "ApiSettings":
        return cls(
            chain_id=os.getenv("TB_CHAIN_ID", DEFAULT_CHAIN_ID),
            deterministic=os.getenv("TB_DETERMINISTIC", "1").lower() not in {"0","false","no"},
            consensus_safe=os.getenv("TB_CONSENSUS_SAFE", "1").lower() not in {"0","false","no"},
            project_root=os.getenv("TB_PROJECT_ROOT", ""),
            artifact_root=os.getenv("TB_ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT),
            enable_lobes=os.getenv("TB_ENABLE_LOBES", "1").lower() not in {"0","false","no"},
            require_signatures=os.getenv("TB_REQUIRE_SIGNATURES", "0").lower() in {"1","true","yes"},
            verify_mempool_signatures=os.getenv("TB_VERIFY_MEMPOOL_SIGNATURES", "0").lower() in {"1","true","yes"},
            cors_origins=[x.strip() for x in os.getenv("TB_API_CORS_ORIGINS", "*").split(",") if x.strip()],
        )

    def to_dict(self) -> Dict[str, Any]: return js(asdict(self))


@dataclass
class StoredExecution:
    execution_hash: str
    transaction: Any
    result: Any
    created_at: str = field(default_factory=_now)
    def to_dict(self) -> Dict[str, Any]: return js(asdict(self))


class TBApiRuntime:
    def __init__(self, settings: Optional[ApiSettings] = None):
        self.settings = settings or ApiSettings.from_env()
        configure_paths(self.settings.project_root or None)
        self.created_at = _now()
        self.runtime_id = hsh({"settings": self.settings.to_dict(), "created_at": self.created_at}, "TB:API:RUNTIME_ID:v1")[:32]
        self.artifact_root = Path(self.settings.artifact_root).expanduser().resolve(); self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.ledger = None; self.mempool = None; self.twin_runtime: Optional[Dict[str, Any]] = None; self.lobe_runtime = None
        self.executions: Dict[str, StoredExecution] = {}; self.last_batch: Dict[str, Any] = {}; self.last_tensor_report: Dict[str, Any] = {}
        self.warnings: List[str] = []
        self._init()

    def _warn(self, x: str) -> None: self.warnings.append(str(x))

    def _init(self) -> None:
        tbk = mods.get("tensorized_blockchain_kernel")
        if tbk and hasattr(tbk, "LedgerMPS"):
            try:
                cfg = tbk.LedgerConfig(chain_id=self.settings.chain_id, require_signatures=self.settings.require_signatures, deterministic_blocks=self.settings.deterministic)
                self.ledger = tbk.LedgerMPS(cfg)
            except Exception as exc: self._warn(f"ledger_init_failed:{exc!r}")
        else: self._warn(f"ledger_unavailable:{errs.get('tensorized_blockchain_kernel')}")
        mp = mods.get("tb_mempool")
        if mp and hasattr(mp, "TransactionMempool"):
            try:
                cfg = mp.MempoolConfig(chain_id=self.settings.chain_id, require_signature=self.settings.require_signatures, verify_signature=self.settings.verify_mempool_signatures)
                self.mempool = mp.TransactionMempool(cfg)
            except Exception as exc: self._warn(f"mempool_init_failed:{exc!r}")
        else: self._warn(f"mempool_unavailable:{errs.get('tb_mempool')}")

    def integrations(self) -> Dict[str, Any]:
        d = {k: {"available": mods.get(k) is not None, "error": errs.get(k, "")} for k in mods}
        d["fastapi"] = {"available": FASTAPI_OK, "error": FASTAPI_ERROR}
        return d

    def health(self) -> Dict[str, Any]:
        s = None
        if self.ledger is not None:
            try: s = self.ledger.summary()
            except Exception as exc: s = {"error": repr(exc)}
        return {"ok": FASTAPI_OK and self.ledger is not None, "version": TB_API_VERSION, "runtime_id": self.runtime_id, "settings": self.settings.to_dict(), "integrations": self.integrations(), "ledger": s, "mempool_pending": None if self.mempool is None else self.mempool.pending_count(), "twin_loaded": self.twin_runtime is not None, "lobe_runtime_loaded": self.lobe_runtime is not None, "execution_count": len(self.executions), "warnings": self.warnings[-20:]}

    def require_ledger(self) -> Any:
        if self.ledger is None: raise RuntimeError("ledger unavailable")
        return self.ledger
    def require_mempool(self) -> Any:
        if self.mempool is None: raise RuntimeError("mempool unavailable")
        return self.mempool
    def require_twin(self) -> Tuple[Any, Any]:
        if not self.twin_runtime: raise RuntimeError("no twin loaded; POST /twins first")
        return self.twin_runtime.get("network"), self.twin_runtime.get("control_plane")

    def ensure_lobes(self) -> Any:
        if self.lobe_runtime is not None: return self.lobe_runtime
        cr = mods.get("cognitive_lobe_runtime")
        if cr is None or not hasattr(cr, "CognitiveLobeRuntime"): raise RuntimeError(f"cognitive_lobe_runtime unavailable:{errs.get('cognitive_lobe_runtime')}")
        cfg = None
        if hasattr(cr, "CognitiveLobeRuntimeConfig"):
            try: cfg = cr.CognitiveLobeRuntimeConfig(deterministic=self.settings.deterministic, consensus_safe=self.settings.consensus_safe, runtime_id=f"{self.runtime_id}_lobes")
            except TypeError: cfg = cr.CognitiveLobeRuntimeConfig()
        self.lobe_runtime = cr.CognitiveLobeRuntime(config=cfg)
        return self.lobe_runtime

    def list_scenarios(self) -> Dict[str, Any]:
        ds = mods.get("tb_demo_scenarios")
        return {"ok": bool(ds and hasattr(ds,"list_scenarios")), "scenarios": [] if not ds else ds.list_scenarios() if hasattr(ds,"list_scenarios") else [], "error": errs.get("tb_demo_scenarios") if not ds else ""}

    def build_scenario(self, req: ScenarioRequest) -> Dict[str, Any]:
        ds = mods.get("tb_demo_scenarios")
        if ds is None or not hasattr(ds, "build_scenario"): raise RuntimeError("tb_demo_scenarios unavailable")
        fx = ds.build_scenario(req.scenario, chain_id=self.settings.chain_id, seed=req.seed)
        payload = fx.to_dict(include_private=False) if hasattr(fx,"to_dict") else js(fx)
        if req.materialize_twin and hasattr(ds, "try_materialize_twin_fixture"):
            payload["materialized_twin"] = ds.try_materialize_twin_fixture(fx)
        return {"ok": True, "fixture": payload}

    def create_twin(self, req: CreateTwinRequest) -> Dict[str, Any]:
        ta = mods.get("twin_anything")
        if ta is None or not hasattr(ta, "TwinAnythingFactory"): raise RuntimeError("twin_anything unavailable")
        fac = ta.TwinAnythingFactory(deterministic=self.settings.deterministic if req.deterministic is None else req.deterministic, consensus_safe=self.settings.consensus_safe if req.consensus_safe is None else req.consensus_safe, project_root=self.settings.project_root or None)
        res = fac.create_twin_runtime(thing_name=req.thing_name, thing_type=req.thing_type, description=req.description, observables=req.observables, process_steps=req.process_steps, resources=req.resources, constraints=req.constraints, components=req.components, environment=req.environment, success_criteria=req.success_criteria, risks=req.risks, actions=req.actions, source_files=req.source_files, links=req.links, tags=req.tags, metadata=req.metadata)
        serial = res.to_serializable(include_network_snapshot=req.include_network_snapshot) if hasattr(res, "to_serializable") else js(res)
        self.twin_runtime = {"network": getattr(res,"network",None), "control_plane": getattr(res,"control_plane",None), "serializable": serial, "created_at": _now()}
        return {"ok": True, "twin": serial}

    def action(self, p: Mapping[str, Any]) -> Any:
        dtk = mods.get("digital_twin_kernel")
        if dtk and hasattr(dtk, "Action"):
            return dtk.Action(kind=str(p.get("kind","")), node_id=p.get("node_id"), payload=p.get("payload"), note=str(p.get("note","")))
        return dict(p)

    def tx_obj(self, payload: Any) -> Any:
        tbk = mods.get("tensorized_blockchain_kernel")
        if tbk and hasattr(tbk, "TensorTransaction") and isinstance(payload, Mapping): return tbk.TensorTransaction.from_dict(payload)
        return payload

    def create_transaction(self, req: CreateTransactionRequest) -> Dict[str, Any]:
        tbk = mods.get("tensorized_blockchain_kernel")
        if tbk is None or not hasattr(tbk, "TensorTransaction"): raise RuntimeError("tensorized_blockchain_kernel unavailable")
        tx = tbk.TensorTransaction.new(sender_id=req.sender_id, sender_sequence=req.sender_sequence, sender_public_key=req.sender_public_key, twin_node_id=req.twin_node_id, action=self.action(req.action), chain_id=self.settings.chain_id, metadata=req.metadata, timestamp=req.timestamp, nonce=req.nonce)
        if req.sign:
            if not req.private_key: raise RuntimeError("private_key required when sign=True")
            tx.sign(req.private_key)
        return {"ok": True, "transaction": tx.to_dict(), "tx_hash": tx.payload_hash()}

    def submit_transaction(self, req: SubmitTransactionRequest) -> Dict[str, Any]:
        dec = self.require_mempool().accept_transaction(req.transaction, strict=req.strict)
        return {"ok": bool(getattr(dec,"accepted",False)), "decision": js(dec), "mempool": js(self.mempool.to_dict()) if self.mempool else {}}

    def batch(self, req: BatchRequest) -> Dict[str, Any]:
        b = self.require_mempool().produce_batch(max_count=req.max_count, max_bytes=req.max_bytes, remove=req.remove)
        self.last_batch = asdictish(b); return {"ok": True, "batch": self.last_batch}

    def lobe_analysis(self, req: LobeAnalysisRequest) -> Dict[str, Any]:
        lr = self.ensure_lobes()
        r = lr.run(twin_snapshot=req.twin_snapshot, transaction=req.transaction, pre_state_commitment=req.pre_state_commitment, post_state_commitment=req.post_state_commitment, policy_receipt=req.policy_receipt, extra=req.extra, raw_vector=req.raw_vector, time_delta=req.time_delta, advance_cfc=req.advance_cfc)
        return {"ok": True, "lobe_result": js(r)}

    def resolve_tx(self, req: ExecuteRequest) -> Any:
        if req.transaction: return req.transaction
        if req.tx_id_or_hash and self.mempool:
            tx = self.mempool.get_transaction(req.tx_id_or_hash)
            if tx is None: raise RuntimeError("transaction not found in mempool")
            return tx
        if req.use_mempool:
            b = self.require_mempool().produce_batch(max_count=max(1, req.max_count), remove=req.remove_from_mempool)
            self.last_batch = asdictish(b); txs = self.last_batch.get("transactions", [])
            if not txs: raise RuntimeError("mempool batch empty")
            return txs[0]
        raise RuntimeError("provide transaction, tx_id_or_hash, or use_mempool=True")

    def execute(self, req: ExecuteRequest) -> Dict[str, Any]:
        ex = mods.get("tb_execution")
        if ex is None or not hasattr(ex, "execute_transaction"): raise RuntimeError("tb_execution unavailable")
        net, cp = self.require_twin(); tx = self.tx_obj(self.resolve_tx(req))
        ncp = cfc = None
        if req.run_lobes and self.settings.enable_lobes:
            try:
                lr = self.ensure_lobes(); ncp = getattr(lr,"ncp_lobe",None); cfc = getattr(lr,"cfc_lobe",None)
            except Exception as exc: self._warn(f"lobe_unavailable:{exc!r}")
        cfg = ex.ExecutionConfig(deterministic=self.settings.deterministic, consensus_safe=self.settings.consensus_safe, require_valid_transaction=req.require_valid_transaction, verify_signature=False, allow_direct_network_fallback=req.allow_direct_network_fallback, emit_lobe_receipts=req.run_lobes, run_ncp_lobe=bool(ncp), run_cfc_lobe=bool(cfc), cfc_time_delta=req.cfc_time_delta, strict=req.strict) if hasattr(ex,"ExecutionConfig") else None
        res = ex.execute_transaction(tx, network=net, control_plane=cp, ncp_lobe=ncp, cfc_lobe=cfc, intent=req.intent, config=cfg)
        eh = res.compute_hash() if hasattr(res,"compute_hash") else getattr(res,"execution_hash","") or hsh(res, "TB:API:EXECUTION:v1")
        self.executions[eh] = StoredExecution(eh, tx, res)
        return {"ok": bool(asdictish(res).get("accepted", True)), "execution_hash": eh, "execution": js(res)}

    def append_block(self, req: AppendBlockRequest) -> Dict[str, Any]:
        led = self.require_ledger(); txs: List[Any] = []; pre = req.pre_state_root; post = req.post_state_root; pol = list(req.policy_receipts); act = list(req.action_receipts); meta = dict(req.metadata)
        if req.execution_hash:
            st = self.executions.get(req.execution_hash)
            if st is None: raise RuntimeError("unknown execution_hash")
            txs = [self.tx_obj(st.transaction)]; rd = asdictish(st.result)
            pre = pre or str(rd.get("pre_state_root", "")); post = post or str(rd.get("post_state_root", ""))
            if not pol and rd.get("policy_receipt"): pol = [rd.get("policy_receipt")]
            if not act and isinstance(rd.get("action_receipts"), list): act = list(rd.get("action_receipts"))
            meta.setdefault("execution_hash", req.execution_hash)
        elif req.transactions: txs = [self.tx_obj(x) for x in req.transactions]
        elif req.transaction: txs = [self.tx_obj(req.transaction)]
        else: raise RuntimeError("append requires transaction(s) or execution_hash")
        if not pre or not post: raise RuntimeError("pre_state_root and post_state_root required")
        block = led.append_block(txs, pre_state_root=pre, post_state_root=post, policy_receipts=pol, action_receipts=act, metadata=meta)
        if self.mempool:
            try: self.mempool.mark_committed([js(t) for t in txs])
            except Exception: pass
        return {"ok": True, "block": js(block), "ledger": led.summary() if hasattr(led,"summary") else {}}

    def tensor_report(self, req: CompressionRequest) -> Dict[str, Any]:
        led = self.require_ledger(); report = led.compress_ledger() if req.run and hasattr(led,"compress_ledger") else getattr(led,"_last_compression_report",{}) or {}; comm = {}
        try:
            tt = led.get_tensor_train(use_compressed=req.use_compressed)
            ttc = mods.get("tb_tensor_commitments")
            if ttc and hasattr(ttc,"tensor_train_commitment"): comm = js(ttc.tensor_train_commitment(tt))
            elif hasattr(tt,"commitment"): comm = js(tt.commitment())
        except Exception as exc: comm = {"error": repr(exc)}
        self.last_tensor_report = {"compression_report": report, "tensor_commitment": comm}
        return {"ok": True, **self.last_tensor_report}

    def export(self, req: ExportArtifactsRequest) -> Dict[str, Any]:
        sid = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in req.export_id)[:96] or "demo_export"
        root = (self.artifact_root / sid).resolve()
        if self.artifact_root not in root.parents and root != self.artifact_root: raise RuntimeError("unsafe export path")
        root.mkdir(parents=True, exist_ok=True); files: Dict[str,str] = {}
        def w(name: str, payload: Any) -> None:
            p = root / name; tmp = p.with_suffix(p.suffix + ".tmp"); tmp.write_text(json.dumps(js(payload), indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"); os.replace(tmp,p); files[name] = str(p)
        if req.include_ledger and self.ledger: w("ledger.json", self.ledger.to_dict(include_compressed_cores=False) if hasattr(self.ledger,"to_dict") else js(self.ledger))
        if req.include_twin_snapshot and self.twin_runtime: w("twin_snapshot.json", self.twin_runtime.get("serializable",{}))
        if req.include_mempool and self.mempool: w("mempool.json", self.mempool.to_dict() if hasattr(self.mempool,"to_dict") else {})
        if req.include_lobe_snapshot and self.lobe_runtime:
            try: snap = self.lobe_runtime.snapshot()
            except Exception as exc: snap = {"error": repr(exc)}
            w("lobe_runtime_snapshot.json", snap)
        if req.include_executions: w("executions.json", {k:v.to_dict() for k,v in self.executions.items()})
        manifest = {"schema":"tb.api.export_manifest.v1", "export_id": sid, "created_at": _now(), "runtime_id": self.runtime_id, "files": files}
        manifest["manifest_hash"] = hsh(manifest, "TB:API:EXPORT_MANIFEST:v1"); w("manifest.json", manifest)
        return {"ok": True, "export_root": str(root), "manifest": manifest, "files": files}


def create_app(settings: Optional[ApiSettings] = None) -> Any:
    if not FASTAPI_OK: raise RuntimeError(f"FastAPI unavailable: {FASTAPI_ERROR}")
    runtime = TBApiRuntime(settings)
    app = FastAPI(title="Tensorized Blockchain Local API", version=TB_API_VERSION, description="Local API boundary for the Tensorized Blockchain demo stack.")
    app.state.tb_runtime = runtime
    if CORSMiddleware: app.add_middleware(CORSMiddleware, allow_origins=runtime.settings.cors_origins or ["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    def rt() -> TBApiRuntime: return app.state.tb_runtime

    @app.exception_handler(Exception)
    async def handler(request: Request, exc: Exception):
        return out({"ok": False, "error": str(exc), "type": exc.__class__.__name__, "path": str(getattr(request.url,"path","")), "traceback": traceback.format_exc() if os.getenv("TB_API_DEBUG") else ""}, 500)

    @app.get("/")
    def root(): return out({"ok": True, "service": "Tensorized Blockchain Local API", "version": TB_API_VERSION, "docs": "/docs"})
    @app.get("/health")
    def health(): return out(rt().health())
    @app.get("/config")
    def config(): return out({"ok": True, "settings": rt().settings.to_dict(), "integrations": rt().integrations()})
    @app.get("/scenarios")
    def scenarios(): return out(rt().list_scenarios())
    @app.post("/scenarios/build")
    def scenario_build(req: ScenarioRequest): return out(rt().build_scenario(req))
    @app.post("/twins")
    def twins(req: CreateTwinRequest): return out(rt().create_twin(req))
    @app.get("/twins/current")
    def twin_current(): return out({"ok": bool(rt().twin_runtime), "twin": None if not rt().twin_runtime else rt().twin_runtime.get("serializable")})
    @app.post("/transactions/create")
    def tx_create(req: CreateTransactionRequest): return out(rt().create_transaction(req))
    @app.post("/transactions/submit")
    def tx_submit(req: SubmitTransactionRequest): return out(rt().submit_transaction(req))
    @app.get("/mempool")
    def mempool():
        mp = rt().require_mempool(); return out({"ok": True, "mempool": mp.to_dict() if hasattr(mp,"to_dict") else js(mp)})
    @app.post("/mempool/batch")
    def mp_batch(req: BatchRequest): return out(rt().batch(req))
    @app.post("/execute")
    def execute(req: ExecuteRequest): return out(rt().execute(req))
    @app.post("/lobes/analyze")
    def lobes(req: LobeAnalysisRequest): return out(rt().lobe_analysis(req))
    @app.get("/lobes/snapshot")
    def lobes_snapshot():
        lr = rt().ensure_lobes(); return out({"ok": True, "snapshot": lr.snapshot() if hasattr(lr,"snapshot") else js(lr)})
    @app.post("/blocks/append")
    def blocks(req: AppendBlockRequest): return out(rt().append_block(req))
    @app.get("/ledger/summary")
    def ledger_summary(validate: bool = False):
        led = rt().require_ledger(); payload = {"ok": True, "summary": led.summary() if hasattr(led,"summary") else js(led)}
        if validate and hasattr(led,"validate_chain"): payload["validation"] = led.validate_chain(verify_tensor_commitments=False, verify_signatures=False)
        return out(payload)
    @app.post("/tensor/compression")
    def tensor(req: CompressionRequest): return out(rt().tensor_report(req))
    @app.get("/validators/votes")
    def votes():
        led = rt().ledger; head = None
        if led and getattr(led,"blocks",None):
            try: head = led.blocks[-1].to_dict()
            except Exception: head = js(led.blocks[-1])
        return out({"ok": True, "head_block_consensus": None if not isinstance(head, Mapping) else head.get("consensus_certificate"), "note": "Use tb_consensus.py for strict quorum generation/verification."})
    @app.post("/artifacts/export")
    def export(req: ExportArtifactsRequest): return out(rt().export(req))
    return app

try:
    app = create_app(ApiSettings.from_env()) if FASTAPI_OK else None
except Exception:
    app = None


def run_self_test() -> Dict[str, Any]:
    if not FASTAPI_OK: raise RuntimeError(FASTAPI_ERROR)
    from fastapi.testclient import TestClient
    a = create_app(ApiSettings(chain_id="project-chimera-api-selftest", artifact_root="artifacts/tb_api_selftest_artifacts", enable_lobes=False, cors_origins=["*"]))
    c = TestClient(a)
    assert c.get("/health").status_code == 200
    assert c.get("/scenarios").status_code == 200
    tw = c.post("/twins", json={"thing_name":"API Selftest Robot","thing_type":"robot","observables":{"status":"nominal"}}); assert tw.status_code == 200, tw.text
    txr = c.post("/transactions/create", json={"sender_id":"selftest","sender_sequence":1,"action":{"kind":"update_node_data","node_id":"5.4","payload":{"temperature_c":42.0},"note":"selftest"}}); assert txr.status_code == 200, txr.text
    tx = txr.json()["transaction"]
    sub = c.post("/transactions/submit", json={"transaction": tx}); assert sub.status_code == 200, sub.text
    ex = c.post("/execute", json={"transaction": tx, "run_lobes": False, "strict": False, "require_valid_transaction": False}); assert ex.status_code == 200, ex.text
    lg = c.get("/ledger/summary"); assert lg.status_code == 200, lg.text
    ep = c.post("/artifacts/export", json={"export_id":"selftest","include_lobe_snapshot":False}); assert ep.status_code == 200, ep.text
    return {"ok": True, "version": TB_API_VERSION, "execution_hash": ex.json().get("execution_hash"), "export_root": ep.json().get("export_root")}


def status_report() -> Dict[str, Any]:
    return {"ok": FASTAPI_OK, "version": TB_API_VERSION, "fastapi_error": FASTAPI_ERROR, "paths": configure_paths(), "integrations": TBApiRuntime(ApiSettings(enable_lobes=False)).integrations()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Tensorized Blockchain API")
    p.add_argument("--mode", choices=["status","self-test","serve"], default="status")
    p.add_argument("--host", default=os.getenv("TB_API_HOST", "127.0.0.1")); p.add_argument("--port", type=int, default=int(os.getenv("TB_API_PORT", "8000"))); p.add_argument("--reload", action="store_true")
    args = p.parse_args(argv)
    try:
        if args.mode == "self-test": print(json.dumps(run_self_test(), indent=2)); return 0
        if args.mode == "serve":
            import uvicorn
            uvicorn.run("tb_api:app", host=args.host, port=args.port, reload=args.reload); return 0
        print(json.dumps(status_report(), indent=2, ensure_ascii=False)); return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()}, indent=2), file=sys.stderr); return 1

if __name__ == "__main__": raise SystemExit(main())
