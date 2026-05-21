#!/usr/bin/env python3
# tb_cli.py
r"""
Project Chimera / Akkurat - Tensorized Blockchain CLI
=====================================================

Production command-line boundary for the Tensorized Blockchain stack.

Responsibilities
----------------
- status
- init-chain
- create-twin
- submit-tx
- run-demo
- validate-chain
- export-ledger
- inspect-block
- inspect-tensor
- run-self-tests

Design notes
------------
This file is intentionally a thin orchestration boundary. It does not own
ledger semantics, twin semantics, tensor algebra, cryptography, or persistence
formats. It imports the production modules when available and reports degraded
status rather than hiding missing dependencies.

The CLI uses JSON for all machine-readable output. Human-readable terminal
formatting is deliberately minimal so the output can be redirected into demo
apps, CI jobs, and artifact pipelines.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib
import json
import os
import platform
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


# =============================================================================
# Version / constants
# =============================================================================

TB_CLI_VERSION = "tb-cli-v1.0-production"
DEFAULT_CHAIN_ID = "tensorized-blockchain-local-v1"
DEFAULT_LEDGER_PATH = "./artifacts/ledger.json"
DEFAULT_TWIN_PATH = "./artifacts/twin_snapshot.json"
DEFAULT_TX_PATH = "./artifacts/transaction.json"
DEFAULT_EXPORT_PATH = "./artifacts/exported_ledger.json"


# =============================================================================
# Import/path helpers
# =============================================================================

def _module_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()


def _add_path(path: Union[str, os.PathLike[str], None], *, prepend: bool = True) -> Optional[Path]:
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


def configure_paths(project_root: Optional[Union[str, os.PathLike[str]]] = None) -> Dict[str, Optional[str]]:
    here = _module_dir()
    candidates: List[Union[str, os.PathLike[str], None]] = [
        project_root,
        os.environ.get("TB_ROOT"),
        os.environ.get("TENSORIZED_BLOCKCHAIN_ROOT"),
        os.environ.get("AKKURAT_ROOT"),
        os.environ.get("LK20_ROOT"),
        here,
        here.parent,
        Path.cwd(),
        Path(tempfile.gettempdir()),
    ]
    found: Dict[str, Optional[str]] = {"module_dir": str(here), "project_root": None if project_root is None else str(Path(project_root).expanduser())}
    for candidate in candidates:
        p = _add_path(candidate, prepend=True)
        if p is None:
            continue
        for name in (
            "tn.py",
            "digital_twin_kernel.py",
            "twin_anything.py",
            "tensorized_blockchain_kernel.py",
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
            "ncp.py",
            "cfc.py",
            "tb_api.py",
        ):
            key = name[:-3] + "_root"
            if found.get(key) is None and (p / name).exists():
                found[key] = str(p)
    _add_path(here, prepend=True)
    return found


_PATHS = configure_paths()


def _import_optional(module_name: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:
        return None, repr(exc)


# =============================================================================
# JSON / IO helpers
# =============================================================================

def _json_safe(obj: Any) -> Any:
    # Prefer the shared canonical serializer when present.
    try:
        tb_canonical, _ = _import_optional("tb_canonical")
        if tb_canonical is not None and hasattr(tb_canonical, "json_safe"):
            return tb_canonical.json_safe(obj)
    except Exception:
        pass
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else str(obj)
    if isinstance(obj, bytes):
        return {"__bytes_hex__": obj.hex()}
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return _json_safe(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "payload") and callable(getattr(obj, "payload")):
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


def _canonical_dumps(obj: Any, *, pretty: bool = False) -> str:
    return json.dumps(_json_safe(obj), indent=2 if pretty else None, sort_keys=not pretty, ensure_ascii=False, separators=None if pretty else (",", ":"))


def _sha256_hex_obj(obj: Any) -> str:
    return hashlib.sha256(_canonical_dumps(obj, pretty=False).encode("utf-8", errors="strict")).hexdigest()


def _atomic_write_text(path: Union[str, os.PathLike[str]], text: str, *, overwrite: bool = True) -> None:
    p = Path(path).expanduser()
    if p.exists() and not overwrite:
        raise FileExistsError(str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def _write_json(path: Union[str, os.PathLike[str]], payload: Any, *, pretty: bool = True, overwrite: bool = True) -> str:
    p = Path(path).expanduser()
    _atomic_write_text(p, _canonical_dumps(payload, pretty=pretty) + "\n", overwrite=overwrite)
    return str(p.resolve())


def _read_json(path: Union[str, os.PathLike[str]]) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _print_json(payload: Any, *, pretty: bool = False) -> None:
    print(_canonical_dumps(payload, pretty=pretty))


def _parse_json_arg(value: str, *, default: Any = None) -> Any:
    if value is None or str(value).strip() == "":
        return copy.deepcopy(default)
    s = str(value).strip()
    if s.startswith("@"):
        return _read_json(s[1:])
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        p = Path(s).expanduser()
        if p.exists():
            return _read_json(p)
        raise


def _status_for_module(module_name: str) -> Dict[str, Any]:
    mod, err = _import_optional(module_name)
    return {
        "available": mod is not None,
        "module": getattr(mod, "__file__", None) if mod is not None else None,
        "error": err,
        "version_attrs": {k: getattr(mod, k) for k in dir(mod) if k.endswith("VERSION")} if mod is not None else {},
    }


def _load_keypair(path: str = "", private_key_hex: str = "") -> Dict[str, str]:
    if private_key_hex:
        tb_crypto, err = _import_optional("tb_crypto")
        if tb_crypto is None:
            raise RuntimeError(f"tb_crypto unavailable: {err}")
        pub = tb_crypto.derive_public_key_hex(private_key_hex)
        return {"private_key_hex": tb_crypto.normalize_private_key_hex(private_key_hex), "public_key_hex": pub, "fingerprint": tb_crypto.key_fingerprint(pub)}
    if not path:
        raise ValueError("A private key is required. Pass --private-key-hex or --keypair-json.")
    payload = _read_json(path)
    if "private_key_hex" not in payload:
        raise ValueError("keypair JSON must contain private_key_hex")
    tb_crypto, err = _import_optional("tb_crypto")
    if tb_crypto is not None and hasattr(tb_crypto, "Ed25519KeyPair"):
        kp = tb_crypto.Ed25519KeyPair.from_private_json(payload)
        return {"private_key_hex": kp.private_key_hex, "public_key_hex": kp.public_key_hex, "fingerprint": kp.fingerprint}
    private = str(payload["private_key_hex"])
    public = str(payload.get("public_key_hex", ""))
    return {"private_key_hex": private, "public_key_hex": public, "fingerprint": str(payload.get("fingerprint", ""))}


# =============================================================================
# Command handlers
# =============================================================================

def cmd_status(args: argparse.Namespace) -> Dict[str, Any]:
    configure_paths(args.project_root or None)
    modules = [
        "tb_config", "tb_canonical", "tb_hashing", "tb_crypto", "tb_schemas", "tb_validation", "tb_storage",
        "tn", "digital_twin_kernel", "twin_anything", "tensorized_blockchain_kernel",
        "ncp", "cfc", "cognitive_lobe_runtime", "lobe_scaler", "tb_tensor_commitments",
        "tb_mempool", "tb_consensus", "tb_execution", "tb_demo_scenarios", "tb_api",
    ]
    status = {m: _status_for_module(m) for m in modules}
    ok = all(status[m]["available"] for m in ("tn", "digital_twin_kernel", "twin_anything", "tensorized_blockchain_kernel", "tb_crypto"))
    return {
        "ok": bool(ok),
        "version": TB_CLI_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "paths": configure_paths(args.project_root or None),
        "modules": status,
    }


def _ledger_config_from_args(args: argparse.Namespace) -> Any:
    tbk, err = _import_optional("tensorized_blockchain_kernel")
    if tbk is None:
        raise RuntimeError(f"tensorized_blockchain_kernel unavailable: {err}")
    cfg = tbk.LedgerConfig(
        chain_id=str(args.chain_id or DEFAULT_CHAIN_ID),
        out_dim=int(args.out_dim),
        in_dim=int(args.in_dim),
        max_bond_dim=int(args.max_bond_dim),
        seed=int(args.seed),
        require_signatures=bool(args.require_signatures),
        compress_on_append=not bool(args.no_compress_on_append),
    )
    return cfg.normalized() if hasattr(cfg, "normalized") else cfg


def cmd_init_chain(args: argparse.Namespace) -> Dict[str, Any]:
    configure_paths(args.project_root or None)
    tbk, err = _import_optional("tensorized_blockchain_kernel")
    if tbk is None:
        raise RuntimeError(f"tensorized_blockchain_kernel unavailable: {err}")
    cfg = _ledger_config_from_args(args)
    ledger = tbk.LedgerMPS(cfg)
    out = Path(args.ledger or DEFAULT_LEDGER_PATH).expanduser()
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"ledger already exists: {out}")
    ledger.save_json(out, include_compressed_cores=bool(args.include_compressed_cores))
    return {
        "ok": True,
        "command": "init-chain",
        "ledger_path": str(out.resolve()),
        "chain_id": cfg.chain_id,
        "summary": ledger.summary(),
    }


def cmd_keygen(args: argparse.Namespace) -> Dict[str, Any]:
    configure_paths(args.project_root or None)
    tb_crypto, err = _import_optional("tb_crypto")
    if tb_crypto is None:
        raise RuntimeError(f"tb_crypto unavailable: {err}")
    kp = tb_crypto.generate_ed25519_keypair(seed=args.seed_value if args.deterministic else None, label=args.label, metadata={"generated_by": TB_CLI_VERSION})
    payload = kp.to_private_json(include_private=not bool(args.public_only))
    if args.output:
        path = _write_json(args.output, payload, pretty=True, overwrite=bool(args.overwrite))
        return {"ok": True, "command": "keygen", "saved": path, "public_identity": kp.public_identity().to_dict()}
    return {"ok": True, "command": "keygen", "keypair": payload if not args.public_only else kp.public_identity().to_dict()}


def cmd_create_twin(args: argparse.Namespace) -> Dict[str, Any]:
    configure_paths(args.project_root or None)
    ta, err = _import_optional("twin_anything")
    if ta is None:
        raise RuntimeError(f"twin_anything unavailable: {err}")
    if args.input_json:
        payload = _read_json(args.input_json)
        if not isinstance(payload, Mapping):
            raise ValueError("--input-json must contain a JSON object")
        kwargs = dict(payload)
    else:
        kwargs = {
            "thing_name": args.thing_name,
            "thing_type": args.thing_type,
            "description": args.description,
            "observables": _parse_json_arg(args.observables_json, default={}) if args.observables_json else {},
            "components": _parse_json_arg(args.components_json, default={}) if args.components_json else {},
            "environment": _parse_json_arg(args.environment_json, default={}) if args.environment_json else {},
            "success_criteria": _parse_json_arg(args.success_json, default={}) if args.success_json else {},
            "tags": [x.strip() for x in str(args.tags or "").split(",") if x.strip()],
        }
    factory = ta.TwinAnythingFactory(
        vector_dim=int(args.vector_dim),
        sketch_dim=min(int(args.sketch_dim), int(args.vector_dim)),
        history_capacity=int(args.history_capacity),
        seed=int(args.seed),
        deterministic=bool(args.deterministic),
        consensus_safe=not bool(args.not_consensus_safe),
        use_tn_projection=not bool(args.disable_tn_projection),
    )
    snapshot = factory.create_twin_snapshot(**kwargs, include_network_snapshot=not bool(args.no_network_snapshot))
    out = Path(args.output or DEFAULT_TWIN_PATH).expanduser()
    saved = _write_json(out, snapshot, pretty=True, overwrite=bool(args.overwrite))
    return {"ok": True, "command": "create-twin", "saved": saved, "summary": {"thing_name": kwargs.get("thing_name"), "thing_type": kwargs.get("thing_type"), "receipt_hash": snapshot.get("creation_receipt", {}).get("receipt_hash")}}


def _build_action(action_kind: str, twin_node_id: str, action_payload: Any, note: str) -> Any:
    dtk, err = _import_optional("digital_twin_kernel")
    if dtk is None:
        raise RuntimeError(f"digital_twin_kernel unavailable: {err}")
    return dtk.Action(str(action_kind), node_id=str(twin_node_id), payload=action_payload, note=str(note or ""))


def cmd_submit_tx(args: argparse.Namespace) -> Dict[str, Any]:
    configure_paths(args.project_root or None)
    tbk, err = _import_optional("tensorized_blockchain_kernel")
    if tbk is None:
        raise RuntimeError(f"tensorized_blockchain_kernel unavailable: {err}")
    ledger = tbk.LedgerMPS.load_json(args.ledger, validate=not bool(args.unsafe_load)) if args.ledger and Path(args.ledger).exists() else None
    chain_id = str(args.chain_id or (ledger.cfg.chain_id if ledger is not None else DEFAULT_CHAIN_ID))
    if args.action_json:
        action_payload_obj = _read_json(args.action_json) if Path(args.action_json).expanduser().exists() else _parse_json_arg(args.action_json, default={})
        if not isinstance(action_payload_obj, Mapping):
            raise ValueError("--action-json must contain a JSON object")
        action_kind = str(action_payload_obj.get("kind", args.action_kind))
        twin_node_id = str(action_payload_obj.get("node_id", args.twin_node_id))
        payload = action_payload_obj.get("payload", {})
        note = str(action_payload_obj.get("note", args.note))
    else:
        action_kind = str(args.action_kind)
        twin_node_id = str(args.twin_node_id)
        payload = _parse_json_arg(args.action_payload_json, default={}) if args.action_payload_json else {}
        note = str(args.note or "")
    keypair = _load_keypair(path=args.keypair_json or "", private_key_hex=args.private_key_hex or "") if (args.keypair_json or args.private_key_hex) else {"private_key_hex": "", "public_key_hex": args.sender_public_key or ""}
    sender_sequence = int(args.sender_sequence)
    if sender_sequence <= 0 and ledger is not None:
        identity_key = f"pub:{keypair.get('public_key_hex')}" if keypair.get("public_key_hex") else f"sender:{args.sender_id}"
        sender_sequence = int(ledger.account_sequences.get(identity_key, 0)) + 1
    if sender_sequence <= 0:
        sender_sequence = 1
    tx = tbk.TensorTransaction.new(
        sender_id=str(args.sender_id),
        sender_public_key=str(keypair.get("public_key_hex") or args.sender_public_key or ""),
        sender_sequence=sender_sequence,
        chain_id=chain_id,
        twin_node_id=twin_node_id,
        action=_build_action(action_kind, twin_node_id, payload, note),
        timestamp=float(args.timestamp),
        nonce=str(args.nonce or f"cli-{chain_id}-{args.sender_id}-{sender_sequence}"),
        metadata={"created_by": TB_CLI_VERSION, **(_parse_json_arg(args.metadata_json, default={}) if args.metadata_json else {})},
    )
    if keypair.get("private_key_hex"):
        tx.sign(keypair["private_key_hex"])
    elif args.unsigned:
        tx.signature_scheme = "unsigned"
    else:
        raise ValueError("Transaction signing required unless --unsigned is explicitly supplied.")

    tx_payload = tx.to_dict()
    result: Dict[str, Any] = {"ok": True, "command": "submit-tx", "transaction": tx_payload, "tx_hash": tx.payload_hash()}

    if args.tx_output:
        result["saved_tx"] = _write_json(args.tx_output, tx_payload, pretty=True, overwrite=bool(args.overwrite))

    if args.append_direct:
        if ledger is None:
            raise ValueError("--append-direct requires --ledger pointing to an existing ledger")
        if not args.pre_state_root or not args.post_state_root:
            raise ValueError("--append-direct requires --pre-state-root and --post-state-root")
        block = ledger.append_block([tx], pre_state_root=args.pre_state_root, post_state_root=args.post_state_root, metadata={"appended_by": TB_CLI_VERSION, "direct_append": True})
        ledger.save_json(args.ledger)
        result["appended_block"] = block.summary()
        result["ledger_saved"] = str(Path(args.ledger).expanduser().resolve())
    return result


def cmd_run_demo(args: argparse.Namespace) -> Dict[str, Any]:
    configure_paths(args.project_root or None)
    if args.fixture_only:
        demo, err = _import_optional("tb_demo_scenarios")
        if demo is None:
            raise RuntimeError(f"tb_demo_scenarios unavailable: {err}")
        if args.output:
            saved = demo.export_scenario_json(args.scenario, args.output, chain_id=args.chain_id or DEFAULT_CHAIN_ID, seed=int(args.seed), overwrite=bool(args.overwrite))
            return {"ok": True, "command": "run-demo", "mode": "fixture-only", "saved": _json_safe(saved)}
        fixture = demo.build_scenario(args.scenario, chain_id=args.chain_id or DEFAULT_CHAIN_ID, seed=int(args.seed))
        return {"ok": True, "command": "run-demo", "mode": "fixture-only", "fixture": fixture.to_dict() if hasattr(fixture, "to_dict") else _json_safe(fixture)}
    tbk, err = _import_optional("tensorized_blockchain_kernel")
    if tbk is None:
        raise RuntimeError(f"tensorized_blockchain_kernel unavailable: {err}")
    report = tbk.run_wiser_demo(output=args.output or None)
    return {"ok": bool(report.get("ok", True)), "command": "run-demo", "report": report}


def cmd_validate_chain(args: argparse.Namespace) -> Dict[str, Any]:
    configure_paths(args.project_root or None)
    tbk, err = _import_optional("tensorized_blockchain_kernel")
    if tbk is None:
        raise RuntimeError(f"tensorized_blockchain_kernel unavailable: {err}")
    ledger = tbk.LedgerMPS.load_json(args.ledger, validate=False)
    check = ledger.validate_chain(verify_tensor_commitments=not bool(args.skip_tensor), verify_signatures=bool(args.verify_signatures))
    return {"ok": bool(check.get("ok", False)), "command": "validate-chain", "ledger": str(Path(args.ledger).expanduser().resolve()), "validation": check}


def cmd_export_ledger(args: argparse.Namespace) -> Dict[str, Any]:
    configure_paths(args.project_root or None)
    tbk, err = _import_optional("tensorized_blockchain_kernel")
    if tbk is None:
        raise RuntimeError(f"tensorized_blockchain_kernel unavailable: {err}")
    ledger = tbk.LedgerMPS.load_json(args.ledger, validate=not bool(args.unsafe_load))
    payload = ledger.to_dict(include_compressed_cores=bool(args.include_compressed_cores))
    out = args.output or DEFAULT_EXPORT_PATH
    saved = _write_json(out, payload, pretty=True, overwrite=bool(args.overwrite))
    return {"ok": True, "command": "export-ledger", "saved": saved, "summary": ledger.summary(), "payload_hash": _sha256_hex_obj(payload)}


def cmd_inspect_block(args: argparse.Namespace) -> Dict[str, Any]:
    configure_paths(args.project_root or None)
    tbk, err = _import_optional("tensorized_blockchain_kernel")
    if tbk is None:
        raise RuntimeError(f"tensorized_blockchain_kernel unavailable: {err}")
    ledger = tbk.LedgerMPS.load_json(args.ledger, validate=not bool(args.unsafe_load))
    block = None
    if args.hash:
        h = str(args.hash).lower().strip()
        for b in ledger.blocks:
            if b.block_hash.lower() == h or b.block_hash.lower().startswith(h):
                block = b
                break
    else:
        idx = int(args.index)
        if idx < 0:
            idx = len(ledger.blocks) + idx
        if 0 <= idx < len(ledger.blocks):
            block = ledger.blocks[idx]
    if block is None:
        raise IndexError("block not found")
    payload = block.to_dict() if args.full else block.summary()
    if args.proof_tx_index is not None:
        payload = {"block": payload, "merkle_proof": block.transaction_proof(int(args.proof_tx_index)).to_dict()}
    return {"ok": True, "command": "inspect-block", "payload": payload}


def cmd_inspect_tensor(args: argparse.Namespace) -> Dict[str, Any]:
    configure_paths(args.project_root or None)
    tbk, err = _import_optional("tensorized_blockchain_kernel")
    if tbk is None:
        raise RuntimeError(f"tensorized_blockchain_kernel unavailable: {err}")
    ledger = tbk.LedgerMPS.load_json(args.ledger, validate=not bool(args.unsafe_load))
    tt = ledger.get_tensor_train(use_compressed=not bool(args.uncompressed))
    commitment = tt.commitment().to_dict() if hasattr(tt, "commitment") else {}
    payload: Dict[str, Any] = {
        "summary": ledger.summary(),
        "tensor_train": tt.describe() if hasattr(tt, "describe") else {},
        "tensor_commitment": commitment,
        "tensor_commitment_hash": getattr(tt.commitment(), "commitment_hash", "") if hasattr(tt, "commitment") else "",
        "last_compression_report": getattr(ledger, "_last_compression_report", {}),
    }
    if args.estimate_norm:
        payload["operator_norm_estimate"] = float(tt.estimate_operator_norm_power(iters=int(args.norm_iters), seed=int(args.seed))) if hasattr(tt, "estimate_operator_norm_power") else None
    return {"ok": True, "command": "inspect-tensor", "payload": payload}


def cmd_run_self_tests(args: argparse.Namespace) -> Dict[str, Any]:
    configure_paths(args.project_root or None)
    results: Dict[str, Any] = {}

    # Internal CLI smoke test first.
    with tempfile.TemporaryDirectory(prefix="tb_cli_selftest_") as td:
        root = Path(td)
        key_path = root / "key.json"
        ledger_path = root / "ledger.json"
        tx_path = root / "tx.json"
        twin_path = root / "twin.json"

        kg_args = argparse.Namespace(project_root=args.project_root, deterministic=True, seed_value="tb-cli-self-test", label="cli_self_test", output=str(key_path), public_only=False, overwrite=True)
        kg = cmd_keygen(kg_args)
        init_args = argparse.Namespace(project_root=args.project_root, ledger=str(ledger_path), chain_id="tb-cli-self-test-chain", out_dim=4, in_dim=4, max_bond_dim=2, seed=13, require_signatures=True, no_compress_on_append=False, include_compressed_cores=False, overwrite=True)
        init = cmd_init_chain(init_args)
        submit_args = argparse.Namespace(
            project_root=args.project_root,
            ledger=str(ledger_path),
            chain_id="",
            action_json="",
            action_kind="update_node_data",
            twin_node_id="5.4",
            action_payload_json='{"temperature_c": 33.0, "status": "cli_self_test"}',
            note="cli_self_test",
            keypair_json=str(key_path),
            private_key_hex="",
            sender_public_key="",
            sender_id="cli_self_test_agent",
            sender_sequence=1,
            timestamp=1.0,
            nonce="cli-self-test-1",
            metadata_json="",
            unsigned=False,
            tx_output=str(tx_path),
            overwrite=True,
            append_direct=True,
            pre_state_root="cli_pre_state_root",
            post_state_root="cli_post_state_root",
            unsafe_load=False,
        )
        tx = cmd_submit_tx(submit_args)
        val_args = argparse.Namespace(project_root=args.project_root, ledger=str(ledger_path), skip_tensor=False, verify_signatures=True)
        val = cmd_validate_chain(val_args)
        twin_args = argparse.Namespace(project_root=args.project_root, input_json="", thing_name="CLI Self Test Twin", thing_type="industrial_machine", description="Self-test twin", observables_json='{"status":"ok"}', components_json="{}", environment_json="{}", success_json="{}", tags="selftest", vector_dim=16, sketch_dim=16, history_capacity=8, seed=13, deterministic=True, not_consensus_safe=False, disable_tn_projection=True, no_network_snapshot=True, output=str(twin_path), overwrite=True)
        twin = cmd_create_twin(twin_args)
        results["tb_cli_internal"] = {"ok": bool(val.get("ok")), "keygen": kg.get("ok"), "init": init.get("ok"), "submit": tx.get("ok"), "validate": val.get("validation", {}), "twin": twin.get("ok")}
        if not bool(val.get("ok")):
            raise AssertionError(f"internal CLI validation failed: {val}")

    if args.fast:
        return {"ok": True, "command": "run-self-tests", "results": results, "mode": "fast"}

    # Run selected module self-tests when available. Keep this bounded and honest.
    test_modules = [
        "tb_hashing", "tb_crypto", "tb_schemas", "tb_validation", "tb_storage", "tb_mempool",
        "tb_consensus", "tb_execution", "tb_demo_scenarios", "tb_tensor_commitments",
        "lobe_scaler", "cognitive_lobe_runtime", "tensorized_blockchain_kernel",
    ]
    for name in test_modules:
        mod, err = _import_optional(name)
        if mod is None:
            results[name] = {"ok": False, "skipped": True, "error": err}
            continue
        fn = getattr(mod, "run_self_test", None) or getattr(mod, "_self_test", None)
        if not callable(fn):
            results[name] = {"ok": True, "skipped": True, "reason": "no self-test function"}
            continue
        try:
            res = fn()
            results[name] = _json_safe(res if res is not None else {"ok": True})
            if isinstance(results[name], Mapping) and results[name].get("ok") is False:
                results[name]["warning"] = "module self-test returned ok=false"
        except Exception as exc:
            results[name] = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc() if args.tracebacks else ""}
            if args.strict:
                raise
    overall_ok = bool(results.get("tb_cli_internal", {}).get("ok", False)) and all(bool(v.get("ok", True)) or bool(v.get("skipped", False)) for v in results.values() if isinstance(v, Mapping))
    return {"ok": overall_ok, "command": "run-self-tests", "results": results, "mode": "full"}


# =============================================================================
# Argument parser
# =============================================================================

def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", default="", help="Optional project root to add to PYTHONPATH.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tensorized Blockchain production CLI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", action="version", version=TB_CLI_VERSION)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("status", help="Report stack status and dependency availability.")
    _common(s)
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("keygen", help="Generate an Ed25519 keypair for demos/tests/operators.")
    _common(s)
    s.add_argument("--output", default="", help="Optional keypair JSON output path.")
    s.add_argument("--label", default="", help="Key label.")
    s.add_argument("--deterministic", action="store_true", help="Use deterministic seed; for demos/tests only.")
    s.add_argument("--seed-value", default="tb-cli-demo-key", help="Deterministic seed value.")
    s.add_argument("--public-only", action="store_true", help="Do not print/save the private key.")
    s.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")
    s.set_defaults(func=cmd_keygen)

    s = sub.add_parser("init-chain", help="Initialize a new ledger JSON file.")
    _common(s)
    s.add_argument("--ledger", default=DEFAULT_LEDGER_PATH)
    s.add_argument("--chain-id", default=DEFAULT_CHAIN_ID)
    s.add_argument("--out-dim", type=int, default=8)
    s.add_argument("--in-dim", type=int, default=8)
    s.add_argument("--max-bond-dim", type=int, default=4)
    s.add_argument("--seed", type=int, default=2027)
    s.add_argument("--require-signatures", action="store_true")
    s.add_argument("--no-compress-on-append", action="store_true")
    s.add_argument("--include-compressed-cores", action="store_true")
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_init_chain)

    s = sub.add_parser("create-twin", help="Create a deterministic serializable twin snapshot.")
    _common(s)
    s.add_argument("--input-json", default="", help="JSON object with TwinAnythingFactory kwargs.")
    s.add_argument("--thing-name", default="Unnamed Thing")
    s.add_argument("--thing-type", default="generic")
    s.add_argument("--description", default="")
    s.add_argument("--observables-json", default="")
    s.add_argument("--components-json", default="")
    s.add_argument("--environment-json", default="")
    s.add_argument("--success-json", default="")
    s.add_argument("--tags", default="")
    s.add_argument("--vector-dim", type=int, default=256)
    s.add_argument("--sketch-dim", type=int, default=96)
    s.add_argument("--history-capacity", type=int, default=64)
    s.add_argument("--seed", type=int, default=2027)
    s.add_argument("--deterministic", action="store_true")
    s.add_argument("--not-consensus-safe", action="store_true")
    s.add_argument("--disable-tn-projection", action="store_true")
    s.add_argument("--no-network-snapshot", action="store_true")
    s.add_argument("--output", default=DEFAULT_TWIN_PATH)
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_create_twin)

    s = sub.add_parser("submit-tx", help="Create/sign a transaction and optionally append it directly to a ledger.")
    _common(s)
    s.add_argument("--ledger", default=DEFAULT_LEDGER_PATH)
    s.add_argument("--chain-id", default="", help="Override chain ID; default reads from ledger.")
    s.add_argument("--sender-id", required=True)
    s.add_argument("--sender-sequence", type=int, default=0, help="0 means infer next sequence from ledger if available.")
    s.add_argument("--sender-public-key", default="")
    s.add_argument("--private-key-hex", default="")
    s.add_argument("--keypair-json", default="")
    s.add_argument("--unsigned", action="store_true", help="Explicitly create an unsigned transaction.")
    s.add_argument("--twin-node-id", default="5.4")
    s.add_argument("--action-kind", default="update_node_data")
    s.add_argument("--action-payload-json", default="{}")
    s.add_argument("--action-json", default="", help="JSON action object or path with kind/node_id/payload/note.")
    s.add_argument("--note", default="")
    s.add_argument("--timestamp", type=float, default=0.0)
    s.add_argument("--nonce", default="")
    s.add_argument("--metadata-json", default="")
    s.add_argument("--tx-output", default=DEFAULT_TX_PATH)
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--append-direct", action="store_true", help="Append directly using supplied pre/post roots; bypasses governed execution.")
    s.add_argument("--pre-state-root", default="")
    s.add_argument("--post-state-root", default="")
    s.add_argument("--unsafe-load", action="store_true")
    s.set_defaults(func=cmd_submit_tx)

    s = sub.add_parser("run-demo", help="Run the canonical Alicante demo or export deterministic fixtures.")
    _common(s)
    s.add_argument("--output", default="")
    s.add_argument("--scenario", default="alicante_fire_department_robot")
    s.add_argument("--chain-id", default=DEFAULT_CHAIN_ID)
    s.add_argument("--seed", type=int, default=2027)
    s.add_argument("--fixture-only", action="store_true")
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_run_demo)

    s = sub.add_parser("validate-chain", help="Validate a ledger JSON file.")
    _common(s)
    s.add_argument("--ledger", default=DEFAULT_LEDGER_PATH)
    s.add_argument("--skip-tensor", action="store_true")
    s.add_argument("--verify-signatures", action="store_true")
    s.set_defaults(func=cmd_validate_chain)

    s = sub.add_parser("export-ledger", help="Export a ledger JSON payload.")
    _common(s)
    s.add_argument("--ledger", default=DEFAULT_LEDGER_PATH)
    s.add_argument("--output", default=DEFAULT_EXPORT_PATH)
    s.add_argument("--include-compressed-cores", action="store_true")
    s.add_argument("--unsafe-load", action="store_true")
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_export_ledger)

    s = sub.add_parser("inspect-block", help="Inspect one block by index or hash prefix.")
    _common(s)
    s.add_argument("--ledger", default=DEFAULT_LEDGER_PATH)
    s.add_argument("--index", type=int, default=-1)
    s.add_argument("--hash", default="")
    s.add_argument("--full", action="store_true")
    s.add_argument("--proof-tx-index", type=int, default=None)
    s.add_argument("--unsafe-load", action="store_true")
    s.set_defaults(func=cmd_inspect_block)

    s = sub.add_parser("inspect-tensor", help="Inspect the ledger TensorTrain commitment and compression state.")
    _common(s)
    s.add_argument("--ledger", default=DEFAULT_LEDGER_PATH)
    s.add_argument("--uncompressed", action="store_true")
    s.add_argument("--estimate-norm", action="store_true")
    s.add_argument("--norm-iters", type=int, default=8)
    s.add_argument("--seed", type=int, default=2027)
    s.add_argument("--unsafe-load", action="store_true")
    s.set_defaults(func=cmd_inspect_tensor)

    s = sub.add_parser("run-self-tests", help="Run CLI and selected stack self-tests.")
    _common(s)
    s.add_argument("--fast", action="store_true", help="Only run bounded CLI smoke tests.")
    s.add_argument("--strict", action="store_true", help="Raise on module self-test failure.")
    s.add_argument("--tracebacks", action="store_true", help="Include tracebacks for failed module tests.")
    s.set_defaults(func=cmd_run_self_tests)

    return p


# =============================================================================
# Main
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        configure_paths(getattr(args, "project_root", "") or None)
        payload = args.func(args)
        _print_json(payload, pretty=bool(getattr(args, "pretty", False)))
        return 0 if bool(payload.get("ok", True)) else 2
    except Exception as exc:
        payload = {
            "ok": False,
            "command": getattr(args, "command", ""),
            "error": repr(exc),
            "traceback": traceback.format_exc() if os.getenv("TB_CLI_TRACEBACK", "").lower() in {"1", "true", "yes"} else "",
        }
        _print_json(payload, pretty=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
