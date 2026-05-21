# Release Notes - Tensorized Blockchain v1.0-alpha

We are pleased to present the initial open-source AI heritage research release of the **Tensorized Blockchain** stack.

## What is in this Release
This release includes the complete, core python script stack (28 scripts total) defining a governed digital-twin ledger:
* **Low-Level Tensor Train (TT/MPO/Tucker) Runtime**: `tn.py`
* **Digital Twin Governance Engine**: `digital_twin_kernel.py`, `twin_anything.py`
* **Ledger Compilation & MPS Kernel**: `tensorized_blockchain_kernel.py`
* **Dual Cognitive Lobes**: NCP (Liquid Reflex) and CfC (Predictive Transition) substrates (`ncp.py`, `cfc.py`, `cognitive_lobe_runtime.py`, `lobe_scaler.py`)
* **Consensus, Mempool & Cryptography**: `tb_consensus.py`, `tb_mempool.py`, `tb_crypto.py`, `tb_hashing.py`, `tb_schemas.py`, `tb_validation.py`, `tb_storage.py`, `tb_execution.py`, `tb_security.py`
* **CLI, API & Demo Dashboard**: `tb_cli.py`, `tb_api.py`, `app.py`
* **Verification & Audit Tooling**: `tb_open_source_audit.py`, `tb_benchmarks.py`, `tb_replay.py`

## Important Disclaimers and Limitations
* **Research Release**: This stack is presented as production-oriented and research-release code, but is NOT yet externally audited for production use.
* **Streamlit Demo App**: Included as `app.py`, clearly marked as an experimental / demo cockpit UI.
* **No Formal Tests Included**: A formal test suite is excluded from this initial repository structure. Do not invent mock tests.
* **Consensus Grade**: The Proof-of-Coherence consensus protocol is research-grade and requires extensive deployment-specific validation.
