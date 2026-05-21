# Tensorized Blockchain

Tensorized Blockchain is an open-source research stack that implements a governed digital-twin ledger. It combines deterministic tensor-network primitives (specifically Matrix Product States / Tensor Trains) with public-key transaction signing, cryptographic hashing, and governed state transitions to resolve ledger state bloat while maintaining a mathematically queryable runtime representation of the system history.

## Current Status
**Research Release / AI Heritage Project / Production-Oriented Stack**  
This stack is a production-oriented research release but has **not** been externally audited. Cryptographic, consensus, and tensor network implementations must be audited before production deployment.

## Architecture

```text
+-----------------------------------------------------------+
|                      CLI / API / App                      |
|           (tb_cli.py, tb_api.py, app.py [dashboard])       |
+-----------------------------------------------------------+
                              |
                              v
+-----------------------------------------------------------+
|          Governance & Cognitive Orchestration             |
|  (digital_twin_kernel.py, twin_anything.py, ncp.py, cfc.py,|
|    cognitive_lobe_runtime.py, lobe_scaler.py)             |
+-----------------------------------------------------------+
                              |
                              v
+-----------------------------------------------------------+
|          Consensus, Mempool & Execution Layers            |
|   (tb_mempool.py, tb_consensus.py, tb_execution.py,       |
|    tb_validation.py, tb_tensor_commitments.py)            |
+-----------------------------------------------------------+
                              |
                              v
+-----------------------------------------------------------+
|             State Compilation & Tensor Engine             |
|   (tensorized_blockchain_kernel.py, tn.py [TensorTrain])  |
+-----------------------------------------------------------+
                              |
                              v
+-----------------------------------------------------------+
|                 Shared Infrastructure                     |
|  (tb_config.py, tb_canonical.py, tb_hashing.py,           |
|   tb_crypto.py, tb_schemas.py, tb_storage.py,             |
|   tb_observability.py, tb_security.py)                    |
+-----------------------------------------------------------+
```

## Included Core Scripts
1. `tn.py` - Low-level deterministic TensorTrain / MPO / Tucker runtime.
2. `digital_twin_kernel.py` - Canonical governed world-state engine.
3. `twin_anything.py` - Materialization of universal object/process twins.
4. `tensorized_blockchain_kernel.py` - Ledger compiled matrix product states (MPS).
5. `ncp.py` - Liquid Reflex Lobe substrate.
6. `cfc.py` - Predictive Transition Lobe substrate.
7. `tb_config.py` - Global constants and stack configuration.
8. `tb_canonical.py` - Deterministic serialization utilities.
9. `tb_hashing.py` - Domain-separated SHA-256 and Merkle hashing.
10. `tb_crypto.py` - Ed25519 signing and verification.
11. `tb_schemas.py` - Pydantic and JSON schema validation bounds.
12. `tb_validation.py` - Strict block and ledger transaction rules.
13. `tb_storage.py` - Ledger file serialization and load checks.
14. `tb_mempool.py` - Transaction staging and ordering queue.
15. `tb_consensus.py` - Proof-of-Coherence consensus engine.
16. `tb_execution.py` - Governed execution state machine.
17. `cognitive_lobe_runtime.py` - Dual-lobe coordinating runtime.
18. `lobe_scaler.py` - Parameter capacity scaling and projection bounds.
19. `tb_tensor_commitments.py` - Deterministic tensor train commitment hash checks.
20. `tb_demo_scenarios.py` - Deterministic fixtures (e.g. Alicante search & rescue).
21. `tb_api.py` - FastAPI server layer.
22. `tb_cli.py` - Production operator CLI.
23. `tb_observability.py` - Unified performance counters and logging.
24. `tb_security.py` - Audit and verification rules.
25. `tb_benchmarks.py` - Matrix/Transaction benchmarks.
26. `tb_replay.py` - Immutable history re-execution.
27. `tb_open_source_audit.py` - Path/secrets leakage scanner.
28. `app.py` - Streamlit demo cockpit.

## Quickstart

```powershell
# 1. Set up virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify module availability
python tb_cli.py status --pretty

# 4. Run CLI self-tests
python tb_cli.py run-self-tests --pretty

# 5. Run the Alicante demo scenario and export results
python tb_cli.py run-demo --output artifacts/demo_ledger.json --pretty

# 6. Run the experimental Streamlit dashboard
streamlit run app.py
```

## CLI Commands
* `status` - Report stack status and dependency availability.
* `init-chain` - Initialize a new ledger JSON file.
* `create-twin` - Create a deterministic serializable twin snapshot.
* `submit-tx` - Create/sign a transaction and append it directly to a ledger.
* `run-demo` - Run the canonical Alicante demo or export deterministic fixtures.
* `validate-chain` - Validate a ledger JSON file.
* `export-ledger` - Export a ledger JSON payload.
* `inspect-block` - Inspect one block by index or hash prefix.
* `inspect-tensor` - Inspect the compiled Tensor Train representation.
* `run-self-tests` - Run basic validation self-tests.

## API Command

```bash
uvicorn tb_api:app --host 127.0.0.1 --port 8000 --reload
```

## Known Limitations
* **No Formal Test Suite**: No formal test suite or tests directory is included yet.
* **App Operational Hardening**: The Streamlit demo dashboard UI is an experimental cockpit and is still undergoing operational hardening.
* **Audit Pending**: The cryptography, storage format, and consensus protocols are research-grade and have not been externally audited.
* **Demo Keys Only**: Key generation and signature verification use local seeds. These keys are not safe for production.
* **Tensor commitments**: Tensorized commitments are experimental research infrastructure and must be evaluated in a deployment context.

## Safety Statement
Dual cognitive lobe outputs (NCP and CfC) are advisory substrates that generate signed action receipts. They do not replace cryptographic block forging and consensus mechanisms.

## License
Permissive MIT License. See [LICENSE](LICENSE) for details.
