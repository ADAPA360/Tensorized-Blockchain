# Demo Guide - Tensorized Blockchain

This guide provides instructions to run the various demonstration paths in the Tensorized Blockchain stack.

---

## 1. CLI Demonstration Path
The CLI allows you to run a deterministic Alicante Fire Department rescue robot scenario:

```powershell
# Run the demo and export the ledger JSON
python tb_cli.py run-demo --output artifacts/demo_ledger.json --pretty

# Validate the generated ledger file
python tb_cli.py validate-chain --ledger artifacts/demo_ledger.json

# Inspect the compiled Tensor Train commitment inside the ledger
python tb_cli.py inspect-tensor --ledger artifacts/demo_ledger.json --pretty
```

## 2. Streamlit Dashboard Cockpit
To view the interactive cockpit dashboard:

```powershell
streamlit run app.py
```
This launches a browser page at `http://localhost:8501`. Use the sidebar to step through the story:
1. Initialize the Chain.
2. Submit a governed telemetry transaction.
3. Forge a cryptographic block.
4. Visualize the validator quorum.
5. Trigger SVD rounding compression over the ledger cores.
6. Audit dual brain lobe outputs.

## 3. REST API Server Path
You can run a local FastAPI backend to interact with the ledger programmatically:

```powershell
# Start the server
uvicorn tb_api:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser to `http://127.0.0.1:8000/docs` to view the interactive Swagger API documentation page.

## Expected Artifacts
Running the demo scenario creates the following artifacts in the `artifacts/` folder:
* `artifacts/demo_ledger.json`: The complete serialized Matrix Product State blockchain log containing keys, headers, block structures, signatures, and committed tensor core values.

## Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit'"
Streamlit is an optional dependency but required to launch the interactive dashboard. Install it via:
```powershell
pip install streamlit
```

### "ModuleNotFoundError: No module named 'fastapi'" or "uvicorn"
The REST API requires both FastAPI and Uvicorn. Install them via:
```powershell
pip install fastapi uvicorn
```

### "ModuleNotFoundError: No module named 'cryptography'"
Required for signature validation. Install it via:
```powershell
pip install cryptography
```

### Tensor Compression / SVD Errors
If you run into dimension mismatch or execution errors during tensor compilation or compression:
1. Run `python tb_cli.py status --pretty` to make sure `tn.py` and `tensorized_blockchain_kernel.py` are loaded cleanly from the local folder.
2. If memory or rounding errors persist, reduce the matrix dimensions in your command parameters (e.g., set `--out-dim 4 --in-dim 4 --max-bond-dim 2`).
