# ADAPA360 Capability Adoption Protocol (CAP) — v0.1

**Status:** Research protocol / proposed integration specification, not a deployed service  
**Date:** 2026-10-17  
**Scope:** `ADAPA360/Tensorized-Blockchain` and, when explicitly selected, related ADAPA360 module variants  
**Owner of script sequence:** Human project lead; the assistant does not choose or silently advance to another module  
**Source-reviewed modules:** `tn.py`, `ncp.py`, `cfc.py`, `cognitive_lobe_runtime.py`, `lobe_scaler.py`, `digital_twin_kernel.py`, `twin_anything.py`, and `tensorized_blockchain_kernel.py`. **AST status:** Python 3.12.3 successfully parsed all 28 `.py` files (27 core modules plus `app.py`) at research revision `6e52251fdc5ced99560b07a24a69cb983f542194`; zero syntax failures. [Evidence](https://github.com/ADAPA360/Tensorized-Blockchain/actions/runs/39018305393).  
**Excluded from active integration:** `app.py`  
**Purpose:** Measure whether repository components, offered as bounded **external computational tools**, make an LLM-assisted workflow more accurate, useful, reproducible or efficient. This does **not** alter the underlying LLM's weights, training, native memory, internal activations, or permissions.

## 1. Non-negotiable principles

1. **Inspect, don't assume.** Distinguish documentation claims, implementation observations, directly executed source-code tests, independent reproductions, and deployed tool results.
2. **A module is not a capability until invoked.** Reading the code, writing a wrapper, or reproducing its mathematics is not evidence that ChatGPT can execute that module as a callable tool.
3. **Track baselines.** A claimed improvement requires a defined matched baseline, task-specific acceptance criteria specified **before** the comparison, and recorded negative results.
4. **No silent autonomy.** No continuous processing, background agent, persistent runtime, or cross-chat state unless a separate authorized environment and integration actually provide it.
5. **User controls sequence.** Analyze only the next module specified by the project lead. Explain findings as they emerge; do not disappear into long silent runs.
6. **Research-grade treatment.** No production claims, scientific novelty claims, safety assurances, or EU AI Act compliance conclusions without the appropriate evidence.
7. **Preserve originals.** Work in isolated branches/containers or independent experiment directories; do not alter the default branch, datasets or production operations without review.

## 2. Per-module adoption workflow

Complete the following **eight gates**, recording every gate in the living research briefs or a future capability register. Record `PASS`, `FAIL`, `BLOCKED`, or `NOT STARTED` with an evidence reference; `PASS` is never inferred from a narrative claim.

### Gate A — Source and provenance
- Record repository, path, branch, immutable commit/blob reference, license, date inspected, and variant relationship.
- Identify relevant earlier versions, external prior art, and user-reported versus independently corroborated development dates.
- Build a function/class inventory and distinguish exports from private helpers.

### Gate B — Dependency and boundary map
- Identify inputs, outputs, numerical precision, filesystem/network side effects, required packages, linked modules, mutable state, checkpoint format and failure modes.
- Classify: **standalone**, **standalone with optional acceleration**, or **requires named dependencies**.
- Name the **specific bounded capability** worth exposing; avoid generic "enhanced intelligence" claims.

### Gate C — Reproducible execution
- Execute the **actual checked-out module** in an isolated environment, pin interpreter/dependency versions and randomness.
- Run existing self-tests, then independent positive/negative tests; capture stdout/stderr, exit status, and full environment and source identifier.
- If repository execution is blocked, mark `BLOCKED`. A NumPy/SciPy reproduction of an algorithm is only a **reproduction**, not a pass for this gate.
- Never fetch/execute unreviewed code with credentials or unrestricted network/filesystem access.

### Gate D — Scientific and systems baselines
- Establish matched alternatives without the ADAPA360 component (e.g., NumPy dense operations, standard NCP/LTC/CfC, static heuristics).
- Define task-specific output correctness, error tolerance, repeatability, robustness, memory, setup and p50/p95 latency including tool-call overhead.
- Include both structured and adversarial/unstructured inputs. Capture cases where tensorization *increases* cost or loses accuracy.
- Compare whole-system memory and latency, not only the tensorized submatrix. Keep CPU/GPU, dtype, batch size and seeds explicit.

### Gate E — Typed, bounded tool adapter
Create a thin wrapper that **does not contain unrelated model/ledger logic**. Every operation specifies:
- `operation`, `schema_version`, `module_sha`, validated `input`, `options`, and optionally an opaque `state_handle`.
- `result`, `warnings`, `error`, `metrics`, `artifact_hashes`, provenance and `state_handle` when applicable.
- Explicit maximum input dimensions, data types, memory/CPU/time budgets, timeout, read-only default, and deterministic failure behavior.
- Separate computational receipt **hashes** from *digitally signed attestations*. Hashes alone do not prove authorship.
- No arbitrary Python evaluation, unrestricted import, shell execution, arbitrary file reads, network fetches, secret access, or implicit writes.

### Gate F — In-chat A/B evaluation
- Run a clearly identified set of tasks: **A, ordinary assistant + baseline computation** versus **B, assistant invoking the actual external module**.
- Compare outputs blind when possible. Use the same input, fixed evaluation rubric, fixed tool and model configuration, and repeated runs.
- Metrics: correctness/utility, cost, total latency, failure rate, and reproducibility. Log when tool use worsens an outcome.
- Do not imply the model's native capabilities changed. A gain is an **external-tool-assisted workflow gain**.

### Gate G — Deployment and persistence (separate authorization)
- Choose local sandbox, separately deployed API/MCP service, or approved cloud runner. Repository access by itself is not execution or deployment.
- Require authentication, explicit data retention policy, secret isolation, resource budgets, observability, and rollback.
- A checkpoint is persistent only if an actual persistent storage service is running; chat messages are not runtime checkpoints.
- **Streaming chat content into an experimental NCP requires explicit user authorization and data minimization**. Begin with synthetic or consented data.

### Gate H — White-paper and engineering handoff
Prepare for every selected script:
1. One-page module dossier: purpose, mathematics, inputs/outputs, dependencies, standalone feasibility, interfaces, and worked example.
2. Novelty assessment: closest prior art, clearly specified differences, provisional subjective rating if used, and reasons to revise it.
3. Evidence table: claimed / source-observed / reproduced / module-executed / independently benchmarked / production-validated.
4. Improvement backlog with priority, exact code locus, design hypothesis, risks, acceptance criteria, tests, and dependencies.
5. Implementation-ready handoff for Codex, Claude Code, Antigravity or equivalent—never an unqualified instruction to refactor everything.

## 3. Capability contracts — initial candidates

These are **adapter proposals, not available ChatGPT tools**.

| Proposed operation | Source | Purpose | Required evidence before marking usable |
| --- | --- | --- | --- |
| `tn.decompose` | `tn.py` | Construct Tensor Train/MPO from a bounded dense operator | Full module self-test, reconstruction tolerance, memory and time baselines |
| `tn.apply` | `tn.py` | Apply committed tensor operator to a vector/batch | Compare with dense reference on identical operator and inputs |
| `tn.verify_artifact` | `tn.py` | Verify hashes/manifest of exact stored tensor cores | Alter-one-byte negative test and verified load roundtrip |
| `ncp.step` | `ncp.py` + `tn.py` | Advance external liquid-inspired recurrent state on encoded input | Deterministic repeated stream, dense/TT comparison, drift and saturation tests |
| `ncp.inspect_state` | `ncp.py` | Inspect health, state commitment and current mode | Ensure non-finite events are not silently hidden; validate semantics |
| `ncp.checkpoint` | `ncp.py` | Export or restore external state | Independent roundtrip, corruption detection, explicit storage policy |

These calls operate only on **external, tool-owned state**. The assistant cannot inspect its own internal activations using them.

## 4. Initial source findings to preserve

**`tn.py`:** `TensorTrain`, TT-SVD, Tucker fusion, deterministic configuration, NPZ manifests and cryptographic commitments exist in source. Core hashes identify the stored *representation*, not necessarily a unique mathematical operator: tensor gauge transformations can represent the same operator with different bytes. Normalize/gauge-fix where required, and do not conflate operator equivalence with byte-identity. Small-scale speedups are not guaranteed.

**`ncp.py`:** `TensorizedNCP.step` advances hidden state with each input, using a tensorized *recurrent* operator when available, with dense fallback. Adaptive time constants still involve a dense hidden-to-hidden matrix `tc_W_h`. The per-step code does not perform online weight optimization: **state adaptation is not automatically learning**. The `is_stable` health flag is a numerical heuristic, not a mathematical stability proof. The class also provides hash-based state/weight manifests and NPZ checkpoints.

**Existing test status:** Both modules contain self-tests in source, but the repository documents the absence of a comprehensive formal test suite. The inspected modules have **not** thereby passed independent verification; preserve any previously reported numerical reproduction as provisional until independently reproduced and versioned.

**Initial architectural novelty estimate:** Prior conversational `7/10` is a *provisional, subjective systems-architecture impression* after examining two scripts, not a validated performance or patentability score. Tensorized RNNs, NCP/LTC, and CfC all have published precedents. The integration hypothesis needs separate evidence.

## 5. Proposed first experiments

### TN-EXP-01 — Operator correctness and cost
- Execute `tn.py` self-tests from an immutable revision.
- Generate a seeded structured operator and seeded random operator of equal shape.
- Compare dense vs TT vector/batch outputs, reconstruction errors, active parameters, **total allocated bytes**, setup time, and p50/p95 apply time.
- Flip signs in two adjacent TT cores to preserve the represented matrix; verify that raw core commitments differ. Record whether a proposed normalization solves this safely.

### NCP-EXP-01 — Streamed temporal response
- Use **synthetic** sensor time series: steady baseline, step change, impulse, noise, missing values and outliers.
- Hold initial operator and seed fixed for TT-vs-dense comparison; explicitly record truncation error if TT is approximate.
- Run sequences with no weight learning; measure hidden-state trajectories, output latency, recovery, numerical faults and retained signal.
- Distinguish compression of recurrent weights from whole-model parameter savings, since `tc_W_h` remains dense.
- Subsequently add a **separately specified** online-learning rule only after the fixed-weight baseline is recorded.

### CHAT-EXP-01 — Actual assistant augmentation
- With explicit consent and an available external tool adapter, compare ordinary analysis against analysis enriched by `ncp.inspect_state` on a consented/synthetic stream.
- Determine whether the extra state improves correct temporal judgments **net of latency and failure overhead**; a different internal numeric state alone is not success.

## 6. Definition of adoption

A module is **SOURCE INSPECTED** after A/B, **EXECUTABLE** after C, **BENCHMARKED** after D, **TOOL-WRAPPED** after E, **IN-CHAT EVALUATED** after F, **DEPLOYED** after G, and **EVIDENCE-DOCUMENTED** after H.

The words **"capability added"** are reserved for a *real, callable, tested operation* with observed benefit; they must never follow code inspection alone.

## 7. Architecture-aware AST audit (verified syntax pass)

- Enumerate all active Python files at one immutable repository revision. Record filenames, SHA and Python parser version.
- For each actual file, run Python `ast.parse` on its complete source. Capture `PASS` or `FAIL`, syntax errors and line numbers; do not treat search, regex, partial excerpts or alternative-language parsers as Python AST success.
- Build an inventory of definitions, imports, base classes, key call sites and cross-module dependencies, with file-and-line evidence. Distinguish static inference from confirmed runtime invocation.
- Look specifically for TT/Tucker/fusion, NCP/CfC temporal feedback, digital-twin transitions, hashing/commitments, policy/governance and checkpoint verification. Trace unused yet intentional integration points such as `TreeTensorFusion` rather than discarding them as utilities.
- **Observed:** GitHub Actions Python 3.12.3 parsed 28/28 files, zero errors, approximately 37,433 lines and 270,167 AST nodes. The reproducible source-only audit is in `.github/workflows/adapa360_ast_audit.yml`; log and JSON artifact are linked in the module briefs. This does **not** validate imports, runtime execution, feature usefulness, or scientific novelty. Preserve source and avoid speculative edits during the semantic review.

## 8. Continuing protocol

- The project lead chooses the sequence. After source review of `tn.py`, `ncp.py`, `cfc.py`, and `cognitive_lobe_runtime.py`, conduct the requested full-stack AST mapping before extending module claims. The Python AST parse has **passed**. Continue per-script semantic reviews and independent runtime gates; dynamic imports are not fully visible in the static module-import graph.
- At each major finding, communicate what the code does, the technical implication, and a specific improvement—**during** the work, not only after it.
- Keep evidence, blockers, and implementation ideas in `docs/RESEARCH_MODULE_BRIEFS.md` until a separate register is justified; do not refer to a nonexistent register as completed.
- An engineering agent may implement approved adapters using this protocol; it must not silently modify default-branch source, treat all hashes as signatures, process private chat streams without permission, or claim successful benchmarks without recorded output.
