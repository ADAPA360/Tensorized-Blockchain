# ADAPA360 Research Module Briefs — Living White-Paper Notes

**Research status:** source-reviewed proof-of-concept modules; no independent production validation  
**Version:** 0.1 — 17 October 2026  
**Repository:** [ADAPA360/Tensorized-Blockchain](https://github.com/ADAPA360/Tensorized-Blockchain/tree/initial-stack-upload)  
**Source reference:** default branch `initial-stack-upload`; verify the actual commit and blob SHA before executing or reproducing experiments.  
**Scope:** This document grows **one reviewed module at a time** in the order chosen by the project lead. It is a compact repository companion to the eventual scientific white paper, **not** a replacement for the underlying code or a claim of scientific priority. Original source modules remain unchanged.

## Project architecture: three complementary pillars

ADAPA360's research programme is organized around **three interconnected but distinct pillars**. This is a proof-of-concept architecture, not yet a validated end-to-end product.

1. **Tensor networks — mathematical construction substrate.** `tn.py` supplies reusable TT/MPO/Tucker representations, operators, contractions, factorization, and verifiable computational artifacts. Tensorization is proposed as a general construction approach, not merely a compression technique.
2. **Liquid neural networks — temporal cognitive processing.** `ncp.py` and `cfc.py` provide complementary liquid-inspired recurrent state processing and time-aware state prediction. `cognitive_lobe_runtime.py` coordinates their inputs and outputs; actual inter-lobe learning and functional advantages remain to be demonstrated.
3. **Blockchain with digital twinning — governed state and traceability.** `digital_twin_kernel.py`, `twin_anything.py`, and the ledger/transaction/consensus modules represent real or simulated entities and record controlled state transitions, receipts and cryptographic commitments. These features can support auditability but do not independently establish safety or legal compliance.

**Integration hypothesis:** These pillars may jointly support a model-agnostic, inspectable computational framework. Each pillar should be assessed independently *and* at its interfaces. The white paper must explain both standalone applications and the added value, if any, of their combination.

---

## 1. `tn.py` — Tensor-network computational substrate

**Purpose and context.** The foundation for ADAPA360's tensorized AI experiments: CPU-first Tensor Train / matrix-product-operator (TT/MPO) transformations and Tucker-style fusion. The design intent is to reuse tensor-network operators as computational building blocks across cognitive, digital-twin and ledger systems, not only to reduce storage.

**Implemented interfaces.** `TensorTrain`, `TTConfig`, `TensorCommitment`, `TTRoundingReport`, `tensor_train_svd`, `TuckerTensor`, `TuckerFusionLayer`, `TreeTensorFusion`, and `DendriticProcessor`. The runtime offers contraction, dense conversion (with guardrails), rounding, reconstruction reporting, NPZ artifact serialization and hash-based commitment checks.

**Dependencies / standalone use.** Requires Python and NumPy; SciPy is optional for CPU decompositions and CuPy for non-consensus GPU work. It is intended to run independently of the ledger and lobes. A built-in smoke/self-test entry point exists: `python tn.py` (not independently verified as passing during this documentation pass).

**Connections.** `ncp.py` can use TT recurrent weights; `cfc.py` can use a TT cell backbone; `digital_twin_kernel.py` uses tensor operators/fusion; `tensorized_blockchain_kernel.py` and `tb_tensor_commitments.py` consume tensor representations and their commitments. Higher layers should own governance, ledger execution, security and tool routing.

**Research differentiation.** The proposed contribution extends beyond operator compression: `TreeTensorFusion` (near line 1478 of the current source) encodes an **experimental two-level cognitive integration topology**, combining `memory`, `sensory`, and `tcfc_pred` through a first `TuckerFusionLayer`, then fusing that result with `vlm_strategy` at a root layer. `TuckerFusionLayer.train` includes a local gradient update, so fusion is not strictly inference-only. This is a fixed tree of factor-projection/Hadamard fusion layers, **not** a general hierarchical Tucker decomposition with arbitrary dimension trees or hierarchical-SVD rank truncation. Its architectural intent needs to be traced in other modules and must not be lost in a TT-focused assessment. TT and hierarchical fusion have earlier scientific precedents, so originality of this particular topology remains to be established.

**Boundaries / next experiments.** Structure determines TT efficiency: random dense operators may not compress or execute faster. Core-byte commitments verify specific stored representations, not necessarily mathematically equivalent tensors in different gauges. Measure matched dense/TT accuracy, total memory, setup and inference latency on structured and unstructured cases; investigate clear representation-versus-operator identity semantics. **No measured in-chat capability gain is yet attributable to a deployed `tn.py` tool.**

**Primary source:** [`tn.py`](https://github.com/ADAPA360/Tensorized-Blockchain/blob/initial-stack-upload/tn.py).

## 2. `ncp.py` — Tensorized liquid-reflex neural substrate

**Purpose and context.** A liquid-inspired, recurrent temporal component positioned as the **reflex/stability lobe** of the prospective cognitive system. It is designed to process numeric streams, maintain hidden state, and expose health and integrity records, without directly making governance or blockchain decisions.

**Implemented interfaces.** A legacy sparse `NeuralCircuitPolicy` implementation and a production-oriented `TensorizedNCP` class. The latter supports `step(inputs, dt, return_state)`, adaptive time constants, bounded state updates, optional TT recurrent operators with dense fallback, optional auxiliary outputs, state/weight hashes, lobe-health receipts, NPZ checkpoints, and deterministic configuration controls.

**Dependencies / standalone use.** Python and NumPy are required. `tn.py` is required for the **tensorized path**; a dense fallback can be selected when configured. `python ncp.py` invokes a built-in self-test (not independently verified as passing during this documentation pass). Input is a numeric vector; higher layers encode structured or conversational inputs.

**Connections.** Consumes `tn.py` operators; `cognitive_lobe_runtime.py` encodes context and calls the NCP pass; `tb_execution.py` can request an NCP health receipt during state execution. The NCP does not independently train a language model or control an assistant's private activations.

**Research differentiation.** The exploratory combination of liquid-inspired recurrent dynamics, TT recurrent operators, and auditable state/weight records is pertinent to ADAPA360's unified tensor-based cognitive architecture. The exact current update is **not identical** to the original conductance-based LTC equations, and tensorized recurrence itself has prior art. Claims of mathematical novelty and advantages require comparison with published NCP/LTC baselines.

**Boundaries / next experiments.** The adaptive gate still contains a dense hidden-to-hidden matrix `tc_W_h`; this limits whole-model savings. The `step()` implementation updates **hidden state, not model weights**: exposure to streaming data is not, by itself, online weight learning. Numerical-health flags are not proofs of dynamical stability, and sanitizing invalid numeric values may hide fault origins. Compare TT/dense recurrent modes, stability under perturbation, recovery, memory retention and genuine feedback-driven learning. **A chat-connected recurrent tool has been proposed but not installed or validated.**

**Primary source:** [`ncp.py`](https://github.com/ADAPA360/Tensorized-Blockchain/blob/initial-stack-upload/ncp.py). **Scientific comparison:** [NCP/LTC/CfC reference implementation](https://github.com/mlech26l/ncps).

## 3. `cfc.py` — Tensorized continuous-time predictive-transition substrate

**Purpose and context.** The proposed **predictive lobe**, complementary to NCP's reflex and stability role. It calculates time-aware transitions of an *internal numeric neural state* and supplies state, diagnostic and integrity records suitable for higher-level inspection.

**Implemented interfaces.** `TCfC_Cell`, `TCfC_Policy`, `ForecastErrorReport`, `CfCTransitionReceipt`, and commitment/health classes. `TCfC_Cell.forward` combines input and prior state, transforms them through a configurable TT/dense backbone, calculates three subsequent dense transformation heads and a time-dependent gate, then applies exponential temporal mixing. `TCfC_Policy.forecast` estimates a next state without advancing the stored policy state; `step` advances state. Checkpoints, hashes and source self-tests are present.

**Dependencies / standalone use.** Requires Python and NumPy; `tn.py` provides the tensorized backbone and a configurable dense fallback exists. `python cfc.py` runs a built-in self-test (not independently verified as passing during this documentation pass). Its inputs are numeric vectors and time intervals, not raw natural language or real-world event outcomes.

**Connections.** Depends on `tn.py`; runs alongside NCP in `cognitive_lobe_runtime.py`; `tb_execution.py` can request predictive transition receipts. The current multi-cell policy provides distinct recurrent states per cell, but cells receive the same external input and do not exchange recurrent state during each update.

**Research differentiation.** The investigation concerns a TT-backed, time-aware, liquid-inspired state transition combined with reproducible checkpoints and governance-oriented receipts. The cell equation differs from the original MIT-associated CfC research implementation, so both mathematical fidelity and functional consequences require comparative study. A forecast of the lobe's next **internal state** is **not automatically** a forecast of a fire, vehicle trajectory, or another external event.

**Boundaries / next experiments.** Three transformation heads remain dense. The self-test includes a predicted-versus-itself comparison, and a path in `tb_execution.py` calls `compare_forecast_actual(predicted_state, predicted_state, ...)`; these do not establish real predictive accuracy. Introduce an independently observed target, pre-register forecast metrics, and compare matched baselines across time gaps. Assess true cell-to-cell interaction and whole-model TT memory/latency separately. **No demonstrated improvement to a deployed assistant is claimed.**

**Primary source:** [`cfc.py`](https://github.com/ADAPA360/Tensorized-Blockchain/blob/initial-stack-upload/cfc.py). **Scientific comparison:** [CfC reference implementation](https://github.com/raminmh/CfC).

## Findings shared across the first three modules

The project proposes a layered tensor-based systems architecture. Its **possible differentiation** lies in the composition of computational primitives, recurrent temporal processors and integrity-verifiable artifacts, not in claiming the underlying TT, NCP or CfC methods were invented here.

**Evidence labels:** `source-observed` for the implementations described above; `research hypothesis` for proposed improvements; `not yet established` for training gains, operational safety, scientific priority and measurable assistant augmentation. A source self-test, a mathematical reproduction and a tested deployed tool are three distinct kinds of evidence.

**Engineering rule:** Preserve existing source, use reviewable changes on separate branches, and delay major rewrites until the complete module-by-module review clarifies authoritative cross-module interfaces. Prefer this **single living document** over one speculative new file per script. The project lead chooses the next script and its review order.

## 4. `cognitive_lobe_runtime.py` — Dual-lobe coordinator and integration gap

**Role.** A bridge between tensorized liquid neural networks and higher-level digital-twin, transaction and receipt contexts. It creates or accepts NCP and CfC lobes, converts context to numerical features, runs the two processors, returns hashes and health/transition receipts, and offers joint checkpoint operations. It is **not** itself a trainable fusion network, learned routing policy or independent governance engine.

**Architecture and data path.** `CognitiveFeatureEncoder.encode` deterministically maps canonicalized twin snapshots, transaction records, commitments, policy receipts and optional metadata into a default **128-dimensional** numeric feature vector. This involves per-byte hash buckets and limited numeric slots with signed-log scaling, clipping and normalization. `encode_features(raw_vector=...)` also supports directly supplied numeric features. `CognitiveLobeRuntime.run` calls `run_ncp_pass` and `run_cfc_pass` with the **same feature vector**, then records two lobe results and a combined run hash. Neither lobe receives the other's output, hidden state or gradients. They are coordinated but **not dynamically coupled**. For architecture-level research, compare this path against `tn.py`'s currently separate `TreeTensorFusion` hierarchy rather than assuming it is connected.

**Implemented interfaces.** `CognitiveLobeRuntimeConfig`, `FeatureEncoderConfig`, `CognitiveFeatureEncoder`, `DualLobeRunResult`, `LobePassResult`, `run`, `run_ncp_pass`, `run_cfc_pass`, `save_checkpoints`, `load_from_checkpoints`, `validate_consensus_safe`, and `run_self_test`. The API offers a cognitive analysis route through `tb_api.py` (`POST /lobes/analyze`); `tb_execution.py` separately accepts NCP/CfC lobe objects as optional execution inputs.

**Dependencies and standalone use.** Python/NumPy; NCP/CfC module dependencies for both lobe paths; shared canonicalization, hashing and schema utilities for receipt creation with some fallback handling. `tn.py` enters transitively through the lobes when TT modes are enabled. Module self-tests exist in source; there is **no verified execution/benchmark artifact in this review**. State exists within the running Python instance, not automatically in a ChatGPT conversation.

**Key caveats and next tests.** The hash-based encoder is **not a semantic embedding**; a changed payload hash influences per-byte bucket placement, potentially making semantically similar inputs quite different. `run_cfc_pass` can compare against an explicit `actual_state`, but the public `run` path does not provide independent ground truth: when it advances CfC, the resulting state is primarily a consistency check against the model's own calculation. Its receipt formatter looks for `relative_error` or `l2_error` while the native CfC `ForecastErrorReport` uses `relative_l2_error`; prioritize a field-semantics regression test. Health scores are heuristic and should not be labeled dynamical stability proofs. `save_checkpoints` writes both NPZ files and a hash manifest, but `load_from_checkpoints` does not visibly enforce verification of the **top-level** manifest before loading: test integrity and rollback in the integrated path. Controlled experiments should test semantic/typed feature encoders, explicit and bounded lobe-to-lobe fusion, accurate forecast observation alignment, and coordinated checkpoint verification.

**Differentiation hypothesis.** The possible systems contribution is a traceable orchestration boundary that may eventually connect tensor-tree fusion, dual temporal processing and digital-twin governance. Current source demonstrates coordination, not demonstrated cognitive emergence or an advantage over simpler parallel models.

**Primary source:** [`cognitive_lobe_runtime.py`](https://github.com/ADAPA360/Tensorized-Blockchain/blob/initial-stack-upload/cognitive_lobe_runtime.py).

## Cross-module architectural questions retained for full-stack review

- Does any runtime instantiate or invoke `TreeTensorFusion`? Identify the actual call paths rather than infer them from comments or class names.
- Are numerical streams represented by stable, typed physical features or hashed serialized records? Measure both approaches.
- Does the path for predicting a post-state receive later *independent observations* or only the model's own post-step state?
- Are serialized state and tensor commitments checked in a single verifiable, canonical way end-to-end, including on restore?
- Where should explicit lobe exchange and hierarchical sensor/memory/strategy fusion take place without bypassing governance?

**Analysis technique:** repeat the **full-stack structural and Python-AST scan** with this architecture-first lens. A source/keyword inventory does not count as a successful `ast.parse` check. See the adoption protocol for what evidence is required before reporting AST or execution checks as passing.

