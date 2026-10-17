# ADAPA360 Research Module Briefs — Living White-Paper Notes

**Research status:** source-reviewed proof-of-concept modules; no independent production validation  
**Version:** 0.2 — 17 October 2026  
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

**Connections.** `ncp.py` can use TT recurrent weights; `cfc.py` can use a TT cell backbone. **Confirmed static call site:** `digital_twin_kernel.py` creates `tn.TuckerFusionLayer` through `_TuckerFusionAdapter` in `FusionMode.TN_TUCKER` and configures that option at twin-tree node `4.1.4` in its platform builder (availability-conditional). This is already a functional *integration path in source*, but **not** the specialized four-input `TreeTensorFusion` class, for which no external construction call was found in the scanned modules. `tensorized_blockchain_kernel.py` and `tb_tensor_commitments.py` also consume tensor representations and their commitments. Higher layers should own governance, ledger execution, security and tool routing.

**Research differentiation.** The proposed contribution extends beyond operator compression: `TreeTensorFusion` (near line 1478 of the current source) encodes an **experimental two-level cognitive integration topology**, combining `memory`, `sensory`, and `tcfc_pred` through a first `TuckerFusionLayer`, then fusing that result with `vlm_strategy` at a root layer. `TuckerFusionLayer.train` includes a local gradient update, so fusion is not strictly inference-only. This is a fixed tree of factor-projection/Hadamard fusion layers, **not** a general hierarchical Tucker decomposition with arbitrary dimension trees or hierarchical-SVD rank truncation. Its architectural intent needs to be traced in other modules and must not be lost in a TT-focused assessment. TT and hierarchical fusion have earlier scientific precedents, so originality of this particular topology remains to be established.

**Boundaries / next experiments.** `TuckerFusionLayer.train` performs local weight updates; its input-gradient calculation uses a factor matrix after in-place modification, which should be checked against finite-difference gradients before broader backpropagation. `digital_twin_kernel.py` additionally exposes `learn_fusion_bias`, a separate scalar feedback update, but these are not evidence of an integrated NCP/CfC online-learning loop. Structure determines TT efficiency: random dense operators may not compress or execute faster. Core-byte commitments verify specific stored representations, not necessarily mathematically equivalent tensors in different gauges. Measure matched dense/TT accuracy, total memory, setup and inference latency on structured and unstructured cases; investigate clear representation-versus-operator identity semantics. **No measured in-chat capability gain is yet attributable to a deployed `tn.py` tool.**

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


## 5. `lobe_scaler.py` — Risk-responsive cognitive resource policy

**Purpose and placement.** A *capacity-policy planner* across the liquid-network and governance pillars. It takes risk/drift/saturation/forecast-error signals, requested operating mode and a resource envelope, then recommends NCP/CfC budget and frequency profiles. Available modes are `dormant`, `reflex`, `operational`, `crisis`, and `deep_cognition`.

**Implemented mechanism.** `LobeScaler.select_mode` maps thresholds to a target mode; `_fit_profiles_to_resource_envelope` contracts proposed budgets when resources are insufficient; `decide` creates hashed `LobeScaleEvent` records, a decision and an event root. NCP drift and saturation, and CfC forecast error, can independently elevate the target to crisis mode. Checkpoint-hash requirements can disallow certain contractions/dormancy decisions.

**Standalone and dependencies.** Source exposes `build_default_scaler`, `recommend_profiles`, CLI and `_self_test`. Low-level decision logic is Python and standard-library based with optional shared schema/hash/config interfaces. This module **does not allocate, train, resize, or deploy an NCP/CfC instance**: its parameter counts are targets/budgets. Any real adaptive scaling needs a separate actuator, measured allocation and safe state migration.

**Novelty and applications.** Combining risk-aware capacity planning with verifiable events, separately for reflex and predictive lobes, could be useful in bandwidth-/power-constrained systems and emergency operations. Risk thresholds and nominal multi-billion parameter budgets are design values, not demonstrated performance, actual model size, or field reliability.

**Open checks.** Verify whether any running orchestrator consumes scale decisions; distinguish desired from allocated parameters; test mode hysteresis and convergence under resource pressure, checkpoint persistence on contraction, and the consistency of per-event approval fields if approval changes during a multi-lobe decision. Source review only; no independent execution results.

**Primary source:** [`lobe_scaler.py`](https://github.com/ADAPA360/Tensorized-Blockchain/blob/initial-stack-upload/lobe_scaler.py).

## 6. `digital_twin_kernel.py` — Governed hierarchical world-state engine

**Purpose and placement.** The core digital-twin model. `TreeTensorNetwork` maintains a hierarchical node topology, latent numeric state, aggregations, sketches, baselines, causal/temporal probes, snapshots, and Merkle-backed state commitments. `AkkuratInterface` and `PolicyEngine` manage simulated and allowed actions; blockchain append/validator consensus remain in separate modules.

**Fusion, with an important correction.** In `_fuse_children`, the `TN_TUCKER` mode uses `_TuckerFusionAdapter`, which instantiates a two-input `tn.TuckerFusionLayer` when available. The platform builder configures this mode for node `4.1.4`; an alternative `TUCKER` mode is used when `tn.py` is unavailable. Other modes include averaging and attention. This confirms an **existing configurable tensor-fusion handoff**, but not invocation of `TreeTensorFusion`'s memory/sensory/CfC/strategy path. `learn_fusion_bias` separately adjusts attention/fusion bias in response to an explicit reward.

**Execution and audit.** `state_commitment()` captures state, topology, metadata, node-state and projector hashes. `governed_actions` filters proposed actions, simulates permitted actions, asks policy approval and generates policy/action receipts. `sandbox_clone` and `deep_clone_for_replay` support pre-action simulation and replay experiments. These are source-observed mechanisms, not evidence of field-safe autonomous actuation.

**Standalone use.** NumPy and Python are required; `tn.py` is optional for certain tensor modes, with fallback. A `_self_test` is present and `python digital_twin_kernel.py` invokes it; that test has not been independently reproduced in this review.

**Novelty and next steps.** The research proposition is linking hierarchical tensor fusion, governed state simulation and inspectable commitments, not inventing digital twins, Tucker fusion or Merkle trees. Prioritize testing deterministic replay and state-root equality, visibility of fusion fallback, training-vs-inference separation, and full policy simulation against negative actions. For fire and rescue, proposed uses include a simulated robot, sensor or incident twin with humans retaining operational authority.

**Primary source:** [`digital_twin_kernel.py`](https://github.com/ADAPA360/Tensorized-Blockchain/blob/initial-stack-upload/digital_twin_kernel.py).

## 7. `twin_anything.py` — Domain-agnostic digital twin factory

**Purpose and placement.** A typed construction bridge making digital twins of objects, equipment, processes, plans, software, organizations and other defined entities. `TwinAnythingFactory.create_twin_runtime` produces a `digital_twin_kernel.TreeTensorNetwork`, `AkkuratInterface`, schema, node map, metadata, source-file manifests and a creation receipt. `create_twin_snapshot` returns a serializable representation; a legacy `create_twin` interface also exists.

**Evidence and integrity.** The factory materializes a universal branch, applies type-specific structure, records inputs, source-manifest roots and pre/post commitments, performs topology validation, and emits a `TwinCreationReceipt`. `TensorNetworkProfileBuilder` inspects optional `tn.py` capabilities; it does not demonstrate that all materialized twins contain learned or accurate physical dynamics.

**Standalone and dependencies.** Python/NumPy and `digital_twin_kernel.py` are needed for actual twin creation; `tn.py` integration is optional and diagnostic/configuration-sensitive. The factory is separately callable from the ledger, while the full governance and transaction life cycle requires additional stack modules.

**Novelty and next steps.** Universal schema-based twin construction with deterministic receipts could be valuable as an adaptation layer for multiple sectors, including emergency services, health logistics, autonomous robotics and transport. This is a platform hypothesis, not evidence that every domain is adequately modeled. Check typed-observable fidelity, source-manifest trust, type-registry coverage, failure handling, serialization invariants and the status of `run_smoke_tests` paths.

**Primary source:** [`twin_anything.py`](https://github.com/ADAPA360/Tensorized-Blockchain/blob/initial-stack-upload/twin_anything.py).


## 8. `tensorized_blockchain_kernel.py` — Cryptographic ledger with compiled tensor representations

**Purpose and role.** A canonical local block ledger for governed digital-twin transitions. `TensorTransaction`, `TensorBlock`, `TransitionReceipt`, Merkle proofs, signatures, replay/sequence guards and `LedgerMPS` form its central interfaces. `GovernedLedgerExecutor` routes candidate actions through `digital_twin_kernel.AkkuratInterface`. `twin_anything.py` is an optional source of materialized twins; `tn.py` and `digital_twin_kernel.py` are required to import the kernel.

**Precisely what is tensorized.** `LedgerMPS._derive_core` hashes canonical per-block transition context into a deterministic numeric vector, reshapes it into an operator-sized matrix, and combines it with seeded bond weights to construct one MPO/TT core per block. `get_tensor_train` assembles the cores; `compress_ledger` requests TT rounding and records a compression report. The **cryptographic block chain remains canonical**. The tensor train is a reproducible, hash-committed derived representation, not the sole ledger source of truth, and a hashed context vector is not automatically a learned representation of physical dynamics.

**Security and consensus boundary.** SHA-256/Merkle commitments and Ed25519 signatures provide integrity/authentication primitives, **not encryption or confidentiality** for 6G communications. `LedgerConfig.require_signatures` defaults to false, and `append_block` can insert an empty `ConsensusCertificate` if none is provided; `TensorBlock.validate_static` bypasses the failed-certificate check when `total_votes == 0`. Therefore the current implementation **does not enforce a validator quorum for all appended blocks**. The separate `tb_consensus.py` contains a transition-based quorum engine, but compatibility and actual use require explicit verification. The `production` string in source version labels does not constitute a deployment assurance.

**Standalone operation.** Python, NumPy, `tn.py` and `digital_twin_kernel.py` are required; Ed25519 signing depends on a cryptography backend, and `twin_anything.py` is optional for demo twin materialization. CLI/self-test and an Alicante fire-department robot scenario exist in source, but neither represents independent field validation.

**Research differentiation and risks.** The distinctive integration hypothesis is pairing governed twin transitions and cryptographically verifiable receipts with reproducible TT/MPO compilation. Prior art includes blockchains, Merkle trees, digital twins and tensor networks separately. Before claiming an operational protocol: enforce configurable signature/quorum invariants, compare `tb_consensus` and this module's competing certificate schemas, build golden vectors for hash/serialization compatibility, and test tampering, replay, deterministic reconstruction, compression errors and reloading. Keep signed attestation distinct from encryption.

**Primary source:** [`tensorized_blockchain_kernel.py`](https://github.com/ADAPA360/Tensorized-Blockchain/blob/initial-stack-upload/tensorized_blockchain_kernel.py).


## 9. `tb_tensor_commitments.py` — Tensor/ledger artifact bridge

**Role.** Converts `tn.py` tensor cores and rounding reports into schema-aware commitments, NPZ artifact integrity reports, and links to block hashes and digital-twin state roots. `hash_tensor_core` commits each array's index, shape, dtype, order, byte length and raw SHA-256 under domain separation; `round_tensor_train_with_commitment` binds pre/post rounding representations and compression reports. It does **not** execute actions, run a blockchain, or perform tensor numerical computations itself.

**Standalone use.** Python/NumPy and a `tn.py`-compatible TensorTrain are necessary for actual tensor artifacts; shared schema/canonical/hash layers have guarded import fallbacks. Source contains CLI/self-tests but these have not been run independently. The integrity API is suitable for a future bounded verification tool.

**Differentiation and checks.** The proposed systems contribution is verifiable continuity from mathematical tensor data to governed state-transition evidence. Hashes verify *exact stored representations*, not tensor gauge-equivalence, source authenticity or confidentiality. Test corruption, endian/dtype normalization, manifest tampering, cross-platform recomputation, and compatibility with the ledger's independently compiled tensor-core commitments.

**Primary source:** [`tb_tensor_commitments.py`](https://github.com/ADAPA360/Tensorized-Blockchain/blob/initial-stack-upload/tb_tensor_commitments.py).

## 10. `tb_execution.py` — Canonical governed transaction-execution boundary

**Role and boundaries.** `CanonicalTransactionExecutor.execute` validates a transaction envelope and expected pre-state root, applies one or more actions through `AkkuratInterface.governed_actions` (unless explicitly configured for direct fallback), and emits pre/post state commitments, action/policy/transaction/transition receipts and optional lobe evidence. This file deliberately does not append blocks or own consensus; the ledger should consume its outputs.

**Neural integration.** `run_ncp_lobe_receipt` advances an NCP lobe using `deterministic_feature_vector` and records health. `run_cfc_transition_receipt` requests a CfC internal-state forecast and commits a transition receipt. However the code calls `compare_forecast_actual(predicted_state, predicted_state, ...)`: comparing an internal prediction with itself does **not** validate actual predictive accuracy. Furthermore the CfC `predicted_post_state_hash` is derived from a neural hidden-state vector while `actual_post_state_hash` is populated with the digital twin's `post_root`; these are **different state spaces**. The receipt needs explicit typed target semantics and a true observed, aligned target before comparative claims.

**Hash interoperability.** `_object_hash` calls `tb_hashing.domain_hash_hex(domain, obj)` positionally, despite the shared function's signature requiring `domain` as a keyword-only argument. The exception is swallowed and a different local hash encoding is used. This is a source-level compatibility risk affecting receipt consistency, not yet an independently executed failure case. A golden-vector test must precede changes to protocol hashes.

**Standalone use and improvement.** Python/NumPy and a network/control plane object are required for meaningful execution; shared schemas and low-level hashing/validation are optional imports with fallback. A self-test uses `digital_twin_kernel.py`, without independent results here. Prioritize a typed forecast-to-observation pipeline, invariant tests for rejected actions and pre-state mismatch, recovery after partial failures, and agreement between execution receipts and consensus/ledger schemas.

**Primary source:** [`tb_execution.py`](https://github.com/ADAPA360/Tensorized-Blockchain/blob/initial-stack-upload/tb_execution.py).

## 11. `tb_consensus.py` — Signed transition quorum and validator workflow

**Role.** A separate transition-based consensus implementation. `TransitionProposal` binds chain ID, height/round, block hash, transaction Merkle root, pre/post digital-twin roots, policy receipt root and tensor commitment hash. `TransitionConsensusEngine` creates/verifies Ed25519 validator votes, checks known validators and duplicates, sums weighted approvals, and issues/verifies certificates; `TransitionCoherenceValidator` checks proposed transition consistency.

**Standalone use and boundaries.** Source calls shared `tb_canonical`, `tb_hashing`, `tb_crypto`, `tb_schemas` and `tb_validation`. It is a decision/certificate engine, not a network transport layer, ledger append routine or neural inference path. Source includes a self-test with demo validators, quorum-failure and tamper tests, but these are not independently run in this research pass.

**Critical integration distinction.** `ConsensusConfig` defaults to requiring signatures, but `require_tensor_commitment_match=False` and `require_transition_receipt=False` by default; policy/tensor validation behavior therefore depends on configuration and the provided objects. `tensorized_blockchain_kernel.py` also defines *different, locally owned* vote/certificate structures and allows default empty certificates on append. A scientifically defensible protocol needs a single authoritative consensus schema and verified, fail-closed linkage from the execution result to a signed quorum and then to an appended block.

**Potential applications and limits.** Verifiable distributed approval could be useful for multi-agency asset-state changes, controlled robot operations and audit trails. This does not establish a deployed distributed consensus network, a secure communications encryption scheme, or safety certification.

**Primary source:** [`tb_consensus.py`](https://github.com/ADAPA360/Tensorized-Blockchain/blob/initial-stack-upload/tb_consensus.py).

## Complete-source AST audit — verified 17 October 2026

A GitHub Actions check on research commit `6e52251fdc5ced99560b07a24a69cb983f542194` ran Python **3.12.3 `ast.parse` on 28/28 Python files**, including all 27 active core files and separately classified `app.py`: **zero parse failures**, approximately **37,433 source lines and 270,167 AST nodes**. The check records per-file source SHA-256, classes, top-level functions, local imports, static call sites and AST-node counts. [Audit run and retained artifact](https://github.com/ADAPA360/Tensorized-Blockchain/actions/runs/39018305393). AST success proves syntax parsing only; no imports, self-tests, benchmarks or operational interactions were executed.

**Static-analysis limitation:** `digital_twin_kernel.py`, `twin_anything.py` and `tb_api.py` use dynamic module loading and/or path bootstrapping, so simple `Import`/`ImportFrom` graphs undercount runtime dependencies; the above findings supplement AST with source-level call tracing.

**Cross-stack interoperability candidate requiring tests:** `tb_hashing.domain_hash_hex(payload, *, domain=...)` requires a keyword-only domain, while some callers, including `tb_execution._object_hash` and `tb_api.hsh`, pass two positional arguments, catch the resulting exception and fall back to a locally different hash formulation. This can undermine cross-module canonical-hash consistency. Do **not** replace cryptographic formatting without a compatibility/migration design and golden-vector tests.

**Review status:** Eleven module briefs now exist; the remaining scripts have completed syntax/structural inventory but have **not** each received a source-semantic or execution-level audit. Record exact evidence before assigning implementation, safety or scientific-priority claims.

## Cross-module architectural questions retained for full-stack review

- Does any runtime instantiate or invoke `TreeTensorFusion`? Identify the actual call paths rather than infer them from comments or class names.
- Are numerical streams represented by stable, typed physical features or hashed serialized records? Measure both approaches.
- Does the path for predicting a post-state receive later *independent observations* or only the model's own post-step state?
- Are serialized state and tensor commitments checked in a single verifiable, canonical way end-to-end, including on restore?
- Where should explicit lobe exchange and hierarchical sensor/memory/strategy fusion take place without bypassing governance?

**Analysis technique:** the full-stack Python-AST parse is now verified (audit run linked above). Continue architecture-aware semantic and runtime reviews, explicitly separating AST success from successful execution.

