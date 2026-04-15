# ESLLM Development Roadmap

**Goal**: Achieve unprecedented deep integration of Echo State Networks and reservoir computing into the foundational architectures at the heart of AI research and LLM technologies, creating an **ESLLM** capable of synchronous *infer-train* context-aware execution with real-time proprioceptive feedback—enabling the model to "learn from experience" based on each interaction.

---

## Definition of Done (Global)

An ESLLM release is complete when:

- [ ] ESLLM can run low-latency, **synchronous infer-train loops** inside `llama-server`.
- [ ] It can ingest structured real-time **proprioceptive feedback** and measurably improve behaviour across interactions without catastrophic instability.
- [ ] It supports **hierarchical reservoir execution** (DTE-inspired multi-level architecture).
- [ ] It optionally integrates as a **hybrid ESN + Transformer/SSM** front-end.
- [ ] It ships with **reproducible evaluations**, rollback safety, and documented operational controls.

---

## Phase 1 — ESLLM Contract & Capability Matrix ✅ (in progress)

### 1.1 Precise Definitions

**Synchronous infer-train**: A token-generation step that *atomically* consists of:
1. **Infer step** — forward pass through the ESN to produce output logits.
2. **Feedback assimilation step** — ingestion of a structured feedback event (correction, reward, tool result, etc.) into a bounded replay buffer.
3. **Train step** — online update of the linear readout `W_out` using the buffered evidence.

The entire cycle is bounded: it must complete within a configurable latency budget per token. The reservoir weights `W_res` and `W_in` are **never** modified during inference—only the readout is adaptive.

**Proprioceptive feedback**: Any signal that originates *from the model's own execution context*, as opposed to passive supervised labels:
- Observed output confidence (entropy of the softmax distribution)
- Latency of the last decode step
- Token-level corrections provided by the user or a verifier
- Structured results from tool / function calls
- Scalar reward or penalty from a RL-style evaluator

Proprioceptive signals are routed through the `llama_esn_feedback_event` API and converted into online gradient / RLS update signals in the train step.

### 1.2 Invariants

| Invariant | Guarantee |
|-----------|-----------|
| **Bounded latency** | Online train step is O(R²) and must complete in < 1 ms for R ≤ 4096 on CPU |
| **Stable reservoir dynamics** | Spectral radius is validated at load time to be in (0, 1) |
| **Non-destructive continual updates** | Only `W_out` is adapted; `W_res`, `W_in` are read-only after model load |
| **Reproducible checkpoints** | State + readout weights can be serialised via `llama_state_*` APIs at any point |
| **Safe online updates** | All updates are bounded by L2 regularisation; a drift detector can trigger rollback |

### 1.3 ESN/ESLLM Capability Matrix

| Capability | Current Status | Target (ESLLM) |
|-----------|---------------|----------------|
| Single-layer ESN inference | ✅ Implemented | ✅ Keep |
| Batch ridge regression training | ✅ Python utility | ✅ Keep |
| Online RLS adaptation | ✅ Python utility | ✅ Runtime (C++) |
| Online SGD adaptation | ✅ Python utility | ✅ Runtime (C++) |
| Parameter validation at load | ✅ Implemented | ✅ Keep |
| Proprioceptive feedback API | ✅ Header-defined stubs | 🔄 Full implementation |
| Synchronous infer-train loop | ❌ Not yet | 🎯 Phase 4 |
| Hierarchical reservoir (DTE) | ❌ Config only | 🎯 Phase 6 |
| ESN–Transformer hybrid | ❌ Not yet | 🎯 Phase 7 |
| Drift detection + rollback | ❌ Not yet | 🎯 Phase 8 |
| Online learning GGUF metadata | ✅ Implemented | ✅ Keep |
| Safety/policy gates | ❌ Not yet | 🎯 Phase 8 |
| Research benchmark suite | ❌ Not yet | 🎯 Phase 9 |
| Full API surface in llama.h | ✅ Stubs implemented | 🔄 Phase 10 |

---

## Phase 2 — Harden the ESN Baseline ✅ (completed)

**Goal**: Ensure the existing single-layer ESN is production-grade before building on top of it.

### 2.1 Expanded Test Coverage

`tests/test-esn.cpp` now contains **23 tests** covering:

- Architecture recognition, recurrent/hybrid/diffusion classification
- All six tensor info mappings (including optional tensors)
- Default hyperparameter values (static + online-learning + hierarchical)
- `n_embd_s()` correctness for ESN models
- `validate_esn_params()` — 14 distinct validation scenarios including boundary values for all numeric parameters, online learning config, and hierarchical settings

### 2.2 Validation Guards

`llama_hparams::validate_esn_params()` is called immediately after loading all ESN GGUF metadata in `src/llama-model.cpp`. It enforces:

| Parameter | Constraint |
|-----------|-----------|
| `esn_reservoir_size` | > 0 |
| `esn_spectral_radius` | in (0, 1) — strict, to guarantee echo state property |
| `esn_sparsity` | in [0, 1] |
| `esn_leaking_rate` | in (0, 1] |
| `esn_input_scaling` | > 0 |
| `esn_feedback_scaling` | ≥ 0 |
| `esn_noise_level` | ≥ 0 |
| `esn_activation_type` | ∈ {0, 1, 2} |
| `esn_online_lr` | > 0 when online learning enabled |
| `esn_online_reg` | ≥ 0 |
| `esn_online_update_mode` | ∈ {0, 1, 2} |
| `esn_online_decay_rate` | in [0, 1) |
| `esn_n_levels` | ≥ 1 |
| `esn_branching_factor` | ≥ 1 when n_levels > 1 |

---

## Phase 3 — Online Learning Primitives ✅ (completed)

**Goal**: Provide a complete, end-to-end substrate for online readout adaptation that can be driven from both Python (training scripts) and C++ (runtime).

### 3.1 New Hyperparameters

All of the following are loaded from GGUF metadata and fall back to safe defaults:

| GGUF key | hparam field | Default | Purpose |
|----------|-------------|---------|---------|
| `esn.online_learning.enabled` | `esn_online_learning` | false | Master switch |
| `esn.online_learning.rate` | `esn_online_lr` | 1e-4 | Step size (RLS/SGD) |
| `esn.online_learning.reg` | `esn_online_reg` | 1e-6 | L2 regularisation |
| `esn.online_learning.buf_size` | `esn_online_buffer_size` | 64 | Replay buffer capacity |
| `esn.online_learning.mode` | `esn_online_update_mode` | 0 (batch_ridge) | Algorithm |
| `esn.online_learning.decay` | `esn_online_decay_rate` | 0.0 | Forgetting factor |
| `esn.freeze_reservoir` | `esn_freeze_reservoir` | true | Lock W_res/W_in |
| `esn.replay_window` | `esn_replay_window` | 256 | Sliding window size |

### 3.2 Python Online Training Utility

`scripts/esn_training.py` supports:

```bash
# RLS online training (efficient for sequential adaptation)
python scripts/esn_training.py --reservoir-size 1024 --online --online-mode rls \
    --forgetting-factor 0.999 --save-online-metadata --output esllm.gguf

# SGD online training (standard gradient descent)
python scripts/esn_training.py --reservoir-size 1024 --online --online-mode sgd \
    --online-lr 1e-4 --weight-decay 1e-5 --output esllm_sgd.gguf
```

Key classes:
- `ESN.init_rls()` — initialises the RLS inverse correlation matrix P = (1/λ)I
- `ESN.rls_update(state, target)` — one online RLS step (O(R²))
- `ESN.sgd_update(state, target)` — one online SGD step (O(V·R))
- `ESN.online_train_sequence(...)` — process a complete token sequence online

### 3.3 C++ API Stubs

Declared in `include/llama.h`:

```c
struct llama_esn_online_params { ... };
struct llama_esn_feedback_event { ... };

struct llama_esn_online_params llama_esn_online_params_default(void);
bool    llama_esn_is_adaptive(const struct llama_model * model);
int32_t llama_esn_submit_feedback(ctx, events, n_events);
int32_t llama_esn_update_readout(ctx, params);
void    llama_esn_reset_adaptation(ctx);
uint32_t llama_esn_reservoir_size(const struct llama_model * model);
uint32_t llama_esn_n_levels(const struct llama_model * model);
```

---

## Phase 4 — Synchronous Infer-Train Execution Loop 🔄 (planned)

**Goal**: Wire the online learning primitives into the core decode pipeline so that every token generation step can optionally trigger an adaptation step.

### 4.1 Changes Required

**`src/llama-context.cpp` / `src/llama-model.cpp`**:
- Add `esn_online_state` struct holding the live RLS/SGD state (P matrix, gradient accumulators, replay buffer).
- After `llama_decode()` returns, expose a hook `llama_esn_do_train_step()` that:
  1. Drains the feedback buffer.
  2. Runs the appropriate online update on `W_out`.
  3. Optionally writes a checkpoint.

**`src/llama-context.h`**:
- Add `esn_online_state_t * esn_online` member to `llama_context`.
- Initialise from `hparams.esn_online_*` when arch is `LLM_ARCH_ESN`.

**Scheduling modes** (controlled by `llama_context_params.esn_train_schedule`):
| Mode | Behaviour |
|------|-----------|
| `ESN_TRAIN_NONE` | Inference only (default for static models) |
| `ESN_TRAIN_ALTERNATING` | Train after every N infer steps |
| `ESN_TRAIN_EVENT_TRIGGERED` | Train only when `llama_esn_submit_feedback` is called |

**Checkpointing**:
- `llama_state_get_data` / `llama_state_set_data` already serialise recurrent memory; extend to also serialise `W_out` and the RLS P matrix.

### 4.2 Determinism Guarantee

The train step must be deterministic given the same feedback sequence. This is achieved by:
- Using exact floating-point arithmetic (no GPU-side atomic adds) for the P matrix update.
- Storing the full replay buffer in the state blob.

---

## Phase 5 — Real-Time Proprioceptive Feedback Channel 🔄 (planned)

**Goal**: Allow every interaction via `llama-server` to produce structured learning signals that feed directly into the online adaptation loop.

### 5.1 Feedback Schema

The `llama_esn_feedback_event` struct (already declared in `include/llama.h`) captures five signal types:

| Type | Signal Value | Use Case |
|------|-------------|----------|
| `CORRECTION` | 0.0 or token IDs | User says "actually, you should have said X" |
| `REWARD` | ± scalar | RL-style reward shaping |
| `CONFIDENCE` | [0, 1] | Model's own softmax entropy (self-supervised) |
| `TOOL_RESULT` | ± scalar | Was the tool call successful? |
| `LATENCY` | ms | Penalise slow generations |

### 5.2 Server Integration

`tools/server` will expose two new REST endpoints:

```
POST /esn/feedback          # Submit a feedback_event batch
POST /esn/adapt             # Trigger an online train step
GET  /esn/state             # Inspect adaptation status
DELETE /esn/adaptation      # Reset to checkpoint weights
```

### 5.3 Privacy & Policy Gates

Before any feedback signal can trigger adaptation:
- PII scrubbing: token sequences in `target_tokens` are filtered through a configurable redaction function.
- Policy gate: adaptation can be disabled per-endpoint via a `no_adapt` flag in context params.
- Rate limiting: a maximum N feedback events per minute per session.

---

## Phase 6 — Hierarchical Reservoir Architecture (DTE Path) 🔄 (planned)

**Goal**: Implement the Deep Tree Echo (see `DTECHO.md`) as a first-class GGUF architecture with multi-level state management.

### 6.1 New Hyperparameters (already added)

| hparam | Default | Purpose |
|--------|---------|---------|
| `esn_n_levels` | 1 (flat ESN) | Number of hierarchical levels |
| `esn_branching_factor` | 3 | Children per parent reservoir node |
| `esn_lateral_links` | false | Same-level lateral connections |
| `esn_top_down_mod` | false | Higher-level modulation of lower levels |

### 6.2 Architecture Changes

**New tensor types** (to be added to `src/llama-arch.h`):
```
LLM_TENSOR_ESN_HIER_UP_WEIGHTS    // up-projection from child to parent
LLM_TENSOR_ESN_HIER_DOWN_WEIGHTS  // top-down modulation weights
LLM_TENSOR_ESN_LATERAL_WEIGHTS    // lateral connections between nodes
```

**Graph builder** `llm_build_dte` inherits from `llm_build_esn` and:
1. Runs all leaf reservoirs in parallel (batch axis = leaf index).
2. Aggregates child states into parent inputs via `W_up`.
3. Optionally applies top-down gating from parent to children via `W_down`.
4. Routes lateral signals via `W_lateral`.

**Memory**: Each reservoir node has its own state slot in `llama_memory_recurrent`; slot indexing is `level * max_nodes_per_level + node_idx`.

### 6.3 GGUF Backward Compatibility

Flat ESN GGUFs (esn_n_levels == 1) load and run identically to before. Hierarchical GGUFs require `esn_n_levels > 1` and the new tensor types.

---

## Phase 7 — Hybrid ESN–LLM Architecture 🔄 (planned)

**Goal**: Allow an ESN to serve as a persistent dynamical memory front-end to a standard Transformer or SSM reasoning layer.

### 7.1 Architecture Topology

```
Input tokens
    │
    ▼
┌─────────────────────────────┐
│  ESN front-end (per-token)  │  ← O(R) per step, constant memory
│  Reservoir state: x(t)      │
└────────────┬────────────────┘
             │  x(t) concatenated with token embedding
             ▼
┌─────────────────────────────┐
│  Transformer / SSM layers   │  ← standard llama.cpp arch
│  (sparse activation)        │
└────────────┬────────────────┘
             │  logits + hidden state h(t)
             ▼
┌─────────────────────────────┐
│  ESN feedback (optional)    │  ← h(t) fed back into reservoir
└─────────────────────────────┘
             │
             ▼
         Output logits
```

### 7.2 New Architecture Enum

```c
LLM_ARCH_ESN_HYBRID   // ESN front-end + Transformer/SSM backend
```

### 7.3 Compute Partitioning

| Component | Frequency | Cost |
|-----------|-----------|------|
| ESN reservoir update | Every token | O(R²) |
| Transformer / SSM forward | Every token (or every K tokens) | O(n² · d) or O(d²) |
| Online readout update | Every N tokens (configurable) | O(R²) |

---

## Phase 8 — Continual Learning Safety & Governance 🔄 (planned)

**Goal**: Ensure online adaptation never degrades model behaviour beyond acceptable bounds.

### 8.1 Drift Detection

A dedicated `esn_drift_monitor` thread tracks:
- **Spectral stability margin**: `||W_out||_F` should not grow unboundedly.
- **Update norm**: Per-step `||ΔW_out||_F` clipped to `esn_online_clip_norm`.
- **Perplexity drift**: Moving average perplexity on a held-out "anchor" evaluation set.

When drift exceeds threshold, the monitor:
1. Emits a warning log.
2. Optionally freezes online learning.
3. Optionally rolls back to the last committed checkpoint.

### 8.2 Rollback Checkpoints

- A "golden" checkpoint (the static GGUF weights) is always held in memory.
- `llama_esn_reset_adaptation()` restores `W_out` to the golden checkpoint weights instantly.
- Periodic automatic checkpoints can be saved with `llama_state_save_file()`.

### 8.3 Policy Controls

| Policy | Effect |
|--------|--------|
| `adaptation=disabled` | No online learning regardless of model settings |
| `adaptation=offline` | Accumulate feedback but don't apply until explicit call |
| `adaptation=live` | Fully automatic synchronous infer-train |
| `max_updates_per_session` | Hard cap on total W_out update steps |

---

## Phase 9 — Research-Grade Evaluation Stack 🔄 (planned)

**Goal**: Provide rigorous, automated benchmarks that measure ESLLM-specific qualities.

### 9.1 Benchmark Suites

| Suite | Metric | Implementation |
|-------|--------|---------------|
| **Adaptation speed** | Steps to 10% perplexity reduction after feedback | `tests/bench-esn-adapt.cpp` |
| **Context-aware correction** | Accuracy improvement after user correction | `tests/bench-esn-correction.cpp` |
| **Long-horizon retention** | Accuracy after 1k, 10k, 100k tokens without re-training | `tests/bench-esn-retention.cpp` |
| **Proprioceptive conditioning** | Quality improvement from latency/reward signals | `tests/bench-esn-proprio.cpp` |
| **Catastrophic forgetting** | Performance on original task after 10 online epochs | `tests/bench-esn-forgetting.cpp` |

### 9.2 ESN-Specific Metrics

- **Reservoir state entropy** `H(x)` — higher entropy = richer representation
- **Spectral stability margin** `1 - ρ(W_res)` — distance from instability
- **Update norm statistics** `||ΔW_out||_F` per step
- **Adaptation ROI** = accuracy gain / compute cost of online updates

### 9.3 Automated Reporting

Each PR triggers `scripts/esn_eval.py --compare baseline` which:
1. Runs all ESN benchmark suites.
2. Computes regression vs the main-branch ESN baseline.
3. Emits a Markdown report artifact to GitHub Actions.

---

## Phase 10 — Productization & Ecosystem Integration 🔄 (planned)

**Goal**: Ensure ESLLM is a first-class citizen in the llama.cpp ecosystem.

### 10.1 Public API (llama.h)

All ESN-specific APIs are declared in the `include/llama.h` ESN section:
- `llama_esn_online_params` struct + `llama_esn_online_params_default()`
- `llama_esn_feedback_event` struct
- `llama_esn_submit_feedback()`, `llama_esn_update_readout()`, `llama_esn_reset_adaptation()`
- `llama_esn_is_adaptive()`, `llama_esn_reservoir_size()`, `llama_esn_n_levels()`

### 10.2 Documentation

- `docs/esn.md` — static ESN model usage + GGUF format reference
- `docs/esllm-roadmap.md` — this document
- `DTECHO.md` — Deep Tree Echo theoretical framework
- `CLAUDE.md` — developer guide for contributors
- Server docs — REST API for feedback and adaptation endpoints

### 10.3 CI Integration

New CI checks:
- `test-esn` runs on every PR (already in `tests/CMakeLists.txt`)
- Python lint + type check for `scripts/esn_training.py`
- Integration test: create ESN GGUF → load → run test-esn → run online training sequence

---

## Implementation File Map

| File | Phase | Changes |
|------|-------|---------|
| `src/llama-arch.h` | 3, 6 | New `LLM_KV_ESN_ONLINE_*` and `LLM_KV_ESN_HIER_*` keys |
| `src/llama-arch.cpp` | 3, 6 | Key name mappings |
| `src/llama-hparams.h` | 2, 3, 6 | New hparam fields, `validate_esn_params()` declaration |
| `src/llama-hparams.cpp` | 2 | `validate_esn_params()` implementation |
| `src/llama-model.cpp` | 2, 3 | Load new hparams, call validation |
| `include/llama.h` | 4, 5, 10 | Full ESN API surface |
| `tests/test-esn.cpp` | 2 | 23 comprehensive tests |
| `scripts/esn_training.py` | 3 | RLS + SGD online training, GGUF metadata |
| `docs/esn.md` | 1 | ESLLM contract, invariants, online training |
| `docs/esllm-roadmap.md` | 1 | This document |
| `DTECHO.md` | 1, 6 | Updated DTE roadmap and progress |

---

## References

1. Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks.
2. Lukoševičius, M. (2012). A practical guide to applying echo state networks.
3. Gallicchio, C. & Micheli, A. (2017). Echo state property of deep reservoir computing networks.
4. Haykin, S. (2002). Adaptive Filter Theory (4th ed.) — RLS algorithm reference.
5. Sutton, R. & Barto, A. (2018). Reinforcement Learning: An Introduction — reward signal design.
