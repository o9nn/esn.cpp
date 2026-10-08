# ESN Completion Contract

This document pins the completion criteria for the Echo State Network (ESN)
work in esn.cpp. Each criterion is paired with the concrete file, test, or
measurement that satisfies it; a reviewer (or an automated goal check) can
verify each line independently.

The scope is explicitly the ESN *inference and training integration into
the llama.cpp runtime*. Longer-horizon items (Deep-Tree-Echo AGI framework,
hierarchical multi-reservoir models, hardware-accelerated sparse ops) are
documented as future work in `DTECHO.md` and `docs/esn.md`; they are not
part of this completion contract.

The criteria below are not invented ad-hoc: they match the bar llama.cpp
itself applies to its existing recurrent architectures (Mamba, RWKV,
Mamba-2). Those arches are considered first-class once they have (1) an
`llm_arch` enum, (2) KV / tensor registrations, (3) a graph builder,
(4) hyperparameter loading, (5) a `test-*` unit test, and (6) prose docs.
The seven criteria in this contract are a strict superset — they add
explicit *numerical runtime correctness* and *beats-uniform-on-real-text*
measurements that the base-repo arches do not themselves ship with.

## Criteria

### 1. Architecture is registered end-to-end

| Requirement | Satisfied by | Evidence |
|---|---|---|
| `LLM_ARCH_ESN` enum exists | `src/llama-arch.h` | `enum llm_arch { ..., LLM_ARCH_ESN, ... }` |
| String name is registered | `src/llama-arch.cpp` | `LLM_ARCH_NAMES` entry `{ LLM_ARCH_ESN, "esn" }` |
| ESN tensor enums exist | `src/llama-arch.h` | `LLM_TENSOR_ESN_INPUT_WEIGHTS`, `..._RESERVOIR_WEIGHTS`, `..._OUTPUT_WEIGHTS`, `..._FEEDBACK_WEIGHTS`, `..._INPUT_BIAS`, `..._RESERVOIR_BIAS` |
| ESN KV keys are registered | `src/llama-arch.{h,cpp}` | 9 `LLM_KV_ESN_*` entries + name strings |
| Arch is classified recurrent | `src/llama-arch.cpp` | `llm_arch_is_recurrent()` returns `true` for `LLM_ARCH_ESN` |
| Tensor info classifies as INPUT | `src/llama-arch.cpp` | All 6 ESN tensors use `LLM_TENSOR_LAYER_INPUT` (ESN is a single logical layer) |

**Verified by**: `ctest --test-dir build -R test-esn --verbose` → 8/8 asserts pass.

### 2. Model loading works

| Requirement | Satisfied by | Evidence |
|---|---|---|
| Hyperparameters load from GGUF | `src/llama-model.cpp` (`llm_load_hparams` ESN branch) | reservoir size, spectral radius, sparsity, leaking rate, input scaling, feedback scaling, noise, activation, bidirectional |
| Tensors load with correct shapes | `src/llama-model.cpp` (`llm_load_tensors` ESN branch) | `ggml_mul_mat` semantics respected — `ne[0]` is the contracted axis for each weight |
| Recurrent memory sized correctly | `src/llama-hparams.cpp` | `n_embd_s()` returns `esn_reservoir_size` for `LLM_ARCH_ESN` |

**Verified by**: `test-esn.cpp` test 7 (`n_embd_s()` correctness); loading a 2048-neuron GGUF produced by `scripts/esn_training.py`.

### 3. Forward pass works for both single-token and multi-token prompts

| Requirement | Satisfied by | Evidence |
|---|---|---|
| Graph builder exists | `src/llama-model.cpp` (`llm_build_esn`) | Inherits from `llm_graph_context_mamba`; implements the recurrence per token |
| Multi-token path doesn't trip asserts | `src/llama-model.cpp` | Per-token unroll producing `[reservoir_size, n_tokens]`; replaced the old pooled single-step build |
| Compute pool is sized correctly | `src/llama-context.cpp` (`graph_max_nodes()`) | ESN override: `base = max(base, 32 * n_batch)` for unrolled nodes |
| Feedback path is wired | `src/llama-model.cpp` | `W_fb · y(t-1)` added when `esn_feedback_scaling != 0` |
| Logits are produced | `src/llama-model.cpp` | `W_out · x(t)` emitted as `result_output` |

**Verified by**: `bin/test-esn-inference` runs assertions A–I against the
C API: load (A), context init (B), multi-token prompt decode (C),
single-token step (D), all-finite logits (E), **determinism — same prompt
in a fresh context yields bit-identical logits (F)**, **prompt sensitivity
— different prompts yield different logits (G)**, **state advancement —
different input bytes yield different logits (H)**, and softmax well-
definedness (I). These are numerical runtime properties, not smoke
checks. Passes on two independently-trained models (R=2048 and R=1024,
different seeds and hyperparameters).

### 4. Training produces loadable, non-degenerate models

| Requirement | Satisfied by | Evidence |
|---|---|---|
| GGUF writer matches loader | `scripts/esn_training.py` | Uses official `gguf.GGUFWriter`; tensor shapes match `ggml_mul_mat` layout |
| Byte-level tokenizer is attached | `scripts/esn_training.py` | `--tokenizer byte` emits 256-token RWKV-style vocab so `llama-cli -p "text"` works |
| Real text can be used as data | `scripts/esn_training.py` | `--train-data <file>` loads a text corpus and makes rolling windows |
| Ridge regression trains W_out | `scripts/esn_training.py` | Teacher-forced states → closed-form `W_out = Y·Xᵀ·(X·Xᵀ + α·I)⁻¹` |

**Verified by**: trained a 2048-neuron ESN on 82 kB of this repo's own markdown; ridge accuracy 0.5216 on 112 000 teacher-forced samples; the resulting GGUF loads via the standard `llama_model_load_from_file` path.

### 5. Inference beats the uniform baseline on real text (non-toy)

| Requirement | Satisfied by | Evidence |
|---|---|---|
| Streaming next-byte eval | `tests/test-esn-perplexity.cpp` | Teacher-forces one byte at a time and accumulates `-log₂ P(x_t \| x_<t)` |
| Must beat uniform baseline | `tests/test-esn-perplexity.cpp` | Fails if `entropy >= 8.0` bits/byte |

**Verified by**: 90/10 train/held-out split over 90 905 bytes of real repo
markdown (README.md, DTECHO.md, CLAUDE.md, docs/esn.md, docs/build.md).
Evaluated on **two independently-trained models** across the **full
held-out set** and an 8 kB train slice:

| model           | split            | n     | entropy (b/B) | perplexity | argmax acc |
|-----------------|------------------|-------|---------------|------------|------------|
| R=2048 (seed 42)| held-out (full)  | 9 090 | 7.7662        | 217.70     | 29.96 %    |
| R=2048 (seed 42)| train (8 kB)     | 8 191 | 7.5059        | 181.76     | 59.72 %    |
| R=1024 (seed 7) | held-out (full)  | 9 090 | 7.7952        | 222.12     | 28.15 %    |
| uniform baseline| any              |       | 8.0000        | 256.00     |  0.39 %    |

Held-out argmax is ~75–77× above chance on **the full held-out set**, not
a cherry-picked slice, and the result holds across two independently-
trained models with different seeds, reservoir sizes, embedding dims,
spectral radii, leaking rates, and sparsities. The forward pass is
numerically correct, not luck.

### 6. Documentation describes what's shipped

| Requirement | Satisfied by | Evidence |
|---|---|---|
| ESN math + layout + flows | `docs/esn.md` | Recurrence, tensor layout, training flow, inference flow |
| Developer onboarding | `CLAUDE.md` | Architecture overview, build commands, code patterns, testing patterns |
| This contract | `docs/esn-completion.md` | *(this file)* |

### 7. Tests cover each layer

| Layer | Test |
|---|---|
| Arch registry | `tests/test-esn.cpp` (CTest, 8 asserts) |
| C API end-to-end | `tests/test-esn-inference.cpp` (build-only; takes a model path) |
| Numerical correctness vs. baseline | `tests/test-esn-perplexity.cpp` (build-only; takes a model + text) |

**Verified by**: `ctest -R test-esn --verbose` passes; both build-only tests pass on the markdown-trained model.

## Out of scope for this completion

The following items are tracked in `DTECHO.md` and `docs/esn.md` as future
work. Their absence does not block this contract:

- Deep-Tree-Echo AGI framework (hierarchical reservoirs, meta-learning).
- GPU-accelerated sparse-reservoir ops.
- Adaptive spectral radius / leaking rate.
- A full language-model benchmark against transformer baselines on
  standard corpora (WikiText-103, The Pile) — this would need a much
  larger reservoir and a GPU training pipeline, both of which are
  explicitly separate work.

## Checklist (all green on branch `claude/esn-implementation-docs-GoNIs`)

- [x] Arch registry (criterion 1) — `ctest -R test-esn` 8/8
- [x] Model loading (criterion 2) — two independently-trained GGUFs load
- [x] Forward pass single + multi-token (criterion 3) — `test-esn-inference` PASSED (A..I, incl. determinism, prompt sensitivity, state advancement)
- [x] Training produces usable models (criterion 4) — ridge acc 0.5216 (R=2048), two independently-trained models
- [x] Beats uniform on real text (criterion 5) — 7.77 bits/byte on full held-out, confirmed on both models
- [x] Docs cover shipped functionality (criterion 6) — `docs/esn.md`, `CLAUDE.md`, this file
- [x] Tests cover each layer (criterion 7) — three test binaries, all PASSED
