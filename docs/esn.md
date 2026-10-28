# Echo State Networks (ESN) in llama.cpp

This document describes the implementation of Echo State Networks (ESNs) in llama.cpp, providing a comprehensive model for reservoir computing within the ggml framework.

> **See also:** [`docs/esn-completion.md`](esn-completion.md) — the explicit
> completion contract for the ESN work: scope, criteria, and the exact
> file / test / measurement that satisfies each criterion.

## What are Echo State Networks?

Echo State Networks are a type of recurrent neural network that belongs to the reservoir computing paradigm. ESNs consist of three main components:

1. **Input Layer**: Projects input signals into the reservoir space
2. **Reservoir**: A large, sparsely connected, randomly initialized recurrent network with fixed weights
3. **Output Layer**: A trainable linear readout layer that maps reservoir states to outputs

The key insight of ESNs is that only the output weights need to be trained, while the reservoir weights remain fixed after initialization according to specific spectral radius constraints.

## Architecture Overview

### Core ESN Equation

The full ESN dynamics implemented by `llm_build_esn` are:

```
x(t+1) = (1-α) * x(t) + α * f(W_res * x(t) + W_in * u(t) + W_fb * y(t-1) + b)
y(t)   = W_out * x(t)
```

Where:
- `x(t)` is the reservoir state at time t
- `u(t)` is the input embedding at time t
- `y(t)` is the output projection at time t
- `W_res` is the reservoir weight matrix (with spectral radius < 1)
- `W_in` is the input weight matrix
- `W_fb` is the optional output-to-reservoir feedback matrix, applied to
  `y(t-1)`, the one-hot of the token emitted at t-1
- `W_out` is the trained output readout matrix
- `b` are the optional input/reservoir bias vectors
- `f` is the activation function (tanh, sigmoid, or leaky-relu)
- `α` is the leaking rate (controls memory vs. adaptation trade-off)

The feedback term `W_fb * y(t-1)` is only included when `esn_feedback_scaling > 0`
and the `esn_feedback_weights` tensor is present in the GGUF file. The token
emitted at t-1 is the input token at t: the sampled token in generation and
the true next token in training (teacher forcing). So `W_fb * y(t-1)` is the
row of `W_fb` for the input token, scaled by `esn_feedback_scaling`, and
training and inference run the same recurrence.

### Model Components

#### Hyperparameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `esn_reservoir_size` | Number of reservoir neurons | - | > 0 |
| `esn_spectral_radius` | Spectral radius of reservoir matrix | 0.95 | (0, 1) |
| `esn_sparsity` | Connectivity sparsity (fraction of zero weights) | 0.1 | [0, 1] |
| `esn_leaking_rate` | Memory vs. adaptation parameter | 1.0 | (0, 1] |
| `esn_input_scaling` | Input signal scaling factor | 1.0 | > 0 |
| `esn_feedback_scaling` | Output-to-reservoir feedback (0 = disabled) | 0.0 | >= 0 |
| `esn_noise_level` | Noise injection for regularization | 0.0 | >= 0 |
| `esn_activation_type` | Activation function (0=tanh, 1=sigmoid, 2=leaky_relu) | 0 | {0,1,2} |
| `esn_bidirectional` | Not supported (inference is forward-only; see limitations) | false | bool |

#### Tensors

| Tensor | Shape | Description |
|--------|-------|-------------|
| `esn_input_weights` | `[reservoir_size, n_embd]` | Maps input embeddings to reservoir |
| `esn_reservoir_weights` | `[reservoir_size, reservoir_size]` | Recurrent reservoir connections |
| `esn_output_weights` | `[n_vocab, reservoir_size]` | Maps reservoir to output vocabulary |
| `esn_feedback_weights` | `[n_vocab, reservoir_size]` | Output-to-reservoir feedback, one row per token (optional) |
| `esn_input_bias` | `[reservoir_size]` | Input projection bias (optional) |
| `esn_reservoir_bias` | `[reservoir_size]` | Reservoir activation bias (optional) |
| `esn_input_table` | `[n_vocab, reservoir_size]` | Whole input drive of each token, read instead of computing it from the tensors above (optional; see below) |

## Implementation Details

### Architecture Integration

ESNs are implemented as a recurrent architecture in llama.cpp, leveraging the existing recurrent memory system used by models like Mamba and RWKV. Key integration points:

- **Architecture Type**: `LLM_ARCH_ESN`
- **Recurrent Memory**: Uses `llama_memory_recurrent` for state management
- **Graph Builder**: `llm_build_esn` implements forward pass computation
- **RoPE Support**: None (returns `LLAMA_ROPE_TYPE_NONE`)

### Forward Pass Implementation

The ESN forward pass (`llm_build_esn` in `src/llama-model.cpp`) consists of:

1. **Input Projection**: `W_in * input_embeddings`, scaled by `esn_input_scaling`,
   plus the optional `esn_input_bias` and `esn_reservoir_bias` (both are
   constant over time, so they are added once per batch, not per timestep)
2. **Optional Feedback**: when `esn_feedback_scaling > 0` and
   `esn_feedback_weights` is loaded, add `esn_feedback_scaling` times the
   `W_fb` row of each input token (a `get_rows`, done for the whole batch
   before the recurrence because it does not depend on the state)

   With token inputs, steps 1 and 2 depend on the token alone. When the
   model has `esn_input_table`, its row for each token replaces them (one
   `get_rows`); embedding inputs still take steps 1 and 2
3. **State Retrieval**: Get previous reservoir state via
   `llama_memory_recurrent_context::get_s_l(0)` and `build_rs()`
4. **Recurrent Projection**: `W_res * x(t)`, added to the input drive
5. **Activation**: Apply the selected activation (`esn_activation_type`:
   tanh, sigmoid, or leaky ReLU with slope 0.01, matching the training script)
6. **Leaky Integration**: `x(t+1) = (1-α)*x(t) + α*f(...)` (skipped when α = 1)
7. **State Storage**: `ggml_cpy` new state back to `esn_states_all` at
   `kv_head * reservoir_size`
8. **Output Stacking**: for multi-token batches, each step's state is
   written into its row of a `[reservoir_size, n_seq_tokens, n_seqs]`
   buffer with `ggml_set_rows`, one row copy per step. Each write returns a
   view of the whole buffer that depends on the previous one, so reading
   the last sees them all. The row index is a per-step view of a CPU input,
   which a GPU backend would copy as one split input per step (and abort
   past `GGML_SCHED_MAX_SPLIT_INPUTS`), so with more than one backend the
   steps are concatenated in a balanced `ggml_concat` tree instead
9. **Output Normalization & Projection**: RMS norm, then
   `W_out * reservoir_state → vocabulary_logits`

Notes on features that live outside the inference path:

- **`esn_noise_level`**: injected into the reservoir update during Python
  training (`scripts/esn_training.py`) for regularization; omitted from the
  deterministic inference graph
- **`esn_bidirectional`**: not supported. Inference is causal, and the
  training script rejects `--bidirectional` and the loader refuses a GGUF
  that sets it
- **`esn_spectral_radius`, `esn_sparsity`**: baked into the reservoir
  weight matrix at training time; the inference graph does not need to see
  them beyond metadata

### Memory Management

ESN state management leverages llama.cpp's recurrent memory system:

- Reservoir states are stored in `llama_memory_recurrent`
- Supports sequence management (copy, clear, remove operations)
- Handles multi-sequence batching
- Provides efficient state serialization/deserialization

## GGUF Model Format

ESN models use the standard GGUF format with ESN-specific metadata:

### Required Metadata Keys

```
# Architecture
general.architecture = "esn"

# Model dimensions
esn.vocab_size = <vocabulary size>
esn.embedding_length = <input embedding dimension>
esn.context_length = <maximum sequence length>

# ESN hyperparameters
esn.reservoir_size = <number of reservoir neurons>
esn.spectral_radius = <reservoir spectral radius>
esn.sparsity = <connection sparsity>
esn.leaking_rate = <leaking integration rate>
esn.input_scaling = <input scaling factor>

# Normalization
esn.attention.layernorm_rms_eps = <RMS norm epsilon>
```

### Required Tensors

```
# Core tensors
token_embd.weight          # [n_embd, n_vocab]
output_norm.weight         # [reservoir_size]
esn_input_weights.weight   # [reservoir_size, n_embd]
esn_reservoir_weights.weight # [reservoir_size, reservoir_size]
esn_output_weights.weight  # [n_vocab, reservoir_size]
```

## Usage Examples

### Model Loading

```cpp
#include "llama.h"

// Standard model loading
llama_backend_init();
auto params = llama_model_default_params();
auto* model = llama_load_model_from_file("esn_model.gguf", params);

// Verify ESN architecture
if (llama_model_arch(model) == LLM_ARCH_ESN) {
    printf("Loaded ESN model successfully\n");
}
```

### Inference

```cpp
// Create context with recurrent memory
auto ctx_params = llama_context_default_params();
ctx_params.n_ctx = 2048;
auto* ctx = llama_new_context_with_model(model, ctx_params);

// Process sequence
std::vector<llama_token> tokens = {1, 2, 3, 4, 5};
llama_batch batch = llama_batch_init(tokens.size(), 0, 1);

for (size_t i = 0; i < tokens.size(); ++i) {
    batch.token[i] = tokens[i];
    batch.pos[i] = i;
    batch.seq_id[i][0] = 0;
    batch.n_seq_id[i] = 1;
    batch.logits[i] = (i == tokens.size() - 1);
}
batch.n_tokens = tokens.size();

// Decode (reservoir state automatically managed)
if (llama_decode(ctx, batch) != 0) {
    fprintf(stderr, "Failed to decode\n");
}

// Get logits for next token prediction
float* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
```

## Performance Characteristics

### Computational Complexity

- **Forward Pass**: O(R² + R*E + R*V) where R=reservoir_size, E=embedding_dim, V=vocab_size
- **Memory**: O(R² + R*E + R*V + R*B) where B=batch_size
- **No Backpropagation**: Only output weights are trainable

### Memory Usage

ESN memory usage scales with:
- Reservoir size squared (for W_res)
- Reservoir × embedding dimension (for W_in)
- Vocabulary × reservoir size (for W_out)
- Recurrent state storage per sequence

### Scaling Properties

ESNs exhibit favorable scaling properties:
- **Linear inference time** in sequence length (vs quadratic for attention)
- **Constant memory** per timestep (vs growing KV cache)
- **Parallel processing** across batch dimension
- **Efficient long sequences** due to recurrent nature

## Limitations and Considerations

### Current Limitations

1. **Training**: Only inference is implemented; training requires external tools
2. **Initialization**: Reservoir weights should satisfy echo state property
3. **Hyperparameter Sensitivity**: Performance depends critically on spectral radius
4. **Limited Context**: No explicit attention mechanism for long-range dependencies
5. **Output feedback is the emitted token, not the readout**: `W_fb` is
   applied to the one-hot of the token emitted at t-1, which training
   (teacher forcing) and inference both know exactly. Feeding back the
   readout's own prediction instead would put `W_out` inside the recurrence
   whose states it is fit to: ridge regression has no closed form for that,
   and refitting on states regenerated with the previous readout did not
   converge in tests. GGUF files with feedback written before
   `esn.feedback_token` (which fed back raw `W_out * x` logits at inference
   and something else in training) are refused at load and need retraining.
6. **No bidirectional reservoirs**: the readout predicts the next token, and
   at position t a backward pass has already read it, so backward-pass
   states would leak the target into training and validation while
   inference runs forward only. `esn_training.py` rejects `--bidirectional`
   and `"bidirectional": true` configs, and the loader refuses GGUF files
   that set `esn.bidirectional`.
7. **One reservoir**: `esn.n_levels` > 1, `esn.lateral_links` and
   `esn.top_down_mod` describe hierarchical reservoirs that the graph does
   not build; the loader refuses models that set them rather than running a
   single reservoir in their place.

### Best Practices

1. **Spectral Radius**: Keep < 1.0 for stability, typically 0.8-0.95
2. **Reservoir Size**: Should be 10-100x larger than input/output dimensions
3. **Sparsity**: 5-20% connectivity often optimal
4. **Leaking Rate**: Use < 1.0 for better temporal processing

## Integration with llama.cpp Ecosystem

### Supported Features

- ✅ Standard model loading and inference
- ✅ Batch processing
- ✅ Sequence management (copy, clear, remove)
- ✅ State serialization/deserialization
- ✅ Multi-backend support (CPU, GPU)
- ✅ Quantization support
- ✅ Server integration

### Ecosystem Compatibility

ESN models integrate seamlessly with:
- **llama-server**: OpenAI-compatible API
- **llama-cli**: Command-line interface
- **llama-bench**: Performance benchmarking
- **llama-quantize**: Model quantization
- **Python bindings**: Through llama-cpp-python

## End-to-End Usage

The full pipeline from an untrained reservoir to running inference:

```bash
# 1. Train (or just initialize) an ESN model and export as GGUF.
#    See "Training ESN Models" below for all flags.
python scripts/esn_training.py \
    --reservoir-size 1024 \
    --spectral-radius 0.95 \
    --leaking-rate 0.3 \
    --activation tanh \
    --train-data corpus.txt \
    --output esn-model.gguf

# 2. Build llama-cli (and the ESN inference smoke test).
cmake -B build
cmake --build build -j --target llama-cli test-esn-inference

# 3. Verify the pipeline on raw token IDs (no tokenizer needed).
build/bin/test-esn-inference esn-model.gguf
```

### Tokenizer

`scripts/esn_training.py` writes a 256-token byte-level tokenizer by
default (the `--tokenizer byte` mode), using the RWKV-style escape format
llama.cpp supports. Any input string is split into its UTF-8 bytes and
each byte becomes its own token, which gives:

- `llama-cli -p "..."` works directly, no extra setup
- `vocab_size` is 256 (the default; any other value is an error with
  this tokenizer)
- Round-trip is exact — a model that memorizes a byte-sequence produces
  that sequence back verbatim

For non-text sequence-learning use cases (where the ESN is driven with
raw token IDs via `llama_batch_get_one`, as in
`tests/test-esn-inference.cpp`), pass `--tokenizer none` to write no text
tokenizer (tokenizer model `no_vocab`). The model loads but `llama-cli -p
"..."` will not work on it.

Switching to a larger pretrained tokenizer (SPM/BPE) is a straightforward
extension of `_write_tokenizer()` — call `writer.add_tokenizer_model(...)`,
`writer.add_token_list(...)`, `writer.add_token_scores(...)`, and
`writer.add_token_types(...)` with the data from any compatible GGUF.

### Hyperparameter split: inference-time vs training-time

- Inference-time (consumed by `llm_build_esn`): activation choice, feedback
  (when `esn_feedback_scaling > 0` and `esn_feedback_weights` are in the GGUF),
  input bias, reservoir bias, leaky integration, input scaling.
- Training-time-only: noise injection, spectral radius scaling, sparsity
  masking.

## Testing

Run ESN-specific tests:

```bash
# Static architecture tests (no model file needed)
cmake --build build --target test-esn
ctest --test-dir build -R test-esn --verbose

# End-to-end inference smoke test (loads a real GGUF, runs llama_decode)
cmake --build build --target test-esn-inference
python scripts/esn_training.py --init-only --reservoir-size 128 \
    --vocab-size 1024 --embedding-dim 64 --output /tmp/esn-smoke.gguf
build/bin/test-esn-inference /tmp/esn-smoke.gguf
```

Static test coverage (`tests/test-esn.cpp`):
- Architecture recognition and name mapping
- Recurrent / hybrid / diffusion classification
- Tensor info mappings for all 7 ESN tensors (weights + biases + feedback
  + input table)
- Hyperparameter defaults (base + extended)
- Tensor name generation
- `n_embd_s()` returns `reservoir_size` for ESN models

Activation accuracy (`tests/test-esn-activations.cpp`, a CTest): the CPU
backend's `tanh` and `sigmoid` against double-precision references, special
values, and multi-row / multi-thread consistency.

End-to-end inference coverage (`tests/test-esn-inference.cpp`):
- GGUF metadata & tensor-shape loading for `arch=esn`
- `llama_memory_recurrent` allocation sized by `n_embd_s()`
- `llm_build_esn` forward graph construction and execution
- `llama_decode` success and finite logits

## Training ESN Models

ESN training is provided via a Python utility in `scripts/esn_training.py`:

```bash
# Install dependencies
pip install numpy

# Basic training on a text file (byte-level next-token prediction);
# --train-data is required unless --init-only
python scripts/esn_training.py --reservoir-size 1024 --train-data corpus.txt --output model.gguf

# With custom configuration
python scripts/esn_training.py --config esn_config.json --output trained.gguf

# Initialize without training (creates random weights)
python scripts/esn_training.py --init-only --reservoir-size 2048 --output untrained.gguf
```

### Training Parameters

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Vocabulary Size | `--vocab-size` | 256 | Token vocabulary size (256 for the byte tokenizer) |
| Embedding Dim | `--embedding-dim` | 512 | Input embedding dimension |
| Reservoir Size | `--reservoir-size` | 1024 | Number of reservoir neurons |
| Spectral Radius | `--spectral-radius` | 0.95 | Reservoir stability parameter |
| Sparsity | `--sparsity` | 0.1 | Reservoir connectivity fraction |
| Leaking Rate | `--leaking-rate` | 0.3 | Memory vs adaptation trade-off |
| Regularization | `--regularization` | 1e-6 | Ridge regression lambda |
| Washout | `--washout` | 100 | Initial timesteps to discard |
| Validation split | `--val-fraction` | 0.1 | Sequences held out to pick lambda and fit the temperature |
| Lambda search | `--reg-grid` | (off) | Comma-separated lambdas, chosen by held-out NLL |
| No calibration | `--no-calibrate` | (off) | Export raw ridge outputs as logits |
| No input table | `--no-input-table` | (off) | Do not store `esn_input_table` (`n_vocab * reservoir_size` floats; inference then computes the input drive) |
| Reservoir type | `--outtype` | `f32` | `f16`, `q8_0` or `q4_0` (16, 8.5 or 4.5 bits per weight) shrink the reservoir and its per-token read; the quantized types need a reservoir size that is a multiple of 32 |

### Training Process

ESN training uses ridge regression on output weights only:

1. **Initialization**: Random reservoir and input weights (fixed). The
   reservoir is rescaled to the target spectral radius, computed exactly up
   to R=2048 and estimated with a 128-step Arnoldi iteration above that
   (`np.linalg.eigvals` is O(R³) and took longer than training at R=8192).
2. **State Collection**: Run input sequences through the reservoir in
   batches (one matrix-matrix product per timestep for the whole batch),
   accumulating `XᵀX` and `XᵀY` without materializing `X`
3. **Ridge Regression**: `W_out = Y^T * X * (X^T * X + λI)^(-1)`; with
   `--reg-grid`, the λ with the lowest held-out NLL is used
4. **Calibration**: ridge regression to one-hot targets produces
   probability-scale outputs, so using them directly as logits gives a
   nearly uniform softmax. A single logit scale β is fitted on the
   held-out sequences (NLL is convex in β) and folded into `W_out`. It does
   not change the argmax.
5. **Refit and Export**: the readout is refit on training + held-out data
   with the chosen λ and saved as GGUF for llama.cpp inference

With `--outtype f16`, `q8_0` or `q4_0`, `W_res` is rounded to that type
before state collection, so the readout is trained on the reservoir
inference runs. The reservoir is random, so rounding it costs little; what
matters is that the readout sees the rounded one. Quantizing an F32-trained
model afterwards (as `llama-quantize` would) puts a different reservoir
under the readout: at R=2048 that took held-out entropy from 4.77 to 6.86
bits/byte, where training on the Q4_0 reservoir gives 4.78.

ggml's CPU backend also rounds the state vector before each `W_res`
product: to F16 for an F16 reservoir, and to Q8_0 blocks (`rint(x * 127 /
amax)` per 32 values, with the scale stored as F16) for Q8_0 and Q4_0 ones.
State collection rounds it the same way, so the Python validation numbers
are what llama.cpp computes on CPU: logits agree to ~1e-6 for Q8_0 and Q4_0
reservoirs, where unrounded states differed by ~5e-3 and changed the
next-byte argmax at 2-22 % of positions. An F16 reservoir still differs by
~1e-4 at some positions, where a state element sits on an F16 rounding
boundary. Other backends round the vector their own way.

### Measured Effect of the Training and Inference Optimizations

Byte-level models (`--tokenizer byte --embedding-dim 128 --spectral-radius
0.9 --leaking-rate 0.3 --sparsity 0.1 --seq-length 256 --washout 32`)
trained on ~86 kB of this repository's markdown and evaluated with
`test-esn-perplexity` on a separate ~9.6 kB held-out file; 4-thread CPU.
"Before" is the previous training script and graph builder.

| | R=2048 before | R=2048 after | R=8192 before | R=8192 after | R=8192 after, F16 |
|---|---:|---:|---:|---:|---:|
| sequences / λ | 500 / 1e-2 | 500 / 1e-2 | 300 / 1e-2 | 300 / 100 (searched) | 300 / 100 |
| training wall time | 67 s | 24 s | 510 s | 128 s | 107 s |
| held-out bits/byte | 7.776 | 5.767 | 7.664 | 5.653 | 5.653 |
| held-out perplexity | 219.1 | 54.4 | 202.7 | 50.3 | 50.3 |
| held-out argmax accuracy | 28.6 % | 28.6 % | 26.0 % | 28.6 % | 28.6 % |
| GGUF size | 20 MB | 20 MB | 281 MB | 281 MB | 147 MB |
| prompt processing, pp512 (tok/s) | 1236 | 3571 | 162 | 247 | 443 |
| generation, tg128 (tok/s) | 2879 | 2701 | 203 | 229 | 404 |

The R=2048 throughput numbers are an A/B of the old and new graph builder
on the same model and machine (`llama-bench -r 10`); generation is
unchanged within noise there, since a single-token step does no output
stacking. At R=8192 generation is bound by reading `W_res` once per token
(256 MiB in F32), so the F16 reservoir nearly doubles it. Fixing λ at
R=8192 matters as much as calibration: at λ=1e-2 the readout overfits
(93.7 % training vs 24.1 % validation accuracy) and calibration alone
reaches only 6.70 bits/byte.

With the Arnoldi spectral-radius estimate, R=16384 is now practical on the
same 4-core machine: an F16 model (λ searched over 10/100/1000) trains in
584 s, reaches 5.635 bits/byte (perplexity 49.7, argmax 29.9 %) on the same
held-out file, and generates at 70 tok/s from a 536 MiB GGUF. The previous
script had not finished initializing it after 30 minutes. That this barely
improves on R=8192 shows the ~67k-sample corpus, not the reservoir, is the
limit at this scale.

### Graph Reuse and Output Stacking

Two inference-side changes, each measured as an A/B on the same machine
(4-thread CPU, medians of five interleaved `llama-bench -r 3` runs,
untrained Q4_0 reservoirs at R ≤ 1024 and the trained Q4_0 models from
the table below at R = 2048 and 8192):

- **Decode graph reuse.** The recurrent-state input never reported that a
  graph could be reused, so llama.cpp rebuilt and re-allocated the graph
  for every generated token of every recurrent architecture (Mamba and
  RWKV as well as ESN). It now allows reuse when the state cells the graph
  was built for (`head`, `rs_z`, `n_rs`) are unchanged, which is the case
  for consecutive decode steps.
- **O(n) output stacking.** Prompt processing stacked the per-token
  states in a balanced `ggml_concat` tree, copying each state log2(n)
  times; it now writes each state once (step 8 above) when the CPU is the
  only backend.

| R | 256 | 1024 | 2048 | 8192 |
|---|---:|---:|---:|---:|
| tg256 (tok/s), rebuild every token → reuse | 16578 → 25732 | 9182 → 11673 | 4437 → 5458 | 597 → 672 |
| pp512 (tok/s), concat tree → `set_rows` | 66057 → 65988 | 11276 → 14196 | 5807 → 7494 | 729 → 723 |

Both leave the logits bit-identical. Stacking costs nothing measurable when
the states stay in cache (R = 256) or when reading `W_res` dominates
(R = 8192).

Without an input table, batched and token-by-token decoding of the same
text take different matmul kernels for `W_in`, so their logits differ by rounding (~1e-5 with an F32
reservoir). With a Q8_0 or Q4_0 reservoir that rounding can flip the
quantization of one state value, after which the later logits of that
sequence differ visibly (up to 0.1 relative to the largest logit in one
`test-esn-inference` sequence).

### Vectorized Activations and the Input Table

At small and medium reservoirs, a token's time went mostly to two things
other than reading `W_res`:

- **The activation.** The CPU backend computed `tanh` with libm's scalar
  `tanhf`, about 20 ns per element here: at R = 1024 it was 14 % of the
  single-thread instructions, more than everything but the `W_res` matvec.
  A state is one row, so the op also ran on one thread while the others
  waited. `tanh` and `sigmoid` in `ggml/src/ggml-cpu/unary-ops.cpp` are now
  branch-free (a minimax polynomial below |x| = 0.625, an inline `exp`
  above), so the compiler vectorizes them: 0.8 ns per element with AVX2,
  within 1.2 ulp of `tanh` and 2 ulp of the logistic function
  (`tests/test-esn-activations.cpp`), like libm's. This applies to every
  model that uses `ggml_tanh` or `ggml_sigmoid` on the CPU.
- **Graph nodes.** A decode step is a short chain of small ops, and each
  op ends with a barrier across threads. The input drive took up to seven
  of them (projection, scale, two bias adds, feedback lookup, scale, add)
  although it depends on the token alone; `scripts/esn_training.py` now
  stores it per token as `esn_input_table` (`--no-input-table` to omit it),
  and the graph reads it with one `get_rows`. A scale by 1 is also no
  longer emitted. For the default 256-token byte vocabulary the table adds
  1 KiB per reservoir unit to the GGUF (+60 % at R = 256 Q4_0, +11 % at
  R = 2048 F32).

Before (libm `tanhf`, no table) → after, on the same machine (4-thread CPU,
medians of eight interleaved `llama-bench -r 3` runs; `--use-bias
--leaking-rate 0.5` reservoirs, Q4_0 at R ≤ 1024, F32 at R = 2048, and the
R = 8192 Q4_0 model of the next table without a table):

| R | 256 | 1024 | 2048 | 8192 |
|---|---:|---:|---:|---:|
| pp512 (tok/s) | 64846 → 123743 | 20361 → 37590 | 4784 → 6000 | 893 → 970 |
| tg256 (tok/s) | 25659 → 32764 | 11956 → 18560 | 3470 → 4440 | 766 → 852 |

Most of this is the activation (pp512 +64 % and tg256 +27 % at R = 1024
from it alone); the table adds 16-18 % to R = 1024 decoding and is within
this machine's noise at R = 256. Logits move by rounding only (≤ 1.1e-6
relative to the largest logit for F32 reservoirs; held-out perplexity and
the C++/Python parity are unchanged).

### Quantized Reservoirs

The same corpus and held-out file, with `--embedding-dim 64 --spectral-radius
0.9 --leaking-rate 0.5 --use-bias --seq-length 256 --washout 32 --reg-grid
0.1,1,10,100` (674 windows; λ=1 was selected for every model), 4-thread CPU,
`llama-bench -r 3`:

| | R=2048 F32 | R=2048 F16 | R=2048 Q8_0 | R=2048 Q4_0 | R=8192 F32 | R=8192 F16 | R=8192 Q8_0 | R=8192 Q4_0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GGUF size | 19 MB | 11 MB | 7 MB | 5 MB | 267 MB | 139 MB | 79 MB | 47 MB |
| held-out bits/byte | 4.772 | 4.772 | 4.779 | 4.783 | 4.331 | 4.330 | 4.336 | 4.346 |
| held-out perplexity | 27.3 | 27.3 | 27.5 | 27.5 | 20.1 | 20.1 | 20.2 | 20.3 |
| held-out argmax accuracy | 38.5 % | 38.5 % | 38.7 % | 38.6 % | 46.5 % | 46.4 % | 46.2 % | 46.4 % |
| pp512 (tok/s) | 4935 | 6982 | 6979 | 7976 | 244 | 592 | 826 | 951 |
| tg128 (tok/s) | 3796 | 5174 | 5770 | 6677 | 220 | 548 | 758 | 922 |

Training time does not depend on the type (47-48 s at R=2048, 448-480 s
at R=8192). At R=8192 a token's work is mostly reading `W_res`, so Q4_0
generates 4.2x faster than F32 and 1.7x faster than F16, for 0.015
bits/byte.

## Related research directions (not part of this implementation)

The items below are a **separate research roadmap**, not pending work on
the ESN runtime shipped in this fork. The ESN implementation described
above (architecture registration, GGUF loader, graph builder, byte
tokenizer, training script, three test binaries, docs) is complete per
[`docs/esn-completion.md`](esn-completion.md); none of the items in this
section are prerequisites for that completion.

These are directions a *downstream* project could take; they would be
new, separable PRs if they were pursued:

1. **Hierarchical ESNs** — Multi-layer reservoir architectures. See
   `DTECHO.md`, which is a vision document describing a possible
   Deep-Tree-Echo architecture built *on top of* this ESN runtime. The
   `esn.n_levels` / `esn.branching_factor` / `esn.lateral_links` /
   `esn.top_down_mod` keys are reserved for it; models that set them are
   refused until it exists.
2. **Advanced initialization** — Dynamic spectral-radius adaptation.
3. **Adaptive hyperparameters** — Online tuning of the leaking rate.
4. **Sparse operations** — A dedicated sparse `ggml_mul_mat` path for
   the reservoir weight matrix (currently it runs as dense).
5. **Larger benchmarks** — Transformer-vs-ESN comparisons on
   WikiText-scale corpora with a GPU training pipeline.

Items 1–5 do not block the goal of integrating ESNs into the llama.cpp
runtime, which is the scope of this fork.

## ESLLM: Synchronous Infer-Train Architecture

`esn.cpp` is evolving towards a fully adaptive **ESLLM** (Echo State Language Model) capable of learning from every interaction in real time. See `docs/esllm-roadmap.md` for the complete multi-phase implementation roadmap.

### Key Concepts

**Synchronous infer-train**: A generation step that atomically interleaves inference with online readout adaptation:

```
Token t arrives
    │
    ├─► [Infer step]  Forward pass → reservoir state x(t) → logits
    │
    ├─► [Feedback]    Optional feedback event (correction, reward, tool result)
    │                 accumulated in bounded replay buffer
    │
    └─► [Train step]  llama_esn_update_readout: per-context W_out only
                      Reservoir weights W_res and W_in are NEVER modified
```

### Online readout adaptation (C API)

A model exported with `--save-online-metadata` (`esn.online_learning.enabled`)
makes every context record the readout input `x` (the RMS-normalized state)
of each position it decodes with logits, keyed by `(seq_id, pos)`; the
newest `esn.replay_window` positions per sequence are kept. Feedback names a
position whose logits were read and what the readout should have produced
there:

```c
// decode, read logits at position p, sample `tok`, ...
llama_esn_feedback_event ev = {
    LLAMA_ESN_FEEDBACK_CORRECTION, /*seq_id*/ 0, /*pos*/ p, /*signal*/ 0.0f,
    &correct_token, 1,
};
llama_esn_submit_feedback(ctx, &ev, 1);

// the mode, rates and window exported with the model, or set your own
llama_esn_online_params op = llama_esn_model_online_params(model);
op.update_mode    = 0;      // 0 ridge, 1 RLS, 2 SGD
op.regularization = 1e-2f;
llama_esn_update_readout(ctx, &op);   // later decodes on ctx use the new W_out
// ...
llama_esn_reset_adaptation(ctx);      // back to the model's W_out
```

The first update copies `W_out` (dequantized to F32) into a buffer owned by
the context, which the graph then reads instead of the model tensor; the
model, and every other context on it, keep the checkpoint. Updates:

| Mode | Objective | Cost per update |
|------|-----------|-----------------|
| 0 ridge | `min Σ wᵢ‖T yᵢ − W xᵢ‖² + λ‖W − W₀‖²` over the newest `replay_window` samples, refit from W₀ (Woodbury: an n×n solve for n samples) | O(V·R·n + n³) |
| 1 RLS | recursively from the current W with forgetting f = `1 − decay_rate`: after k samples `min Σ f^(k−i) wᵢ‖T yᵢ − W xᵢ‖² + f^k λ‖W − W₀‖²`, keeping P = (f^k λI + Σ f^(k−i) w x xᵀ)⁻¹. With decay 0 it is ridge over every sample since the last restart; with decay > 0 the pull toward W₀ fades with the samples (use ridge for a fixed anchor) | O(R² + V·R) per sample, R² doubles of memory |
| 2 SGD | one weighted cross-entropy step on softmax(W x) per sample, plus `regularization`·(W − W₀) | O(V·R) per sample |

`T` is `esn.output_temperature`: `esn_training.py` fits `W_out` to one-hot
targets and folds the calibrated softmax temperature into it, so the
exported readout is fit to `T · onehot` and ridge/RLS keep that scale. A
readout without the key (SGD-trained or untrained) can only be adapted with
SGD. Corrections and tool results are samples of weight 1; rewards weight
the emitted token by the reward (negative only with SGD); confidence
weights it by a value in [0, 1]. Latency events are rejected: they carry
no token target. A readout stored in one of the CPU's extra buffer types
(REPACK, AMX) cannot be read back; load such models with
`use_extra_bufts = false` (`--no-repack`).
The adapted readout is not included in `llama_state_*` data.

Recordings follow the sequence: decoding a position again (after
`llama_memory_seq_rm` or a memory clear) discards what was recorded at and
after it, even when the new tokens produce no logits, and feedback is
refused (-3) for positions past the newest one the context's memory holds,
and for anything recorded before the memory replaced the sequence's content
(`llama_state_*set_data`/`load_file`, `llama_memory_seq_cp` onto it).
A batch of feedback that would overflow `esn.online_learning.buf_size` is
refused (-5) before anything is copied. If a least-squares solve is
numerically singular (λ below the rounding of `Σ w x xᵀ`), the update
returns -9 and leaves the readout and the pending samples as they were;
raise `regularization`. A LoRA on `esn_output_weights` and an adapted
readout exclude each other: the update returns -10 while such an adapter
is set, and `llama_set_adapter_lora` returns -1 for one while the readout
is adapted (`llama_esn_reset_adaptation` lifts it).
`tests/test-esn-online.cpp` checks each mode against its closed form, and
the rewind, singular-solve and LoRA rules.

**Proprioceptive feedback**: Signals originating from the model's own execution context:
- `CORRECTION` — explicit user or verifier correction
- `REWARD` — scalar RL-style reward/penalty
- `CONFIDENCE` — model's softmax entropy (self-supervised)
- `TOOL_RESULT` — structured outcome of a function call
- `LATENCY` — observed generation latency

### ESLLM Invariants

| Invariant | Guarantee |
|-----------|-----------|
| Bounded cost | Train step cost is set by the mode (table above), independent of context length |
| Stable dynamics | Spectral radius validated to be in (0, 1) at load time |
| Non-destructive | Only W_out is adapted; W_res, W_in are read-only |
| Reversible | `llama_esn_reset_adaptation` returns a context to the checkpoint W_out |
| Regularized | Ridge is anchored to the checkpoint by λ‖W − W₀‖²; RLS by the same term decayed by `(1 − decay_rate)^k` (fixed at decay 0); SGD optionally |

### Capability Matrix

| Capability | Status |
|-----------|--------|
| Single-layer ESN inference | ✅ |
| Parameter validation at load | ✅ |
| Online RLS/SGD (Python) | ✅ |
| Online learning GGUF metadata | ✅ |
| Feedback API + online readout updates (C, ridge/RLS/SGD) | ✅ |
| Synchronous infer-train runtime loop (in a CLI/server) | 🔄 Phase 4 |
| Hierarchical reservoir (DTE) | 🔄 Phase 6 |
| ESN–Transformer hybrid | 🔄 Phase 7 |
| Safety governance + rollback | 🔄 Phase 8 |

## References

1. Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks. GMD Technical Report 148.
2. Lukoševičius, M., & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training. Computer Science Review, 3(3), 127-149.
3. Verstraeten, D., Schrauwen, B., D'Haene, M., & Stroobandt, D. (2007). An experimental unification of reservoir computing methods. Neural networks, 20(3), 391-403.
