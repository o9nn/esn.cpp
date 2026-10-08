# Echo State Networks (ESN) in llama.cpp

This document describes the implementation of Echo State Networks (ESNs) in llama.cpp, providing a comprehensive model for reservoir computing within the ggml framework.

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
- `W_fb` is the optional output-to-reservoir feedback matrix
- `W_out` is the trained output readout matrix
- `b` are the optional input/reservoir bias vectors
- `f` is the activation function (tanh, sigmoid, or leaky-relu)
- `α` is the leaking rate (controls memory vs. adaptation trade-off)

The feedback term `W_fb * y(t-1)` is only included when `esn_feedback_scaling > 0`
and the `esn_feedback_weights` tensor is present in the GGUF file.

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
| `esn_bidirectional` | Enable bidirectional processing | false | bool |

#### Tensors

| Tensor | Shape | Description |
|--------|-------|-------------|
| `esn_input_weights` | `[reservoir_size, n_embd]` | Maps input embeddings to reservoir |
| `esn_reservoir_weights` | `[reservoir_size, reservoir_size]` | Recurrent reservoir connections |
| `esn_output_weights` | `[n_vocab, reservoir_size]` | Maps reservoir to output vocabulary |
| `esn_feedback_weights` | `[reservoir_size, n_vocab]` | Output-to-reservoir feedback (optional) |
| `esn_input_bias` | `[reservoir_size]` | Input projection bias (optional) |
| `esn_reservoir_bias` | `[reservoir_size]` | Reservoir activation bias (optional) |

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
   plus optional `esn_input_bias`
2. **State Retrieval**: Get previous reservoir state via
   `llama_memory_recurrent_context::get_s_l(0)` and `build_rs()`
3. **Recurrent Projection**: `W_res * x(t)`
4. **Optional Feedback**: When `esn_feedback_scaling > 0` and
   `esn_feedback_weights` is loaded, add
   `esn_feedback_scaling * W_fb * (W_out * x(t))` to the pre-activation
5. **Optional Reservoir Bias**: Add `esn_reservoir_bias`
6. **Activation**: Apply the selected activation (`esn_activation_type`:
   tanh, sigmoid, or gelu as a leaky-relu proxy)
7. **Leaky Integration**: `x(t+1) = (1-α)*x(t) + α*f(...)`
8. **State Storage**: `ggml_cpy` new state back to `esn_states_all` at
   `kv_head * reservoir_size`
9. **Output Normalization & Projection**: RMS norm, then
   `W_out * reservoir_state → vocabulary_logits`

Notes on features that live outside the inference path:

- **`esn_noise_level`**: injected into the reservoir update during Python
  training (`scripts/esn_training.py`) for regularization; omitted from the
  deterministic inference graph
- **`esn_bidirectional`**: affects how `W_res` is initialized at training
  time; inference remains strictly causal/forward to preserve streaming
  semantics
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
- `vocab_size` is pinned to 256 (the script will override larger values
  with a warning so training/inference stay consistent)
- Round-trip is exact — a model that memorizes a byte-sequence produces
  that sequence back verbatim

For non-text sequence-learning use cases (where the ESN is driven with
raw token IDs via `llama_batch_get_one`, as in
`tests/test-esn-inference.cpp`), pass `--tokenizer none` to emit the
`no_vocab` stub instead. The model loads but `llama-cli -p "..."` will
not work on it.

Switching to a larger pretrained tokenizer (SPM/BPE) is a straightforward
extension of `_write_tokenizer()` — call `writer.add_tokenizer_model(...)`,
`writer.add_token_list(...)`, `writer.add_token_scores(...)`, and
`writer.add_token_types(...)` with the data from any compatible GGUF.

### Hyperparameter split: inference-time vs training-time

- Inference-time (consumed by `llm_build_esn`): activation choice, feedback
  (when `esn_feedback_scaling > 0` and `esn_feedback_weights` are in the GGUF),
  input bias, reservoir bias, leaky integration, input scaling.
- Training-time-only: noise injection, bidirectional reservoir
  initialization, spectral radius scaling, sparsity masking.

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
- Tensor info mappings for all 6 ESN tensors (weights + biases + feedback)
- Hyperparameter defaults (base + extended)
- Tensor name generation
- `n_embd_s()` returns `reservoir_size` for ESN models

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

# Basic training with sample data
python scripts/esn_training.py --reservoir-size 1024 --output model.gguf

# With custom configuration
python scripts/esn_training.py --config esn_config.json --output trained.gguf

# Initialize without training (creates random weights)
python scripts/esn_training.py --init-only --reservoir-size 2048 --output untrained.gguf
```

### Training Parameters

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Vocabulary Size | `--vocab-size` | 32000 | Token vocabulary size |
| Embedding Dim | `--embedding-dim` | 512 | Input embedding dimension |
| Reservoir Size | `--reservoir-size` | 1024 | Number of reservoir neurons |
| Spectral Radius | `--spectral-radius` | 0.95 | Reservoir stability parameter |
| Sparsity | `--sparsity` | 0.1 | Reservoir connectivity fraction |
| Leaking Rate | `--leaking-rate` | 0.3 | Memory vs adaptation trade-off |
| Regularization | `--regularization` | 1e-6 | Ridge regression lambda |
| Washout | `--washout` | 100 | Initial timesteps to discard |

### Training Process

ESN training uses ridge regression on output weights only:

1. **Initialization**: Random reservoir and input weights (fixed)
2. **State Collection**: Run input sequences through reservoir
3. **Ridge Regression**: `W_out = Y^T * X * (X^T * X + λI)^(-1)`
4. **GGUF Export**: Save trained model for llama.cpp inference

## Future Enhancements

Potential future improvements:

1. **Hierarchical ESNs**: Multi-layer reservoir architectures (see `DTECHO.md`)
2. **Advanced Initialization**: Dynamic spectral radius adaptation
3. **Adaptive Parameters**: Dynamic leaking rate tuning
4. **Sparse Operations**: Optimized sparse matrix operations for reservoir computation
5. **Deep Tree Echo**: Hierarchical AGI architecture exploration

## References

1. Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks. GMD Technical Report 148.
2. Lukoševičius, M., & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training. Computer Science Review, 3(3), 127-149.
3. Verstraeten, D., Schrauwen, B., D'Haene, M., & Stroobandt, D. (2007). An experimental unification of reservoir computing methods. Neural networks, 20(3), 391-403.