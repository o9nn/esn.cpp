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

The ESN dynamics are governed by:

```
x(t+1) = (1-α) * x(t) + α * tanh(W_res * x(t) + W_in * u(t))
```

Where:
- `x(t)` is the reservoir state at time t
- `u(t)` is the input at time t  
- `W_res` is the reservoir weight matrix (with spectral radius < 1)
- `W_in` is the input weight matrix
- `α` is the leaking rate (controls memory vs. adaptation trade-off)

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

The ESN forward pass consists of:

1. **Input Projection**: `W_in * input_embeddings`
2. **State Retrieval**: Get previous reservoir state from recurrent memory
3. **Reservoir Update**: Apply ESN dynamics with leaky integration
4. **State Storage**: Store new reservoir state for next timestep
5. **Output Projection**: `W_out * reservoir_state → vocabulary_logits`

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

## Testing

Run ESN-specific tests:

```bash
# Build and run ESN tests
cmake --build build --target test-esn
ctest --test-dir build -R test-esn --verbose
```

Test coverage includes:
- Architecture recognition
- Tensor mapping validation  
- Hyperparameter initialization
- Recurrent memory integration
- Forward pass computation

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