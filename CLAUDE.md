# CLAUDE.md - Development Guide for esn.cpp

This file provides guidance for Claude Code and other AI assistants working on this codebase.

## Project Overview

**esn.cpp** is a fork of llama.cpp extended with **Echo State Network (ESN)** support for reservoir computing. The project integrates ESN architectures into the llama.cpp ecosystem, enabling efficient inference of reservoir computing models alongside traditional transformer-based LLMs.

## Architecture

### Core ESN Components

The ESN implementation follows the standard reservoir computing paradigm:

```
x(t+1) = (1-α) * x(t) + α * tanh(W_res * x(t) + W_in * u(t))
```

**Key Files:**
- `src/llama-arch.h` - Architecture enums (`LLM_ARCH_ESN`, `LLM_TENSOR_ESN_*`)
- `src/llama-arch.cpp` - Architecture mappings and recognition functions
- `src/llama-hparams.h` - ESN hyperparameters (reservoir_size, spectral_radius, etc.)
- `src/llama-model.cpp` - Model loading and forward pass (`llm_build_esn`)
- `docs/esn.md` - Comprehensive ESN documentation
- `tests/test-esn.cpp` - ESN architecture tests
- `scripts/esn_training.py` - Python utility for ESN training and GGUF creation

### ESN Hyperparameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `esn_reservoir_size` | 0 | Number of reservoir neurons |
| `esn_spectral_radius` | 0.95 | Echo state property (must be < 1) |
| `esn_sparsity` | 0.1 | Reservoir connectivity (10%) |
| `esn_leaking_rate` | 1.0 | Memory vs adaptation trade-off |
| `esn_input_scaling` | 1.0 | Input signal scaling |

### Tensor Layout

```
token_embd.weight         [n_embd, n_vocab]
esn_input_weights.weight  [reservoir_size, n_embd]
esn_reservoir_weights.weight  [reservoir_size, reservoir_size]
output_norm.weight        [reservoir_size]
esn_output_weights.weight [n_vocab, reservoir_size]
```

## Build Commands

```bash
# Configure
cmake -B build

# Build all
cmake --build build -j

# Build ESN tests only
cmake --build build --target test-esn

# Run ESN tests
ctest --test-dir build -R test-esn --verbose

# Build main CLI
cmake --build build --target llama-cli
```

## Code Patterns

### Adding New ESN Features

1. **New hyperparameters**: Add to `llama_hparams` in `src/llama-hparams.h`
2. **New tensors**: Add to `llm_tensor` enum in `src/llama-arch.h`, then add mappings in `src/llama-arch.cpp`
3. **Forward pass changes**: Modify `llm_build_esn` class in `src/llama-model.cpp`
4. **GGUF metadata**: Add key to `llm_kv` enum and implement loading in model loading switch

### ESN Forward Pass Location

The ESN forward pass is implemented in `src/llama-model.cpp` around line 19338:

```cpp
struct llm_build_esn : public llm_graph_context_mamba {
    llm_build_esn(const llama_model & model, const llm_graph_params & params)
        : llm_graph_context_mamba(params) {
        // Forward pass implementation
    }
};
```

### Testing Patterns

```cpp
// Test architecture recognition
llm_arch esn_arch = llm_arch_from_string("esn");
assert(esn_arch == LLM_ARCH_ESN);

// Test tensor info
const llm_tensor_info & info = llm_tensor_info_for(LLM_TENSOR_ESN_INPUT_WEIGHTS);
assert(info.layer == LLM_TENSOR_LAYER_INPUT);
```

## Training ESN Models

Use the Python training utility:

```bash
# Basic training
python scripts/esn_training.py --reservoir-size 1024 --output model.gguf

# With custom config
python scripts/esn_training.py --config config.json

# Initialize only (no training)
python scripts/esn_training.py --init-only --output untrained.gguf
```

## Common Tasks

### Debugging ESN Issues

1. Check reservoir spectral radius is < 1.0
2. Verify tensor shapes match expected dimensions
3. Enable debug callbacks with `cb(tensor, "name", -1)`
4. Run test-esn to verify architecture recognition

### Adding New Recurrent Architectures

ESN is recognized as recurrent via `llm_arch_is_recurrent()`. To add similar:

1. Add arch enum to `llm_arch` in `src/llama-arch.h`
2. Add to `LLM_ARCH_NAMES` map in `src/llama-arch.cpp`
3. Add to `llm_arch_is_recurrent()` switch in `src/llama-arch.cpp`
4. Implement graph builder inheriting from `llm_graph_context_mamba`

## Performance Considerations

- ESN has **O(R²)** complexity in reservoir size
- Memory scales with **R² + R*E + R*V** (R=reservoir, E=embedding, V=vocab)
- Linear inference time in sequence length (vs quadratic for attention)
- Constant memory per timestep (no KV cache growth)

## Future Enhancements

Current limitations and planned improvements:

1. **Training Support** - Ridge regression training (see `scripts/esn_training.py`)
2. **Hierarchical ESNs** - Multi-layer reservoir architectures
3. **Adaptive Parameters** - Dynamic leaking rate and spectral radius
4. **Sparse Operations** - Optimized sparse matrix ops for reservoir

## Related Documentation

- `docs/esn.md` - Full ESN technical documentation
- `DTECHO.md` - Deep Tree Echo AGI concepts
- `README.md` - Main project documentation
- `CONTRIBUTING.md` - Contribution guidelines

## Code Style

Follow llama.cpp conventions:
- 4-space indentation
- `snake_case` for functions and variables
- `PascalCase` for classes and types
- Prefix ESN-specific items with `esn_` or `ESN_`
- Use ggml tensor operations for GPU compatibility

## Contact

For ESN-specific issues, check:
- GitHub Issues for esn.cpp repository
- Original llama.cpp documentation for base functionality
- Reservoir computing literature for theoretical background
