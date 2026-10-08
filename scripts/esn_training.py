#!/usr/bin/env python3
"""
ESN Training and Model Creation Utility for esn.cpp

This script provides training capabilities for Echo State Networks and
conversion to GGUF format for use with llama.cpp.

Echo State Networks only require training the output weights (W_out),
while the reservoir weights (W_res) and input weights (W_in) remain fixed
after initialization according to spectral radius constraints.

Features:
- Multiple activation functions (tanh, sigmoid, leaky_relu)
- Output feedback connections for generative tasks
- Bias vectors for input and reservoir
- Bidirectional reservoir processing
- Ridge regression training with regularization

Usage:
    python esn_training.py --config esn_config.json
    python esn_training.py --reservoir-size 1000 --train-data data.txt
    python esn_training.py --reservoir-size 2048 --activation sigmoid --use-bias

Author: esn.cpp contributors
License: MIT
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable
import os
import sys

# Prefer the in-tree gguf package (gguf-py/) so this script matches the
# llama.cpp build exactly, even in an environment where the published
# `gguf` wheel is older than this fork.
_HERE = Path(__file__).resolve().parent
_GGUF_PY = _HERE.parent / "gguf-py"
if _GGUF_PY.is_dir():
    sys.path.insert(0, str(_GGUF_PY))

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError as e:
    raise SystemExit(
        "The 'gguf' package is required. Install the one bundled with this\n"
        "repository by running: pip install -e gguf-py"
    ) from e

# Activation type constants (must match llama-hparams.h)
ACTIVATION_TANH = 0
ACTIVATION_SIGMOID = 1
ACTIVATION_LEAKY_RELU = 2


class ESNConfig:
    """Configuration for Echo State Network."""

    def __init__(
        self,
        vocab_size: int = 32000,
        embedding_dim: int = 512,
        reservoir_size: int = 1024,
        spectral_radius: float = 0.95,
        sparsity: float = 0.1,
        leaking_rate: float = 0.3,
        input_scaling: float = 1.0,
        feedback_scaling: float = 0.0,
        noise_level: float = 0.0,
        activation_type: int = ACTIVATION_TANH,
        bidirectional: bool = False,
        use_bias: bool = False,
        context_length: int = 2048,
        regularization: float = 1e-6,
        seed: int = 42,
        tokenizer: str = "byte",
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leaking_rate = leaking_rate
        self.input_scaling = input_scaling
        self.feedback_scaling = feedback_scaling
        self.noise_level = noise_level
        self.activation_type = activation_type
        self.bidirectional = bidirectional
        self.use_bias = use_bias
        self.context_length = context_length
        self.regularization = regularization
        self.seed = seed
        self.tokenizer = tokenizer  # "byte" | "none"
        if self.tokenizer == "byte" and self.vocab_size != 256:
            # Byte tokenizer must have vocab == 256. Auto-correct and warn
            # rather than silently producing a broken model.
            print(f"[ESNConfig] tokenizer='byte' requires vocab_size=256; "
                  f"overriding vocab_size {self.vocab_size} -> 256")
            self.vocab_size = 256

    @classmethod
    def from_json(cls, path: str) -> "ESNConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config = json.load(f)
        return cls(**config)

    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    def __repr__(self) -> str:
        return f"ESNConfig({self.__dict__})"


def get_activation_fn(activation_type: int) -> Callable:
    """Get activation function based on type."""
    if activation_type == ACTIVATION_SIGMOID:
        return lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    elif activation_type == ACTIVATION_LEAKY_RELU:
        return lambda x: np.where(x > 0, x, 0.01 * x)
    else:  # ACTIVATION_TANH
        return np.tanh


class ESN:
    """
    Echo State Network implementation for training.

    The ESN follows the standard reservoir computing paradigm:
    - Fixed random reservoir with sparse connectivity
    - Spectral radius < 1 for echo state property
    - Only output weights are trained via ridge regression

    Extended features:
    - Multiple activation functions
    - Output feedback connections
    - Bias vectors
    - Bidirectional processing
    """

    def __init__(self, config: ESNConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.activation_fn = get_activation_fn(config.activation_type)

        # Initialize weights
        self.W_in = None      # Input weights
        self.W_res = None     # Reservoir weights
        self.W_out = None     # Output weights (trained)
        self.W_fb = None      # Feedback weights (optional)
        self.b_in = None      # Input bias (optional)
        self.b_res = None     # Reservoir bias (optional)
        self.tok_embd = None  # Token embeddings
        self.output_norm = None  # Output normalization

        self._initialized = False

    def initialize(self) -> None:
        """Initialize all weight matrices according to ESN principles."""
        print(f"Initializing ESN with reservoir size {self.config.reservoir_size}")
        print(f"  Activation: {['tanh', 'sigmoid', 'leaky_relu'][self.config.activation_type]}")
        print(f"  Feedback scaling: {self.config.feedback_scaling}")
        print(f"  Bidirectional: {self.config.bidirectional}")
        print(f"  Use bias: {self.config.use_bias}")

        # Token embeddings (random initialization, to be learned or loaded)
        self.tok_embd = self.rng.standard_normal(
            (self.config.embedding_dim, self.config.vocab_size)
        ).astype(np.float32) * 0.02

        # Input weights: dense random matrix scaled by input_scaling
        self.W_in = self.rng.standard_normal(
            (self.config.reservoir_size, self.config.embedding_dim)
        ).astype(np.float32) * self.config.input_scaling

        # Reservoir weights: sparse random matrix with specified connectivity
        self._init_reservoir_weights()

        # Output normalization (RMS norm weights)
        self.output_norm = np.ones(self.config.reservoir_size, dtype=np.float32)

        # Output weights: in a proper ESN these are the ONLY trainable
        # parameters and are typically fitted via ridge regression on
        # collected reservoir states. We seed them with small random values
        # so an un-trained model still produces non-zero logits and the end-
        # to-end inference path can be smoke-tested without first running a
        # full training pass. Ridge regression (see .train()) overwrites
        # them with the fitted values, so this initial seed has no effect
        # once the model is trained.
        self.W_out = self.rng.standard_normal(
            (self.config.vocab_size, self.config.reservoir_size)
        ).astype(np.float32) * 0.02

        # Optional: feedback weights
        if self.config.feedback_scaling > 0:
            self.W_fb = self.rng.standard_normal(
                (self.config.reservoir_size, self.config.vocab_size)
            ).astype(np.float32) * self.config.feedback_scaling
            print(f"  Initialized feedback weights")

        # Optional: bias vectors
        if self.config.use_bias:
            self.b_in = np.zeros(self.config.reservoir_size, dtype=np.float32)
            self.b_res = self.rng.standard_normal(
                self.config.reservoir_size
            ).astype(np.float32) * 0.1
            print(f"  Initialized bias vectors")

        self._initialized = True
        print("ESN initialization complete")

    def _init_reservoir_weights(self) -> None:
        """
        Initialize reservoir weights with sparse connectivity and
        proper spectral radius scaling.
        """
        size = self.config.reservoir_size
        sparsity = self.config.sparsity

        # Create sparse connectivity mask
        connectivity = self.rng.random((size, size)) < sparsity

        # Random weights for connected elements
        weights = self.rng.standard_normal((size, size)).astype(np.float32)
        weights *= connectivity  # Apply sparsity mask

        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(weights)
        current_spectral_radius = np.max(np.abs(eigenvalues))

        if current_spectral_radius > 0:
            scale = self.config.spectral_radius / current_spectral_radius
            weights *= scale

        self.W_res = weights

        # Verify spectral radius
        eigenvalues = np.linalg.eigvals(self.W_res)
        actual_sr = np.max(np.abs(eigenvalues))
        print(f"  Reservoir spectral radius: {actual_sr:.4f} (target: {self.config.spectral_radius})")
        print(f"  Reservoir sparsity: {1 - np.count_nonzero(self.W_res) / self.W_res.size:.2%} zeros")

    def forward(
        self,
        embeddings: np.ndarray,
        state: Optional[np.ndarray] = None,
        prev_output: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ESN forward pass.

        Args:
            embeddings: Input embeddings [n_tokens, embedding_dim]
            state: Previous reservoir state [reservoir_size] or None
            prev_output: Previous output for feedback [vocab_size] or None

        Returns:
            states: Reservoir states for all timesteps [n_tokens, reservoir_size]
            new_state: Final reservoir state [reservoir_size]
        """
        if not self._initialized:
            raise RuntimeError("ESN not initialized. Call initialize() first.")

        n_tokens = embeddings.shape[0]
        alpha = self.config.leaking_rate

        if state is None:
            state = np.zeros(self.config.reservoir_size, dtype=np.float32)

        states = np.zeros((n_tokens, self.config.reservoir_size), dtype=np.float32)

        for t in range(n_tokens):
            # Project input to reservoir space
            input_proj = self.W_in @ embeddings[t]

            # Add input bias if present
            if self.b_in is not None:
                input_proj += self.b_in

            # Reservoir recurrence
            reservoir_input = self.W_res @ state + input_proj

            # Add feedback if enabled
            if self.W_fb is not None and prev_output is not None:
                reservoir_input += self.W_fb @ prev_output

            # Add reservoir bias if present
            if self.b_res is not None:
                reservoir_input += self.b_res

            # Add noise if specified
            if self.config.noise_level > 0:
                reservoir_input += self.rng.standard_normal(
                    self.config.reservoir_size
                ).astype(np.float32) * self.config.noise_level

            # Apply activation function
            activated = self.activation_fn(reservoir_input)

            # Leaky integration: x(t+1) = (1-α)*x(t) + α*f(...)
            state = (1 - alpha) * state + alpha * activated

            states[t] = state

            # Update prev_output for next iteration (if using feedback)
            if self.W_fb is not None:
                # Simple softmax to get output distribution
                logits = self.W_out @ state
                prev_output = np.exp(logits - np.max(logits))
                prev_output /= prev_output.sum()

        return states, state

    def forward_bidirectional(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Bidirectional ESN forward pass.

        Processes sequence in both directions and concatenates states.

        Args:
            embeddings: Input embeddings [n_tokens, embedding_dim]

        Returns:
            states: Combined bidirectional states [n_tokens, 2*reservoir_size]
        """
        # Forward pass
        states_fwd, _ = self.forward(embeddings)

        # Backward pass
        states_bwd, _ = self.forward(embeddings[::-1])
        states_bwd = states_bwd[::-1]  # Reverse to align with forward

        # Concatenate
        return np.concatenate([states_fwd, states_bwd], axis=1)

    def collect_states(self, token_sequences: np.ndarray) -> np.ndarray:
        """
        Collect reservoir states for training data.

        Args:
            token_sequences: Token IDs [n_sequences, seq_length]

        Returns:
            all_states: Collected states [total_tokens, reservoir_size]
        """
        all_states = []

        for seq in token_sequences:
            # Convert tokens to embeddings
            embeddings = self.tok_embd[:, seq].T  # [seq_length, embedding_dim]

            # Get reservoir states
            if self.config.bidirectional:
                states = self.forward_bidirectional(embeddings)
            else:
                states, _ = self.forward(embeddings)
            all_states.append(states)

        return np.vstack(all_states)

    def train(
        self,
        token_sequences: np.ndarray,
        target_sequences: np.ndarray,
        washout: int = 100
    ) -> float:
        """
        Train output weights using ridge regression.

        This is the key insight of ESNs: only the output layer needs
        to be trained, and it can be done with a simple linear method.

        Args:
            token_sequences: Input token IDs [n_sequences, seq_length]
            target_sequences: Target token IDs [n_sequences, seq_length]
            washout: Number of initial timesteps to discard

        Returns:
            training_accuracy: Accuracy after training
        """
        if not self._initialized:
            raise RuntimeError("ESN not initialized. Call initialize() first.")

        print(f"Collecting reservoir states (washout={washout})...")

        # Collect reservoir states and targets
        all_states = []
        all_targets = []

        for seq_idx, (in_seq, tgt_seq) in enumerate(zip(token_sequences, target_sequences)):
            if seq_idx % 100 == 0:
                print(f"  Processing sequence {seq_idx}/{len(token_sequences)}")

            # Convert tokens to embeddings
            embeddings = self.tok_embd[:, in_seq].T

            # Get reservoir states
            if self.config.bidirectional:
                states = self.forward_bidirectional(embeddings)
            else:
                states, _ = self.forward(embeddings)

            # Remove washout period
            states = states[washout:]
            targets = tgt_seq[washout:]

            all_states.append(states)
            all_targets.extend(targets)

        X = np.vstack(all_states)  # [n_samples, reservoir_size] or [n_samples, 2*reservoir_size]
        y = np.array(all_targets)  # [n_samples]

        state_dim = X.shape[1]
        print(f"Training output weights on {X.shape[0]} samples (state dim: {state_dim})...")

        # Apply RMS normalization to states
        rms = np.sqrt(np.mean(X ** 2, axis=-1, keepdims=True) + 1e-6)
        X_norm = X / rms

        # For bidirectional, we need to adjust the output weights
        if self.config.bidirectional:
            # Combine forward and backward contributions
            X_norm = X_norm[:, :self.config.reservoir_size] + X_norm[:, self.config.reservoir_size:]

        X_norm *= self.output_norm

        # Convert targets to one-hot
        Y = np.zeros((len(y), self.config.vocab_size), dtype=np.float32)
        for i, target in enumerate(y):
            Y[i, target] = 1.0

        # Ridge regression: W_out = Y^T * X * (X^T * X + λI)^(-1)
        reg = self.config.regularization * np.eye(self.config.reservoir_size)
        XtX = X_norm.T @ X_norm + reg
        XtY = X_norm.T @ Y

        # Solve linear system
        self.W_out = np.linalg.solve(XtX, XtY).T

        # Calculate training accuracy
        predictions = X_norm @ self.W_out.T
        pred_tokens = np.argmax(predictions, axis=-1)
        accuracy = np.mean(pred_tokens == y)

        print(f"Training complete. Accuracy: {accuracy:.4f}")

        return accuracy

    def save_gguf(self, path: str) -> None:
        """
        Save the ESN model as a GGUF file compatible with esn.cpp's
        ``llm_build_esn`` graph builder.

        Metadata keys use the ``esn.*`` namespace matching the KV mappings
        declared in ``src/llama-arch.cpp``:

          - ``esn.reservoir_size``   (uint32, required)
          - ``esn.spectral_radius``  (float32, required)
          - ``esn.sparsity``         (float32, required)
          - ``esn.leaking_rate``     (float32, required)
          - ``esn.input_scaling``    (float32, required)
          - ``esn.feedback_scaling`` (float32, optional)
          - ``esn.noise_level``      (float32, optional)
          - ``esn.activation_type``  (uint32, optional: 0=tanh, 1=sigmoid, 2=leaky_relu)
          - ``esn.bidirectional``    (bool, optional)
          - ``esn.attention.layernorm_rms_eps`` (float32, required by loader)

        Tensors emitted (``.weight`` suffix per llama.cpp convention):
          - ``token_embd.weight``          [n_embd, n_vocab]
          - ``esn_input_weights.weight``   [n_embd, reservoir_size]
          - ``esn_reservoir_weights.weight`` [reservoir_size, reservoir_size]
          - ``output_norm.weight``         [reservoir_size]
          - ``esn_output_weights.weight``  [reservoir_size, n_vocab]
          - ``esn_feedback_weights.weight``  (optional)
          - ``esn_input_bias.weight``        (optional)
          - ``esn_reservoir_bias.weight``    (optional)
        """
        if not self._initialized:
            raise RuntimeError("ESN not initialized. Call initialize() first.")

        print(f"Saving model to {path}")

        writer = GGUFWriter(path=path, arch="esn")

        # General metadata
        writer.add_name("ESN Model")

        # Base model dimensions — use the standard key helpers so llama.cpp's
        # auto-derived KV keys resolve correctly (they map arch -> key).
        writer.add_vocab_size(self.config.vocab_size)
        writer.add_embedding_length(self.config.embedding_dim)
        writer.add_context_length(self.config.context_length)
        writer.add_block_count(1)
        writer.add_layer_norm_rms_eps(1e-6)

        # Tokenizer. Two paths are supported:
        #
        #   - "byte" (default): a 256-token byte-level tokenizer written in
        #     the RWKV escape format (\xNN per byte). Produces a working
        #     text-prompt path: any input string is split into its UTF-8
        #     bytes, each mapped to its byte-id. Requires vocab_size == 256.
        #
        #   - "none": emits the "no_vocab" stub. The model loads but text
        #     tokenization is unavailable; drive it with raw token IDs via
        #     llama_batch_get_one. Useful when the ESN is being used as a
        #     non-text sequence learner.
        #
        self._write_tokenizer(writer)

        # ESN-specific hyperparameters (match LLM_KV_ESN_* in llama-arch.cpp)
        writer.add_uint32("esn.reservoir_size",    self.config.reservoir_size)
        writer.add_float32("esn.spectral_radius",  self.config.spectral_radius)
        writer.add_float32("esn.sparsity",         self.config.sparsity)
        writer.add_float32("esn.leaking_rate",     self.config.leaking_rate)
        writer.add_float32("esn.input_scaling",    self.config.input_scaling)
        writer.add_float32("esn.feedback_scaling", self.config.feedback_scaling)
        writer.add_float32("esn.noise_level",      self.config.noise_level)
        writer.add_uint32("esn.activation_type",   self.config.activation_type)
        writer.add_bool("esn.bidirectional",       self.config.bidirectional)

        # Tensors (always .weight-suffixed to match tn(..., "weight") in C++).
        #
        # The gguf writer reverses the numpy shape when it emits tensor info
        # (GGUF ne[0] is fastest-varying). The C++ loader creates these with
        # {ne[0], ne[1]} directly, so numpy shape (a, b) ends up as GGUF
        # [b, a]. The required Python shapes are therefore the REVERSE of
        # the C++ create_tensor shapes:
        #
        #   token_embd.weight            C++ {n_embd, n_vocab}
        #       -> numpy (n_vocab, n_embd)        [we train (n_embd, n_vocab), so transpose]
        #   esn_input_weights.weight     C++ {n_embd, reservoir}
        #       -> numpy (reservoir, n_embd)      [matches training layout]
        #   esn_reservoir_weights.weight C++ {reservoir, reservoir}
        #       -> numpy (reservoir, reservoir)   [square]
        #   output_norm.weight           C++ {reservoir}                  [1-D]
        #   esn_output_weights.weight    C++ {reservoir, n_vocab}
        #       -> numpy (n_vocab, reservoir)     [matches training layout]
        #   esn_feedback_weights.weight  C++ {n_vocab, reservoir}
        #       -> numpy (reservoir, n_vocab)     [matches training layout]
        #   esn_input_bias.weight        C++ {reservoir}                  [1-D]
        #   esn_reservoir_bias.weight    C++ {reservoir}                  [1-D]
        def _f32(x: np.ndarray) -> np.ndarray:
            return np.ascontiguousarray(x, dtype=np.float32)

        writer.add_tensor("token_embd.weight",            _f32(self.tok_embd.T))
        writer.add_tensor("esn_input_weights.weight",     _f32(self.W_in))
        writer.add_tensor("esn_reservoir_weights.weight", _f32(self.W_res))
        writer.add_tensor("output_norm.weight",           _f32(self.output_norm))
        writer.add_tensor("esn_output_weights.weight",    _f32(self.W_out))

        if self.W_fb is not None:
            writer.add_tensor("esn_feedback_weights.weight", _f32(self.W_fb))
        if self.b_in is not None:
            writer.add_tensor("esn_input_bias.weight",       _f32(self.b_in))
        if self.b_res is not None:
            writer.add_tensor("esn_reservoir_bias.weight",   _f32(self.b_res))

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        print(f"Model saved successfully ({os.path.getsize(path) / 1e6:.1f} MB)")

    def _write_tokenizer(self, writer: GGUFWriter) -> None:
        """Write the tokenizer section to the GGUF file.

        Two modes:
          - "byte": a 256-token byte-level tokenizer using the RWKV escape
            format. Any text input is split into UTF-8 bytes and each byte
            becomes its own token. llama-cli -p "..." works.
          - "none": the "no_vocab" stub. Load-only, raw-token-id driven.
        """
        mode = self.config.tokenizer

        if mode == "none":
            writer.add_tokenizer_model("no_vocab")
            return

        if mode != "byte":
            raise ValueError(f"Unknown tokenizer mode: {mode!r} (expected 'byte' or 'none')")

        # Byte-level RWKV-escape tokenizer:
        # - 256 tokens, one per byte value 0..255
        # - Token text uses the RWKV escape format llama.cpp expects
        #   (see llama_unescape_rwkv_token in src/llama-vocab.cpp):
        #     \xNN for arbitrary bytes
        #     \t \n \r for the standard whitespace escapes (kept readable)
        #     printable ASCII bytes written literally
        tokens = []
        for b in range(256):
            if b == 0x09:
                tokens.append("\\t")
            elif b == 0x0A:
                tokens.append("\\n")
            elif b == 0x0D:
                tokens.append("\\r")
            elif b == 0x5C:  # backslash itself must be escaped
                tokens.append("\\\\")
            elif 0x20 <= b < 0x7F:
                tokens.append(chr(b))
            else:
                tokens.append(f"\\x{b:02x}")

        # All byte tokens are "normal" (type 1). No merges, scores, or
        # special-token IDs are needed for the RWKV tokenizer type —
        # the vocab is just a flat lookup from byte to id.
        token_types = [1] * 256

        writer.add_tokenizer_model("rwkv")
        writer.add_token_list(tokens)
        writer.add_token_types(token_types)
        # No BOS/EOS by default — users can set these via CLI if needed.
        writer.add_add_bos_token(False)
        writer.add_add_eos_token(False)


def create_sample_training_data(
    vocab_size: int,
    n_sequences: int = 1000,
    seq_length: int = 128,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample training data for testing.

    In practice, you would load real tokenized text data here.
    """
    rng = np.random.default_rng(seed)

    # Random token sequences (replace with real data)
    input_seqs = rng.integers(0, vocab_size, (n_sequences, seq_length))

    # Target is next token prediction (shifted by 1)
    target_seqs = np.roll(input_seqs, -1, axis=1)

    return input_seqs, target_seqs


def main():
    parser = argparse.ArgumentParser(
        description="ESN Training and Model Creation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python esn_training.py --reservoir-size 1024 --output model.gguf

  # With sigmoid activation and bias
  python esn_training.py --reservoir-size 2048 --activation sigmoid --use-bias

  # Generative ESN with feedback
  python esn_training.py --reservoir-size 1024 --feedback-scaling 0.5

  # Bidirectional ESN
  python esn_training.py --reservoir-size 512 --bidirectional
        """
    )

    # Config options
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--output", type=str, default="esn_model.gguf",
                        help="Output GGUF file path")

    # ESN hyperparameters
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--reservoir-size", type=int, default=1024)
    parser.add_argument("--spectral-radius", type=float, default=0.95)
    parser.add_argument("--sparsity", type=float, default=0.1)
    parser.add_argument("--leaking-rate", type=float, default=0.3)
    parser.add_argument("--input-scaling", type=float, default=1.0)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)

    # Extended ESN features
    parser.add_argument("--activation", type=str, default="tanh",
                        choices=["tanh", "sigmoid", "leaky_relu"],
                        help="Reservoir activation function")
    parser.add_argument("--feedback-scaling", type=float, default=0.0,
                        help="Output-to-reservoir feedback scaling (0 = disabled)")
    parser.add_argument("--noise-level", type=float, default=0.0,
                        help="Noise injection level for regularization")
    parser.add_argument("--bidirectional", action="store_true",
                        help="Use bidirectional reservoir processing")
    parser.add_argument("--use-bias", action="store_true",
                        help="Use bias vectors for input and reservoir")

    # Training options
    parser.add_argument("--train-data", type=str, help="Path to training data")
    parser.add_argument("--n-sequences", type=int, default=1000,
                        help="Number of sequences for sample data")
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--washout", type=int, default=100)

    # Tokenizer
    parser.add_argument("--tokenizer", choices=["byte", "none"], default="byte",
                        help="Tokenizer type. 'byte' (default) emits a 256-"
                             "token byte-level tokenizer so llama-cli -p "
                             "\"text\" works. 'none' emits the no_vocab stub "
                             "for raw-token-id driven use.")

    # Modes
    parser.add_argument("--init-only", action="store_true",
                        help="Only initialize weights, don't train")
    parser.add_argument("--save-config", type=str,
                        help="Save config to JSON file")

    args = parser.parse_args()

    # Map activation name to type
    activation_map = {
        "tanh": ACTIVATION_TANH,
        "sigmoid": ACTIVATION_SIGMOID,
        "leaky_relu": ACTIVATION_LEAKY_RELU
    }

    # Load or create config
    if args.config:
        config = ESNConfig.from_json(args.config)
    else:
        config = ESNConfig(
            vocab_size=args.vocab_size,
            embedding_dim=args.embedding_dim,
            reservoir_size=args.reservoir_size,
            spectral_radius=args.spectral_radius,
            sparsity=args.sparsity,
            leaking_rate=args.leaking_rate,
            input_scaling=args.input_scaling,
            feedback_scaling=args.feedback_scaling,
            noise_level=args.noise_level,
            activation_type=activation_map[args.activation],
            bidirectional=args.bidirectional,
            use_bias=args.use_bias,
            context_length=args.context_length,
            regularization=args.regularization,
            seed=args.seed,
            tokenizer=args.tokenizer,
        )

    # Save config if requested
    if args.save_config:
        config.to_json(args.save_config)
        print(f"Config saved to {args.save_config}")

    print(f"ESN Configuration: {config}")

    # Create and initialize ESN
    esn = ESN(config)
    esn.initialize()

    if not args.init_only:
        # Load or create training data
        if args.train_data:
            print(f"Loading training data from {args.train_data}")
            # TODO: Implement actual data loading
            input_seqs, target_seqs = create_sample_training_data(
                config.vocab_size, args.n_sequences, args.seq_length, config.seed
            )
        else:
            print("Using sample training data (for testing only)")
            input_seqs, target_seqs = create_sample_training_data(
                config.vocab_size, args.n_sequences, args.seq_length, config.seed
            )

        # Train the model
        accuracy = esn.train(input_seqs, target_seqs, washout=args.washout)
        print(f"Final training accuracy: {accuracy:.4f}")

    # Save to GGUF
    esn.save_gguf(args.output)
    print(f"\nModel saved to {args.output}")
    print("\nTo use with llama.cpp:")
    print(f"  ./llama-cli -m {args.output} -p \"Your prompt here\"")


if __name__ == "__main__":
    main()
