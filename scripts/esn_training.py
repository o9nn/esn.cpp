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
import struct
import os

# GGUF format constants
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

# GGML types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8

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
        seed: int = 42
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

        # Output weights: initialized to zero, will be trained
        self.W_out = np.zeros(
            (self.config.vocab_size, self.config.reservoir_size),
            dtype=np.float32
        )

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
        Save the trained ESN model to GGUF format.

        Args:
            path: Output path for the GGUF file
        """
        if not self._initialized:
            raise RuntimeError("ESN not initialized. Call initialize() first.")

        print(f"Saving model to {path}")

        with open(path, 'wb') as f:
            # Write GGUF header
            self._write_gguf_header(f)

            # Write tensors
            self._write_tensors(f)

        print(f"Model saved successfully ({os.path.getsize(path) / 1e6:.1f} MB)")

    def _write_gguf_header(self, f) -> None:
        """Write GGUF file header with metadata."""
        # Magic and version
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))

        # Count tensors
        n_tensors = 5  # tok_embd, W_in, W_res, output_norm, W_out
        if self.W_fb is not None:
            n_tensors += 1
        if self.b_in is not None:
            n_tensors += 1
        if self.b_res is not None:
            n_tensors += 1

        # Count KV pairs
        n_kv = 16  # Base metadata + extended ESN params

        f.write(struct.pack('<Q', n_tensors))
        f.write(struct.pack('<Q', n_kv))

        # Write metadata
        self._write_string_kv(f, "general.architecture", "esn")
        self._write_string_kv(f, "general.name", "ESN Model")

        self._write_uint32_kv(f, "esn.vocab_size", self.config.vocab_size)
        self._write_uint32_kv(f, "esn.embedding_length", self.config.embedding_dim)
        self._write_uint32_kv(f, "esn.context_length", self.config.context_length)
        self._write_uint32_kv(f, "esn.reservoir_size", self.config.reservoir_size)
        self._write_uint32_kv(f, "esn.block_count", 1)

        self._write_float32_kv(f, "esn.spectral_radius", self.config.spectral_radius)
        self._write_float32_kv(f, "esn.sparsity", self.config.sparsity)
        self._write_float32_kv(f, "esn.leaking_rate", self.config.leaking_rate)
        self._write_float32_kv(f, "esn.input_scaling", self.config.input_scaling)
        self._write_float32_kv(f, "esn.feedback_scaling", self.config.feedback_scaling)
        self._write_float32_kv(f, "esn.noise_level", self.config.noise_level)
        self._write_uint32_kv(f, "esn.activation_type", self.config.activation_type)
        self._write_bool_kv(f, "esn.bidirectional", self.config.bidirectional)
        self._write_float32_kv(f, "esn.attention.layernorm_rms_eps", 1e-6)

    def _write_tensors(self, f) -> None:
        """Write tensor data to GGUF file."""
        tensors = [
            ("token_embd.weight", self.tok_embd),
            ("esn_input_weights.weight", self.W_in),
            ("esn_reservoir_weights.weight", self.W_res),
            ("output_norm.weight", self.output_norm),
            ("esn_output_weights.weight", self.W_out),
        ]

        # Add optional tensors
        if self.W_fb is not None:
            tensors.append(("esn_feedback_weights.weight", self.W_fb))
        if self.b_in is not None:
            tensors.append(("esn_input_bias.weight", self.b_in))
        if self.b_res is not None:
            tensors.append(("esn_reservoir_bias.weight", self.b_res))

        # Calculate tensor info offset
        offset = f.tell()

        # Write tensor metadata
        for name, tensor in tensors:
            self._write_tensor_info(f, name, tensor.shape, GGML_TYPE_F32, offset)
            offset += tensor.nbytes

        # Align to 32 bytes
        alignment = 32
        current_pos = f.tell()
        padding = (alignment - (current_pos % alignment)) % alignment
        f.write(b'\x00' * padding)

        # Write tensor data
        for name, tensor in tensors:
            f.write(tensor.astype(np.float32).tobytes())

    def _write_string_kv(self, f, key: str, value: str) -> None:
        """Write string key-value pair."""
        key_bytes = key.encode('utf-8')
        f.write(struct.pack('<Q', len(key_bytes)))
        f.write(key_bytes)
        f.write(struct.pack('<I', 8))  # Type = string
        value_bytes = value.encode('utf-8')
        f.write(struct.pack('<Q', len(value_bytes)))
        f.write(value_bytes)

    def _write_uint32_kv(self, f, key: str, value: int) -> None:
        """Write uint32 key-value pair."""
        key_bytes = key.encode('utf-8')
        f.write(struct.pack('<Q', len(key_bytes)))
        f.write(key_bytes)
        f.write(struct.pack('<I', 4))  # Type = uint32
        f.write(struct.pack('<I', value))

    def _write_float32_kv(self, f, key: str, value: float) -> None:
        """Write float32 key-value pair."""
        key_bytes = key.encode('utf-8')
        f.write(struct.pack('<Q', len(key_bytes)))
        f.write(key_bytes)
        f.write(struct.pack('<I', 6))  # Type = float32
        f.write(struct.pack('<f', value))

    def _write_bool_kv(self, f, key: str, value: bool) -> None:
        """Write bool key-value pair."""
        key_bytes = key.encode('utf-8')
        f.write(struct.pack('<Q', len(key_bytes)))
        f.write(key_bytes)
        f.write(struct.pack('<I', 7))  # Type = bool
        f.write(struct.pack('<?', value))

    def _write_tensor_info(self, f, name: str, shape: tuple, dtype: int, offset: int) -> None:
        """Write tensor info to GGUF."""
        name_bytes = name.encode('utf-8')
        f.write(struct.pack('<Q', len(name_bytes)))
        f.write(name_bytes)
        f.write(struct.pack('<I', len(shape)))
        for dim in shape:
            f.write(struct.pack('<Q', dim))
        f.write(struct.pack('<I', dtype))
        f.write(struct.pack('<Q', offset))


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
            seed=args.seed
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
