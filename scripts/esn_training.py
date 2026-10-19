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
import logging
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
    from gguf import GGUFWriter
except ImportError as e:
    raise SystemExit(
        "The 'gguf' package is required. Install the one bundled with this\n"
        "repository by running: pip install -e gguf-py"
    ) from e

logger = logging.getLogger("esn-training")

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
            logger.warning(f"[ESNConfig] tokenizer='byte' requires vocab_size=256; "
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


# Up to this size, exact np.linalg.eigvals (O(R^3)) is cheap; above it, it
# dominates initialization (minutes at R=8192, hours at R=16384).
_EXACT_EIGVALS_MAX_SIZE = 2048

# Bound on the floats held per batch of reservoir states during training.
_STATE_BATCH_FLOATS = 1 << 27


def spectral_radius(W: np.ndarray, krylov_dim: int = 128, seed: int = 0) -> float:
    """Largest eigenvalue modulus of a square matrix.

    Exact for small matrices; above _EXACT_EIGVALS_MAX_SIZE uses the outermost
    Ritz value of a `krylov_dim`-step Arnoldi factorization (within 0.2% of
    exact at R=4096 for ESN reservoirs, at O(krylov_dim * R^2) cost).
    """
    n = W.shape[0]
    if n <= _EXACT_EIGVALS_MAX_SIZE:
        return float(np.max(np.abs(np.linalg.eigvals(W))))

    m = min(krylov_dim, n)
    rng = np.random.default_rng(seed)
    Q = np.zeros((m + 1, n), dtype=np.float64)
    H = np.zeros((m + 1, m), dtype=np.float64)
    q = rng.standard_normal(n)
    Q[0] = q / np.linalg.norm(q)
    for j in range(m):
        v = (W @ Q[j].astype(W.dtype)).astype(np.float64)
        for _ in range(2):  # classical Gram-Schmidt, applied twice for stability
            h = Q[: j + 1] @ v
            v -= h @ Q[: j + 1]
            H[: j + 1, j] += h
        H[j + 1, j] = np.linalg.norm(v)
        if H[j + 1, j] < 1e-12:  # invariant subspace found
            m = j + 1
            break
        Q[j + 1] = v / H[j + 1, j]
    return float(np.max(np.abs(np.linalg.eigvals(H[:m, :m]))))


def softmax_nll(Z: np.ndarray, y: np.ndarray, beta: float) -> float:
    """Mean negative log-likelihood (nats) of targets y under softmax(beta * Z)."""
    S = beta * Z.astype(np.float64)
    m = S.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(S - m).sum(axis=1))
    return float(np.mean(lse - S[np.arange(len(y)), y]))


def fit_temperature(Z: np.ndarray, y: np.ndarray) -> float:
    """Scale beta > 0 minimizing softmax_nll(Z, y, beta).

    The NLL is convex in beta, hence unimodal in log(beta), so a golden-section
    search over a wide bracket finds the global minimum.
    """
    lo, hi = np.log(1e-2), np.log(1e4)
    g = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = softmax_nll(Z, y, np.exp(a)), softmax_nll(Z, y, np.exp(b))
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = softmax_nll(Z, y, np.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = softmax_nll(Z, y, np.exp(b))
    return float(np.exp((lo + hi) / 2.0))


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
    """

    def __init__(self, config: ESNConfig):
        if config.bidirectional:
            # At position t a backward pass has already read token t+1, the
            # readout's target, while llm_build_esn runs forward only.
            raise ValueError(
                "bidirectional reservoirs are not supported: the readout predicts "
                "the next token, which backward-pass states have already seen")
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.activation_fn = get_activation_fn(config.activation_type)

        # Weights. initialize() sets every non-optional one; until then they
        # are empty placeholders, so their type is a plain ndarray.
        self.W_in: np.ndarray = np.empty((0, 0), dtype=np.float32)      # Input weights
        self.W_res: np.ndarray = np.empty((0, 0), dtype=np.float32)     # Reservoir weights
        self.W_out: np.ndarray = np.empty((0, 0), dtype=np.float32)     # Output weights (trained)
        self.W_fb: Optional[np.ndarray] = None    # Feedback weights [vocab, reservoir] (optional)
        self.b_in: Optional[np.ndarray] = None    # Input bias (optional)
        self.b_res: Optional[np.ndarray] = None   # Reservoir bias (optional)
        self.tok_embd: np.ndarray = np.empty((0, 0), dtype=np.float32)  # Token embeddings
        self.output_norm: np.ndarray = np.empty(0, dtype=np.float32)    # Output normalization
        # Ridge regression fits W_out to one-hot targets, so W_out @ x is on a
        # probability scale; inference treats it as logits. train() fits this
        # scale on held-out data and save_gguf() folds it into W_out.
        self.output_temperature = 1.0
        # dtype the reservoir matrix is stored in by save_gguf().
        self.reservoir_dtype = np.float32

        self._initialized = False

    def initialize(self) -> None:
        """Initialize all weight matrices according to ESN principles."""
        logger.info(f"Initializing ESN with reservoir size {self.config.reservoir_size}")
        logger.info(f"  Activation: {['tanh', 'sigmoid', 'leaky_relu'][self.config.activation_type]}")
        logger.info(f"  Feedback scaling: {self.config.feedback_scaling}")
        logger.info(f"  Use bias: {self.config.use_bias}")

        # Token embeddings (random initialization, to be learned or loaded)
        self.tok_embd = self.rng.standard_normal(
            (self.config.embedding_dim, self.config.vocab_size)
        ).astype(np.float32) * 0.02

        # Input weights: dense random matrix. input_scaling is applied in the
        # forward pass, as llm_build_esn does with esn.input_scaling; baking it
        # in here as well would apply it twice at inference.
        self.W_in = self.rng.standard_normal(
            (self.config.reservoir_size, self.config.embedding_dim)
        ).astype(np.float32)

        # Reservoir weights: sparse random matrix with specified connectivity
        self._init_reservoir_weights()

        # Output normalization (RMS norm weights)
        self.output_norm = np.ones(self.config.reservoir_size, dtype=np.float32)

        # Output weights W_out: the only trainable parameter of the ESN.
        # Initialized with small Gaussian values N(0, 0.02²), which is a
        # standard choice (matches e.g. GPT-2 weight init). train() fits
        # these via ridge regression on collected reservoir states; any
        # subsequent save() emits the fitted weights.
        self.W_out = self.rng.standard_normal(
            (self.config.vocab_size, self.config.reservoir_size)
        ).astype(np.float32) * 0.02

        # Optional: feedback weights, one row per token since the fed-back
        # output is a one-hot (see _input_drive). Unscaled, like W_in:
        # feedback_scaling is applied once, in the forward pass.
        if self.config.feedback_scaling > 0:
            self.W_fb = self.rng.standard_normal(
                (self.config.vocab_size, self.config.reservoir_size)
            ).astype(np.float32)
            logger.info("  Initialized feedback weights")

        # Optional: bias vectors
        if self.config.use_bias:
            self.b_in = np.zeros(self.config.reservoir_size, dtype=np.float32)
            self.b_res = self.rng.standard_normal(
                self.config.reservoir_size
            ).astype(np.float32) * 0.1
            logger.info("  Initialized bias vectors")

        self._initialized = True
        logger.info("ESN initialization complete")

    def _init_reservoir_weights(self) -> None:
        """
        Initialize reservoir weights with sparse connectivity and
        proper spectral radius scaling.
        """
        size = self.config.reservoir_size
        sparsity = self.config.sparsity

        # Draw in row chunks to avoid full-size float64 temporaries. Chunked
        # draws consume the RNG stream exactly like one (size, size) draw, so
        # a given seed yields the same reservoir as before.
        chunk = max(1, (1 << 24) // size)
        connectivity = np.empty((size, size), dtype=bool)
        for r in range(0, size, chunk):
            connectivity[r:r + chunk] = self.rng.random((min(chunk, size - r), size)) < sparsity

        weights = np.empty((size, size), dtype=np.float32)
        for r in range(0, size, chunk):
            weights[r:r + chunk] = self.rng.standard_normal((min(chunk, size - r), size))
            weights[r:r + chunk] *= connectivity[r:r + chunk]
        del connectivity

        radius = spectral_radius(weights)
        if radius > 0:
            weights *= self.config.spectral_radius / radius

        self.W_res = weights

        method = "exact" if size <= _EXACT_EIGVALS_MAX_SIZE else "Arnoldi estimate"
        logger.info(f"  Reservoir spectral radius: {self.config.spectral_radius} "
                    f"(unscaled radius {radius:.4f}, {method})")
        logger.info(f"  Reservoir sparsity: {1 - np.count_nonzero(self.W_res) / self.W_res.size:.2%} zeros")

    def _input_drive(self, embeddings: np.ndarray, tokens: Optional[np.ndarray]) -> np.ndarray:
        """
        The part of the reservoir pre-activation that does not depend on the
        state: s·W_in·u(t) + b_in + b_res, plus the output feedback.

        The fed-back output y(t-1) is the one-hot of the token emitted at
        t-1, which is the input token at t: the true next token in training
        (teacher forcing) and the sampled token in generation. W_fb·y(t-1)
        is therefore the row W_fb[token_t], and training and llm_build_esn
        run the same recurrence. (Feeding back the readout's own prediction
        would make the states depend on W_out, which is fit to those states.)

        Args:
            embeddings: Input embeddings [..., embedding_dim]
            tokens: Token IDs of the inputs [...]; required with feedback

        Returns:
            drive: [..., reservoir_size]
        """
        drive = (embeddings @ self.W_in.T) * self.config.input_scaling
        if self.b_in is not None:
            drive += self.b_in
        if self.b_res is not None:
            drive += self.b_res
        if self.W_fb is not None:
            if tokens is None:
                raise ValueError("output feedback is enabled: pass the input token IDs")
            drive += self.config.feedback_scaling * self.W_fb[tokens]
        return drive

    def forward(
        self,
        embeddings: np.ndarray,
        state: Optional[np.ndarray] = None,
        tokens: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ESN forward pass.

        Args:
            embeddings: Input embeddings [n_tokens, embedding_dim]
            state: Previous reservoir state [reservoir_size] or None
            tokens: Token IDs of the inputs [n_tokens]; required when output
                feedback is enabled (see _input_drive)

        Returns:
            states: Reservoir states for all timesteps [n_tokens, reservoir_size]
            new_state: Final reservoir state [reservoir_size]
        """
        if not self._initialized:
            raise RuntimeError("ESN not initialized. Call initialize() first.")

        n_tokens = embeddings.shape[0]
        alpha = self.config.leaking_rate

        x = np.zeros(self.config.reservoir_size, dtype=np.float32) if state is None else state

        drive = self._input_drive(embeddings, tokens)
        states = np.zeros((n_tokens, self.config.reservoir_size), dtype=np.float32)

        for t in range(n_tokens):
            # Reservoir recurrence
            reservoir_input = self.W_res @ x + drive[t]

            # Add noise if specified
            if self.config.noise_level > 0:
                reservoir_input += self.rng.standard_normal(
                    self.config.reservoir_size
                ).astype(np.float32) * self.config.noise_level

            # Apply activation function
            activated = self.activation_fn(reservoir_input)

            # Leaky integration: x(t+1) = (1-α)*x(t) + α*f(...)
            x = (1 - alpha) * x + alpha * activated

            states[t] = x

        return states, x

    def forward_batch(self, embeddings: np.ndarray, tokens: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass over a batch of equal-length sequences, from a zero state.

        Equivalent to forward() on each sequence up to float rounding, but
        runs one matrix-matrix product per timestep for the whole batch
        instead of one matrix-vector product per sequence. With noise
        enabled, noise is drawn per timestep for the whole batch, so the
        noise values differ from forward()'s.

        Args:
            embeddings: Input embeddings [batch, n_tokens, embedding_dim]
            tokens: Token IDs of the inputs [batch, n_tokens]; required when
                output feedback is enabled

        Returns:
            states: Reservoir states [batch, n_tokens, reservoir_size]
        """
        if not self._initialized:
            raise RuntimeError("ESN not initialized. Call initialize() first.")

        batch, n_tokens, _ = embeddings.shape
        size = self.config.reservoir_size
        alpha = self.config.leaking_rate

        # The input drive does not depend on the state: one product for all steps.
        drive = self._input_drive(embeddings, tokens)

        state = np.zeros((batch, size), dtype=np.float32)
        states = np.empty((batch, n_tokens, size), dtype=np.float32)
        for t in range(n_tokens):
            pre = state @ self.W_res.T
            pre += drive[:, t]
            if self.config.noise_level > 0:
                pre += (self.rng.standard_normal((batch, size)).astype(np.float32)
                        * self.config.noise_level)
            state = (1 - alpha) * state + alpha * self.activation_fn(pre)
            states[:, t] = state

        return states

    def _state_batches(self, token_sequences: np.ndarray):
        """Yield (start, states [batch, seq_length, reservoir_size]) over token_sequences."""
        n_tokens = token_sequences.shape[1]
        batch = max(1, _STATE_BATCH_FLOATS // (n_tokens * self.config.reservoir_size))
        embd = self.tok_embd.T  # [vocab_size, embedding_dim]
        for start in range(0, len(token_sequences), batch):
            tokens = token_sequences[start:start + batch]
            yield start, self.forward_batch(embd[tokens], tokens)

    def _readout_features(self, states: np.ndarray) -> np.ndarray:
        """Map states [n, reservoir_size] to the RMS-normalized vectors W_out
        reads, matching llm_build_esn's output_norm."""
        X = states / np.sqrt(np.mean(states ** 2, axis=-1, keepdims=True) + 1e-6)
        X *= self.output_norm
        return X

    def collect_states(self, token_sequences: np.ndarray) -> np.ndarray:
        """
        Collect reservoir states for training data.

        Args:
            token_sequences: Token IDs [n_sequences, seq_length]

        Returns:
            all_states: Collected states [total_tokens, reservoir_size]
        """
        return np.concatenate(
            [s.reshape(-1, s.shape[-1]) for _, s in self._state_batches(token_sequences)])

    def _normal_equations(self, inputs, targets, washout, keep_rows):
        """Accumulate X^T X and X^T Y over readout features, batch by batch,
        without materializing X. Also returns the first keep_rows (X, y)."""
        size, vocab = self.config.reservoir_size, self.config.vocab_size
        XtX = np.zeros((size, size), dtype=np.float64)
        XtY = np.zeros((size, vocab), dtype=np.float64)
        kept_X, kept_y, n_rows = [], [], 0
        for start, states in self._state_batches(inputs):
            logger.info(f"  Processing sequence {start}/{len(inputs)}")
            X = self._readout_features(states[:, washout:].reshape(-1, states.shape[-1]))
            y = targets[start:start + len(states), washout:].reshape(-1)
            Y = np.zeros((len(y), vocab), dtype=np.float32)
            Y[np.arange(len(y)), y] = 1.0
            XtX += X.T @ X
            XtY += X.T @ Y
            take = min(keep_rows - min(n_rows, keep_rows), len(y))
            if take > 0:
                kept_X.append(X[:take].copy())
                kept_y.append(y[:take])
            n_rows += len(y)
        return XtX, XtY, n_rows, np.concatenate(kept_X), np.concatenate(kept_y)

    @staticmethod
    def _ridge_solve(XtX: np.ndarray, XtY: np.ndarray, lam: float) -> np.ndarray:
        """W_out [V, R] = ((X^T X + lam I)^-1 X^T Y)^T; XtX is left unchanged."""
        idx = np.diag_indices_from(XtX)
        diag = XtX[idx].copy()
        XtX[idx] += lam
        W = np.linalg.solve(XtX, XtY)
        XtX[idx] = diag
        return W.T.astype(np.float32)

    def train(
        self,
        token_sequences: np.ndarray,
        target_sequences: np.ndarray,
        washout: int = 100,
        val_fraction: float = 0.1,
        calibrate: bool = True,
        reg_grid: Optional[list] = None,
    ) -> float:
        """
        Train output weights using batch ridge regression.

        This is the key insight of ESNs: only the output layer needs
        to be trained, and it can be done with a simple linear method.

        With at least 10 sequences, the last val_fraction of them are held
        out (after a one-sequence gap, since text windows overlap by half) to
        choose the ridge penalty from reg_grid and to fit the output
        temperature; the readout is then refit on train + validation data.

        Args:
            token_sequences: Input token IDs [n_sequences, seq_length]
            target_sequences: Target token IDs [n_sequences, seq_length]
            washout: Number of initial timesteps to discard
            val_fraction: Fraction of sequences held out for selection
            calibrate: Fit the logit scale on held-out data (output_temperature)
            reg_grid: Candidate ridge penalties, chosen by held-out NLL

        Returns:
            training_accuracy: Accuracy on (a sample of) the training data
        """
        if not self._initialized:
            raise RuntimeError("ESN not initialized. Call initialize() first.")
        if washout >= token_sequences.shape[1]:
            raise ValueError(
                f"washout ({washout}) must be smaller than the sequence length "
                f"({token_sequences.shape[1]}), or no states are left to train on")

        n_seq = len(token_sequences)
        n_val = 0
        if (calibrate or reg_grid) and val_fraction > 0 and n_seq >= 10:
            n_val = max(1, int(round(n_seq * val_fraction)))
        n_train = n_seq - n_val - (1 if n_val else 0)
        keep_rows = max(1, _STATE_BATCH_FLOATS // self.config.reservoir_size)

        logger.info(f"Collecting reservoir states (washout={washout})...")
        XtX, XtY, n_rows, X_fit, y_fit = self._normal_equations(
            token_sequences[:n_train], target_sequences[:n_train], washout, keep_rows)

        lam = self.config.regularization
        if n_val:
            val_XtX, val_XtY, n_val_rows, X_val, y_val = self._normal_equations(
                token_sequences[n_seq - n_val:], target_sequences[n_seq - n_val:],
                washout, keep_rows)
            bits = 1.0 / np.log(2.0)

            if reg_grid:
                logger.info("  ridge penalty   validation NLL (bits/token, temperature-fitted)")
                scores = []
                for cand in reg_grid:
                    Z = X_val @ self._ridge_solve(XtX, XtY, cand).T
                    scores.append(softmax_nll(Z, y_val, fit_temperature(Z, y_val)) * bits)
                    logger.info(f"  {cand:13.3g}   {scores[-1]:.4f}")
                lam = float(reg_grid[int(np.argmin(scores))])
                self.config.regularization = lam
                logger.info(f"  selected ridge penalty {lam:.3g}")

            Z = X_val @ self._ridge_solve(XtX, XtY, lam).T
            if calibrate:
                self.output_temperature = fit_temperature(Z, y_val)
            logger.info(f"Validation ({n_val_rows} tokens): "
                        f"accuracy {np.mean(np.argmax(Z, axis=1) == y_val):.4f}, "
                        f"NLL {softmax_nll(Z, y_val, 1.0) * bits:.4f} -> "
                        f"{softmax_nll(Z, y_val, self.output_temperature) * bits:.4f} bits/token "
                        f"at logit scale {self.output_temperature:.4g}")

            XtX += val_XtX
            XtY += val_XtY
            n_rows += n_val_rows
            del val_XtX, val_XtY
        elif calibrate:
            logger.info("Skipping readout calibration: need at least 10 sequences to hold out from")

        logger.info(f"Training output weights on {n_rows} samples "
                    f"(state dim: {self.config.reservoir_size}, ridge penalty {lam:.3g})...")
        self.W_out = self._ridge_solve(XtX, XtY, lam)

        accuracy = float(np.mean(np.argmax(X_fit @ self.W_out.T, axis=1) == y_fit))
        logger.info(f"Training complete. Accuracy: {accuracy:.4f}"
                    + (f" (first {len(y_fit)} samples)" if len(y_fit) < n_rows else ""))

        return accuracy

    def set_reservoir_dtype(self, dtype) -> None:
        """Store W_res as dtype in the GGUF, and round it now so the readout is
        trained on the same reservoir that inference will run."""
        self.reservoir_dtype = np.dtype(dtype)
        self.W_res = self.W_res.astype(self.reservoir_dtype).astype(np.float32)

    # ------------------------------------------------------------------
    # Online / incremental learning (Phase 3 of ESLLM roadmap)
    # ------------------------------------------------------------------

    def init_rls(self) -> None:
        """
        Initialise the Recursive Least Squares (RLS) online adaptation state.

        Call this once before the first call to ``rls_update()``.
        The correlation matrix P is set to (1/λ)*I where λ is the
        regularisation coefficient from config.
        """
        rsz = self.config.reservoir_size
        lam = max(self.config.regularization, 1e-9)
        # Inverse correlation matrix P:  P = (1/λ) I
        self._rls_P = np.eye(rsz, dtype=np.float32) * (1.0 / lam)
        logger.info(f"RLS initialised: reservoir_size={rsz}, λ={lam:.2e}")

    def rls_update(
        self,
        reservoir_state: np.ndarray,
        target_token: int,
        forgetting_factor: float = 1.0
    ) -> float:
        """
        Perform one online RLS update of the output weights.

        Implements the standard RLS algorithm with forgetting factor λ_f:

            k     = P x / (λ_f + x^T P x)
            error = e_{target} - W_out^T x          (one-hot error)
            W_out += error^T k
            P     = (P - k x^T P) / λ_f

        Args:
            reservoir_state:  Current reservoir state vector [reservoir_size]
            target_token:     Ground-truth next token index
            forgetting_factor: Exponential forgetting rate in (0, 1].
                               1.0 = no forgetting; <1.0 = faster adaptation.

        Returns:
            squared_error: Scalar prediction error before update.
        """
        if not hasattr(self, '_rls_P'):
            raise RuntimeError("Call init_rls() before rls_update().")

        x = reservoir_state.reshape(-1, 1).astype(np.float32)  # [R, 1]

        # Compute RLS gain vector
        Px = self._rls_P @ x                                      # [R, 1]
        denom = forgetting_factor + float(np.dot(x.ravel(), Px.ravel()))  # scalar
        k = Px / denom                               # [R, 1]

        # Current logit prediction
        logits = (self.W_out @ x).ravel()            # [V]
        target_one_hot = np.zeros_like(logits)
        target_one_hot[target_token] = 1.0

        # Prediction error before update
        error = target_one_hot - logits              # [V]
        sq_error = float(np.dot(error, error))

        # Update output weights: W_out += error^T k  →  outer product
        self.W_out += np.outer(error, k.ravel())     # [V, R]

        # Update inverse correlation matrix
        self._rls_P = (self._rls_P - k @ (x.T @ self._rls_P)) / forgetting_factor

        return sq_error

    def sgd_update(
        self,
        reservoir_state: np.ndarray,
        target_token: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0
    ) -> float:
        """
        Perform one online SGD update of the output weights (cross-entropy loss).

        Args:
            reservoir_state: Current reservoir state [reservoir_size]
            target_token:    Ground-truth next token index
            learning_rate:   Step size.
            weight_decay:    L2 regularisation coefficient.

        Returns:
            cross_entropy_loss: Scalar loss before update.
        """
        x = reservoir_state.astype(np.float32)       # [R]
        logits = self.W_out @ x                      # [V]

        # Softmax
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(np.clip(logits_shifted, -30, 0))
        probs = exp_logits / (exp_logits.sum() + 1e-9)

        ce_loss = -float(np.log(probs[target_token] + 1e-9))

        # Gradient of cross-entropy w.r.t. W_out:  grad = (p - e_t) outer x
        grad = np.outer(probs, x)                    # [V, R]
        grad[target_token] -= x                      # subtract x for target class

        # Apply update
        self.W_out -= learning_rate * grad
        if weight_decay > 0.0:
            self.W_out -= learning_rate * weight_decay * self.W_out

        return ce_loss

    def online_train_sequence(
        self,
        token_sequence: np.ndarray,
        target_sequence: np.ndarray,
        mode: str = "rls",
        learning_rate: float = 1e-4,
        forgetting_factor: float = 1.0,
        weight_decay: float = 0.0,
        washout: int = 0
    ) -> Dict[str, Any]:
        """
        Online training on a single token sequence using the specified mode.

        Supports ``rls`` (Recursive Least Squares) or ``sgd``
        (Stochastic Gradient Descent).  The reservoir weights are kept fixed.

        Args:
            token_sequence:   Input token IDs [seq_length]
            target_sequence:  Target token IDs [seq_length]
            mode:             ``"rls"`` or ``"sgd"``
            learning_rate:    Used only for SGD.
            forgetting_factor: Used only for RLS (exponential forgetting).
            weight_decay:     L2 penalty (SGD only).
            washout:          Number of initial steps to skip.

        Returns:
            stats: Dict with "mean_loss", "n_updates", "final_state".
        """
        if not self._initialized:
            raise RuntimeError("ESN not initialized. Call initialize() first.")

        if mode == "rls" and not hasattr(self, '_rls_P'):
            self.init_rls()

        embeddings = self.tok_embd[:, token_sequence].T  # [T, E]
        drive = self._input_drive(embeddings, token_sequence)
        state = np.zeros(self.config.reservoir_size, dtype=np.float32)
        losses = []

        for t, tgt in enumerate(target_sequence):
            # Update reservoir state
            res_in = self.W_res @ state + drive[t]
            activated = self.activation_fn(res_in)
            state = (1 - self.config.leaking_rate) * state + self.config.leaking_rate * activated

            if t < washout:
                continue

            if mode == "rls":
                loss = self.rls_update(state, int(tgt), forgetting_factor)
            elif mode == "sgd":
                loss = self.sgd_update(state, int(tgt), learning_rate, weight_decay)
            else:
                raise ValueError(f"Unknown online training mode: {mode!r}. Use 'rls' or 'sgd'.")

            losses.append(loss)

        return {
            "mean_loss":   float(np.mean(losses)) if losses else 0.0,
            "n_updates":   len(losses),
            "final_state": state.copy(),
        }

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
          - ``esn.feedback_token``   (bool, with feedback: fed-back output is the input token)
          - ``esn.noise_level``      (float32, optional)
          - ``esn.activation_type``  (uint32, optional: 0=tanh, 1=sigmoid, 2=leaky_relu)
          - ``esn.bidirectional``    (bool, optional; always false)
          - ``esn.attention.layernorm_rms_eps`` (float32, required by loader)

        Tensors emitted (``.weight`` suffix per llama.cpp convention):
          - ``token_embd.weight``          [n_embd, n_vocab]
          - ``esn_input_weights.weight``   [n_embd, reservoir_size]
          - ``esn_reservoir_weights.weight`` [reservoir_size, reservoir_size]
          - ``output_norm.weight``         [reservoir_size]
          - ``esn_output_weights.weight``  [reservoir_size, n_vocab]
          - ``esn_feedback_weights.weight``  [reservoir_size, n_vocab] (optional)
          - ``esn_input_bias.weight``        (optional)
          - ``esn_reservoir_bias.weight``    (optional)
        """
        if not self._initialized:
            raise RuntimeError("ESN not initialized. Call initialize() first.")

        logger.info(f"Saving model to {path}")

        writer = GGUFWriter(path=path, arch="esn")

        # General metadata
        writer.add_name("ESN Model")
        # llama_ftype: W_res holds nearly all parameters, so an F16 reservoir
        # makes the model MOSTLY_F16 (1); otherwise ALL_F32 (0).
        writer.add_file_type(1 if self.reservoir_dtype == np.float16 else 0)

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
        if self.W_fb is not None:
            # Marks the feedback definition of _input_drive; llama.cpp refuses
            # feedback models without it, whose training did not match inference.
            writer.add_bool("esn.feedback_token", True)
        writer.add_float32("esn.noise_level",      self.config.noise_level)
        writer.add_uint32("esn.activation_type",   self.config.activation_type)
        writer.add_bool("esn.bidirectional",       self.config.bidirectional)

        if getattr(self.config, "online_learning", False):
            writer.add_bool("esn.online_learning.enabled", True)
            writer.add_float32("esn.online_learning.rate", self.config.online_lr)
            writer.add_float32("esn.online_learning.reg", self.config.online_reg)
            writer.add_uint32("esn.online_learning.buf_size", self.config.online_buffer_size)
            writer.add_uint32("esn.online_learning.mode", self.config.online_update_mode)
            writer.add_float32("esn.online_learning.decay", self.config.online_decay_rate)
            writer.add_bool("esn.freeze_reservoir", self.config.freeze_reservoir)
            writer.add_uint32("esn.replay_window", self.config.replay_window)

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
        #   esn_feedback_weights.weight  C++ {reservoir, n_vocab}
        #       -> numpy (n_vocab, reservoir)     [matches training layout]
        #   esn_input_bias.weight        C++ {reservoir}                  [1-D]
        #   esn_reservoir_bias.weight    C++ {reservoir}                  [1-D]
        def _f32(x: np.ndarray) -> np.ndarray:
            return np.ascontiguousarray(x, dtype=np.float32)

        # W_res is the only tensor read once per token for every token, so it
        # may be stored as F16 (see set_reservoir_dtype). The fitted softmax
        # temperature is folded into W_out so inference needs no extra op.
        writer.add_tensor("token_embd.weight",            _f32(self.tok_embd.T))
        writer.add_tensor("esn_input_weights.weight",     _f32(self.W_in))
        writer.add_tensor("esn_reservoir_weights.weight",
                          np.ascontiguousarray(self.W_res, dtype=self.reservoir_dtype))
        writer.add_tensor("output_norm.weight",           _f32(self.output_norm))
        writer.add_tensor("esn_output_weights.weight",
                          _f32(self.W_out * self.output_temperature))

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

        logger.info(f"Model saved successfully ({os.path.getsize(path) / 1e6:.1f} MB)")

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


def load_text_training_data(
    path: str,
    vocab_size: int,
    seq_length: int = 128,
    stride: Optional[int] = None,
    max_sequences: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a text file and turn it into training sequences for the
    byte-level ESN. The whole file is read as UTF-8 bytes, split into
    overlapping windows of length ``seq_length + 1``, and returned as
    (input, target) pairs where target[i] = input[i+1] (next-byte
    prediction).

    Only valid when ``vocab_size == 256`` (byte-level); the caller is
    responsible for matching that to the model's vocab size.
    """
    if vocab_size != 256:
        raise ValueError(
            f"load_text_training_data requires vocab_size=256, got {vocab_size}"
        )

    with open(path, "rb") as f:
        data = f.read()
    if len(data) < seq_length + 1:
        raise ValueError(
            f"Training file too short ({len(data)} bytes); need at least "
            f"{seq_length + 1}"
        )

    tokens = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
    if stride is None:
        stride = max(1, seq_length // 2)

    inputs = []
    targets = []
    for start in range(0, len(tokens) - seq_length, stride):
        window = tokens[start : start + seq_length + 1]
        inputs.append(window[:-1])
        targets.append(window[1:])
        if max_sequences is not None and len(inputs) >= max_sequences:
            break

    inputs_arr  = np.stack(inputs,  axis=0)
    targets_arr = np.stack(targets, axis=0)

    logger.info(
        f"Loaded {len(inputs_arr)} windows of length {seq_length} from "
        f"{path} ({len(data)} bytes)"
    )
    return inputs_arr, targets_arr


def main():
    parser = argparse.ArgumentParser(
        description="ESN Training and Model Creation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic batch ridge regression training
  python esn_training.py --reservoir-size 1024 --output model.gguf

  # With sigmoid activation and bias
  python esn_training.py --reservoir-size 2048 --activation sigmoid --use-bias

  # Generative ESN with feedback
  python esn_training.py --reservoir-size 1024 --feedback-scaling 0.5

  # Online RLS training (infer-train / ESLLM)
  python esn_training.py --reservoir-size 1024 --online --online-mode rls \\
      --forgetting-factor 0.999 --output online_model.gguf

  # Online SGD training
  python esn_training.py --reservoir-size 1024 --online --online-mode sgd \\
      --online-lr 1e-4 --output online_sgd_model.gguf
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
                        help="Not supported: backward-pass states have seen the "
                             "next token the readout predicts")
    parser.add_argument("--use-bias", action="store_true",
                        help="Use bias vectors for input and reservoir")

    # Training options
    parser.add_argument("--train-data", type=str, help="Path to training data")
    parser.add_argument("--n-sequences", type=int, default=1000,
                        help="Number of sequences for sample data")
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--washout", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of sequences held out to select the "
                             "regularization and fit the softmax temperature")
    parser.add_argument("--reg-grid", type=str, default=None,
                        help="Comma-separated regularization values to try "
                             "(picked by held-out NLL); default: --regularization only")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Do not fit a softmax temperature (export raw ridge outputs as logits)")
    parser.add_argument("--outtype", choices=["f32", "f16"], default="f32",
                        help="Storage type of the reservoir matrix in the GGUF")

    # Tokenizer
    parser.add_argument("--tokenizer", choices=["byte", "none"], default="byte",
                        help="Tokenizer type. 'byte' (default) emits a 256-"
                             "token byte-level tokenizer so llama-cli -p "
                             "\"text\" works. 'none' emits the no_vocab stub "
                             "for raw-token-id driven use.")

    # Online learning options (Phase 3 – ESLLM infer-train substrate)
    parser.add_argument("--online", action="store_true",
                        help="Use online incremental training instead of batch ridge regression")
    parser.add_argument("--online-mode", type=str, default="rls",
                        choices=["rls", "sgd"],
                        help="Online learning algorithm: 'rls' (Recursive Least Squares) or 'sgd'")
    parser.add_argument("--online-lr", type=float, default=1e-4,
                        help="Learning rate for online SGD training")
    parser.add_argument("--forgetting-factor", type=float, default=1.0,
                        help="RLS forgetting factor in (0, 1]; 1.0 = no forgetting")
    parser.add_argument("--online-reg", type=float, default=1e-6,
                        help="Regularisation coefficient for online learning initialisation")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="L2 weight decay for online SGD")
    parser.add_argument("--save-online-metadata", action="store_true",
                        help="Embed online-learning config in GGUF metadata so llama.cpp "
                             "enables adaptation at runtime")

    # Modes
    parser.add_argument("--init-only", action="store_true",
                        help="Only initialize weights, don't train")
    parser.add_argument("--save-config", type=str,
                        help="Save config to JSON file")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.bidirectional:
        parser.error("--bidirectional is not supported: at position t the backward "
                     "pass has already read token t+1, the readout's target, and "
                     "inference runs forward only")

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
        logger.info(f"Config saved to {args.save_config}")

    logger.info(f"ESN Configuration: {config}")

    # Create and initialize ESN
    esn = ESN(config)
    esn.initialize()
    if args.outtype == "f16":
        esn.set_reservoir_dtype(np.float16)

    if not args.init_only:
        # Load or create training data
        if args.train_data:
            logger.info(f"Loading training data from {args.train_data}")
            input_seqs, target_seqs = load_text_training_data(
                args.train_data,
                config.vocab_size,
                seq_length=args.seq_length,
                max_sequences=args.n_sequences,
            )
        else:
            logger.info("Using sample training data (for testing only)")
            input_seqs, target_seqs = create_sample_training_data(
                config.vocab_size, args.n_sequences, args.seq_length, config.seed
            )

        if args.online:
            # ---------------------------------------------------------- #
            # Online / incremental training (Phase 3 – ESLLM infer-train) #
            # ---------------------------------------------------------- #
            logger.info(f"\nOnline training mode: {args.online_mode.upper()}")
            if args.online_mode == "rls":
                # Override regularisation for RLS initialisation
                config.regularization = args.online_reg
                esn.config.regularization = args.online_reg
                esn.init_rls()

            total_updates = 0
            total_loss = 0.0
            for seq_idx, (in_seq, tgt_seq) in enumerate(zip(input_seqs, target_seqs)):
                stats = esn.online_train_sequence(
                    in_seq, tgt_seq,
                    mode=args.online_mode,
                    learning_rate=args.online_lr,
                    forgetting_factor=args.forgetting_factor,
                    weight_decay=args.weight_decay,
                    washout=args.washout,
                )
                total_updates += stats["n_updates"]
                total_loss    += stats["mean_loss"] * stats["n_updates"]
                if seq_idx % 100 == 0:
                    mean_so_far = total_loss / max(total_updates, 1)
                    logger.info(f"  Seq {seq_idx}/{len(input_seqs)}  "
                                f"mean_loss={mean_so_far:.4f}  updates={total_updates}")

            mean_loss = total_loss / max(total_updates, 1)
            logger.info(f"\nOnline training complete.  "
                        f"Total updates: {total_updates}  Mean loss: {mean_loss:.4f}")

            # Embed online-learning metadata in the GGUF file if requested
            if args.save_online_metadata:
                config.online_learning      = True
                config.online_lr            = args.online_lr
                config.online_reg           = args.online_reg
                config.online_buffer_size   = 64
                config.online_update_mode   = 0 if args.online_mode == "rls" else 2
                # decay_rate is the complement of the forgetting factor, clamped to [0, 1)
                # A forgetting_factor of 1.0 means no forgetting → decay_rate = 0.0
                # A forgetting_factor approaching 0.0 means fast forgetting → decay_rate approaches 1.0
                # We clamp to < 1.0 to satisfy the validation constraint.
                raw_decay = 1.0 - args.forgetting_factor
                config.online_decay_rate    = min(max(raw_decay, 0.0), 0.9999)
                config.freeze_reservoir     = True
                config.replay_window        = 256
                esn.config = config
                logger.info("Online learning metadata will be embedded in GGUF.")
        else:
            # Batch ridge regression (original mode)
            reg_grid = None
            if args.reg_grid:
                reg_grid = [float(v) for v in args.reg_grid.split(",") if v.strip()]
            accuracy = esn.train(input_seqs, target_seqs, washout=args.washout,
                                 val_fraction=args.val_fraction,
                                 calibrate=not args.no_calibrate,
                                 reg_grid=reg_grid)
            logger.info(f"Final training accuracy: {accuracy:.4f}")

    # Save to GGUF
    esn.save_gguf(args.output)
    logger.info(f"\nModel saved to {args.output}")
    logger.info("\nTo use with llama.cpp:")
    logger.info(f"  ./llama-cli -m {args.output} -p \"Your prompt here\"")


if __name__ == "__main__":
    main()
