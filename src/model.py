"""
The MLP architecture and training utilities (report Sec. 3B3, Table III,
Appendix B).

Requires TensorFlow/Keras (see requirements.txt). Import of this module is
deferred by the experiment scripts until after argument parsing, so
`--help` works even without TensorFlow installed.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "TensorFlow is required for src.model. Install it with "
        "`pip install -r requirements.txt`."
    ) from exc

from . import config
from .states import real_vector_to_rho, rho_to_real_vector
from .measurements import pauli_expectations
from .noise import additive_gaussian_noise, counts_from_stokes_binomial
from .measurements import stokes_from_counts


def build_qst_mlp(
    n_qubits: int,
    hidden_layers: Sequence[int] = tuple(config.HIDDEN_LAYERS),
    dropout_rate: float = config.DROPOUT_RATE,
    learning_rate: float = config.LEARNING_RATE,
) -> "keras.Model":
    """Build the supervised MLP estimator: noisy Stokes vector in, vectorised
    density matrix out (report Fig. 4). ReLU hidden layers with dropout
    after each of the first two, linear output, Adam + MSE.

    The default hyperparameters are "Model 5" from Table III -- the
    architecture selected after comparing five candidates (Fig. 10) and
    refining the learning rate/dropout with Bayesian optimisation.
    """
    input_dim = 4 ** n_qubits
    output_dim = 2 * (2 ** n_qubits) ** 2

    inputs = keras.Input(shape=(input_dim,), name="stokes_input")
    x = inputs
    for i, units in enumerate(hidden_layers):
        x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        if dropout_rate > 0 and i < len(hidden_layers) - 1:
            x = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)
    outputs = layers.Dense(output_dim, activation="linear", name="rho_output")(x)

    model = keras.Model(inputs, outputs, name=f"mlp_qst_{n_qubits}qubit")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_reference_mlp(
    n_qubits: int,
    hidden_layers: Sequence[int] = tuple(config.REFERENCE_HIDDEN_LAYERS),
    learning_rate: float = config.REFERENCE_LEARNING_RATE,
) -> "keras.Model":
    """The large, unregularised reference architecture used only for the
    architecture-comparison plot (report Fig. 10/11, "Model 4"). It
    overfits more than the selected dropout model and is not used for any
    of the headline results."""
    input_dim = 4 ** n_qubits
    output_dim = 2 * (2 ** n_qubits) ** 2

    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for units in hidden_layers:
        x = layers.Dense(units, activation="relu")(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)

    model = keras.Model(inputs, outputs, name=f"mlp_qst_ref_{n_qubits}qubit")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def make_noisy_stokes_inputs(
    y: np.ndarray,
    n_qubits: int,
    n_shots: int,
    sigma: float = 0.0,
    missing_p: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Turn a batch of vectorised target density matrices into noisy Stokes
    inputs under the abstract noise model (matched to `noise.simulate_counts`),
    for training/evaluating the NN and the traditional methods on
    identical noise realisations."""
    if rng is None:
        rng = np.random.default_rng()

    x_noisy = np.zeros((y.shape[0], 4 ** n_qubits), dtype=float)
    for i, y_vec in enumerate(y):
        rho = real_vector_to_rho(y_vec)
        s_true = pauli_expectations(rho, n_qubits)
        if sigma > 0:
            s_true, _ = additive_gaussian_noise(s_true, sigma, n_qubits, rng=rng)
        counts = counts_from_stokes_binomial(s_true, n_qubits, n_shots=n_shots, rng=rng)
        x_noisy[i] = stokes_from_counts(counts, n_qubits)
    return x_noisy


def train_model(
    model: "keras.Model",
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    validation_split: float = config.VALIDATION_SPLIT,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    verbose: int = 0,
) -> "keras.callbacks.History":
    """Train with early stopping on validation loss (report Sec. 3B3)."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=verbose
    )
    return model.fit(
        x_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=verbose,
    )
