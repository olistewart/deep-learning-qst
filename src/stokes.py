"""
Stokes (linear inversion) reconstruction, and the physical projection
shared by every reconstruction method (report Sec. 3B2, Eq. 19-20).
"""

import numpy as np

from .states import get_pauli_ops


def make_physical(rho_raw: np.ndarray) -> np.ndarray:
    """Project a raw (possibly unphysical) matrix onto the nearest valid
    density matrix via rho_hat = A^dagger A / Tr(A^dagger A) (Eq. 20).

    This guarantees Hermiticity, positive semi-definiteness and unit trace
    for any input, and is used to enforce physicality on the raw Stokes
    estimate and on the neural network's raw output alike.
    """
    a = rho_raw
    aha = a.conj().T @ a
    tr = np.trace(aha)
    if np.isclose(tr, 0):
        d = a.shape[0]
        return np.eye(d, dtype=complex) / d
    return aha / tr


def eigen_clip(rho_pred: np.ndarray) -> np.ndarray:
    """Alternative physical projection: symmetrise, clip negative
    eigenvalues to zero, and renormalise the trace. Provided for
    completeness (report Sec. 3B3) -- `make_physical` is used by default
    throughout the experiments to match the reported results."""
    rho_pred = (rho_pred + rho_pred.conj().T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(rho_pred)
    eigenvalues[eigenvalues < 0] = 0
    rho_physical = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
    rho_physical /= np.trace(rho_physical)
    return rho_physical


def reconstruct_from_stokes(s: np.ndarray, n_qubits: int, physical: bool = True) -> np.ndarray:
    """Linear inversion: rho_hat = (1/d) * sum_k S_k P_k (Eq. 19), followed
    by an optional physical projection (the raw estimate is not guaranteed
    to be a valid density matrix under noise)."""
    d = 2 ** n_qubits
    _, ops = get_pauli_ops(n_qubits)

    rho_hat = np.zeros((d, d), dtype=complex)
    for k, p in enumerate(ops):
        rho_hat += s[k] * p
    rho_hat /= d

    if physical:
        rho_hat = make_physical(rho_hat)
    return rho_hat
