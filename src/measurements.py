"""
Projective measurements and expectation-value estimation.

Builds the two-outcome Pauli POVM used throughout this work, computes ideal
Pauli expectation values (Stokes parameters) from a density matrix, and
converts finite-shot measurement counts back into an estimated Stokes
vector. Noise is applied separately in `noise.py`; this module handles the
noiseless measurement model (report Sec. 2A, Eq. 2-4).
"""

from typing import List, Optional, Tuple

import numpy as np

from .states import get_pauli_ops


def pauli_expectations(rho: np.ndarray, n_qubits: int) -> np.ndarray:
    """Ideal Pauli expectation values <P_k> = Tr(rho P_k) for k = 0..4^n-1.

    S[0] is always 1 (identity / normalisation). This is the "Stokes
    vector" S in Eq. 10-11 of the report.
    """
    _, ops = get_pauli_ops(n_qubits)
    s = np.empty(len(ops), dtype=float)
    for k, p in enumerate(ops):
        s[k] = np.trace(rho @ p).real
    return s


def build_pauli_two_outcome_povm(n_qubits: int) -> List[Tuple[np.ndarray, tuple]]:
    """Two-outcome POVM {E_k,+/-} = (I +/- P_k)/2 for every non-identity
    Pauli operator P_k (report Eq. 4, Appendix A). The identity (k=0) is
    skipped: its expectation is always 1 and carries no information.

    Returns a list of (E, (k, sign)) pairs, ordered as
    [E_{1,+}, E_{1,-}, E_{2,+}, E_{2,-}, ...] -- this ordering is shared by
    every function in this package that consumes or produces "counts".
    """
    d = 2 ** n_qubits
    _, ops = get_pauli_ops(n_qubits)
    eye = np.eye(d, dtype=complex)

    povm = []
    for k in range(1, len(ops)):
        pk = ops[k]
        povm.append((0.5 * (eye + pk), (k, +1)))
        povm.append((0.5 * (eye - pk), (k, -1)))
    return povm


def stokes_from_counts(counts: np.ndarray, n_qubits: int, eps: float = 1e-12) -> np.ndarray:
    """Estimate the Stokes vector from measurement counts (Eq. 13).

    `counts` is the flat [n(1,+), n(1,-), n(2,+), n(2,-), ...] array
    produced by `noise.counts_from_stokes_binomial` (or one of the noisy
    variants in `noise.py`), aligned with `build_pauli_two_outcome_povm`.
    """
    counts = np.asarray(counts, dtype=np.int64)
    _, ops = get_pauli_ops(n_qubits)
    m = len(ops) - 1
    if counts.size != 2 * m:
        raise ValueError(f"Expected counts length {2 * m}, got {counts.size}")

    n_plus = counts[0::2].astype(float)
    n_minus = counts[1::2].astype(float)
    denom = n_plus + n_minus

    e_hat = (n_plus - n_minus) / (denom + eps)

    s_hat = np.zeros(len(ops), dtype=float)
    s_hat[0] = 1.0
    s_hat[1:] = np.clip(e_hat, -1.0, 1.0)
    return s_hat
