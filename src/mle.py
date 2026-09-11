"""
Maximum-likelihood estimation via the iterative RhoR algorithm (report
Sec. 3B1, Eq. 16-18).
"""

import time
from typing import Optional, Tuple, Union

import numpy as np

from .measurements import build_pauli_two_outcome_povm


def _hermitize_and_normalize(rho: np.ndarray) -> np.ndarray:
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if np.abs(tr) < 1e-15:
        raise ValueError("Trace became ~0 during MLE iteration.")
    return rho / tr


def reconstruct_mle_from_counts(
    counts: np.ndarray,
    n_qubits: int,
    max_iter: int = 150,
    tol: float = 1e-8,
    eps: float = 1e-12,
    return_diagnostics: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """Reconstruct the density matrix maximising the log-likelihood of the
    observed counts, via the fixed-point iteration

        R(rho) = sum_i (n_i / p_i(rho)) E_i,     rho <- R rho R / Tr(R rho R)

    (Eq. 17), initialised at the maximally mixed state. `R rho R` is
    positive semi-definite by construction for any Hermitian `rho`, so
    physicality holds at every iteration up to numerical rounding.
    """
    d = 2 ** n_qubits
    povm = build_pauli_two_outcome_povm(n_qubits)

    counts = np.asarray(counts, dtype=np.int64)
    if counts.size != len(povm):
        raise ValueError(f"Counts length {counts.size} does not match POVM length {len(povm)}.")

    rho = np.eye(d, dtype=complex) / d
    e_list = [ei for (ei, _) in povm]
    n_list = counts.astype(float)

    logl_prev = None
    diff = np.inf
    start = time.perf_counter()

    for it in range(max_iter):
        p = np.array([np.real(np.trace(rho @ ei)) for ei in e_list], dtype=float)
        p = np.clip(p, eps, None)

        weights = n_list / p
        r = np.zeros((d, d), dtype=complex)
        for w, ei in zip(weights, e_list):
            r += w * ei

        rho_new = r @ rho @ r
        rho_new = _hermitize_and_normalize(rho_new)

        diff = np.linalg.norm(rho_new - rho, ord="fro") / (np.linalg.norm(rho, ord="fro") + eps)

        p_new = np.array([np.real(np.trace(rho_new @ ei)) for ei in e_list], dtype=float)
        p_new = np.clip(p_new, eps, None)
        logl = float(np.sum(n_list * np.log(p_new)))

        rho = rho_new
        logl_prev = logl

        if diff < tol:
            if return_diagnostics:
                return rho, {
                    "iters_used": it + 1,
                    "rel_diff": diff,
                    "logL": logl,
                    "elapsed_time": time.perf_counter() - start,
                }
            return rho

    if return_diagnostics:
        return rho, {
            "iters_used": max_iter,
            "rel_diff": diff,
            "logL": logl_prev,
            "elapsed_time": time.perf_counter() - start,
        }
    return rho
