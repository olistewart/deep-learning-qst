"""
Quantum state generation.

Everything needed to build physically valid n-qubit density matrices:
Pauli operators and their tensor products, several state families (named
states, a trapped-ion-inspired random circuit, Haar-random pure states,
Ginibre-random mixed states), a mixture sampler that draws from these
families, and the vectorisation used as the neural network's training
target.

Corresponds to report Sec. 3A1 and Appendix C.
"""

from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg

# --------------------------------------------------------------------------
# Single-qubit operators and tensor products (Appendix C1)
# --------------------------------------------------------------------------
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_1Q = [I, X, Y, Z]


def kron_all(mats: List[np.ndarray]) -> np.ndarray:
    """Kronecker (tensor) product of a list of matrices, left to right."""
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def op_on_qubit(single_op: np.ndarray, n: int, q: int) -> np.ndarray:
    """Embed a single-qubit operator on qubit `q` of an n-qubit register."""
    return kron_all([single_op if i == q else I for i in range(n)])


# --------------------------------------------------------------------------
# n-qubit Pauli basis, cached per qubit count (Eq. 4)
# --------------------------------------------------------------------------
_PAULI_CACHE: Dict[int, Tuple[List[tuple], List[np.ndarray]]] = {}


def get_pauli_ops(n: int) -> Tuple[List[tuple], List[np.ndarray]]:
    """Return (index_tuples, operators) for the 4^n n-qubit Pauli basis.

    ``index_tuples[k]`` gives the (axis_1, ..., axis_n) indices (0=I, 1=X,
    2=Y, 3=Z) of the k-th operator; ``operators[0]`` is always the identity.
    """
    if n in _PAULI_CACHE:
        return _PAULI_CACHE[n]
    idx_tuples = list(product(range(4), repeat=n))
    ops = [kron_all([PAULI_1Q[i] for i in idx]) for idx in idx_tuples]
    _PAULI_CACHE[n] = (idx_tuples, ops)
    return _PAULI_CACHE[n]


# --------------------------------------------------------------------------
# Density-matrix construction and vectorisation (Appendix C2, Eq. C3-C4)
# --------------------------------------------------------------------------
def rho_from_ket(psi: np.ndarray) -> np.ndarray:
    """Pure-state density matrix rho = |psi><psi| from a state vector."""
    return np.outer(psi, psi.conj())


def rho_to_real_vector(rho: np.ndarray) -> np.ndarray:
    """Vectorise a density matrix as [Re(rho).flatten(), Im(rho).flatten()].

    This 2*d^2-dimensional real vector is the neural network's training
    target (Eq. C4).
    """
    d = rho.shape[0]
    real = np.reshape(rho.real, d * d, order="F")
    imag = np.reshape(rho.imag, d * d, order="F")
    return np.concatenate([real, imag], axis=0)


def real_vector_to_rho(v: np.ndarray) -> np.ndarray:
    """Inverse of `rho_to_real_vector`."""
    d2 = v.shape[0] // 2
    d = int(np.sqrt(d2))
    real = v[:d2].reshape(d, d, order="F")
    imag = v[d2:].reshape(d, d, order="F")
    return real + 1j * imag


# --------------------------------------------------------------------------
# State families (Appendix C3)
# --------------------------------------------------------------------------
def haar_pure_rho(n: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random pure state (Eq. C5-C6)."""
    d = 2 ** n
    psi = rng.normal(size=d) + 1j * rng.normal(size=d)
    psi /= np.linalg.norm(psi)
    return rho_from_ket(psi)


def ginibre_mixed_rho(n: int, rng: np.random.Generator, k: Optional[int] = None) -> np.ndarray:
    """Random mixed state from the Ginibre ensemble (Eq. C7-C8)."""
    d = 2 ** n
    if k is None:
        k = d
    g = rng.normal(size=(d, k)) + 1j * rng.normal(size=(d, k))
    a = g @ g.conj().T
    return a / np.trace(a)


def ket_ghz(n: int) -> np.ndarray:
    """n-qubit GHZ state (Eq. C13)."""
    d = 2 ** n
    psi = np.zeros(d, complex)
    psi[0] = 1 / np.sqrt(2)
    psi[-1] = 1 / np.sqrt(2)
    return psi


def ket_w(n: int) -> np.ndarray:
    """n-qubit W state (Eq. C14)."""
    d = 2 ** n
    psi = np.zeros(d, complex)
    for k in range(n):
        idx = 1 << (n - 1 - k)  # |10...0>, |01...0>, ...
        psi[idx] = 1 / np.sqrt(n)
    return psi


def ket_bell(kind: str = "phi+") -> np.ndarray:
    """One of the four maximally-entangled two-qubit Bell states (Eq. C9-C12)."""
    psi = np.zeros(4, complex)
    if kind == "phi+":
        psi[0] = 1 / np.sqrt(2)
        psi[3] = 1 / np.sqrt(2)
    elif kind == "phi-":
        psi[0] = 1 / np.sqrt(2)
        psi[3] = -1 / np.sqrt(2)
    elif kind == "psi+":
        psi[1] = 1 / np.sqrt(2)
        psi[2] = 1 / np.sqrt(2)
    elif kind == "psi-":
        psi[1] = 1 / np.sqrt(2)
        psi[2] = -1 / np.sqrt(2)
    else:
        raise ValueError(f"Unknown Bell kind: {kind!r}")
    return psi


# --------------------------------------------------------------------------
# Trapped-ion-inspired random circuit (Appendix C4, Eq. C15-C23)
# --------------------------------------------------------------------------
def global_rotation(n: int, axis: str, theta: float) -> np.ndarray:
    """Global single-qubit rotation U = R_axis(theta)^{⊗n} (Eq. C16-C17)."""
    if axis == "X":
        s = X
    elif axis == "Y":
        s = Y
    elif axis == "Z":
        s = Z
    else:
        raise ValueError("axis must be X/Y/Z")
    u1 = scipy.linalg.expm(-1j * theta / 2 * s)
    return kron_all([u1 for _ in range(n)])


def ms_entangler_effective(n: int, theta: float) -> np.ndarray:
    """All-to-all XX-type entangling gate, U = exp(-i*theta/2 * sum_{i<j} X_i X_j)
    (Eq. C18-C19, "Molmer-Sorensen"-inspired)."""
    h = np.zeros((2 ** n, 2 ** n), complex)
    for i in range(n):
        xi = op_on_qubit(X, n, i)
        for j in range(i + 1, n):
            xj = op_on_qubit(X, n, j)
            h += xi @ xj
    return scipy.linalg.expm(-1j * (theta / 2) * h)


def ion_like_circuit_rho(n: int, rng: np.random.Generator, depth: int = 2) -> np.ndarray:
    """Randomised trapped-ion-inspired circuit state (Eq. C20-C23).

    Starting from |0...0>, apply `depth` layers of independently sampled
    global Y/X rotations followed by an all-to-all XX entangler, then
    return the resulting pure-state density matrix.
    """
    d = 2 ** n
    psi = np.zeros(d, complex)
    psi[0] = 1.0

    u = np.eye(d, dtype=complex)
    for _ in range(depth):
        u = global_rotation(n, "Y", rng.uniform(0, 2 * np.pi)) @ u
        u = global_rotation(n, "X", rng.uniform(0, 2 * np.pi)) @ u
        u = ms_entangler_effective(n, rng.uniform(0, np.pi / 2)) @ u

    psi = u @ psi
    return rho_from_ket(psi)


# --------------------------------------------------------------------------
# State-family mixture sampler and dataset construction (Appendix C5)
# --------------------------------------------------------------------------
def sample_state(n: int, rng: np.random.Generator, mix: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Draw one n-qubit density matrix from a mixture of state families.

    `mix` gives relative weights over {"named", "ion", "haar", "ginibre"};
    weights are renormalised to sum to 1. Defaults to pure ion-like circuit
    states, matching the report's primary ensemble.
    """
    if mix is None:
        mix = {"named": 0.0, "ion": 1.0, "haar": 0.0, "ginibre": 0.0}

    keys = list(mix.keys())
    w = np.array([mix[k] for k in keys], dtype=float)
    w = w / w.sum()
    choice = rng.choice(keys, p=w)

    if choice == "named":
        if n == 2:
            psi = ket_bell(rng.choice(["phi+", "phi-", "psi+", "psi-"]))
            return rho_from_ket(psi)
        family = rng.choice(["ghz", "w"])
        psi = ket_ghz(n) if family == "ghz" else ket_w(n)
        return rho_from_ket(psi)

    if choice == "ion":
        return ion_like_circuit_rho(n, rng, depth=int(rng.integers(1, 4)))

    if choice == "haar":
        return haar_pure_rho(n, rng)

    if choice == "ginibre":
        return ginibre_mixed_rho(n, rng)

    raise ValueError(f"Bad mixture choice: {choice!r}")


def make_dataset(
    n_states: int,
    n_qubits: int,
    seed: int = 0,
    mix: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a supervised dataset of (ideal Pauli expectations, target density matrix).

    Returns
    -------
    X : ndarray, shape (n_states, 4**n_qubits)
        Ideal (noiseless) Pauli expectation values / Stokes parameters.
    y : ndarray, shape (n_states, 2 * (2**n_qubits)**2)
        Vectorised target density matrices (see `rho_to_real_vector`).
    """
    from .measurements import pauli_expectations  # local import: avoids a cycle

    rng = np.random.default_rng(seed)
    d = 2 ** n_qubits
    X = np.zeros((n_states, 4 ** n_qubits), dtype=float)
    y = np.zeros((n_states, 2 * d * d), dtype=float)

    for i in range(n_states):
        rho = sample_state(n_qubits, rng, mix=mix)
        X[i] = pauli_expectations(rho, n_qubits)
        y[i] = rho_to_real_vector(rho)
    return X, y
