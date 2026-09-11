"""
Synthetic noise channels.

Three phenomenological SPAM (state-preparation-and-measurement) error
sources are implemented at the abstract level used for the two-qubit
sweeps (report Sec. 3A3, Table I):

    * additive Gaussian perturbation of the Stokes parameters (preparation/control)
    * finite-shot binomial sampling (projection noise, always present)
    * "pepper" / missing-measurement noise (readout)

and their trapped-ion-inspired refinements used for the scalability
benchmark (report Sec. 3C3, Appendix D):

    * coherent overrotation (state-level unitary misrotation)
    * local phase-flip channel (dephasing)
    * bit-flip readout error (fluorescence misclassification)

`simulate_counts` and `simulate_counts_trapped_ion` are the two top-level
entry points that chain these into the full "true state -> noisy counts"
pipeline used by every experiment script.
"""

from typing import Optional, Tuple

import numpy as np
import scipy.linalg

from .states import I, X, Y, Z, kron_all, op_on_qubit
from .measurements import pauli_expectations, get_pauli_ops


# --------------------------------------------------------------------------
# Preparation/control noise: additive Gaussian perturbation (Eq. 14)
# --------------------------------------------------------------------------
def additive_gaussian_noise(
    s: np.ndarray,
    sigma: float,
    n_qubits: int,
    keep_first: bool = True,
    clip: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float]:
    """Perturb each Stokes component with iid N(0, sigma^2) noise, then
    clip back into the physical [-1, 1] range."""
    if rng is None:
        rng = np.random.default_rng()
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    s = np.asarray(s, dtype=float)
    noise = rng.normal(0.0, sigma, size=s.shape)
    if keep_first:
        noise[..., 0] = 0.0

    s_noisy = s + noise
    if clip:
        s_noisy = np.clip(s_noisy, -1.0, 1.0)
    return s_noisy, sigma


# --------------------------------------------------------------------------
# State-level (CPTP) noise channels used by the trapped-ion pipeline
# --------------------------------------------------------------------------
def depolarize_rho(rho: np.ndarray, p: float, n_qubits: int) -> np.ndarray:
    """Global depolarising channel rho -> (1-p) rho + p * I/d."""
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1].")
    d = 2 ** n_qubits
    return (1 - p) * rho + p * (np.eye(d, dtype=complex) / d)


def apply_kraus_channel(rho: np.ndarray, kraus_ops) -> np.ndarray:
    out = np.zeros_like(rho, dtype=complex)
    for k in kraus_ops:
        out += k @ rho @ k.conj().T
    return out


def local_phase_flip(rho: np.ndarray, p_phi: float, n_qubits: int) -> np.ndarray:
    """Independent local phase-flip channel on every qubit (Eq. D4-D5):
    rho -> (1-p_phi) rho + p_phi * Z rho Z, applied qubit by qubit."""
    if not (0.0 <= p_phi <= 1.0):
        raise ValueError("p_phi must be in [0,1].")
    k0_1 = np.sqrt(1 - p_phi) * I
    k1_1 = np.sqrt(p_phi) * Z

    rho_out = rho
    for q in range(n_qubits):
        k0 = op_on_qubit(k0_1, n_qubits, q)
        k1 = op_on_qubit(k1_1, n_qubits, q)
        rho_out = apply_kraus_channel(rho_out, [k0, k1])
    return rho_out


def coherent_overrotation(
    rho: np.ndarray,
    n_qubits: int,
    sigma_theta: float,
    axis: str = "X",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Independent random unitary misrotation on each qubit, angle ~
    N(0, sigma_theta^2) about `axis` (Eq. D2-D3)."""
    if rng is None:
        rng = np.random.default_rng()
    if sigma_theta < 0:
        raise ValueError("sigma_theta must be non-negative.")
    if axis not in ("X", "Y", "Z"):
        raise ValueError("axis must be X/Y/Z.")

    s = {"X": X, "Y": Y, "Z": Z}[axis]
    deltas = rng.normal(0.0, sigma_theta, size=n_qubits)

    u_list = [scipy.linalg.expm(-1j * (dt / 2) * s) for dt in deltas]
    u = kron_all(u_list)
    return u @ rho @ u.conj().T


# --------------------------------------------------------------------------
# Projection noise: finite-shot binomial sampling (always present, Eq. 12)
# --------------------------------------------------------------------------
def counts_from_stokes_binomial(
    s: np.ndarray, n_qubits: int, n_shots: int = 500, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Simulate N_shots repeated projective measurements of every
    non-identity Pauli observable, given (possibly noisy) Stokes values."""
    if rng is None:
        rng = np.random.default_rng()

    s = np.asarray(s, dtype=float)
    _, ops = get_pauli_ops(n_qubits)
    m = len(ops) - 1
    counts = np.empty(2 * m, dtype=np.int64)

    out = 0
    for k in range(1, len(ops)):
        p_plus = np.clip(0.5 * (1.0 + float(s[k])), 0.0, 1.0)
        n_plus = rng.binomial(n_shots, p_plus)
        counts[out] = n_plus
        counts[out + 1] = n_shots - n_plus
        out += 2
    return counts


# --------------------------------------------------------------------------
# Readout noise: abstract "pepper" model and trapped-ion bit-flip model
# --------------------------------------------------------------------------
def apply_missing_measurements_to_counts(
    counts: np.ndarray, missing_p: float, n_shots: int, rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """"Pepper" readout noise (report Sec. 3A3): replace a random fraction
    `missing_p` of Pauli observables with uninformative (p=0.5) counts,
    i.e. a complete loss of information for that observable."""
    if rng is None:
        rng = np.random.default_rng()
    if not (0.0 <= missing_p <= 1.0):
        raise ValueError("missing_p must be in [0,1].")

    counts = np.asarray(counts, dtype=np.int64).copy()
    m = counts.size // 2
    mask = rng.random(m) < missing_p

    idx = np.where(mask)[0]
    if idx.size > 0:
        n_plus = rng.binomial(n_shots, 0.5, size=idx.size)
        counts[2 * idx] = n_plus
        counts[2 * idx + 1] = n_shots - n_plus
    return counts, mask


def apply_readout_flip_to_counts(
    counts: np.ndarray, p_readout: float, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Trapped-ion bit-flip readout error (Eq. D9-D11): a fraction
    `p_readout` of detected outcomes is misassigned between the bright and
    dark states."""
    if rng is None:
        rng = np.random.default_rng()
    if not (0.0 <= p_readout <= 1.0):
        raise ValueError("p_readout must be in [0,1].")

    counts = np.asarray(counts, dtype=np.int64).copy()
    n_plus = counts[0::2]
    n_minus = counts[1::2]

    k_plus = rng.binomial(n_plus, p_readout)   # + -> -
    k_minus = rng.binomial(n_minus, p_readout)  # - -> +

    counts[0::2] = n_plus - k_plus + k_minus
    counts[1::2] = n_minus - k_minus + k_plus
    return counts


# --------------------------------------------------------------------------
# Top-level pipelines: true state -> noisy measurement counts
# --------------------------------------------------------------------------
def simulate_counts(
    rho_true: np.ndarray,
    n_qubits: int,
    n_shots: int,
    gauss_sigma: float = 0.0,
    missing_p: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Abstract two-qubit noise pipeline (report Sec. 3A3):

        rho_true -> S_true -> [additive Gaussian] -> binomial sampling
        -> [pepper / missing-measurement]
    """
    if rng is None:
        rng = np.random.default_rng()

    s_true = pauli_expectations(rho_true, n_qubits)

    if gauss_sigma > 0:
        s_work, _ = additive_gaussian_noise(s_true, gauss_sigma, n_qubits, keep_first=True, rng=rng)
    else:
        s_work = s_true

    counts = counts_from_stokes_binomial(s_work, n_qubits, n_shots=n_shots, rng=rng)

    if missing_p > 0:
        counts, _ = apply_missing_measurements_to_counts(counts, missing_p, n_shots, rng=rng)
    return counts


def simulate_counts_trapped_ion(
    rho_true: np.ndarray,
    n_qubits: int,
    n_shots: int,
    p_depol: float = 0.0,
    p_phi: float = 0.0,
    sigma_theta: float = 0.0,
    axis: str = "X",
    p_readout: float = 0.0,
    missing_p: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Trapped-ion-inspired combined noise pipeline (report Sec. 3C3,
    Appendix D):

        rho_true -> [depolarising] -> [phase-flip] -> [coherent overrotation]
        -> S -> binomial sampling -> [bit-flip readout] -> [missing measurements]
    """
    if rng is None:
        rng = np.random.default_rng()

    rho = rho_true
    if p_depol > 0:
        rho = depolarize_rho(rho, p_depol, n_qubits)
    if p_phi > 0:
        rho = local_phase_flip(rho, p_phi, n_qubits)
    if sigma_theta > 0:
        rho = coherent_overrotation(rho, n_qubits, sigma_theta, axis=axis, rng=rng)

    s = pauli_expectations(rho, n_qubits)
    counts = counts_from_stokes_binomial(s, n_qubits, n_shots=n_shots, rng=rng)

    if p_readout > 0:
        counts = apply_readout_flip_to_counts(counts, p_readout, rng=rng)

    mask = None
    if missing_p > 0:
        counts, mask = apply_missing_measurements_to_counts(counts, missing_p, n_shots, rng=rng)

    return counts, mask
