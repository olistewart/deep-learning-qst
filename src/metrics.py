"""
Fidelity, timing, and cross-method comparison helpers.
"""

import time
from contextlib import contextmanager
from typing import Iterator, Tuple

import numpy as np
import scipy.linalg


def fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """Quantum-state fidelity F(rho1, rho2) = [Tr(sqrt(sqrt(rho1) rho2 sqrt(rho1)))]^2
    (Eq. 15). Symmetric in its two arguments; F=1 iff rho1 == rho2."""
    sqrt_rho1 = scipy.linalg.sqrtm(rho1)
    inner = scipy.linalg.sqrtm(sqrt_rho1 @ rho2 @ sqrt_rho1)
    return float(np.trace(inner).real ** 2)


@contextmanager
def timer() -> Iterator["_Elapsed"]:
    """Context manager measuring wall-clock time.

    >>> with timer() as t:
    ...     do_work()
    >>> t.elapsed
    """
    start = time.perf_counter()
    result = _Elapsed()
    try:
        yield result
    finally:
        result.elapsed = time.perf_counter() - start


class _Elapsed:
    elapsed: float = 0.0


def mean_std(values) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(np.std(values))


def normalised_improvement(
    mean_a: float, std_a: float, mean_b: float, std_b: float
) -> Tuple[float, float]:
    """Percentage improvement of method A over method B, with propagated
    uncertainty (report Appendix E, Eq. E1/E3):

        I = 100 * (F_A - F_B) / F_B
    """
    improvement = 100.0 * (mean_a - mean_b) / mean_b
    sigma = 100.0 * np.sqrt((std_a / mean_b) ** 2 + (mean_a * std_b / mean_b ** 2) ** 2)
    return float(improvement), float(sigma)


def speedup_factor(mean_time_a: float, mean_time_b: float) -> float:
    """How many times faster method B is than method A (T_A / T_B)."""
    return float(mean_time_a / mean_time_b)
