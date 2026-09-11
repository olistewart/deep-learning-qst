"""
Central configuration for the deep-learning-enhanced QST pipeline.

Every numerical constant used to produce the results in the accompanying
report lives here, rather than scattered through the experiment scripts.
Values are grouped by the stage of the pipeline they control:

    dataset generation -> ideal-benchmark noise -> abstract noise sweeps
    -> trapped-ion operating point -> model architecture / training
    -> MLE solver -> scalability sweep

Every experiment script accepts a ``--quick`` flag which overrides the
relevant block below with a small, fast configuration suitable for a
sub-minute smoke test (see ``QUICK_OVERRIDES``). Nothing here is required
to reproduce the report's headline numbers *exactly* (five-run Monte Carlo
averages of stochastic training are not bitwise reproducible across
machines/BLAS backends), but with these defaults the reconstructed trends
and approximate magnitudes match Stewart (2026), "Deep Learning-Enhanced
Quantum State Estimation of Simulated Multi-Qubit Systems".
"""

from dataclasses import dataclass, field
from typing import Dict, List


# --------------------------------------------------------------------------
# Dataset generation (report Sec. 3A, Appendix C)
# --------------------------------------------------------------------------
N_TRAIN = 10_000          # training states for the 2-qubit model
N_TEST = 1_000             # held-out test states (report uses 5000 for the
                            # headline KDE plot in Fig. 6 -- bump N_TEST if
                            # you want to match that figure exactly)
TRAIN_SEED = 42
TEST_SEED = 43

# State-family mixture passed to `states.sample_state`. The report's primary
# ensemble is the trapped-ion-inspired random circuit of Sec. 3A1 (Eq. 8);
# Haar-random pure states, Ginibre mixed states and named states (Bell/GHZ/W)
# are also implemented (Appendix C) and can be mixed in for robustness
# checks by editing these weights.
STATE_MIX: Dict[str, float] = {
    "named": 0.0,
    "ion": 1.0,
    "haar": 0.0,
    "ginibre": 0.0,
}

# --------------------------------------------------------------------------
# Two-qubit "ideal benchmark" (report Sec. 4A, Fig. 6): finite-shot noise
# only, no additional SPAM error.
# --------------------------------------------------------------------------
IDEAL_N_QUBITS = 2
IDEAL_N_SHOTS = 100
IDEAL_N_TEST = 5_000

# --------------------------------------------------------------------------
# Two-qubit independent noise sweeps (report Sec. 4B, Table I)
# --------------------------------------------------------------------------
SWEEP_N_QUBITS = 2
SWEEP_NUM_RUNS = 5

SIGMA_SWEEP: List[float] = [0.00, 0.06, 0.12, 0.18, 0.24, 0.30]     # additive Gaussian std
SHOTS_SWEEP: List[int] = [10, 50, 100, 500, 1000, 5000]              # finite-shot count
PEPPER_SWEEP: List[float] = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]     # missing-measurement fraction

# Shot count held fixed while sweeping sigma/pepper (large enough that
# projection noise is negligible relative to the swept channel).
SWEEP_FIXED_SHOTS = 10_000

# --------------------------------------------------------------------------
# Trapped-ion-inspired operating point (report Sec. 3C3, Table II,
# Appendix D) -- used by the scalability benchmark.
# --------------------------------------------------------------------------
ION_N_SHOTS = 100
ION_P_PHI = 0.06          # local phase-flip strength (laser frequency noise / detuning)
ION_SIGMA_THETA = 0.08    # coherent overrotation std (addressing error)
ION_P_READOUT = 0.02      # bit-flip readout error (fluorescence misclassification)
ION_AXIS = "X"

# --------------------------------------------------------------------------
# Scalability benchmark (report Sec. 4C, Fig. 8)
# --------------------------------------------------------------------------
SCALABILITY_N_QUBITS: List[int] = [2, 3, 4]
SCALABILITY_NUM_RUNS = 5
SCALABILITY_N_TRAIN = 10_000
SCALABILITY_N_TEST = 1_000

# --------------------------------------------------------------------------
# Neural-network architecture and training (report Sec. 3B3, Table III,
# Appendix B2 -- "Model 5", the dropout-regularised, Bayesian-optimised MLP)
# --------------------------------------------------------------------------
HIDDEN_LAYERS: List[int] = [256, 256, 128]
DROPOUT_RATE = 0.2
LEARNING_RATE = 1.5567e-3
BATCH_SIZE = 128
EPOCHS = 60
EARLY_STOPPING_PATIENCE = 10
VALIDATION_SPLIT = 0.1

# Reference architecture used only for the architecture-comparison plot
# in `experiments/train_model.py` (report Fig. 10/11, Model 4).
REFERENCE_HIDDEN_LAYERS: List[int] = [1024, 1024]
REFERENCE_LEARNING_RATE = 1e-3

# --------------------------------------------------------------------------
# Maximum-likelihood estimation (RhoR algorithm, report Sec. 3B1, Eq. 17)
# --------------------------------------------------------------------------
MLE_MAX_ITER = 150     # report settles on 100 for the 2-qubit benchmark and
                        # uses 150 for the scalability benchmark; the
                        # convergence trade-off is explored in Fig. 2.
MLE_TOL = 1e-8

# --------------------------------------------------------------------------
# Fidelity is evaluated on the same test ensemble across methods; how many
# independent Monte Carlo repeats to average over for the headline numbers.
# --------------------------------------------------------------------------
DEFAULT_NUM_RUNS = 5


@dataclass
class QuickOverrides:
    """Small, fast values used when a script is run with --quick."""
    n_train: int = 200
    n_test: int = 100
    epochs: int = 5
    num_runs: int = 1
    mle_max_iter: int = 30
    sigma_sweep: List[float] = field(default_factory=lambda: [0.0, 0.2])
    shots_sweep: List[int] = field(default_factory=lambda: [50, 1000])
    pepper_sweep: List[float] = field(default_factory=lambda: [0.0, 0.2])
    scalability_n_qubits: List[int] = field(default_factory=lambda: [2, 3])


QUICK = QuickOverrides()
