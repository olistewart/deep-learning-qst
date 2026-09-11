"""Deep learning-enhanced quantum state estimation -- core library.

See the top-level README for an overview. Modules:

    states           quantum state generation and vectorisation
    measurements     the Pauli POVM and ideal/estimated expectation values
    noise            synthetic SPAM noise channels
    stokes           linear-inversion (Stokes) reconstruction
    mle              maximum-likelihood (RhoR) reconstruction
    model            the MLP estimator and training utilities (needs TensorFlow)
    metrics          fidelity, timing, and comparison helpers
    interpretability neuron/projector cosine-similarity analysis
    plotting         shared matplotlib styling helpers
    config           all numerical constants used in the experiments
"""
