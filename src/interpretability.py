"""
Neuron/projector interpretability analysis (report Sec. 3C2, Appendix B3).

Quantifies how closely each hidden-layer neuron's weight vector aligns
with the physical Pauli-projector basis used for tomography, via cosine
similarity. High-dimensional hidden layers are reduced with PCA before
comparison. Pure analysis functions only -- plotting lives in
`experiments/interpretability.py`.
"""

from collections import Counter
from typing import List, Optional, Tuple

import numpy as np

from .states import get_pauli_ops


def pauli_coeff_vector(operator: np.ndarray, n_qubits: int) -> np.ndarray:
    """Expand `operator` in the n-qubit Pauli basis: operator = sum_j c_j P_j,
    with c_j = Tr(operator @ P_j) / d (Eq. 24 precursor)."""
    d = 2 ** n_qubits
    _, ops = get_pauli_ops(n_qubits)
    return np.array([np.trace(operator @ p).real / d for p in ops])


def build_projector_pauli_matrix(n_qubits: int) -> Tuple[np.ndarray, List[str]]:
    """Pauli-basis coefficient matrix for every two-outcome projector
    E_{k,+/-} = (I +/- P_k)/2, one row per projector, plus human-readable
    labels like "XZ +"."""
    d = 2 ** n_qubits
    idx_tuples, ops = get_pauli_ops(n_qubits)
    eye = np.eye(d, dtype=complex)
    pauli_names = ["I", "X", "Y", "Z"]

    rows, labels = [], []
    for k in range(1, len(ops)):
        pk = ops[k]
        label_base = "".join(pauli_names[i] for i in idx_tuples[k])
        rows.append(pauli_coeff_vector(0.5 * (eye + pk), n_qubits))
        labels.append(f"{label_base} +")
        rows.append(pauli_coeff_vector(0.5 * (eye - pk), n_qubits))
        labels.append(f"{label_base} -")

    return np.vstack(rows), labels


def get_layer_neurons(model, layer_index: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the weight vectors (one row per neuron) and biases of a
    Dense hidden layer at `layer_index` in a Keras functional model."""
    if not (0 < layer_index < len(model.layers)):
        raise ValueError(f"Layer index {layer_index} out of bounds or is the input layer.")
    layer = model.layers[layer_index]
    w, b = layer.get_weights()
    return w.T, b  # (units, input_dim), (units,)


def normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / (norms + 1e-12)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = normalize_rows(a)
    b = normalize_rows(b)
    return a @ b.T


def best_projector_matches(similarity: np.ndarray, labels: List[str]) -> List[dict]:
    """For each neuron (row of `similarity`), find the projector with the
    largest absolute cosine similarity."""
    best_idx = np.argmax(np.abs(similarity), axis=1)
    results = []
    for i, j in enumerate(best_idx):
        results.append(
            {
                "neuron": i,
                "projector": labels[j],
                "cosine_similarity": float(similarity[i, j]),
                "abs_similarity": float(abs(similarity[i, j])),
            }
        )
    return results


def run_neuron_projector_analysis(
    model, n_qubits: int = 2, layer_index: int = 1
) -> Tuple[List[dict], np.ndarray]:
    """Full single-layer analysis pipeline: extract neuron weight vectors
    (PCA-reduced to 4**n_qubits dimensions if needed), compare against the
    Pauli-projector basis, and return the best-match results plus the raw
    similarity matrix."""
    neuron_matrix, _ = get_layer_neurons(model, layer_index)
    projector_matrix, labels = build_projector_pauli_matrix(n_qubits)
    expected_dim = projector_matrix.shape[1]

    if neuron_matrix.shape[1] > expected_dim:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=expected_dim)
        neuron_matrix = pca.fit_transform(neuron_matrix)

    similarity = cosine_similarity_matrix(neuron_matrix, projector_matrix)
    results = best_projector_matches(similarity, labels)
    return results, similarity


def projector_match_frequencies(results: List[dict]) -> Counter:
    """Count how often each projector label is the best match across all
    neurons in a layer (report Fig. 13)."""
    return Counter(r["projector"] for r in results)
