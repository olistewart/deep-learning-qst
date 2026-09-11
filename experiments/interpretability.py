#!/usr/bin/env python3
"""
Neuron/projector interpretability analysis: trains a two-qubit NN
estimator and measures how closely each hidden-layer neuron's weight
vector aligns with the physical Pauli-projector basis (report Sec. 3C2,
Fig. 5/13, Appendix B3).

Usage
-----
    python experiments/interpretability.py
    python experiments/interpretability.py --quick
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    from src import config, interpretability, plotting, states
    from src.model import build_qst_mlp, make_noisy_stokes_inputs, train_model

    n_qubits = 2  # interpretability analysis is only defined for the 2-qubit projector basis
    n_train = config.QUICK.n_train if args.quick else config.N_TRAIN
    epochs = config.QUICK.epochs if args.quick else config.EPOCHS

    print(f"[interpretability] n_qubits={n_qubits} n_train={n_train} epochs={epochs}")

    _, y_train = states.make_dataset(n_train, n_qubits, seed=config.TRAIN_SEED, mix=config.STATE_MIX)
    rng = np.random.default_rng(0)
    x_train = make_noisy_stokes_inputs(y_train, n_qubits, n_shots=config.IDEAL_N_SHOTS, rng=rng)

    print("Training model...")
    model = build_qst_mlp(n_qubits)
    train_model(model, x_train, y_train, epochs=epochs, verbose=0)

    # Dense hidden layers sit at odd indices in the functional model:
    # [Input, Dense, Dropout, Dense, Dropout, Dense, Dense(output)]
    hidden_layer_indices = [1, 3, 5]

    plotting.apply_style()
    import matplotlib.pyplot as plt
    from scipy import stats

    fig, axes = plt.subplots(1, len(hidden_layer_indices), figsize=(5 * len(hidden_layer_indices), 4))
    if len(hidden_layer_indices) == 1:
        axes = [axes]

    summary = {}
    for ax, layer_idx in zip(axes, hidden_layer_indices):
        results, _ = interpretability.run_neuron_projector_analysis(model, n_qubits=n_qubits, layer_index=layer_idx)
        values = np.array([r["abs_similarity"] for r in results])
        median = float(np.median(values))
        summary[f"layer_{layer_idx}"] = {
            "n_neurons": len(values),
            "median_abs_cosine_similarity": median,
            "mean_abs_cosine_similarity": float(values.mean()),
        }
        print(f"Layer {layer_idx}: {len(values)} neurons, median |cos sim| = {median:.3f}")

        ax.hist(values, bins=20, density=True, alpha=0.5, color="paleturquoise", edgecolor="black")
        if len(values) > 1 and values.std() > 1e-9:
            kde = stats.gaussian_kde(values)
            xs = np.linspace(values.min() - 0.05, values.max() + 0.05, 300)
            ax.plot(xs, kde(xs), color="darkcyan", linewidth=2)
        ax.axvline(median, color="black", linestyle="--", label=f"Median: {median:.2f}")
        ax.set_xlabel("|Best cosine similarity|")
        ax.set_ylabel("Neuron density")
        ax.set_title(f"Hidden layer (index {layer_idx})")
        ax.legend()
        plotting.strip_spines(ax)

    fig_path = args.output_dir / "figures" / "interpretability_cosine_similarity.png"
    plotting.savefig(fig, fig_path)
    print(f"Saved figure to {fig_path}")

    import json

    table_path = args.output_dir / "tables" / "interpretability.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {table_path}")


if __name__ == "__main__":
    main()
