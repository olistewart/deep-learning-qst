#!/usr/bin/env python3
"""
Train the MLP quantum-state estimator on a single qubit count and evaluate
its fidelity on a held-out noisy test set (report Sec. 3B3, Fig. 3-4).

Usage
-----
    python experiments/train_model.py --n-qubits 2
    python experiments/train_model.py --quick          # fast smoke test

Saves a loss-curve figure to results/figures/, a fidelity summary to
results/tables/, and the trained model to results/models/.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=2)
    parser.add_argument("--quick", action="store_true", help="tiny/fast configuration for a smoke test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    from src import config, metrics, plotting, states
    from src.model import build_qst_mlp, make_noisy_stokes_inputs, train_model
    from src.stokes import make_physical

    n_qubits = args.n_qubits
    n_train = config.QUICK.n_train if args.quick else config.N_TRAIN
    n_test = config.QUICK.n_test if args.quick else config.N_TEST
    epochs = config.QUICK.epochs if args.quick else config.EPOCHS
    n_shots = config.IDEAL_N_SHOTS

    print(f"[train_model] n_qubits={n_qubits} n_train={n_train} n_test={n_test} "
          f"epochs={epochs} n_shots={n_shots} quick={args.quick}")

    print("Generating datasets...")
    _, y_train = states.make_dataset(n_train, n_qubits, seed=config.TRAIN_SEED, mix=config.STATE_MIX)
    _, y_test = states.make_dataset(n_test, n_qubits, seed=config.TEST_SEED, mix=config.STATE_MIX)

    rng = np.random.default_rng(args.seed)
    print("Simulating noisy measurements...")
    x_train = make_noisy_stokes_inputs(y_train, n_qubits, n_shots=n_shots, rng=rng)
    x_test = make_noisy_stokes_inputs(y_test, n_qubits, n_shots=n_shots, rng=rng)

    print("Building and training model...")
    model = build_qst_mlp(n_qubits)
    history = train_model(model, x_train, y_train, epochs=epochs, verbose=2)

    print("Evaluating fidelity on held-out test set...")
    y_pred = model.predict(x_test, verbose=0)

    fidelities = []
    for i in range(n_test):
        rho_true = states.real_vector_to_rho(y_test[i])
        rho_pred = states.real_vector_to_rho(y_pred[i])
        rho_pred = make_physical(rho_pred)
        fidelities.append(metrics.fidelity(rho_true, rho_pred))

    mean_f, std_f = metrics.mean_std(fidelities)
    print(f"Mean fidelity: {mean_f:.4f} +/- {std_f:.4f}")

    # -- figures --------------------------------------------------------
    plotting.apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(history.history["loss"], label="Training")
    ax.plot(history.history["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.legend()
    plotting.strip_spines(ax)
    fig_path = args.output_dir / "figures" / f"train_loss_{n_qubits}qubit.png"
    plotting.savefig(fig, fig_path)
    print(f"Saved loss curve to {fig_path}")

    # -- tables / model ---------------------------------------------------
    table_path = args.output_dir / "tables" / f"train_model_{n_qubits}qubit.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w") as f:
        json.dump(
            {
                "n_qubits": n_qubits,
                "n_train": n_train,
                "n_test": n_test,
                "n_shots": n_shots,
                "epochs_trained": len(history.history["loss"]),
                "mean_fidelity": mean_f,
                "std_fidelity": std_f,
            },
            f,
            indent=2,
        )
    print(f"Saved fidelity summary to {table_path}")

    model_path = args.output_dir / "models" / f"mlp_{n_qubits}qubit.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()
