#!/usr/bin/env python3
"""
Two-qubit "ideal benchmark": compare Stokes reconstruction, MLE, and the
NN estimator under finite-shot projection noise only (report Sec. 4A,
Fig. 6). This establishes the feasibility of deep-learning-based QST in
the simplest noise setting before the fuller sweeps in `noise_sweeps.py`.

Usage
-----
    python experiments/ideal_benchmark.py
    python experiments/ideal_benchmark.py --quick
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    from src import config, metrics, plotting, states
    from src.mle import reconstruct_mle_from_counts
    from src.model import build_qst_mlp, make_noisy_stokes_inputs, train_model
    from src.noise import simulate_counts
    from src.measurements import stokes_from_counts
    from src.stokes import make_physical, reconstruct_from_stokes

    n_qubits = config.IDEAL_N_QUBITS
    n_shots = config.IDEAL_N_SHOTS
    n_train = config.QUICK.n_train if args.quick else config.N_TRAIN
    n_test = config.QUICK.n_test if args.quick else config.IDEAL_N_TEST
    epochs = config.QUICK.epochs if args.quick else config.EPOCHS
    mle_max_iter = config.QUICK.mle_max_iter if args.quick else config.MLE_MAX_ITER

    print(f"[ideal_benchmark] n_qubits={n_qubits} n_shots={n_shots} n_train={n_train} "
          f"n_test={n_test} quick={args.quick}")

    _, y_train = states.make_dataset(n_train, n_qubits, seed=config.TRAIN_SEED, mix=config.STATE_MIX)
    _, y_test = states.make_dataset(n_test, n_qubits, seed=config.TEST_SEED, mix=config.STATE_MIX)
    rho_test = [states.real_vector_to_rho(v) for v in y_test]

    rng_train = np.random.default_rng(1)
    rng_test = np.random.default_rng(2)

    print("Training NN estimator...")
    x_train = make_noisy_stokes_inputs(y_train, n_qubits, n_shots=n_shots, rng=rng_train)
    model = build_qst_mlp(n_qubits)
    train_model(model, x_train, y_train, epochs=epochs, verbose=0)

    print("Generating noisy test measurements (shared across methods)...")
    counts_test = [simulate_counts(rho, n_qubits, n_shots=n_shots, rng=rng_test) for rho in rho_test]
    x_test_noisy = np.array([stokes_from_counts(c, n_qubits) for c in counts_test])

    print("Running NN inference...")
    t0 = time.perf_counter()
    y_pred_nn = model.predict(x_test_noisy, verbose=0)
    nn_time = time.perf_counter() - t0

    fidelities = {"stokes": [], "mle": [], "nn": []}
    times = {"stokes": [], "mle": [], "nn": [nn_time]}

    print("Reconstructing with Stokes and MLE, evaluating fidelity...")
    for i, rho_true in enumerate(rho_test):
        counts = counts_test[i]

        t0 = time.perf_counter()
        s_hat = stokes_from_counts(counts, n_qubits)
        rho_stokes = reconstruct_from_stokes(s_hat, n_qubits, physical=True)
        times["stokes"].append(time.perf_counter() - t0)
        fidelities["stokes"].append(metrics.fidelity(rho_true, rho_stokes))

        t0 = time.perf_counter()
        rho_mle = reconstruct_mle_from_counts(counts, n_qubits, max_iter=mle_max_iter, tol=config.MLE_TOL)
        times["mle"].append(time.perf_counter() - t0)
        fidelities["mle"].append(metrics.fidelity(rho_true, rho_mle))

        rho_nn = make_physical(states.real_vector_to_rho(y_pred_nn[i]))
        fidelities["nn"].append(metrics.fidelity(rho_true, rho_nn))

    summary = {}
    for method in ("stokes", "mle", "nn"):
        mean_f, std_f = metrics.mean_std(fidelities[method])
        total_t = float(np.sum(times[method])) if method != "nn" else nn_time
        summary[method] = {"mean_fidelity": mean_f, "std_fidelity": std_f, "total_time_s": total_t}
        print(f"  {method:6s}: F = {mean_f:.4f} +/- {std_f:.4f}   total time = {total_t:.3f} s")

    # -- figure: KDE-style fidelity distribution comparison (Fig. 6) -----
    plotting.apply_style()
    import matplotlib.pyplot as plt
    from scipy import stats

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"stokes": "#4C72B0", "mle": "#55A868", "nn": "#C44E52"}
    for method, color in colors.items():
        vals = np.array(fidelities[method])
        if len(vals) > 1 and np.std(vals) > 1e-9:
            kde = stats.gaussian_kde(vals)
            xs = np.linspace(max(0, vals.min() - 0.02), min(1, vals.max() + 0.02), 400)
            ax.plot(xs, kde(xs), label=method.upper(), color=color)
            ax.fill_between(xs, kde(xs), alpha=0.15, color=color)
        ax.axvline(vals.mean(), color=color, linestyle=":", linewidth=1)
    ax.set_xlabel("Fidelity F")
    ax.set_ylabel("Probability density (KDE)")
    ax.legend()
    plotting.strip_spines(ax)
    fig_path = args.output_dir / "figures" / "ideal_benchmark_fidelity_kde.png"
    plotting.savefig(fig, fig_path)
    print(f"Saved figure to {fig_path}")

    table_path = args.output_dir / "tables" / "ideal_benchmark.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w") as f:
        json.dump({"n_qubits": n_qubits, "n_shots": n_shots, "n_test": n_test, **summary}, f, indent=2)
    print(f"Saved summary to {table_path}")


if __name__ == "__main__":
    main()
