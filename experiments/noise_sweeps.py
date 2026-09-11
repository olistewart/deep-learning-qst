#!/usr/bin/env python3
"""
Two-qubit independent noise sweeps: additive Gaussian preparation/control
noise, finite-shot projection noise, and "pepper" (missing-measurement)
readout noise, swept one at a time with the other two held at their
noiseless values (report Sec. 4B, Table I, Fig. 7).

For each noise value the NN is retrained on data generated under that same
noise condition (deep learning here is a *matched-noise* estimator, not a
noise-agnostic one -- see report Sec. 4B discussion), then all three
methods are evaluated on a shared noisy test set and averaged over several
independent runs.

This is the most expensive script per sweep point (18 points x num_runs
NN trainings); use --quick for a fast, non-representative smoke test, and
see the README's reproduction-hierarchy section for realistic runtimes.

Usage
-----
    python experiments/noise_sweeps.py --quick
    python experiments/noise_sweeps.py --channel sigma
    python experiments/noise_sweeps.py            # all three channels
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_sweep(channel, values, n_qubits, n_train, n_test, epochs, num_runs, mle_max_iter):
    from src import metrics, states
    from src.mle import reconstruct_mle_from_counts
    from src.model import build_qst_mlp, train_model
    from src.noise import simulate_counts
    from src.measurements import stokes_from_counts
    from src.stokes import make_physical, reconstruct_from_stokes
    from src.config import SWEEP_FIXED_SHOTS, TRAIN_SEED, TEST_SEED, STATE_MIX, MLE_TOL

    _, y_train = states.make_dataset(n_train, n_qubits, seed=TRAIN_SEED, mix=STATE_MIX)
    _, y_test = states.make_dataset(n_test, n_qubits, seed=TEST_SEED, mix=STATE_MIX)
    rho_test = [states.real_vector_to_rho(v) for v in y_test]

    def noise_kwargs(value):
        if channel == "sigma":
            return dict(n_shots=SWEEP_FIXED_SHOTS, gauss_sigma=value, missing_p=0.0)
        if channel == "shots":
            return dict(n_shots=int(value), gauss_sigma=0.0, missing_p=0.0)
        if channel == "pepper":
            return dict(n_shots=SWEEP_FIXED_SHOTS, gauss_sigma=0.0, missing_p=value)
        raise ValueError(channel)

    results = []
    for value in values:
        print(f"\n=== channel={channel} value={value} ===")
        run_summaries = {"stokes": [], "mle": [], "nn": []}

        for run in range(num_runs):
            rng_train = np.random.default_rng(hash((channel, value, run, "train")) % (2**32))
            rng_test = np.random.default_rng(hash((channel, value, run, "test")) % (2**32))

            kwargs = noise_kwargs(value)
            x_train = np.array([
                stokes_from_counts(
                    simulate_counts(states.real_vector_to_rho(y), n_qubits, rng=rng_train, **kwargs),
                    n_qubits,
                )
                for y in y_train
            ])

            model = build_qst_mlp(n_qubits)
            train_model(model, x_train, y_train, epochs=epochs, verbose=0)

            counts_test = [
                simulate_counts(rho, n_qubits, rng=rng_test, **kwargs) for rho in rho_test
            ]
            x_test = np.array([stokes_from_counts(c, n_qubits) for c in counts_test])
            y_pred_nn = model.predict(x_test, verbose=0)

            f_stokes, f_mle, f_nn = [], [], []
            for i, rho_true in enumerate(rho_test):
                counts = counts_test[i]
                s_hat = stokes_from_counts(counts, n_qubits)

                rho_stokes = reconstruct_from_stokes(s_hat, n_qubits, physical=True)
                f_stokes.append(metrics.fidelity(rho_true, rho_stokes))

                rho_mle = reconstruct_mle_from_counts(counts, n_qubits, max_iter=mle_max_iter, tol=MLE_TOL)
                f_mle.append(metrics.fidelity(rho_true, rho_mle))

                rho_nn = make_physical(states.real_vector_to_rho(y_pred_nn[i]))
                f_nn.append(metrics.fidelity(rho_true, rho_nn))

            run_summaries["stokes"].append(float(np.mean(f_stokes)))
            run_summaries["mle"].append(float(np.mean(f_mle)))
            run_summaries["nn"].append(float(np.mean(f_nn)))

            print(f"  run {run + 1}/{num_runs}: "
                  f"Stokes={run_summaries['stokes'][-1]:.4f} "
                  f"MLE={run_summaries['mle'][-1]:.4f} "
                  f"NN={run_summaries['nn'][-1]:.4f}")

        point = {"value": float(value)}
        for method in ("stokes", "mle", "nn"):
            mean_f, std_f = metrics.mean_std(run_summaries[method])
            point[f"{method}_mean_fidelity"] = mean_f
            point[f"{method}_std_fidelity"] = std_f
        results.append(point)

    return results


def main() -> None:
    from src import config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", choices=["sigma", "shots", "pepper", "all"], default="all")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    n_qubits = config.SWEEP_N_QUBITS
    n_train = config.QUICK.n_train if args.quick else config.N_TRAIN
    n_test = config.QUICK.n_test if args.quick else config.N_TEST
    epochs = config.QUICK.epochs if args.quick else config.EPOCHS
    num_runs = config.QUICK.num_runs if args.quick else config.SWEEP_NUM_RUNS
    mle_max_iter = config.QUICK.mle_max_iter if args.quick else config.MLE_MAX_ITER

    sweeps = {
        "sigma": config.QUICK.sigma_sweep if args.quick else config.SIGMA_SWEEP,
        "shots": config.QUICK.shots_sweep if args.quick else config.SHOTS_SWEEP,
        "pepper": config.QUICK.pepper_sweep if args.quick else config.PEPPER_SWEEP,
    }
    channels = list(sweeps) if args.channel == "all" else [args.channel]

    all_results = {}
    for channel in channels:
        all_results[channel] = run_sweep(
            channel, sweeps[channel], n_qubits, n_train, n_test, epochs, num_runs, mle_max_iter
        )

        table_path = args.output_dir / "tables" / f"noise_sweep_{channel}.json"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with open(table_path, "w") as f:
            json.dump(all_results[channel], f, indent=2)
        print(f"Saved {channel} sweep results to {table_path}")

    # -- figure: one panel per swept channel (report Fig. 7) -------------
    from src import plotting

    plotting.apply_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(channels), figsize=(5 * len(channels), 4), squeeze=False)
    axes = axes[0]
    labels = {"sigma": "Additive Gaussian std (sigma)", "shots": "No. of shots (N)", "pepper": "Pepper fraction (f)"}
    colors = {"stokes": "#4C72B0", "mle": "#55A868", "nn": "#C44E52"}

    for ax, channel in zip(axes, channels):
        data = all_results[channel]
        x = [p["value"] for p in data]
        for method, color in colors.items():
            means = [p[f"{method}_mean_fidelity"] for p in data]
            stds = [p[f"{method}_std_fidelity"] for p in data]
            ax.errorbar(x, means, yerr=stds, label=method.upper(), color=color, marker="o", capsize=3)
        if channel == "shots":
            ax.set_xscale("log")
        ax.set_xlabel(labels[channel])
        ax.set_ylabel("Mean fidelity")
        ax.legend()
        plotting.strip_spines(ax)

    fig_path = args.output_dir / "figures" / "noise_sweeps.png"
    plotting.savefig(fig, fig_path)
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
