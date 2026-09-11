#!/usr/bin/env python3
"""
Scalability benchmark: compare Stokes, MLE, and the NN estimator across
n in {2, 3, 4} qubits under a single trapped-ion-inspired combined noise
condition (report Sec. 3C3, Sec. 4C, Fig. 8, Table II/Appendix D).

This is the headline result of the project: at n=4 the NN reconstructs
the full 1000-state test set roughly 5x faster than Stokes and ~8000x
faster than MLE, at higher fidelity than both. It is also the most
expensive script to run at full scale -- MLE alone takes on the order of
half an hour per run at n=4 (max_iter=150, 1000 states). Use --quick for
a fast, non-representative smoke test.

Usage
-----
    python experiments/scalability.py --quick
    python experiments/scalability.py --n-qubits 2 3 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    from src import config, metrics, plotting, states
    from src.mle import reconstruct_mle_from_counts
    from src.model import build_qst_mlp, train_model
    from src.noise import simulate_counts_trapped_ion
    from src.measurements import stokes_from_counts
    from src.stokes import make_physical, reconstruct_from_stokes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, nargs="+", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    n_qubits_list = args.n_qubits or (
        config.QUICK.scalability_n_qubits if args.quick else config.SCALABILITY_N_QUBITS
    )
    n_train = config.QUICK.n_train if args.quick else config.SCALABILITY_N_TRAIN
    n_test = config.QUICK.n_test if args.quick else config.SCALABILITY_N_TEST
    epochs = config.QUICK.epochs if args.quick else config.EPOCHS
    num_runs = config.QUICK.num_runs if args.quick else config.SCALABILITY_NUM_RUNS
    mle_max_iter = config.QUICK.mle_max_iter if args.quick else config.MLE_MAX_ITER

    noise_kwargs = dict(
        n_shots=config.ION_N_SHOTS,
        p_phi=config.ION_P_PHI,
        sigma_theta=config.ION_SIGMA_THETA,
        axis=config.ION_AXIS,
        p_readout=config.ION_P_READOUT,
    )
    print(f"[scalability] n_qubits={n_qubits_list} n_train={n_train} n_test={n_test} "
          f"num_runs={num_runs} noise={noise_kwargs}")

    all_results = []
    for n in n_qubits_list:
        print(f"\n{'=' * 60}\n n = {n} qubits\n{'=' * 60}")
        _, y_train = states.make_dataset(n_train, n, seed=config.TRAIN_SEED, mix=config.STATE_MIX)
        _, y_test = states.make_dataset(n_test, n, seed=config.TEST_SEED, mix=config.STATE_MIX)
        rho_test = [states.real_vector_to_rho(v) for v in y_test]

        stokes_f, mle_f, nn_f = [], [], []
        stokes_t, mle_t, nn_t = [], [], []

        for run in range(num_runs):
            rng_train = np.random.default_rng(run * 3)
            rng_test = np.random.default_rng(run * 3 + 1)

            x_train = np.array([
                stokes_from_counts(
                    simulate_counts_trapped_ion(
                        states.real_vector_to_rho(y), n, rng=rng_train, **noise_kwargs
                    )[0],
                    n,
                )
                for y in y_train
            ])

            model = build_qst_mlp(n)
            train_model(model, x_train, y_train, epochs=epochs, verbose=0)

            counts_test = [
                simulate_counts_trapped_ion(rho, n, rng=rng_test, **noise_kwargs)[0] for rho in rho_test
            ]
            x_test = np.array([stokes_from_counts(c, n) for c in counts_test])

            t0 = time.perf_counter()
            y_pred_nn = model.predict(x_test, verbose=0)
            nn_total_time = time.perf_counter() - t0

            run_stokes_f, run_mle_f, run_nn_f = [], [], []
            run_stokes_t, run_mle_t = [], []

            for i, rho_true in enumerate(rho_test):
                counts = counts_test[i]

                t0 = time.perf_counter()
                s_hat = stokes_from_counts(counts, n)
                rho_stokes = reconstruct_from_stokes(s_hat, n, physical=True)
                run_stokes_t.append(time.perf_counter() - t0)
                run_stokes_f.append(metrics.fidelity(rho_true, rho_stokes))

                t0 = time.perf_counter()
                rho_mle = reconstruct_mle_from_counts(counts, n, max_iter=mle_max_iter, tol=config.MLE_TOL)
                run_mle_t.append(time.perf_counter() - t0)
                run_mle_f.append(metrics.fidelity(rho_true, rho_mle))

                rho_nn = make_physical(states.real_vector_to_rho(y_pred_nn[i]))
                run_nn_f.append(metrics.fidelity(rho_true, rho_nn))

            stokes_f.append(float(np.mean(run_stokes_f)))
            mle_f.append(float(np.mean(run_mle_f)))
            nn_f.append(float(np.mean(run_nn_f)))
            stokes_t.append(float(np.sum(run_stokes_t)))
            mle_t.append(float(np.sum(run_mle_t)))
            nn_t.append(float(nn_total_time))

            print(f"  run {run + 1}/{num_runs}: "
                  f"Stokes F={stokes_f[-1]:.4f} T={stokes_t[-1]:.2f}s | "
                  f"MLE F={mle_f[-1]:.4f} T={mle_t[-1]:.2f}s | "
                  f"NN F={nn_f[-1]:.4f} T={nn_t[-1]:.2f}s")

        result = {"n_qubits": n, "num_runs": num_runs, "n_test": n_test, "noise": noise_kwargs}
        for method, fids, times_ in (
            ("stokes", stokes_f, stokes_t),
            ("mle", mle_f, mle_t),
            ("nn", nn_f, nn_t),
        ):
            mean_f, std_f = metrics.mean_std(fids)
            mean_t, std_t = metrics.mean_std(times_)
            result[method] = {
                "mean_fidelity": mean_f,
                "std_fidelity": std_f,
                "mean_time": mean_t,
                "std_time": std_t,
            }
        all_results.append(result)

        table_path = args.output_dir / "tables" / f"scalability_{n}qubit.json"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with open(table_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved {n}-qubit result to {table_path}")

    # -- figure: fidelity and inference-time scaling (report Fig. 8) -----
    plotting.apply_style()
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"stokes": "#4C72B0", "mle": "#55A868", "nn": "#C44E52"}
    ns = [r["n_qubits"] for r in all_results]

    for method, color in colors.items():
        means = [r[method]["mean_fidelity"] for r in all_results]
        stds = [r[method]["std_fidelity"] for r in all_results]
        ax1.errorbar(ns, means, yerr=stds, label=method.upper(), color=color, marker="o", capsize=3)

        t_means = [r[method]["mean_time"] for r in all_results]
        t_stds = [r[method]["std_time"] for r in all_results]
        ax2.errorbar(ns, t_means, yerr=t_stds, label=method.upper(), color=color, marker="o", capsize=3)

    ax1.set_xlabel("Number of qubits (n)")
    ax1.set_ylabel("Mean fidelity F")
    ax1.legend()
    plotting.strip_spines(ax1)

    ax2.set_xlabel("Number of qubits (n)")
    ax2.set_ylabel("Mean test-set inference time (s)")
    ax2.set_yscale("log")
    ax2.legend()
    plotting.strip_spines(ax2)

    fig_path = args.output_dir / "figures" / "scalability.png"
    plotting.savefig(fig, fig_path)
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
