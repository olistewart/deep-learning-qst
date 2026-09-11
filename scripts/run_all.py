#!/usr/bin/env python3
"""
Top-level pipeline runner.

Two modes:

  --quick (default)
      A self-contained, sub-minute demonstration of the full pipeline —
      generate states, apply noise, reconstruct with Stokes / MLE / a
      freshly (briefly) trained NN, compute fidelity, save one figure and
      one table. This is the "quick demonstration" tier of the README's
      reproduction hierarchy; it does NOT reproduce the report's numbers,
      only the mechanics of the pipeline.

  --full
      Runs the four experiment scripts in `experiments/` back to back with
      their full-scale (paper) configuration. This reproduces the report's
      headline results but is computationally expensive -- see the README
      for realistic runtimes, and consider running the scripts individually
      instead so a failure in one does not lose earlier progress.

Usage
-----
    python scripts/run_all.py               # quick demo (~30-60s on CPU)
    python scripts/run_all.py --full         # full reproduction (slow)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def run_quick_demo(output_dir: Path) -> None:
    import numpy as np

    from src import config, metrics, states
    from src.mle import reconstruct_mle_from_counts
    from src.model import build_qst_mlp, make_noisy_stokes_inputs, train_model
    from src.noise import simulate_counts
    from src.measurements import stokes_from_counts
    from src.stokes import make_physical, reconstruct_from_stokes

    n_qubits = 2
    n_shots = config.IDEAL_N_SHOTS
    q = config.QUICK

    print(f"1/6  Generating {q.n_train} training states and {q.n_test} test states (n={n_qubits})...")
    _, y_train = states.make_dataset(q.n_train, n_qubits, seed=config.TRAIN_SEED, mix=config.STATE_MIX)
    _, y_test = states.make_dataset(q.n_test, n_qubits, seed=config.TEST_SEED, mix=config.STATE_MIX)
    rho_test = [states.real_vector_to_rho(v) for v in y_test]

    print(f"2/6  Simulating noisy measurements ({n_shots} shots/state)...")
    rng = np.random.default_rng(0)
    x_train = make_noisy_stokes_inputs(y_train, n_qubits, n_shots=n_shots, rng=rng)
    counts_test = [simulate_counts(rho, n_qubits, n_shots=n_shots, rng=rng) for rho in rho_test]
    x_test = np.array([stokes_from_counts(c, n_qubits) for c in counts_test])

    print(f"3/6  Training NN estimator ({q.epochs} epochs)...")
    model = build_qst_mlp(n_qubits)
    train_model(model, x_train, y_train, epochs=q.epochs, verbose=0)
    y_pred_nn = model.predict(x_test, verbose=0)

    print("4/6  Reconstructing with Stokes, MLE, and the NN...")
    fidelities = {"stokes": [], "mle": [], "nn": []}
    for i, rho_true in enumerate(rho_test):
        counts = counts_test[i]
        s_hat = stokes_from_counts(counts, n_qubits)

        rho_stokes = reconstruct_from_stokes(s_hat, n_qubits, physical=True)
        fidelities["stokes"].append(metrics.fidelity(rho_true, rho_stokes))

        rho_mle = reconstruct_mle_from_counts(counts, n_qubits, max_iter=q.mle_max_iter, tol=config.MLE_TOL)
        fidelities["mle"].append(metrics.fidelity(rho_true, rho_mle))

        rho_nn = make_physical(states.real_vector_to_rho(y_pred_nn[i]))
        fidelities["nn"].append(metrics.fidelity(rho_true, rho_nn))

    print("5/6  Summary (small-sample demo -- not representative of report numbers):")
    summary = {}
    for method, vals in fidelities.items():
        mean_f, std_f = metrics.mean_std(vals)
        summary[method] = {"mean_fidelity": mean_f, "std_fidelity": std_f}
        print(f"     {method:6s}: F = {mean_f:.4f} +/- {std_f:.4f}")

    print("6/6  Saving results...")
    import json

    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "tables" / "quick_demo.json", "w") as f:
        json.dump({"n_qubits": n_qubits, "n_shots": n_shots, **summary}, f, indent=2)

    from src import plotting

    plotting.apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    methods = list(summary)
    means = [summary[m]["mean_fidelity"] for m in methods]
    stds = [summary[m]["std_fidelity"] for m in methods]
    ax.bar(methods, means, yerr=stds, capsize=4, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("Mean fidelity")
    ax.set_ylim(0, 1.05)
    plotting.strip_spines(ax)
    plotting.savefig(fig, output_dir / "figures" / "quick_demo.png")

    print(f"\nDone. Results saved under {output_dir}/")


def run_full(output_dir: Path) -> None:
    scripts = [
        "experiments/train_model.py",
        "experiments/ideal_benchmark.py",
        "experiments/noise_sweeps.py",
        "experiments/scalability.py",
        "experiments/interpretability.py",
    ]
    for script in scripts:
        print(f"\n{'#' * 70}\n# Running {script}\n{'#' * 70}")
        t0 = time.time()
        subprocess.run(
            [sys.executable, str(REPO_ROOT / script), "--output-dir", str(output_dir)],
            check=True,
            cwd=REPO_ROOT,
        )
        print(f"# {script} finished in {time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="run the full-scale reproduction (slow)")
    parser.add_argument(
        "--quick", action="store_true", help="run the quick demonstration (default; explicit for symmetry)"
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()

    if args.full:
        print("Running FULL-SCALE reproduction. This can take a long time "
              "(the scalability benchmark alone can take hours on CPU). "
              "See README.md for a breakdown by script.\n")
        run_full(args.output_dir)
    else:
        print("Running QUICK demonstration (~30-60s on CPU). "
              "Pass --full for the full-scale reproduction.\n")
        run_quick_demo(args.output_dir)


if __name__ == "__main__":
    main()
