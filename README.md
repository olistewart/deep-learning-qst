# Deep Learning-Enhanced Quantum State Estimation

A supervised deep-learning framework for reconstructing multi-qubit quantum states from noisy tomographic measurements, benchmarked against the two standard classical approaches: linear Stokes reconstruction and maximum-likelihood estimation (MLE).

This repository is the cleaned-up, reproducible codebase behind my MPhys research project at Durham University (supervised by Dr. Will Yeadon). The full write-up, with derivations and extended results, is in [`report/Deep_Learning_QSE_report.pdf`](report/Deep_Learning_QSE_report.pdf); this README summarises the method and shows how to reproduce the experiments.

## Motivation

Quantum state tomography (QST) reconstructs a system's density matrix from repeated measurements. Both standard approaches degrade under realistic experimental conditions: Stokes reconstruction is a direct linear inversion, so measurement noise propagates straight into the estimate, while MLE's iterative likelihood optimisation becomes prohibitively slow as the number of qubits grows (the number of measurement operators scales as 4^n). This project asks whether a multilayer perceptron (MLP), trained to map noisy measurement data directly to density matrices, can be more accurate *and* more scalable than either.

## Approach

1. Generate physically valid quantum states from a randomised trapped-ion-inspired circuit (with Haar-random, Ginibre-random, and named Bell/GHZ/W states also implemented for extensibility).
2. Simulate tomographic measurements in the complete n-qubit Pauli basis, and apply configurable synthetic noise: additive Gaussian preparation/control noise, finite-shot projection noise, and readout noise (as an abstract "pepper" model, and as a trapped-ion bit-flip model).
3. Train a supervised MLP (256-256-128 hidden units, dropout-regularised, Bayesian-optimised learning rate) to map noisy measurements directly to density matrices.
4. Benchmark the network against Stokes reconstruction and MLE (the iterative RhoR algorithm) on reconstruction fidelity and inference time.
5. Test robustness to each noise channel independently, then jointly under a combined noise condition derived from the reported error budget of a real ⁴⁰Ca⁺ trapped-ion processor.
6. Evaluate scaling from two to four qubits under that combined noise condition.
7. Probe interpretability by comparing hidden-layer neuron weight vectors against the physical Pauli-projector measurement basis.

## Key results

Averaged over five runs of a 1000-state test set at four qubits, under the trapped-ion-inspired combined noise condition (report Sec. 4C):

| Method | Mean fidelity | Inference time |
|---|---|---|
| **NN**    | **0.950** | **0.26 s** |
| Stokes    | 0.792     | 1.33 s |
| MLE       | 0.712     | ~2287 s (38 min) |

The network is roughly 20% more accurate than Stokes reconstruction and 33% more accurate than MLE, while reconstructing the full test set about 5x faster than Stokes and 8000x faster than MLE. Its advantage over both classical methods also holds across each noise channel tested independently (additive Gaussian, finite-shot, and readout noise; report Sec. 4B), and across two, three, and four qubits (report Sec. 4C). The full numerical results, including the two-qubit noise sweeps and the interpretability analysis, are in the report.

The project is a methodological simulation study rather than a hardware demonstration: the noise models are physically motivated but phenomenological, and the framework is designed so that alternative state families and experimental noise models can be substituted in for future application to real devices.

## Repository structure

```
deep-learning-qse/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── report/                  Full project report (PDF)
├── src/                     Core library
│   ├── config.py            All numerical constants used by the experiments
│   ├── states.py            State generation, Pauli operators, vectorisation
│   ├── measurements.py      Pauli POVM, ideal/estimated expectation values
│   ├── noise.py             Synthetic SPAM noise channels
│   ├── stokes.py            Linear-inversion reconstruction + physical projection
│   ├── mle.py                Maximum-likelihood (RhoR) reconstruction
│   ├── model.py              MLP architecture and training utilities
│   ├── metrics.py            Fidelity, timing, comparison helpers
│   ├── interpretability.py   Neuron/projector cosine-similarity analysis
│   └── plotting.py           Shared matplotlib styling
├── experiments/              One reproducible script per experiment
│   ├── train_model.py
│   ├── ideal_benchmark.py
│   ├── noise_sweeps.py
│   ├── scalability.py
│   └── interpretability.py
├── results/
│   ├── figures/               Generated plots (git-ignored except .gitkeep)
│   └── tables/                Generated JSON result tables (git-ignored except .gitkeep)
├── notebooks/
│   └── analysis.ipynb         Optional: load and plot results/tables/*.json
└── scripts/
    └── run_all.py             Quick end-to-end demo, or --full for everything
```

Each `experiments/*.py` script follows the same flow: generate states → simulate ideal measurements → apply noise → reconstruct with Stokes / MLE / the NN → compute fidelity and timing → save a JSON table to `results/tables/` and a figure to `results/figures/`.

## Installation

```bash
git clone <repository-url>
cd deep-learning-qse

python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

Everything runs on CPU; TensorFlow will use a GPU automatically if one is available and configured, but none of the experiments require it.

## Configuration

All the numbers that matter -- dataset sizes, noise-sweep ranges (report Table I), the trapped-ion operating point (report Table II), model hyperparameters, and MLE settings -- live in one place, [`src/config.py`](src/config.py), rather than scattered as magic numbers through the experiment scripts. If you want to know exactly what parameters produced a given result, or to change one, that's the file to look at. Every experiment script also accepts a `--quick` flag, which swaps in the small, fast values from `config.QUICK` for a smoke test.

## Reproducing the experiments

Reconstruction with real training data is CPU-intensive -- mainly because MLE is iterative and its cost is non-negligible even for one state, let alone a 1000-state test set at four qubits (see the table above: ~38 minutes for MLE alone at n=4). Reproduce at whichever level suits your purpose:

**1. Quick demonstration** (~30-60 seconds on a laptop CPU). Runs the full pipeline end to end -- state generation → noise → a briefly-trained NN → reconstruction → fidelity -- on a tiny dataset. This checks that everything is wired up correctly; the numbers it prints are not meaningful.

```bash
python scripts/run_all.py --quick
```

**2. Main experiments** (minutes to a couple of hours each, depending on your machine). Reproduces the report's main figures at full scale. Run whichever you're interested in independently -- each is self-contained and safe to interrupt without affecting the others:

```bash
python experiments/train_model.py           # trains and evaluates the 2-qubit model, Fig. 3/4
python experiments/ideal_benchmark.py        # Stokes vs MLE vs NN, finite-shot noise only, Fig. 6
python experiments/noise_sweeps.py           # three independent noise sweeps, Fig. 7
python experiments/interpretability.py       # neuron/projector cosine-similarity analysis, Fig. 5
```

**3. Full-scale reproduction, including scalability** (can take several hours on CPU, dominated by MLE at n=4). This reproduces the paper's headline scalability result but should not be assumed to finish quickly:

```bash
python experiments/scalability.py
# or, to run every experiment back to back:
python scripts/run_all.py --full
```

Every script also takes `--quick` individually (e.g. `python experiments/scalability.py --quick`) if you want to sanity-check just that one experiment fast. Add `--n-qubits 2 3` (etc.) to `scalability.py` to restrict which system sizes it runs.

Note that training is stochastic (random weight initialisation, random noise realisations, and dropout), so re-running any script will not reproduce the report's numbers bit-for-bit -- but the trends and approximate magnitudes should match closely across repeated runs, consistent with the five-run Monte Carlo averaging used throughout the report.

### Results

Each script writes a JSON table to `results/tables/` and a figure to `results/figures/`. `notebooks/analysis.ipynb` is an optional notebook that just loads and re-plots those JSON tables -- handy for comparing runs or restyling a figure without re-running an experiment.

## Extending the framework

The state generator (`src/states.py`), noise channels (`src/noise.py`), and reconstruction methods (`src/stokes.py`, `src/mle.py`, `src/model.py`) are all independent of each other, so it's straightforward to substitute in a different state family, a device-specific noise model, or a different network architecture without touching the rest of the pipeline. `src/config.py` is the single place to adjust the noise operating point, sweep ranges, or training hyperparameters.

## Citation

If you build on this work, please cite:

> O. Stewart, "Deep Learning-Enhanced Quantum State Estimation of Simulated Multi-Qubit Systems," MPhys Project, Department of Physics, Durham University, 2026.

## License

MIT -- see [LICENSE](LICENSE).
