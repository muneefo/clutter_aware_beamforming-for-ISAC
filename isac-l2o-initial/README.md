# Initial near-field RIS--ISAC L2O experiment

This is a runnable **nominal, ideal-information, synthetic prototype**, with analytical derivatives and actual held-out simulation results. It implements learned initialization, learned update steps, and their combination. It does not solve the robust PhD proposal or establish deployment latency.

## Read first
- `derivation.pdf` / `derivation.tex`: full signal model, optimization, derivatives, receive-filter solution, fixed-direction power repair, actual results, and limitations.
- `CODEX_PROMPT.md`: staged prompt for extending this verified baseline.
- `results/simulation-summary.pdf`: vector plot generated from raw saved outputs.
- `results/summary.csv`: per-method/per-seed results including failed cases.
- `results/metadata.json`: configuration, seeds, environment, and training budget.

## Reproduce
Requires Python 3.12, NumPy, SciPy, and Matplotlib. Tested versions are in `requirements.txt`.

```sh
OPENBLAS_NUM_THREADS=1 python3 validate.py
OPENBLAS_NUM_THREADS=1 python3 experiment.py
OPENBLAS_NUM_THREADS=1 python3 extra_checks.py
OPENBLAS_NUM_THREADS=1 python3 report.py
```

These commands overwrite `results/`; preserve the distributed results before rerunning. Numerical randomness is seeded; timings depend on the host and workload.

For LaTeX, compile `derivation.tex` with pdfLaTeX, BibTeX, pdfLaTeX, pdfLaTeX, or use `latexmk -pdf derivation.tex`. All required results figures and the generated results-section.tex are included. No external paper images are embedded.

## Methods
- `model.py`: complex128 physical model, analytical gradients, exact receive-filter subproblem, fixed-direction QoS power allocation, and optimization layers.
- `validate.py`: central-difference gradients including the receiver envelope; independent generalized-eigenvalue check; accepted-state constraints and monotone incumbent check.
- `experiment.py`: 27-setting validation-tuned conventional baseline; CEM training on 64 geometries, validation on 32, testing on 128 plus 128 clutter-shift cases; three learning seeds; batch-one timing.
- `extra_checks.py`: independent Monte Carlo covariance check, limited dimension-transfer timing, inclusive component profiling.
- `report.py`: figures, bootstrap paired comparisons, and measured LaTeX result text.

Learning uses a small softmax dictionary initializer (9 parameters), per-layer steps (16 parameters), or both (25 parameters). Offline training uses CEM, not backpropagation; no PyTorch neural network is claimed. All online derivatives are analytic. Fixed-direction power repair and nominal verification are part of every method.

Failures of the initial zero-forcing construction are reported and remain failures; they do not certify that the original joint optimization problem is infeasible. The feasibility percentage is therefore affected by this limited initialization policy. The 80-step reference is a heuristic, not a global optimizer.

Clutter-shift geometries use a separate seed and are not paired copies of nominal geometries. Confidence intervals in the main paired plot resample geometry with trained weights held fixed. Solver timings exclude acquisition, estimation, hardware reconfiguration, and queueing.

The current learned initializer requires K=2. N/M transfer is a small diagnostic with fixed physical aperture; it is not evidence for arbitrary-size networks or realistic antenna coupling.
