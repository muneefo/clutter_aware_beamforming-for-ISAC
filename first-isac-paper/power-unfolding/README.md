# Unfolding the power allocation in clutter-aware ISAC

This is an independent Python experiment on the fixed transmit proposal. It uses the 0.6 base step of the fixed proposal, the analytic MVDR receiver, 32 tones, 16 transmit and 16 receive antennas, a 100 W budget and the tutorial's estimated clutter kernel. Every final design is scored again on the supplied true kernel. All methods share each draw, candidate bank, receiver, and starting allocation within the isolated block test. The earlier MATLAB fixedProposal and learnedProposal results are separate.

## Subproblem and algorithm

For a fixed receive filter and user-tagged candidate beam directions, let p be the vector of nonnegative power fractions across all tones. Columns have unit norm. The fixed-bank target and clutter coefficients are a and c. The positive scalar multiplying the following objective is removed for conditioning and restored for scoring. The problem is

    maximize    (aᵀp)/(1+cᵀp)
    subject to  Bp >= d,  p >= 0,  sum(p) <= 1.

B is block diagonal by tone. For a given user, a column tagged for that user contributes desired gain divided by the target SINR; all other columns contribute negative interference gain. d contains receiver noise after normalization. Safety margins for power and target SINR match the original allocation routine. The reference solves the equivalent Charnes–Cooper LP with SciPy HiGHS; the included checks compare the optimum to the pre-existing master implementation. On all held-out cases the approximate bank objectives are checked against the exact LP and a dual upper bound.

Each of eight layers takes a primal simplex-projected step and an extrapolated nonnegative dual step:

    p_next = project_simplex(p + alpha * (a - eta*c + Bᵀy))
    y_next = max(0, y + beta * (d - B*(2*p_next - p)))

eta is the best feasible normalized design ratio found to date. Projection enforces a unit total power fraction. A separately decoded copy is used for scoring: add missing power to each user's zero-forcing beam until its tightened SINR inequality holds. If the total exceeds budget, mix with the minimum-power feasible zero-forcing anchor to bring the sum back to one. Both endpoints of the mix satisfy the QoS inequalities (up to floating-point tolerance). The homogeneous zero-forcing cross-user gains are checked numerically. The algorithm retains its best feasible allocation under the design kernel. Recovered physical beamformers undergo a separate SINR and power audit; raw dual iterates themselves need not be optimal.

The fixed control uses one training-tuned pair (alpha,beta) in every layer. The unfolded version starts from this control and learns four pairs, each shared by two successive layers. The eight learned positive scalars are global across scenes, loads and tones. No additional input network or ground-truth clutter is used at inference. Fitting maximizes the average returned design SCNR in dB using multiplicative coordinate search on positive step sizes. The validation set chooses among the tied initialization and the three untied passes. The fixed control is tuned by grid search and coordinate refinement on the same training scenes. This tests a learned layer schedule against a tuned tied schedule, and does not establish superiority over every possible deterministic schedule.

## Data splits and comparisons

The independent user-channel draws have disjoint seed ranges for training (31001–31008), validation (32001–32004) and testing (33001–33012). Training and validation use K=4,8,12 at 4 and 16 dB. The isolated held-out block sweep covers K=4,8,12 at 0,4,8,12,16,20 dB: 216 scene-condition cases using 12 seeds, not 216 independent environments. Each fixed-bank comparison uses the first allocation of the outer solver, with the same user channels across thresholds within a load. All experiments condition on the single supplied environment kernel pair; generalization to other environments is untested.

For the full solver, only allocation 1 of 6 is replaced; five later LPs remain exact. All arms execute three outer receiver updates and two allocation steps per outer iteration with no early exit. The first step supplies its own dual variables for direction generation; for approximate solvers the discrete-bank maximum gives a price and is used only to suggest directions. No approximate stopping test is used. The remaining LPs supply their exact dual information. The full evaluation uses the first eight held-out seeds, all three loads and an 8 dB requirement (24 scene-condition cases). Its comparator is the all-exact fixed-proposal LP solver at the same budget, not the authors' AO-SDP algorithm. Checking a replacement in later allocations would require training/evaluating on the evolving direction banks.

For the optional isolated-block fallback, any nonnegative y yields the additive upper bound on the normalized ratio gap

    r = max(0, max(0, max_i [a - eta*c + Bᵀy]_i) - dᵀy - eta).

Indeed, any feasible p obeys (a-eta*c)ᵀp - eta <= r, and its denominator is at least one. The certified dB gap is at most 10 log10(1+r/eta). The fallback runs an exact LP and recovers/audits its beams whenever this certificate exceeds 0.1 dB or the physical audit fails. This bound concerns the estimated-kernel finite-bank allocation, not post-recovery true SCNR or full AO optimality. Timing for fallback is in certifiedMS. The default end-to-end hybrid has no such fallback.

## Reproduce the simulation

Tested with Python 3.12, NumPy 2.3.5, SciPy 1.17.0, threadpoolctl, pandas and matplotlib. From this directory:

    python experiment.py /path/to/Clutter-Aware-ISAC-Tutorial
    python report.py

The input file data/covMat_BLP.mat is from https://github.com/RangLiu0706/Clutter-Aware-ISAC-Tutorial and has SHA-256 99744880b1a6fe5e5f8a0ef57c1738be79fb0ad28629e68db2bfe3a8df67472d. Source code and model schedules are included. Original datasets and previous experiment output folders are not required apart from this input file.

## Reading the outputs

- results/model.json and training.json give the fitted parameters, splits and search history.
- results/block_trials.csv gives the exact, tied and unfolded outcomes per held-out case. bankGapDB is the normalized model objective gap in dB against the exact reference on that bank. trueGapDB scores recovered beams and the same fixed receiver with the true kernel. A negative true gap is possible when the design kernel differs from the scoring kernel.
- results/hybrid_trials.csv evaluates the physically returned transmit/receive pairs after six allocations.
- results/block_summary.csv and hybrid_summary.csv contain pooled and per-load comparisons. Individual design-quality hits and true-kernel quality hits require a gap no greater than 0.1 dB. The paired mean-gain intervals resample whole seed clusters with all their operating conditions.
- results/unfolding_results.pdf contains the first-allocation SCNR sweep and the end-to-end SCNR and runtime comparison.

Times are warm CPU wall times, methods interleaved in random order on each draw. The isolated allocator time starts with a prepared fixed-bank subproblem. Returned-block timing also includes beam recovery and the physical audit. Full-solver timing starts from the supplied channel and includes initialization, constructing the bank, MVDR, allocations, direction generation, beam recovery and final audit. It excludes drawing channels and scoring against hidden true clutter. Three timing repeats are used per isolated case and two per hybrid case; summary medians and p95 use per-case median times. Python/HiGHS and MATLAB/CVX times are separate benchmarks.
