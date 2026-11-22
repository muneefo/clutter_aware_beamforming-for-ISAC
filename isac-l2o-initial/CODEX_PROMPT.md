# Continue the validated near-field RIS--ISAC L2O prototype

You are working with a research prototype for Muneef Mohammed's PhD discussion with Prof. Hesham ElSawy. Read `README.md`, `derivation.tex`, `model.py`, `validate.py`, `experiment.py`, `extra_checks.py`, and all saved result metadata before editing. Treat the accompanying numerical results as a small synthetic pilot, not a published result or a globally optimal solution. Do not overwrite the frozen baseline results; create a new run directory and preserve reproducibility.

## Goal
Develop a mathematically consistent, reproducible, constraint-aware L2O solver for clutter-aware near-field RIS-assisted ISAC. Combine learned initialization and learned iterative updates where evidence supports doing so. Priorities are feasible service, scalable computation, and low latency. Do not add movable/fluid antennas, UAV trajectory optimization, NOMA, or additional system features until the present model is validated.

## Existing implementation: reproduce before extending
- One reciprocal BS array; one passive RIS; K=2 single-antenna users; one point target; L=3 independent random point clutter scatterers.
- N=6, M=12, wavelength 0.1 m; exact element-to-point spherical distances. This is a short-range 2D synthetic scenario, not a calibrated UAV network.
- Forward channel convention: v_j=d_j+G^T diag(exp(i theta)) r_j; received user signal is v_j^T W s+n. Do not silently change transpose to Hermitian transpose.
- E[ss^H]=I; Q=WW^H; total transmit power ||W||_F^2<=1.
- Echo y=alpha_0 v_0 v_0^T x + sum_l alpha_l v_l v_l^T x+n. Scattering coefficients are independent zero-mean with variances beta_j. Thus Rc=sum_l beta_l ||v_l^T W||^2 v_l v_l^H + sigma_r^2 I. Clutter covariance changes with W and RIS phases.
- SCNR=(u^H Rt u)/(u^H Rc u), Rt=beta_0 ||v_0^T W||^2 v_0 v_0^H. u is proportional to solve(Rc,v_0), then unit-normalized.
- Communication SINR_k=|v_k^T w_k|^2/(sum_{i!=k}|v_k^T w_i|^2+sigma_c^2); Gamma=2.
- Loss=-log(SCNR)+rho sum_k [log(Gamma)-log(SINR_k)]_+^2. A finite penalty is NOT a constraint guarantee.
- Analytical Wirtinger beamforming and real phase derivatives are implemented. Verify the factor of two between Wirtinger and Cartesian gradients.
- Candidate steps project onto the power ball and undergo exact fixed-direction power allocation. For normalized beam directions t_i, F_ki=Gamma |v_k^T t_i|^2/|v_k^T t_k|^2 for i!=k, diagonal zero; b_k=Gamma sigma_c^2/|v_k^T t_k|^2. Check rho(F)<1; solve p_min=(I-F)^(-1)b. Reject if sum(p_min)>P. Common scaling to P preserves nominal SINR feasibility. This is not a projection onto the full joint feasible set.
- Reject infeasible updates; retain a verified feasible incumbent. An unsuccessful initializer is not a proof of global infeasibility.
- Current L2O learns 16 layer-specific steps, nine initializer parameters, or both. Initialization mixes target/user phase-focus dictionaries with geometry-conditioned softmax weights. It is not a GNN. Offline training uses derivative-free CEM, not backpropagation. The NumPy baseline must remain runnable without PyTorch.

## Stage 1: independent mathematical and code audit
1. Run `validate.py`, reproduce saved results with the supplied seeds, and verify all equations against the implementation.
2. Add independent tests for the direct and RIS-only channel limits; zero-clutter receive filter; one-user power allocation; degenerate/ill-conditioned user channels; and infeasible initializations. Guard zero desired power and zero normalization denominators explicitly.
3. Verify both RIS legs in target/clutter derivatives, the forward-channel convention, PSD covariance construction, and the conditions under which the receiver envelope gradient is valid.
4. Derive the model's units and path-loss/RCS scaling explicitly. Assess whether the RIS scalar model and array spacing are adequate. If correcting physics, make it a new documented model version and rerun comparisons; do not silently reinterpret old results.
5. Document the independent-scatterer ensemble assumption. Do not label its covariance as a measured-clutter estimate.

## Stage 2: strengthen the conventional baseline and feasibility restoration
1. Implement a stronger alternating reference method with a derived constrained W-subproblem for fixed RIS and receiver. One candidate is an SDP relaxation in per-stream covariance matrices, with a fractional-programming transformation where justified. Derive it fully before coding. Account for rank-one recovery and report any relaxation bound only for its actual subproblem.
2. Keep all randomization, recovery, and repair costs in timing. Verify recovered beamformers against original SINR/power constraints.
3. Add a bounded Phase-I feasibility restoration method when zero forcing fails. Distinguish initialization failure, numerical failure, budget exhaustion, and certified infeasibility. Never claim global infeasibility from failure of a heuristic.
4. Compare solver quality at both equal iteration counts and equal measured wall-clock budgets. Treat the current 80-step backtracking heuristic as a reference, not a global oracle.

## Stage 3: differentiable L2O
1. Implement a complex128 PyTorch reference and cross-check forward outputs and autograd against the NumPy analytical gradients. Use real/imaginary parameterizations where necessary; preserve the physical model.
2. Learn bounded positive per-layer step sizes first. Compare this with the frozen CEM baseline.
3. Add a learned initializer using only inputs legitimately available to the selected model. For the ideal-information baseline, label target geometry as known. For the later estimated-information model, do not leak ground truth to the controller.
4. Explain how gradients traverse receiver solves and power allocation. Any straight-through estimator, detached block, surrogate repair, or implicit differentiation must be explicitly documented and ablated. Nonsmooth rejection and best-iterate selection require special care.
5. Preserve exact nominal acceptance checks at inference. A differentiable penalty or predicted confidence score must not replace original-constraint validation.
6. Train initializer-only, update-only, and combined variants independently, with matched datasets and declared tuning budgets. Do not assume the combined model will win: the initial pilot favors inside-only learning on paired sensing quality.

## Stage 4: profiling and expensive-block acceleration
1. Profile batch-one execution and the costs of feature construction, initialization, receiver solve, gradient evaluation, repair, and checking. State which timings overlap; do not sum inclusive profiles.
2. Replace expensive blocks only after evidence identifies a bottleneck. Investigate learned preconditioning or a fixed number of conjugate-gradient steps for the receiver solve, with residual criteria and exact fallback.
3. Re-derive gradients if an approximate receiver replaces the exact argmax: the exact-receiver envelope gradient cannot automatically be reused as the derivative of an approximate-receiver pipeline.
4. Measure error, feasible solution quality, fallback frequency, memory, and total solver latency. Do not use operation count alone as a speedup claim.

## Stage 5: uncertainty and operational evaluation
1. Define an observation/estimation process and calibrate channel/target/clutter uncertainty before calling the model robust.
2. Implement either a justified bounded-uncertainty formulation or a chance-constrained formulation with a stated probability model. Finite scenario checks are empirical evidence, not universal certificates.
3. Define a sensing detector and report detection probability at fixed false-alarm probability. Do not equate SCNR with detection probability.
4. Add acquisition, estimation, RIS configuration, and queueing only when those processes are explicitly modeled or measured. Keep solver-only and observation-to-action latency separate.
5. Evaluate geometry-conditioned joint sensing/communication/deadline reliability. Distinguish spatial randomness from conditional uncertainty and do not multiply dependent success probabilities.
6. Introduce externally moving targets and stale observations only after the static model is validated. Optimizing a UAV trajectory is outside the current target-tracking scope.

## Experimental requirements
- Keep disjoint training, validation, test, and shifted-distribution sets with saved geometry seeds and complete configuration files.
- Start by reproducing the supplied 64/32/128/128 pilot, then scale the datasets and number of seeds after checking resource limits.
- Report all failed cases. Report feasibility coverage separately from SCNR conditional on success. Use a shared-feasible subset for paired objective gains and state its size.
- For methods with different success sets, do not compare conditional averages as if they were paired measurements.
- Use bootstrap uncertainty intervals with a clearly stated resampling unit and report training-seed variation separately.
- Include batch-one timings, warm-up, repeated trials, hardware/software/precision/thread settings, and training cost. Synchronize accelerators for GPU timing.
- Extend N and M first; do not claim variable-K generalization from the existing three-dictionary initializer.
- Do not fabricate favorable results, optimality gaps, convergence guarantees, or novelty statements. State precisely what each experiment establishes.

## Deliverables
1. Tested source code, pinned environment, configuration files, checkpoints, and frozen raw per-instance results.
2. A derivation note in LaTeX and compiled PDF with the exact implemented model and constraints.
3. Publication-quality vector figures generated directly from recorded results.
4. A table identifying inherited assumptions, new changes, tests passed, and unresolved limitations.
5. A concise research assessment: whether L2O improves feasible quality at matched runtime, which component produces the improvement, and which failure modes remain.

Work in stages. First reproduce and audit; then implement the next justified extension. Continue with authorized implementation choices, but document every departure from the supplied physical model. Never report an unexecuted experiment as completed.
