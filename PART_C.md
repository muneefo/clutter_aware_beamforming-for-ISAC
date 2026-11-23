# Part C statistical evaluation

## Static cohort and Pareto frontier

Part C used the pinned benchmark inputs and the Part A Python environment lock. The 96 references are 32 matched channel seeds per setting across four users at 8 and 12 dB and six users at 8 dB. The earlier 9001–9032 study is archived under Part 0. The new Python cohort reuses the historical MATLAB seed labels 10001–10024 and extends through 10032. The raw files identify the new numerical stack and timing lanes.

The AO reference uses the tutorial's random covariance start, 30×30 caps and early stops. Each method's endpoint is scored on recovered physically feasible streams and its returned receiver. Design gaps use the estimated kernel. True gaps score with the ideal kernel. A gap is the signed per-draw reference-minus-method difference. The design and true thresholds are 0.867 and 0.561 dB from the Part A audit. Each per-condition table treats seeds as the sampling units and reports medians, both quartiles, bootstrap 95% intervals, and exact Clopper–Pearson intervals for quality hits. The empirical CDFs use feasible reference pairs.

All nine core method groups returned 32 feasible pairs. At 3×2, the priced LP met the design threshold on all 32 draws in each setting. The true-SCNR counts were 31/32, 32/32 and 29/32 in the order above. At 5×3, the median design gaps were 0.076, 0.091 and 0.103 dB and the median true gaps were 0.012, 0.064 and 0.142 dB. The 1×1 priced LP had zero design or true hits at all three settings. The true hit count at four users and 12 dB decreased from 32 to 31 between 3×2 and 5×3 while the paired median true gap decreased.

The Pareto plot uses the four-user, 8 dB subset. It keeps the Gaussian-start AO, structured AO, structured Charnes–Cooper AO, per-tone SDP and priced LP across the five registered budget labels. Charnes–Cooper uses one transmit SDP per outer iteration. Same-inner-cap prefixes reuse one measured traversal with separate endpoint recovery, and their clocks exclude intermediate checkpoint evaluation. Filled markers denote endpoints nondominated by median time and median design gap on eligible pairs. The Gaussian-start 3×2 endpoint and priced LP at 1×1, 2×1, 3×2 and 5×3 lie on this empirical frontier.

All 1,088 static endpoints were attempted. The physical gate excluded the Gaussian 2×1 endpoint on seed 10006 and the per-tone 5×3 and 8×3 endpoints on seed 10008. The two per-tone labels share one converged traversal. Its normalized subproblem residual passed the SDP gate while the returned streams missed the nominal SINR gate. The corresponding points retain 31 pairs, and the raw records preserve the failures. The rest of the static groups retain 32 pairs.

## Correlated tracking and array scaling

Each correlated chain draws its user distances once. The initial Gaussian channels and the three innovations are shared across the two rho values, with innovations scaled by each user's fixed path loss. Each comparison uses a fresh offline Gaussian AO reference on that frame. The AO initialization is derived with the Part A mapping from chain seed plus 100000 times frame index. The measured time is the time to the first physically feasible outer checkpoint within the design threshold. The LP solver receives an evaluation stop target from that offline reference. The time excludes the reference optimization.

Both reuse arms prepend the same previous-frame basic directions to the current structured dictionary. The basis arm also maps the previous simplex basis to the retained columns, scale variable and constraints. The current-frame structured dictionary supplies feasible directions and the LP reoptimizes the transported basis for the changed channel. CIs on timing and paired ratios resample complete chains. Exact hit-rate CIs use independent chain outcomes at each frame, or a chain-level indicator for all three successful frames.

The array experiment uses the archived scatterer angles and powers with the factorized ideal clutter operator at every size, and isotropic additive disturbance at the archived mean sensor power. This construction holds the prescribed model fixed as normalized ULA steering is extended. The factorized ideal operator matched the pinned vectorized ideal operator at 16 antennas in a numerical check. Power, communication noise, target coefficient and distances follow the registered rules. The measured-size plot uses a common 1×1 cap. This controlled scaling model differs from the static benchmark's estimated kernel and full additive covariance.

All 96 scored frames in each rho condition reached the target for each of the three arms. HiGHS accepted all 192 requested basis transfers. At rho = 0.99, reusing the basis reduced median initial simplex pivots from 203.5 with dictionary reuse alone to 4. The paired time ratio versus dictionary reuse alone was 1.031× [1.007, 1.044]. The fresh-to-basis ratio was 0.988× [0.962, 1.013], with median times of 0.072 s fresh and 0.073 s with reuse. At rho = 0.90, the fresh-to-basis ratio was 0.937× [0.925, 0.947], and the reuse arm took longer. The two-panel figure shows the measured time to the registered target, including the distribution tails.

All three array methods returned 32 feasible designs at every size, with no timeouts. At 64 antennas per array, median time was 0.153 s for the priced LP, 1.164 s for joint SDP and 6.753 s for per-tone SDP. These are 1×1-cap timings. The plot's whiskers show the interquartile range on the observed design times.

## Artifacts

`results/partc-2026-11-22/summary/` contains the numerical tables, the flattened static, correlated and array-design rows, and three vector figures.

- `core-ecdf.pdf` shows times and the two score-gap distributions for the core cohorts.
- `pareto.pdf` shows all five budget labels for each method and the observed nondominated endpoints.
- `warm-start-scaling.pdf` shows the measured tracking and array-size results.

Input hashes, hardware, threads and solver versions are recorded in the raw cohort manifests. The README gives the commands to regenerate the results. Part D awaits the stage review.
