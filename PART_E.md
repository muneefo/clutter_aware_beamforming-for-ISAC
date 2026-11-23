# Part E learning-to-optimize ablation

## Original SPSA diagnostic

The source of the Draw 112 comparison is `first-isac-paper/three-method-sinr/pilot_scnr.csv`. It records 13.5516 dB for both fixed and learned proposals at the displayed precision and 13.5214305868786 dB for AO, with four users, an 8 dB threshold and a 100 W power limit. The separate `objective-feasibility/pilot_matlab.csv` contains the same displayed fixed and learned values. The contemporaneous checkout, public refs and workspace lack the original 58-parameter `trained_model.mat` or `outWU`. The archive provides a score comparison. The fractions of SPSA atoms with positive power, the phase-invariant cosine similarities and the frozen SPSA holdout effects are unmeasured. The code accepts the original checkpoint for a separate replay and records its hash.

## Teacher and deployment receivers

Each synthetic scene draws its own independent map and channel seed streams. The training maps and unseen test maps use the ideal Eq. (8) kernel. Its Choi matrix is PSD as the Gram matrix of vectorized Kraus operators. The explicit positive-eigenvalue Choi projection changed the pilot covariance response by 1.93 × 10⁻¹⁵ relative Frobenius norm. The original estimated and ideal maps form a separate benchmark environment. All environments keep the target, noise, additive covariance and array dimensions from the pinned tutorial.

The teacher alternates exact MVDR with column generation under the tightened QoS and safe power constraints. For a fixed receiver, the oracle prices each tone and owner over the full complex unit sphere. A numerical upper bound uses the normalization dual, the corrected scale-variable price and a bound on summed scaled power derived from the minimum clutter eigenvalue. The lower bound uses independently audited feasible powers. An outer endpoint is accepted as a teacher when the upper objective is within 0.01 dB of the entering feasible covariance's objective at its current MVDR receiver. The final restricted-versus-full covariance gap must also be at most 0.01 dB. These gates concern block convergence.

The fast implementation reuses the tone's base eigendecomposition across owners and solves a positive rank-one secular equation per user. The numerical guard requires a checked upper root and uses full Hermitian pricing for unresolved or nearly degenerate updates. Across 76,608 pilot and retry eigenproblems, the largest dense-versus-secular principal price difference was 1.86 × 10⁻¹¹ and the largest Rayleigh score error on a returned direction was 1.42 × 10⁻¹³. The checked eigenvalue upper was at least the dense principal value at every tested point. The teacher uses an independent IPM re-solve on restricted masters when the persistent simplex model reports a numerical failure. The first attempt is preserved.

The deployment policy sees the ZF seed's MVDR receiver $U_0$ through the first physics master's duals. It receives features from the same first master in training and testing. The converged teacher receiver and its dual matrices enter the learning loss only. The archived label includes that receiver, projected teacher dual matrices and atoms above the registered $10^{-8}P_s$ activity threshold.

## Learning losses and matched allocation

The shared 12→8→4 tanh network predicts real coefficients for four unit complex directions, the user's ZF and channel directions, the target steering vector, and the target projected into the multiuser null space. Features use projected pricing Rayleigh values, channel overlaps, normalized log QoS and noise, and mean features pooled over the users at each tone. A fixed bias of 1.25 on the target coefficient sets the initialization scale. The shared network and pooled inputs are permutation equivariant. On three pilot scenes, all parameter gradients for both losses matched centered finite differences with maximum scaled error 4.66 × 10⁻¹¹. The largest observed user-permutation error was 1.94 × 10⁻¹⁴.

The first loss minimizes the negative reduced cost of the predicted atom under the deployment-round matrix. The second adds a teacher term weighted by the active scaled primal masses for each user and tone. At a fixed teacher optimum, the envelope derivative with respect to a beam direction is its scaled power times the derivative of its Lagrangian quadratic. The training term uses the teacher dual quadratic projected into the four-direction span, with the active mass and the teacher dual held fixed. Both arms start from the same weights and use the same five epochs and shuffled scene batches.

Every evaluation arm executes the same number of masters. The first nonterminal pass matches counts and owner tags by adding one candidate per user plus an analytic sensing candidate on each tone. Random, the zero-output gradient heuristic and each trained network replace only the first user proposals. Later passes use analytic pricing, with exactly one column per tone and owner so the bank sizes match. The final pricing pass is diagnostic. All deployed streams use the common rank reduction and physical acceptance checks. Inference time includes construction of the policy features, the forward pass and the normalized beam directions. The fixed-receiver restricted LP optimum is nondecreasing as columns are appended under the same objective and constraints. Recovered-stream scores across different dictionaries and receiver updates depend on feasibility, rank reduction and acceptance.

## Teacher coverage and numerical retries

All 5,000 training scenes yielded labels. The largest final fixed-receiver gap was 0.009799 dB and the largest upper bound on the last alternating gain was 0.0099997 dB. The teacher used 195,792 restricted LP iterations, with a median of 36 per scene. Alternating convergence took a median of four outer iterations and at most 21. The full sweep required 15 independent IPM re-solves after a simplex failure. Each recovered within its current alternating run.

The first sweep attempted 1,233 teacher labels before the uniform numerical rerun, including three unknown solver statuses. Those three scenes converged on the rerun. Among 1,230 scenes converged in both attempts, the maximum difference in final teacher SCNR was 0.0000021 dB. Both attempt logs and the independent oracle check are archived.

The median angle between the stacked deployment and teacher receive filters was 69.88° [69.58°, 70.11°], with a 95% bootstrap interval over training scenes. Both policies trained for five epochs and 785 Adam steps. Features were computed once per training scene from its first physics LP.

## Held-out evaluation

All 1,920 endpoints passed the physical stream checks. The held-out cohort had 64 independent map and channel draws. The original benchmark cohort had 64 independent channel draws with four users at 8 dB. At 1×1 all arms gave the same SCNR because the terminal master has no deployable proposal pass. At 2×1 and 3×2 the comparison holds solve count, atom count and owner tags fixed.

The table reports paired medians relative to analytic pricing, with 95% bootstrap intervals. A time ratio below one means shorter measured design time.

| Environment | Budget | Learned policy | Design SCNR change (dB) | Design time ratio |
| --- | --- | --- | ---: | ---: |
| Held-out maps | 2×1 | Reduced cost | −0.255 [−0.384, −0.184] | 0.903 [0.880, 0.922] |
| Held-out maps | 2×1 | Envelope | −0.209 [−0.298, −0.055] | 0.881 [0.875, 0.905] |
| Held-out maps | 3×2 | Reduced cost | −0.012 [−0.031, −0.005] | 0.976 [0.966, 0.995] |
| Held-out maps | 3×2 | Envelope | −0.012 [−0.030, −0.004] | 0.967 [0.952, 0.980] |
| Original benchmark | 2×1 | Reduced cost | −0.748 [−0.846, −0.679] | 0.943 [0.929, 0.956] |
| Original benchmark | 2×1 | Envelope | −0.740 [−0.834, −0.677] | 0.921 [0.910, 0.938] |
| Original benchmark | 3×2 | Reduced cost | −0.044 [−0.084, −0.011] | 0.979 [0.965, 0.996] |
| Original benchmark | 3×2 | Envelope | −0.052 [−0.087, −0.021] | 0.964 [0.953, 0.977] |

Both learned policies traded lower measured time for lower design SCNR at each budget with deployable proposals. Median feature and inference time ranged from 0.69 to 0.79 ms. Neither learned policy jointly improved the two measured criteria over deterministic pricing. At 2×1 on the original map, the median true SCNR changes were −0.889 and −0.885 dB for reduced cost and envelope, respectively. At 3×2 the two median true changes were both −0.038 dB at three decimals, and their intervals included zero. The tables retain the count-matched random and gradient controls. Both also had lower median design SCNR than analytic pricing at 2×1 and 3×2.

In the direct paired comparison of the two networks, the magnitude of the median SCNR change from adding envelope supervision was below 0.000061 dB for both scores in every cohort and budget. Across the two budgets with a deployable proposal pass, none of the envelope policy's 34,176 first-pass user proposal placements exceeded the positive-power threshold at any downstream LP solve. The reduced-cost arm's median ever-active fraction ranged from 7.1% to 27.7% across budgets and cohorts. Those activity statistics concern the two new policies. The original SPSA atom diagnostic remains pending its checkpoint.

## Artifacts

`results/parte-2026-11-23/` contains the 5,000 labeled training scenes, per-scene label hashes, trained weights, solver and gradient traces, and all 1,920 measured endpoints. Its `summary/` directory contains the paired tables and `proposal-ablation.pdf`. The README gives the regeneration commands.
