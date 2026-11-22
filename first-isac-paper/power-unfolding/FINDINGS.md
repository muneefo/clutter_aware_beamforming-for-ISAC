# Held-out findings

The eight-layer unfolded power allocator learned a global, layer-dependent step schedule. The fixed control was tuned to (0.5, 5.6). In order of deployment, the selected pairs were (0.125, 7.84), (0.0625, 5.6), (4, 1.4), and (5.6, 22.4); each pair is applied twice. See model.json for full precision.

| First allocation only; 216 matched cases | Exact LP | Fixed steps | Learned steps |
|---|---:|---:|---:|
| Median design-objective gap, dB | 0 | 0.544 | 0.055 |
| Design gap at most 0.1 dB | 216 | 9 | 126 |
| Feasible physical beams | 216 | 216 | 216 |
| Median allocation + recovery + audit, ms | 28.18 | 7.90 | 7.97 |

Across these cases, the mean paired true-SCNR improvement of learned over fixed was 0.704 dB, and the median paired improvement was 0.215 dB. The 95% bootstrap interval for the mean, resampling by channel seed, was [0.605, 0.796] dB; only 12 distinct held-out seeds and one kernel pair were used. This comparison is raw performance without exact fallback. Requiring an observable design-gap certificate of 0.1 dB forced an exact fallback in 202/216 learned cases and raised the median measured cost to 36.39 ms. An inconclusive certificate does not establish an actual quality failure.

| Three-outer-by-two-inner solver; 24 cases at 8 dB | All exact | Fixed steps first | Learned steps first |
|---|---:|---:|---:|
| Median true-SCNR shortfall from all-exact, dB | 0 | 0.207 | 0.314 |
| True-SCNR shortfall at most 0.1 dB | 24 | 9 | 9 |
| Feasible physical beams | 24 | 24 | 24 |
| Median full-solver time, ms | 390.80 | 377.50 | 377.26 |

The full-solver mean paired learned-minus-fixed true SCNR was −0.018 dB (seed-cluster bootstrap interval [−0.061, +0.040] dB, eight distinct seeds). These data demonstrate a local benefit from the trained schedule. They do not establish an end-to-end learned quality or timing advantage over the fixed-step hybrid. Subsequent power allocations still use the exact LP, and direction generation consumes the first allocator's approximate duals. The causal contribution of the dual error has not been isolated.

Runtime is specific to this Python CPU benchmark (Intel Xeon Platinum 8573C, BLAS thread limit 1). The reference here is the exact LP in the engineered fixed-proposal solver; the separate MATLAB AO comparison remains in the original experiment.
