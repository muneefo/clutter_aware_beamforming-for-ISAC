## A1 · Reconciliation

True covariance SCNR uses the ideal kernel and the returned receiver. The run averages in linear units before converting to dB. Parentheses give successfully solved draws out of attempts. Changes are paired medians in dB with 95% bootstrap intervals.

| Factor | 8 dB mean SCNR | 8 dB paired change | 12 dB mean SCNR | 12 dB paired change |
| --- | --- | --- | --- | --- |
| Digitized Fig. 12(b), estimated K=4 | 15.10 | — | 15.10 | — |
| Python AO, pinned settings | 13.515 (32/32) | 0.000 [0.000, 0.000] | 13.497 (32/32) | 0.000 [0.000, 0.000] |
| AO caps 5×3 | 13.544 (32/32) | -0.003 [-0.015, 0.006] | 13.521 (32/32) | -0.003 [-0.013, 0.012] |
| Outer tolerance 0.1 dB | 13.475 (32/32) | -0.007 [-0.036, -0.000] | 13.454 (32/32) | -0.005 [-0.031, -0.001] |
| Inner relative tolerance 0.01 | 13.515 (32/32) | -0.000 [-0.000, 0.000] | 13.496 (32/32) | -0.000 [-0.001, -0.000] |
| Resampled distances | 13.515 (32/32) | 0.000 [-0.000, 0.001] | 13.497 (32/32) | 0.000 [-0.000, 0.002] |
| Distance range 100–500 m | 13.514 (32/32) | -0.001 [-0.002, -0.001] | 13.495 (32/32) | -0.003 [-0.003, -0.002] |
| Path-loss exponent 3.1 | 13.508 (32/32) | -0.008 [-0.009, -0.006] | 13.483 (32/32) | -0.014 [-0.016, -0.011] |
| Power cap 199.526 W | 13.541 (32/32) | 0.051 [0.040, 0.058] | 13.536 (32/32) | 0.049 [0.040, 0.058] |
| Target coefficient only, ×1.995262 | 16.515 | +3.000 by normalization | 16.497 | +3.000 by normalization |

## A2 · Initialization sensitivity

| Channel seed | AO solves | Design spread (dB) | True SCNR spread (dB) |
| --- | --- | --- | --- |
| 13001 | 20/20 | 0.566 | 0.378 |
| 13002 | 20/20 | 0.342 | 0.218 |
| 13003 | 20/20 | 0.434 | 0.280 |
| Quality tolerance | — | 0.867 | 0.561 |

## A3 · Stream recovery and paired gaps

Gaps are AO minus the 5×3 priced LP for physically feasible pairs. Estimates are paired medians in dB with 95% bootstrap intervals. A material truncation change is at least 0.001 dB in absolute value.

| SINR (dB) | AO solves | Rank-excess designs | Design changes | True changes | Design gap | True gap |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 32 | 0 | 0 | 0 | 0.069 [-0.061, 0.161] | 0.056 [-0.089, 0.152] |
| 12 | 32 | 0 | 0 | 0 | 0.066 [-0.044, 0.166] | 0.056 [-0.052, 0.143] |

## A4 · Margin and CSI errors

The three percentages are the mean across all draws of the fraction of 100 perturbed channels that satisfy every nominal SINR requirement. The SCNR cost is the paired median versus zero explicit tightening, with its 95% bootstrap interval.

| SINR (dB) | Tightening (dB) | Achieved median (dB) | SCNR cost (dB) | 1% CSI error | 3% CSI error | 5% CSI error |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | legacy | 0.000087 | 0.0000 [0.0000, 0.0000] | 0.000% | 0.000% | 0.000% |
| 8 | 0 | -0.000000 | 0.0000 [0.0000, 0.0000] | 0.000% | 0.000% | 0.000% |
| 8 | 0.25 | 0.250000 | 0.0022 [0.0012, 0.0034] | 7.906% | 0.000% | 0.000% |
| 8 | 0.5 | 0.500000 | 0.0029 [0.0020, 0.0062] | 21.438% | 1.062% | 0.031% |
| 8 | 1 | 1.000000 | 0.0066 [0.0045, 0.0120] | 38.531% | 4.125% | 0.312% |
| 12 | legacy | 0.000087 | 0.0000 [0.0000, 0.0000] | 0.000% | 0.000% | 0.000% |
| 12 | 0 | -0.000000 | 0.0000 [0.0000, 0.0000] | 0.000% | 0.000% | 0.000% |
| 12 | 0.25 | 0.250000 | 0.0014 [0.0005, 0.0023] | 2.188% | 0.000% | 0.000% |
| 12 | 0.5 | 0.500000 | 0.0026 [0.0014, 0.0036] | 5.375% | 0.000% | 0.000% |
| 12 | 1 | 1.000000 | 0.0059 [0.0030, 0.0075] | 10.594% | 0.156% | 0.000% |
