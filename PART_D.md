# Part D column generation and pricing

## D1. Terminal fixed-receiver certificate

The study replays the 96 matched Part C scenes with 1×1, 3×2 and 5×3 B8 budgets. Each of the three operating conditions has 32 seed pairs. The 288 replays matched the archived design and true SCNR scores, nominal certificate and terminal bank size. Each continuation used the receiver from the last restricted master, before the final MVDR update. It kept that master's objective normalization, estimated kernel, tightened SINR targets, safe power budget and terminal bank. The continuation added directions priced above \(10^{-11}\) and performed 4–21 solves. All 288 references reached an audited residual gap of at most \(10^{-4}\) dB.

The plot compares the nominal bound \(10\log_{10}(1+\epsilon/\psi)\) against \(10\log_{10}(L_\mathrm{ref}/\psi)\), where \(L_\mathrm{ref}\) is the independently checked feasible objective at the reference endpoint. Every nominal bound also exceeded the upper end of its reference gap interval. The axes use a linear region near zero and logarithmic scaling above \(10^{-4}\) dB.

| B8 budget | Range of condition median bounds (dB) | Range of condition median gaps (dB) | LP solves and pricing passes | Passes that add columns |
| --- | ---: | ---: | ---: | ---: |
| 1×1 | 18.57–19.02 | 1.009–1.257 | 1 | 0 |
| 3×2 | 7.70–9.63 | 0.0233–0.0283 | 6 | 5 |
| 5×3 | 0.100–0.229 | 0.0002–0.0004 | 15 | 14 |

The certificate was conservative at every tested budget. At 3×2, the median ratio to the observed gap was 262–410 across conditions using the registered 0.001 dB gap floor. At 5×3, only 23 of 96 gaps cleared that floor. The absolute difference is easier to interpret at those small gaps.

### Derivation

Write the scaled master as maximizing \(\bar a^\top y\) over \(y,t\ge0\), with \(By\ge dt\), \(\sum_j y_j\le t\) and \(c^\top y + t=1\). Each column \(j=(n,o,v)\) contributes the unit rank-one direction \(vv^H\) at tone \(n\) with owner \(o\). The vector \(d\) is the normalized communication noise and \(c_j\ge0\) comes from the estimated clutter operator. The reward \(\bar a_j\) uses the frozen positive normalization. Let \(\mu\ge0\) and \(\lambda\ge0\) be the QoS and power prices. In exact arithmetic, the normalization constraint's dual price equals the restricted optimum \(\psi>0\). The scale-variable dual constraint gives \(\lambda-\psi-\mu^\top d\le0\).

The reduced cost of a column is \(r_j=\bar a_j-\psi c_j+\mu^\top B_{\cdot j}-\lambda\). For each tone and owner its maximum over complex unit vectors is the largest eigenvalue of the Hermitian pricing matrix minus \(\lambda\). Define \(\epsilon=\max(0,\sup_j r_j)\) over all these directions. For any feasible allocation over the full dictionary,

\[
\bar a^\top y
\le \psi + (\lambda-\psi-\mu^\top d)t + \epsilon\sum_j y_j
\le \psi+\epsilon .
\]

The last step uses \(\sum_j y_j\le t\le1\). The restricted allocation gives the lower objective \(\psi\), so the ratio gives the nominal dB bound. The bound charges the largest excess to all available scaled power and replaces \(\sum_j y_j\) by one. Both steps can make it loose. The argument applies to full per-owner covariances because each PSD block admits a finite decomposition into nonnegative rank-one terms.

### Numerical audit

The independent upper calculation uses the actual equality dual \(\psi_D\), nonnegative multipliers and full-sphere spectral excess \(\epsilon_D\). It adds \(t_{\max}\bigl(\epsilon_D+[\lambda-\psi_D-\mu^\top d]_+\bigr)\) and a roundoff allowance to \(\psi_D\). Here \(t_{\max}=1/(1+\min(0,c_{\min}))\) uses the smallest scaled estimated clutter eigenvalue. The observed minimum was 1.181. A separate feasible lower witness clips negative roundoff powers and mixes with a strictly feasible zero-forcing allocation when required. This is an empirical numerical audit. The first sweep flagged 15 witness-check failures. The archived repeat evaluated the residual after each correction and completed all 288 cases.

The largest terminal primal–equality-dual objective difference was \(9.42\times10^{-12}\) in scaled objective units. Independently recomputed decoded QoS residuals found two large shortfalls relative to normalized noise, despite zero reported solver row residuals. They occurred at 5×3 on seed 10005 with six users at 8 dB and seed 10020 with four users at 12 dB. Mixing the decoded allocations with the ZF witness lowered their objective by 0.192 and 0.155 dB, respectively. The 288 returned stream designs passed the Part C physical gate after the original acceptance checks.

A post hoc HiGHS interior-point re-solve with crossover on every identical terminal bank bracketed each restricted optimum within \(7.02\times10^{-10}\) dB. The largest difference between the archived raw objective and the independent feasible lower objective was \(2.43\times10^{-9}\) dB. The largest rechecked normalized QoS shortfall before witness correction was \(4.76\times10^{-9}\). The widest resulting full-dictionary gap interval was \(9.9912\times10^{-5}\) dB. All 288 nominal upper objectives exceeded their feasible reference objective. The independent solves corroborated the objective values used for the gap plot.

## D2. Atom activity and power

An atom is active when its decoded power exceeds \(10^{-8}P_s\), approximately \(1\,\mu\mathrm W\). At 3×2 the per-condition median added-column count ranged from 649 to 968. Of those, 19.8–19.9% were active at the terminal master and 84.0–85.8% were active at least once. At 5×3 the median count ranged from 1,680 to 2,506. Terminal activity was 7.6–7.7%, and ever-active fractions were 89.0–89.8%. Priced atoms carried all decoded terminal LP power on every 3×2 and 5×3 endpoint. At 1×1 the initial dictionary carried all decoded power and the active fraction for priced atoms is undefined. Every budget's last pricing pass was for certification only.

The activity measurements describe the original solver allocations. For a basic optimum of this master the \(NK\) QoS rows, power row and normalization allow at most \(NK+1\) positive beam variables alongside the positive scale variable. A growing bank can therefore have a falling active fraction even when its columns carry all power. The atom logs identify tone, owner, addition iteration, decoded watts and whether each column was ever active.

## D3. Initial and random dictionaries

Under the same terminal receiver and tightened constraints across the Part D conditions, the paired median estimated-kernel gain of B8 over the initial bank was 1.99–2.18 dB at 3×2 and 2.21–2.56 dB at 5×3. B8's gains over the count- and owner-matched random bank have the same median ranges to two decimal places. Each of the twelve paired median comparisons has a bootstrap interval entirely above zero.

The archived one-thread Part B end-to-end runs give the following priced design SCNR gains. Intervals are 95% paired bootstrap intervals with 32 draws per row.

| Budget | B8 minus B6 (dB) | B8 minus B7 (dB) |
| --- | ---: | ---: |
| 3×2 | 1.484 [1.393, 1.574] | 1.484 [1.391, 1.574] |
| 5×3 | 1.552 [1.470, 1.683] | 1.552 [1.474, 1.683] |

On the ideal-kernel scoring pass, paired median gains were 1.747 dB at 3×2 and 1.884 dB at 5×3 against either control to three decimals. The B7 schedule matches B8 by tone, owner, round and count.

## D4. Scope

The certificate applies to the transmit covariance optimum for the fixed receive filters \(U\), under the estimated clutter kernel \(\widehat V_n\) and the stated tightened constraints, before rank truncation. The global optimum of the joint nonconvex transceiver problem is outside the certificate's scope.

## Artifacts

`results/partd-2026-11-22/summary/` contains `certificate-vs-gap.pdf`, `tables.md` with per-condition medians and intervals, flattened terminal and recheck records, and paired Part B differences. The shard directories preserve the solve traces, compressed per-atom power logs, cross-checks and environment manifests. The README gives regeneration commands.
