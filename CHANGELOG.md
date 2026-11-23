# IEEE Letters revision changelog

The comparison is against the supplied manuscript and repository revision `901213023ce5815e1e470769bfe0dd348a42dba0` as identified in the audit. The archived baseline PDF, MATLAB Table I and source aggregates remain available for comparison.

## Manuscript changes

- Reframed the contribution around block covariance linearity, exact linear-fractional reformulation and analytic reduced-cost pricing. Added related work on semidefinite relaxation and rank-one recovery, quadratic transforms and first-order SDP methods.
- Separated target bearing `theta_0`, master value `psi`, total-power price `lambda`, largest eigenvalue, SCNR `eta`, Dinkelbach ratio and scatterer power `nu`. Added a compact symbol table and retypeset Eqs. (16c), (16d) and (21).
- Replaced the main historical headline with matched Python results. The main text carries the benchmark reconciliation, timing ladder, measured Pareto frontier, same-receiver certificate comparison, correlated tracking and array scaling, and the count-matched learning ablation. The supplement retains the historical Table I and audit limitations.
- Described design time as slow-timescale block design. The paper gives the measured gap to a 125 µs NR slot at 120 kHz and states the quadratic waveform dependence in symbol-level STAP/SLP.
- Added asset regeneration commands, input checksums, verified bibliographic identifiers and an abstract/conclusion claim audit.
- Identified learning to optimize explicitly in the abstract, keywords, ablation and supplement. The main text distinguishes the two training objectives and specifies where the network enters the LP workflow. The visual revision adds the architecture, fixed-budget pseudocode and a quantitative interpretation of the paired SCNR and time changes.

- Added a native MATLAB threshold sweep for analytic pricing and both frozen L2O arms, with optional CVX/MOSEK AO–SDP, paired intervals and vector plots. MATLAB outputs use a separate run folder.

## Negative and unexpected findings

- The pinned Python replay remains 1.58–1.60 dB below the digitized tutorial ordinate. The tested one-factor controls did not explain the difference. The figure-generation calibration and Monte Carlo inputs remain unavailable.
- The historical 15-LP aggregate in question stores 1.3826 s and displays 1.383 s. The 1.39 s and 1.395 s values require their original draft or log.
- Intentional SINR tightening buys limited robustness under the sampled CSI errors. The per-tone SDP was slower than the matched joint stack and its original timed blocks certified 28/32 draws per setting. Rebuilding the Charnes–Cooper master also cost time relative to the persistent comparison.
- At correlation 0.99, basis reuse sharply reduced the initial pivot count while its wall-time ratio to a fresh start had an interval spanning one. At 0.90 the reuse arm took longer than cold.
- All audited terminal certificate comparisons held, but the bounds were loose. Two terminal decoded allocations showed QoS shortfalls. Independent re-solves confirmed the restricted objective values and all recovered streams passed the physical gate.
- The two new learned policies traded SCNR for time. The envelope model's first-pass user atoms had zero allocations above the activity threshold. The original SPSA diagnostics still require the archived checkpoint.
- MATLAB/CVX/MOSEK timing, cross-stack tolerance equivalence and measured CVX parsing overhead remain open. The release includes the MATLAB runner and its static validation.

The separate study reports and raw archives record the matched pair counts and uncertainty. [Claim audit](CLAIMS.md) · [Reference review](REFERENCES.md)
