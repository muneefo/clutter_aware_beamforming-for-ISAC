# Absolute SCNR and physical feasibility in clutter-aware ISAC

Open **objective_feasibility_sweeps.pdf** to see all four figures. Each page is also supplied as an individual vector PDF and a PNG.

| Figure | Observations and interpretation |
|---|---|
| absolute_objective | Left: median and interquartile range over 24 fresh Python trajectories, K=4, minimum SINR 8 dB and a 100 W power budget. Iteration zero is the feasible initialization. At each subsequent point the transmit beams have been updated and the receiver has been recomputed. Blue uses the estimated clutter kernel used by the solver. Green scores the same returned beams and receivers with the true kernel. Right: MATLAB final true SCNR from the four printed pilot cases, showing AO, physics at 15 LPs, and fixed and learned proposals at six LPs. The two panels use different experiments. |
| beam_feasibility | Counts from the saved MATLAB Monte Carlo summary. Each cell counts returned beams meeting both the communication and transmit-power tests across all timing repeats for a configuration. Each of the three fast-solver budgets gave the same feasibility count. AO was run once per scene and operating condition. The same 24 channel seeds were used for each condition. Failure to return validated beams does not establish infeasibility of the underlying mathematical problem. |
| scnr_vs_power | 30 to 50 dBm in 2 dB increments, six users, 10 dB minimum communication SINR. Compares joint ISAC, radar only, heuristic and communication only. |
| scnr_vs_sinr | 0 through 20 dB in 4 dB increments at 50 dBm, for 4, 8 and 12 users. Solid and dashed curves show designs obtained using the true and estimated clutter kernels. |

## Display the results in MATLAB

Extract the ZIP and set MATLAB's current folder to the extracted folder. To display the two supplied sweeps and the outer-iteration trajectory, run:

    plot_objective_sweeps();

To plot the full absolute scores and physical constraint residuals from a saved MATLAB benchmark, run:

    plot_saved_objective_feasibility(R.folder);

If R is no longer in memory, run the function without arguments and select the folder containing **raw.csv**, for example paper_mc_v1 or paper_smoke. The output folder **absolute_and_feasibility** contains three vector PDFs and one per-scene CSV. The viewer computes per-scene medians over timing repeats and uses matching feasible scenes for the four methods at each budget in the two objective panels. Feasibility uses all attempted scenes for each method. The margin panels include any finite recorded values, including those from unsuccessful returns. The dashed lines are the experiment's acceptance tolerances. Negative power slack means transmitted power exceeded 100 W. Distinct outer-by-inner budgets are independent runs. Only the outer-update trajectory plot represents within-run convergence.

## Scope of the new simulations

The new curves use the tutorial’s 32-tone, 16-transmit/16-receive configuration and its supplied covMat_BLP.mat. Each power/SINR point uses eight user-channel draws, seeds 11001 through 11008, with a fixed pair of true and estimated environmental kernels. Seeds are reused across operating points. These draws characterize variability of the user channels conditional on that environment. Results are arithmetic means of linear SCNR converted to dB. For each condition and method, sweep_means.csv gives both attempted and usable draw counts. All 464 joint beam designs in the two sweeps passed the returned-beam power and SINR checks. The unconstrained radar, heuristic, and communication curves are reference designs without that joint feasibility test.

The joint curve in the supplied power and SINR figures is the physics-based solver with up to three outer updates and two LPs per outer update. The Python radar reference alternates receive updates and a closed-form, power-only transmit update using the best single-tone generalized Rayleigh quotient. It starts from equal power with target steering. The heuristic adds target steering to maximum-ratio communication beams. The communication-only design uses maximum-ratio transmit beams with a receive filter designed using the true kernel. The original tutorial's figures use its alternating-optimization implementation. These experiments use its operating axes and compare physical beam designs with their own solver, random draws, and averaging. For the power study the target coefficient is scaled with transmit power following the tutorial implementation. The path-loss exponent is 3.1 for the power study and 2.7 for the SINR study, with user distances in [100,500] and [50,400] metres, respectively.

When only the objective panel's right side is considered, its four MATLAB pilot scores have the precision printed in the console. The complete MATLAB per-scene absolute values and residuals are available to the viewer on the machine holding raw.csv. A median absolute score cannot be recovered from a median paired gap alone.

## Optional new MATLAB runs

The two plotting functions require only the saved data. To rerun new optimization sweeps, make the extracted package current, use your existing repository root and trained outWU model, and use the same LP and CVX setup as your preceding benchmark:

    folder = run_objective_sweeps(repo,outWU);
    plot_objective_sweeps(folder,"learnedProposal");
    plot_objective_sweeps(folder,"fixedProposal");
    plot_objective_sweeps(folder,"physics");

To also run the original, slower joint and radar AO solvers:

    folderAO = run_objective_sweeps(repo,outWU,struct('authorsAO',true));
    plot_objective_sweeps(folderAO,"AO");

The driver writes one checkpoint per operating point and seed, then assembles sweep_trials.csv and outer_trace.csv. You may choose smaller seed lists with options.seeds and options.traceSeeds. Use a new options.folder when changing settings. The original AO history is a design-kernel covariance trajectory, so the MATLAB viewer shows that quantity, while the final joint scores evaluate recovered beams. The MATLAB driver and plotting functions passed syntax checks; the included new numerical sweeps were executed in Python.

## Data and source

- **full_run_summary.csv** contains the archived MATLAB summary at its printed precision.
- **pilot_matlab.csv** contains the four saved, returned-beam pilot scores.
- **outer_trace.csv** contains actual outer updates for seeds 10001 through 10024.
- **sweep_trials.csv** contains per-draw objective scores and joint-beam audits.
- **sweep_means.csv** contains the points used in the two operating curves.

Python regeneration requires Python, NumPy (2.x), SciPy, pandas, threadpoolctl, matplotlib, and the original covariance data. Run from the extracted folder:

    python run_sweeps.py /path/to/Clutter-Aware-ISAC-Tutorial --workers 2
    python make_plots.py

Tutorial source and data: https://github.com/RangLiu0706/Clutter-Aware-ISAC-Tutorial
