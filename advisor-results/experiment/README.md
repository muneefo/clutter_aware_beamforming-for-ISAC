# Paired clutter controls and comparison with AO
The PDF manuscript and vector plots use existing MATLAB AO measurements and new Python runs. The full 3-by-2 MATLAB SINR sweep shown in the uploaded screenshot has no matching per-seed CSV in these files. Supply its sweep_trials.csv to plot_sinr_three_methods(folder) for a numerical plot with auditable counts, or run the included run_sinr_three_methods(repo,outWU) using the original MATLAB benchmark setup and trained model. The scene-index pilot and archived summary support the AO panel delivered here.

From this experiment directory, with NumPy 2.x, SciPy, pandas, Matplotlib and threadpoolctl installed, run:
    python clutter_test.py /path/to/Clutter-Aware-ISAC-Tutorial
    python controls.py /path/to/Clutter-Aware-ISAC-Tutorial
    python depth.py /path/to/Clutter-Aware-ISAC-Tutorial
    python figures.py

Input: authors' data/covMat_BLP.mat, sha256 99744880b1a6fe5e5f8a0ef57c1738be79fb0ad28629e68db2bfe3a8df67472d
Source repository: https://github.com/RangLiu0706/Clutter-Aware-ISAC-Tutorial . The input covariance file and authors' MATLAB software are distributed separately there.

trials.csv and controls.csv store individual scene-condition outcomes. Absent and Blind reuse the same physical beams with different scoring kernels. RX only updates that blind receiver. Thermal only reoptimizes with the thermal variance supplied in para.sigma2. ZF, RZF and MRT use minimum feasible user powers and a target-aligned null-space sensing beam. RZF uses a fixed 0.1 regularization on normalized channels. depth.csv preserves every layer-prefix measurement; model.json contains the original frozen schedule. All Python runs condition on one scattering environment. No MATLAB or AO rerun was necessary for these controls.
