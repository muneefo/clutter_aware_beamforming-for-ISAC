# Sensing SCNR versus required communication SINR

The supplied PDF shows the measured K=4, 8 dB pilot only. Its AO rows come from trials.csv and its two six-LP proposal scores from the printed fixedProposal and learnedProposal benchmark, matched on scene 111 through 114. The fixed and learned labels in trials.csv refer to the earlier three-layer method. The paper's fast proposals use a different solver. The pilot CSV preserves the printed precision of those proposal scores.

## Plot an existing three-method sweep

Extract the ZIP and add its folder to your MATLAB path. If your run_objective_sweeps run included authorsAO=true, give its output directory to:

    [fig,S] = plot_sinr_three_methods(folder);

## Run the SINR sweep on your PC

    [fig,S,folder] = run_sinr_three_methods(repo,outWU);

The second argument can also be a path to trained_model.mat from the W/U run. The authors' tutorial functions, covariance data, a working LP backend, and the original AO CVX solver setup are required. Results are saved under ao_fixed_learned_sinr_50dBm in the repository. Completed channel-condition checkpoints are reused on restart. For this grid there are 288 AO solves (3 loads, 6 thresholds, 8 seeds, 2 kernels) and 576 fast solves.

Axes are communication SINR requirement (dB) and true-kernel sensing SCNR (dB). Both proposal solvers use up to three outer updates and two LPs per update. Each panel fixes user count and the kernel available for design. All three methods receive identical user channels and are scored on the same true clutter kernel, using their returned beams. At each threshold, SCNR is averaged in linear units over channel draws where all three methods returned beams that passed the experiment's power and minimum user-SINR checks, then converted to dB. The summary CSV gives attempted, individually successful, and paired counts. Unavailable points stay missing.
