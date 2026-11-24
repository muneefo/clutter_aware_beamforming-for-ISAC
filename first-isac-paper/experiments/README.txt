Run the matched MATLAB experiment
=================================
Place run_paper_experiments.m and plot_paper_results.m together and add their
folder to the MATLAB path. Use the same tutorial repository, covariance file
and enabled CVX SDP backend as in the four-scene experiment. If needed,
select your installed SDP solver in the authors' solve_TX_Dinkelbach.m.
The fast-branch LP backend is tested automatically. The driver contains
its solver; no extra copy of fast_tutorial_beams.m is needed.

R = run_paper_experiments(repo,outWU);
stats = plot_paper_results(R.folder);

If starting a new MATLAB session, set repo to your tutorial root and pass
the full path to the saved trained_model.mat in place of outWU.

A quick end-to-end smoke test in a separate folder:
opt = struct('seeds',10001:10002,'conditions',[4 8],'budgets',[3 2], ...
             'repeats',1,'folder',fullfile(repo,'paper_smoke'));
R = run_paper_experiments(repo,outWU,opt);
stats = plot_paper_results(R.folder);

Defaults: 24 fresh channels per condition; [users, SINR in dB] = [4,8],
[4,12], [6,8]; [outer,inner] = [1,1], [3,2], [5,3]. One reference AO
solve and three timing repetitions for each fast configuration per scene.
The first repetition randomizes AO with the nine fast methods. Remaining
repetitions randomize fast methods. One separate warm-up scene per condition.
At full settings this is 72 timed AO solves and 1944 fast solves, plus
warm-up. Start with the smoke test. The full call automatically resumes
completed scene/condition checkpoints in repo/paper_mc_v1. Use one MATLAB
process per output folder. A changed manifest requires a new folder.
To rerun a failed scene, move its checkpoint out of the result directory
and repeat the command. Solver errors are retained in raw.csv.
Do not pool different user-count or SINR settings when assessing learning.
Freeze the original weights across conditions; results at K=6 or 12 dB are
transfer tests. Seeds vary the communication channels, not the clutter map.
Use new seeds and a new folder if the current test set informs a redesign.

Outputs:
 raw.csv: all attempted solves, feasibility, scores, margins, power, times.
 scenes.csv: within-scene median time over repeats; missing/failed outcomes.
 summary.csv: feasibility, hits, comparable AO count, gaps and speedups.
 learning.csv: learned vs fixed proposals matched on scene, condition and budget.
 paired_intervals.csv: percentile bootstrap intervals over paired feasible
 channel draws; learned-only and fixed-only quality hits.
 quality_time_*.pdf: runtime vs success frontier for each condition.

The auxiliary Python budget experiment in the manuscript is reproducible with
numpy, scipy, threadpoolctl and the tutorial checkout:
    python heldout_budget.py /path/to/Clutter-Aware-ISAC-Tutorial
heldout_budget.csv contains the 64 actual Python evaluations. The Python
experiment covers the physics solver only. Four paired MATLAB measurements
were supplied from the author's PC. No new MATLAB sweep has been executed here.

Recreating the training and plots:
 - Original tutorial and required data: https://github.com/RangLiu0706/Clutter-Aware-ISAC-Tutorial
   The source revision checked for this package is 0f864c45b4775fbe01789b313d81f22bebf65177.
 - To recreate outWU if you have no saved weights, use the included original
   training driver: outWU = run_tutorial_WU(repo,struct('depth',3,'rxSteps',3));
   It runs the anchor-repair pilot too. If you already have outWU or the
   corresponding saved trained_model.mat, pass it directly.
 - learning_gain_*.pdf: empirical distribution of the paired learned-minus-fixed
   true SCNR for each LP budget, grouped by user count and threshold.
 - python plot_budget.py heldout_budget.csv budget-gains.pdf recreates the
   manuscript figure using numpy and matplotlib.

Capture a representative transmit beampattern
=============================================
The scene checkpoints above contain scalar results only. To retain the actual
beamformers, run the dedicated one-scene capture in a fresh folder:

    P = run_beampattern_capture(repo, fullfile(paperResults,'manifest.mat'));

Set paperResults to the completed paper_mc_v1 folder. A trained_model.mat path
or the in-memory outWU structure can also be passed. The default is seed 10009
at four users and 8 dB, using
the 15-allocation budget. This is the lower-index scene adjacent to the median
learned-policy SCNR gap. The function reruns AO, physics pricing, the fixed
proposal and the learned proposal, then saves their W and U matrices. It
exports a vector PDF, a PNG and a CSV with the target response, strongest
dominant-clutter response and their difference. The three red dotted guides
mark the strongest separated cold-clutter directions in covMat_BLP.mat.
