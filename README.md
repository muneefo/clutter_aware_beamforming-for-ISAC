# Clutter-aware beamforming for ISAC
The current manuscript is in twc-resources/paper. The MATLAB and Python solver benchmarks are in first-isac-paper and advisor-results/experiment. The latest calibration experiment and measurements are in calibration-study/code. Earlier models and experiment versions are retained in their respective directories.

The solvers use the authors' benchmark and covariance data from https://github.com/RangLiu0706/Clutter-Aware-ISAC-Tutorial . Supply that checkout as the repo argument to the experiment drivers.

## IEEE Letters Part 0 replay

`AUDIT.md` records the provenance of the named manuscript, equation checks, and
the historical MATLAB inputs needed to repeat its optimization. `PROTOCOL.md`
records the study design and stage reviews.

Use CPython 3.12.14 on Linux and the pinned tutorial checkout. From this
repository root, replace `../Clutter-Aware-ISAC-Tutorial` with its location if
already cloned.

```bash
git clone https://github.com/RangLiu0706/Clutter-Aware-ISAC-Tutorial.git ../Clutter-Aware-ISAC-Tutorial
git -C ../Clutter-Aware-ISAC-Tutorial checkout 0f864c45b4775fbe01789b313d81f22bebf65177
python3.12 -m venv /tmp/isac-part0-venv
/tmp/isac-part0-venv/bin/pip install --require-hashes -r environment/requirements-part0.lock
MPLCONFIGDIR=/tmp/isac-part0-matplotlib PYTHONDONTWRITEBYTECODE=1 \
  /tmp/isac-part0-venv/bin/python scripts/part0_replay.py \
  ../Clutter-Aware-ISAC-Tutorial results/part0-replay-new
```

Pass a fresh destination each time. The command reconstructs Table I and Fig. 1
from the MATLAB aggregate archive. It reruns the Python 9001–9032 study and
regenerates Fig. 3 from the new raw CSV. The run manifest and comparisons are in
the same output directory. The committed replay is under
`results/part0-replay-2026-11-22/`.

The archived MATLAB command for the Table I numerical experiment requires the
original `trained_model.mat` (or `outWU`) and the MATLAB/CVX stack documented in
`environment/matlab-part0.lock.toml`. Assign `repo` and `checkpoint` to those
paths and use a fresh results directory from the manuscript repository root.

```matlab
addpath('first-isac-paper/experiments')
R = run_paper_experiments(repo, checkpoint, ...
    struct('folder', fullfile(pwd,'results','part0-matlab-replay-new')));
plot_paper_results(R.folder)
```

The aggregate-based Fig. 2 plot can be rebuilt in MATLAB using the archived
`summary.csv` and `paired_intervals.csv`. Copy those inputs into a fresh
directory under `results/` and pass that directory to `plot_full_isac_results`
from `first-isac-paper/result-figures/`.

## IEEE Letters Part A benchmark and CSI audit

Use the same pinned tutorial checkout and CPython as Part 0. Install
`environment/requirements-parta.lock` into a separate Python environment.
Use fresh destination subdirectories and set `TUTORIAL` to the tutorial root.
The two AO shards can run independently on separate CPU cores.

```bash
export PYTHONDONTWRITEBYTECODE=1
python scripts/parta_run.py ao "$TUTORIAL" results/parta-new/ao-0 --backend incremental --shard 0 --shards 2
python scripts/parta_run.py ao "$TUTORIAL" results/parta-new/ao-1 --backend incremental --shard 1 --shards 2
python scripts/parta_run.py csi "$TUTORIAL" results/parta-new/csi
python scripts/parta_run.py slow-time "$TUTORIAL" results/parta-new/slow-time --backend incremental
curl -fL https://arxiv.org/pdf/2602.10537v2 -o /tmp/isac-liu-tutorial-v2.pdf
python scripts/parta_digitize.py /tmp/isac-liu-tutorial-v2.pdf results/parta-new/fig12b-digitized.csv
python scripts/parta_report.py results/parta-new results/parta-new/summary.json \
  --design-csv results/parta-new/ao-designs.csv
python scripts/parta_tables.py results/parta-new/summary.json results/parta-new/tables.md
```

Fig. 12(b) digitization uses Poppler's `pdftocairo`. The raw `ao.jsonl` files
contain all attempted designs and each SDP's numerical audit.
`csi-perturbations.csv` contains the individual perturbed-channel margins.
`PART_A.md` summarizes the completed Part A findings.
The startup diagnostics and earlier solver cohorts are retained separately
from the paired experiment. See the Part A amendments in `PROTOCOL.md`.

## IEEE Letters Part B attribution

Use the Part A lockfile for the Python ladder. Set `TUTORIAL` to the pinned
tutorial checkout and use fresh output paths. Run the two timing commands
sequentially on an idle machine.

```bash
export PYTHONDONTWRITEBYTECODE=1
python scripts/partb_run.py "$TUTORIAL" results/partb-new/t1 --threads 1
python scripts/partb_run.py "$TUTORIAL" results/partb-new/t2 --threads 2
python scripts/partb_run.py "$TUTORIAL" results/partb-new/b4-t1 --threads 1 --b4-persistent
python scripts/partb_run.py "$TUTORIAL" results/partb-new/b4-t2 --threads 2 --b4-persistent
python scripts/partb_report.py results/partb-new results/partb-new/summary
```

Add `--pilot` to run only the warm-up draw and its native SCS check. The
preregistered recovery diagnostic accepts the first run's design file.

```bash
python scripts/partb_run.py "$TUTORIAL" results/partb-new/diagnostic-t1 --threads 1 \
  --recover-b5-from results/partb-new/t1/designs.jsonl
python scripts/partb_run.py "$TUTORIAL" results/partb-new/diagnostic-t2 --threads 2 \
  --recover-b5-from results/partb-new/t2/designs.jsonl
```

The MATLAB entry point uses the original tutorial B1 and the existing
`fast_tutorial_beams` LP with explicit B6, B7 or B8 selection. It requires
MATLAB with Optimization Toolbox, CVX and MOSEK on the path. Run each
thread setting in a fresh MATLAB process.

```matlab
addpath('scripts')
partb_matlab("/path/to/Clutter-Aware-ISAC-Tutorial","results/partb-new/matlab-t1",1)
```

Use `matlab-t2` and thread cap `2` for the two-thread block. The warm-up
profiler captures inclusive function times separately from measured wall time.
The first Python report is archived under `results/partb-2026-11-22/`.

## IEEE Letters Part C statistical evaluation

Use the Part A Python lock, the pinned tutorial, and fresh paths. The two static
shards used CPU 0 and CPU 1 as independent single-core timing lanes. The
correlated-channel and scaling commands can run on the same separate lanes.
Change the CPU indices to allowed distinct cores on a different machine.

```bash
export PYTHONDONTWRITEBYTECODE=1
python scripts/partc_run.py static "$TUTORIAL" results/partc-new/static-0 --cpu 0 --shard 0 --shards 2
python scripts/partc_run.py static "$TUTORIAL" results/partc-new/static-1 --cpu 1 --shard 1 --shards 2
python scripts/partc_run.py warm "$TUTORIAL" results/partc-new/warm --cpu 0
python scripts/partc_run.py scaling "$TUTORIAL" results/partc-new/scaling --cpu 1
python scripts/partc_report.py results/partc-new results/partc-new/summary
python scripts/partc_tables.py results/partc-new/summary/summary.json results/partc-new/summary/tables.md
```

Each execution mode accepts `--pilot` with a fresh output path. The summary
command generates the empirical CDF, Pareto and warm-start/scaling PDFs and
the flattened data. The manifests and raw traces retain coverage, timing lanes
and solver metadata. The benchmark condition and the scaling-only physical
model are specified in the Part C protocol.

## IEEE Letters Part D fixed-receiver certification

Use the Part A Python lock, the pinned tutorial checkout, and the Part C
records. Set `TUTORIAL` to the tutorial path. Use fresh output paths and
select allowed CPU indices when running the shards concurrently.

```bash
export PYTHONDONTWRITEBYTECODE=1
python scripts/partd_run.py "$TUTORIAL" results/partd-new/cert-0 --cpu 0 --shard 0 --shards 2
python scripts/partd_run.py "$TUTORIAL" results/partd-new/cert-1 --cpu 1 --shard 1 --shards 2
python scripts/partd_run.py "$TUTORIAL" results/partd-new/recheck-0 --cpu 0 --recheck results/partd-new/cert-0/bounds.jsonl
python scripts/partd_run.py "$TUTORIAL" results/partd-new/recheck-1 --cpu 1 --recheck results/partd-new/cert-1/bounds.jsonl
python scripts/partd_report.py results/partd-new results/partd-new/summary
```

Use `--pilot` with a separate destination for the nine-combination
startup check. The terminal atom data are in the gzip-compressed CSVs.
`bounds.jsonl` holds the original restricted solves and separate same-receiver
continuations. The archived first-pass audits are in `audit-v1-*`.
The independent interior-point checks of the terminal restricted LPs are in `recheck-*`.
`summary/` contains the vector plot, flat endpoint results, paired ablation
reanalysis and bootstrap tables. `PART_D.md` describes the certificate and
its scope.

## IEEE Letters Part E held-out clutter learning ablation

Use the Part A Python lock and the pinned tutorial. Set `TUTORIAL` to that
checkout, `TEACHER` to a writable working directory and `RUN` to a fresh
results path. Select available CPU indices. Each shard uses one CPU.

```bash
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TEACHER"
python scripts/parte_teacher.py "$TUTORIAL" "$TEACHER/parte-train-0" --split train --cpu 0 --shard 0 --shards 2
python scripts/parte_teacher.py "$TUTORIAL" "$TEACHER/parte-train-1" --split train --cpu 1 --shard 1 --shards 2
python scripts/parte_bundle.py "$TEACHER" "$RUN"
python scripts/parte_policy.py "$TUTORIAL" "$TEACHER" "$RUN/training" --cpu 0
python scripts/parte_evaluate.py "$TUTORIAL" "$RUN/training" "$RUN/evaluation-0" --cpu 0 --shard 0 --shards 2
python scripts/parte_evaluate.py "$TUTORIAL" "$RUN/training" "$RUN/evaluation-1" --cpu 1 --shard 1 --shards 2
python scripts/parte_report.py "$RUN" "$RUN/summary"
```

To train from the archived labels, extract every ZIP under `RUN/labels/`
into `TEACHER` and run the policy step with a new destination. The archives
preserve each scene's fixed receive filters, active atom indices, powers and
directions, and its projected teacher dual matrices. The training arrays are
derived locally from those labels and the first physics master. Per-scene
hashes are recorded in the label manifest and the training input ledger.

For a pilot, run the teacher with `--split pilot`, then pass that directory
as the teacher source to `parte_policy.py --pilot`. The latter logs numerical
gradient and permutation checks. Use `parte_oracle_check.py TUTORIAL OUT.json`
to reproduce the dense-oracle checks. The evaluation script accepts the
original MATLAB checkpoint with `--checkpoint` and records its hash and the
Draw 111–114 atom-level diagnostics when available.

## IEEE Letters Part F manuscript and appendix

The revised sources are `first-isac-paper/manuscript/main.tex` and
`first-isac-paper/manuscript/appendix.tex`. The archived manuscript PDF and
historical Table I remain in their original locations. The latest PDFs,
with the original MATLAB AO baseline and the workstation L2O sweep, are in
`results/letters-matlab-workstation/`. AO time and coverage come from the AO rows of
`first-isac-paper/result-figures/summary.csv`. The separate AO score sweep is
`twc-resources/scnr_vs_sinr_three_methods.csv`. The workstation L2O figure
uses `results/letters-matlab-workstation/l2o-summary.csv`. The original release PDFs, Letter-size
figures, table inputs and provenance manifest are in
`results/partf-2026-11-23/`. The nine verified references, abstract and
conclusion evidence mapping, and changes from the prior draft are in
`REFERENCES.md`, `CLAIMS.md` and `CHANGELOG.md`.

From the repository root, run the following with Python from the Part A lock
and the tools recorded in `environment/manuscript.lock.toml`. This rebuilds
both PDFs and all new table rows and figure variants in a fresh directory.
It reads the recorded experiment results without rerunning optimization.

```bash
partf_out=$(mktemp -d)
mkdir -p "$partf_out/pdf"
MPLCONFIGDIR="$partf_out/matplotlib" PYTHONDONTWRITEBYTECODE=1 \
  python scripts/partf_assets.py "$partf_out/assets"
(
  cd first-isac-paper/manuscript
  for partf_source in main appendix; do
    partf_name=$partf_source
    if [ "$partf_source" = main ]; then partf_name=letters; fi
    latexmk -pdf -interaction=nonstopmode -halt-on-error \
      -usepretex="\\def\\LetterAssets{$partf_out/assets}" \
      -jobname="$partf_name" -outdir="$partf_out/pdf" "$partf_source.tex"
  done
)
```

The asset command formats the reconciliation and CSI tables from Part A,
the ladder and component table from Part B, the core table and Pareto and
tracking/scaling plots from Part C, the certificate plot from Part D's
terminal masters, and both learning tables from Part E. The supplement
directly includes the archived Table I reconstruction, Part C empirical CDF
and Part E ablation plot. The preceding sections give their generation commands.

### MATLAB timing and SCNR versus required SINR

In MATLAB R2022a or later, add the scripts directory and pass the pinned
tutorial folder. The full sweep uses CVX/MOSEK for AO–SDP and either Optimization
Toolbox `linprog` or licensed MOSEK `mosekopt` for the three matched LP arms. Use a fresh MATLAB process
per thread setting and a fresh output folder.

```matlab
addpath('scripts')
tutorial = "../Clutter-Aware-ISAC-Tutorial";
T = run_matlab_sinr_sweep(tutorial, "results/matlab-threshold-preview", ...
    SINRdB=[6 8 10], Seeds=16001:16003, Repeats=1, IncludeAO=false);
```

For the full run, use a new destination and omit the overrides. Defaults
are four users, thresholds 4:2:14 dB, 32 scenes, three repetitions, a
3×2 LP budget and one MATLAB computation thread. Set `Threads=2` in a
separate run for a second thread setting. The frozen reduced-cost and
envelope policies in `results/matlab-sweep-2026-11-23/l2o-weights.mat`
were exported from the Part E checkpoints. The harness checks the native
forward pass against a packaged parity fixture before timing. The 4 dB
point extends below the policies' 6–14 dB training thresholds.

The runner logs `environment.json`, `warmup.csv` and an append-only
`raw.csv`. The console reports feasible runs, rejected designs and runtime
errors separately. Inspect `status`, `message`, `minMarginDB` and `powerW`
in `raw.csv` to trace individual failures. The design clock includes
initialization, solver work, pricing, inference, recovery and acceptance.
Endpoint scoring and file writing fall outside that clock. The LP arms
share the exact solve and column counts and use inference only after the
first nonterminal master. The sweep checks the power LP solution and its
inequality multiplier before timing. It tries explicit `1e-9` tolerances
first, then the MATLAB release defaults. All LP arms in the run use the
same passing configuration. When native MATLAB linprog is unavailable or
fails the probe, the sweep tries MOSEK dual simplex through mosekopt with
explicit primal and dual tolerances of 1e-9. The sweep prioritizes the Optimization Toolbox
copy of `linprog` on its temporary MATLAB path. The manifest records the
solver file, selected algorithm, tolerance mode, startup failures, version
information and requested thread caps.

Under `plots/`, `scnr-vs-sinr.pdf` separates estimated- and ideal-kernel
SCNR. `timing-vs-sinr.pdf` shows design time and feasible-scene coverage.
Each point uses the common scenes for which all requested repetitions
were feasible across methods. Times are paired per-scene medians and
SCNRs are averaged in linear units across scenes. `summary.csv` stores
confidence intervals, paired score changes and time ratios. `paired.csv`
stores the eligible scene-level comparisons when at least one exists.
Plot a partial or completed run again with
`plot_matlab_sinr_sweep(runFolder, freshPlotFolder)`.

Re-export the weights using the Part A Python lock and a new filename.

```bash
python scripts/export_l2o_matlab.py results/parte-2026-11-23/training /tmp/l2o-weights.mat
```

The manuscript's architecture and pseudocode describe the measured Part E
algorithm. Build the standalone diagram from the manuscript source folder
with `latexmk -pdf -outdir=FRESH_OUTPUT l2o-flowchart.tex`. The new MATLAB
curves require a licensed run of the sweep.

### Run just the two new L2O policies

Use the `letters-revision` checkout or extract the [MATLAB L2O bundle](https://github.com/muneefo/clutter_aware_beamforming-for-ISAC/raw/refs/heads/letters-revision/results/matlab-sweep-2026-11-23/isac-l2o-matlab.zip).

The sweep uses either licensed MOSEK `mosekopt` or Optimization Toolbox
`linprog`, plus a separate tutorial
folder. An extracted ZIP or a Git checkout works. MATLAB verifies the
covariance and the tutorial functions used by the sweep against the pinned
revision and records their hashes in `environment.json`. If you need the
files, extract the [pinned tutorial ZIP](https://github.com/RangLiu0706/Clutter-Aware-ISAC-Tutorial/archive/0f864c45b4775fbe01789b313d81f22bebf65177.zip).

From any MATLAB working directory, select the folder containing `scripts`
and the tutorial root containing `data` and `function`. Start with two scenes
at one threshold to check the installation.

```matlab
bundle = uigetdir(pwd, 'Select the extracted bundle or letters-revision repository');
assert(ischar(bundle) && isfolder(fullfile(bundle,'scripts')), ...
    'Select the folder containing scripts.');
tutorial = uigetdir(pwd, 'Select the Clutter-Aware-ISAC-Tutorial folder');
assert(ischar(tutorial), 'Select the tutorial folder.');
cd(bundle)
addpath(fullfile(bundle,'scripts'),'-begin')
clear run_l2o_matlab_sweep run_matlab_sinr_sweep plot_matlab_sinr_sweep
preview = fullfile(bundle,"l2o-preview-"+string(datetime("now","Format","yyyyMMdd-HHmmss")));
T = run_l2o_matlab_sweep(tutorial,preview,Users=4,SINRdB=8,Seeds=11001:11002,Repeats=1);
```

If the preview reports errors, the console prints the first error for each
seed. The `K4/estimated/raw.csv` file lists every trial and its status.
The console and `K4/estimated/environment.json` identify the LP solver
configuration used for both policies. The manifest also records the
selected solver path in `powerSolverFile`.

Once the preview succeeds, launch the full sweep into a fresh folder.

```matlab
T = run_l2o_matlab_sweep(tutorial,fullfile(bundle,'l2o-thresholds'));
```

This uses the archived MATLAB sweep grid (4, 8 and 12 users, 0:4:20 dB,
seeds 11001–11008), the estimated design kernel, and six LP solves
(`Budget=[3 2]`). It times the two frozen Part E policies, reduced cost
and reduced cost plus envelope supervision. Both use the first physics
LP's dual features to propose atoms through a shared MLP. The following LP
assigns power.

Use `IncludeAnalytic=true` for an analytic-pricing reference using the
same counts. Use `DesignKernels=["estimated","perfect"]` to run both
archived design-kernel panels. Both settings report the true SCNR under
the ideal kernel.

Each `K<users>/<design-kernel>/` folder contains `raw.csv` with per-run
timings, `plots/scnr-vs-sinr.pdf`, and `plots/timing-vs-sinr.pdf`.
The root `summary.csv` combines all settings.
The `referenceMethod` column identifies the baseline for paired changes
and time ratios. In the two-policy sweep that baseline is the reduced-cost
policy. The policy training domain was 3–6 users and 6–14 dB. The
additional loads and thresholds in the archived grid are transfer tests.
