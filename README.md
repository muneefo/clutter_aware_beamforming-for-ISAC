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
