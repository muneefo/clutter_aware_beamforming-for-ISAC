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
