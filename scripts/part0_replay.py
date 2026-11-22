"""Replay the archived Python experiment and reconstruct Table I and Fig. 1."""
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

# Set library limits before importing NumPy or SciPy, including in child runs.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import matplotlib
import numpy as np
from scipy.optimize._highspy import _core
from threadpoolctl import threadpool_info

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "first-isac-paper/experiments"
MANUSCRIPT = ROOT / "first-isac-paper/manuscript"
TUTORIAL_REVISION = "0f864c45b4775fbe01789b313d81f22bebf65177"
COV_SHA256 = "99744880b1a6fe5e5f8a0ef57c1738be79fb0ad28629e68db2bfe3a8df67472d"
CONDITIONS = [(4, 8), (4, 12), (6, 8)]
BUDGETS = [1, 6, 15]
MODES = ["physics", "fixedProposal", "learnedProposal"]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_if_available(path):
    source = Path(path)
    return source.read_text().strip() if source.is_file() else None


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def environment(tutorial, command):
    highs = _core._Highs()
    cpuinfo = read_if_available("/proc/cpuinfo") or ""
    cpu = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)
    inputs = [
        EXPERIMENTS / "heldout_budget.py",
        EXPERIMENTS / "verify_fast_beams.py",
        EXPERIMENTS / "plot_budget.py",
        EXPERIMENTS / "heldout_budget.csv",
        MANUSCRIPT / "mc-summary.csv",
        MANUSCRIPT / "main.tex",
        ROOT / "environment/requirements-part0.lock",
    ]
    return {
        "started_utc": utc_now(),
        "repo_commit_at_run": git(ROOT, "rev-parse", "HEAD"),
        "protocol_commit": git(ROOT, "log", "-1", "--format=%H", "--", "PROTOCOL.md"),
        "tutorial_commit": git(tutorial, "rev-parse", "HEAD"),
        "covariance_sha256": sha256(tutorial / "data/covMat_BLP.mat"),
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "os": platform.freedesktop_os_release().get("PRETTY_NAME"),
        "kernel": platform.release(),
        "cpu_model": cpu.group(1) if cpu else platform.processor(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cgroup_cpu_max": read_if_available("/sys/fs/cgroup/cpu.max"),
        "cgroup_memory_max_bytes": read_if_available("/sys/fs/cgroup/memory.max"),
        "gpu_used": False,
        "gpu_device_nodes": [str(p) for p in sorted(Path("/dev").glob("nvidia*"))],
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("numpy", "scipy", "matplotlib", "threadpoolctl")
        },
        "blas": [
            {k: lib[k] for k in ("internal_api", "version", "threading_layer", "num_threads")}
            for lib in threadpool_info()
        ],
        "thread_environment": {
            name: os.environ[name]
            for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "solver": {
            "interface": "scipy.optimize.linprog(method='highs')",
            "version": highs.version(),
            "githash": highs.githash(),
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
            "unchanged_native_defaults": {
                key: highs.getOptionValue(key)[1] for key in ("solver", "threads", "parallel")
            },
        },
        "available_matlab": shutil.which("matlab"),
        "available_octave": shutil.which("octave"),
        "warmup_seed": 8700,
        "draw_seed_range": [9001, 9032],
        "outer_inner_budgets": [[3, 2], [5, 3]],
        "command": [str(arg) for arg in command],
    }


def run_untouched(command):
    thread_counts = set()
    with subprocess.Popen(command, cwd=ROOT) as process:
        while process.poll() is None:
            try:
                status = read_if_available(f"/proc/{process.pid}/status") or ""
            except (FileNotFoundError, ProcessLookupError):
                status = ""
            match = re.search(r"^Threads:\s+(\d+)$", status, re.M)
            if match:
                thread_counts.add(int(match.group(1)))
            time.sleep(0.1)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    return sorted(thread_counts)


def paired_rows(path):
    raw = read_csv(path)
    rows = {(int(r["seed"]), int(r["budget"])): r for r in raw}
    keys = {(seed, budget) for seed in range(9001, 9033) for budget in (6, 15)}
    if len(raw) != len(keys) or rows.keys() != keys:
        raise ValueError(f"Incomplete or repeated seed/budget pairs in {path}")
    return rows


def summarize(rows):
    scores = {
        budget: [float(rows[seed, budget]["trueDB"]) for seed in range(9001, 9033)]
        for budget in (6, 15)
    }
    gain = np.subtract(scores[15], scores[6])
    return {
        "feasible_designs": sum(int(row["feasible"]) for row in rows.values()),
        "median_true_dB": {b: float(np.median(values)) for b, values in scores.items()},
        "paired_gain_dB_q1_median_q3": np.quantile(gain, [0.25, 0.5, 0.75]).tolist(),
        "positive_gain_draws": int(np.count_nonzero(gain > 0)),
        "median_seconds": {
            b: float(np.median([float(r["seconds"]) for key, r in rows.items() if key[1] == b]))
            for b in (6, 15)
        },
    }


def compare_budget(fresh):
    archived = paired_rows(EXPERIMENTS / "heldout_budget.csv")
    replay = paired_rows(fresh)
    numerical = ("trueDB", "designDB", "marginDB", "powerW")
    archive_summary = summarize(archived)
    replay_summary = summarize(replay)
    displayed = {
        "median_true_dB": {6: 13.437, 15: 13.463},
        "paired_gain_dB_q1_median_q3": [0.034, 0.068, 0.116],
        "positive_gain_draws": 29,
        "feasible_designs": 64,
    }
    display_matches = {
        "median_true_dB": {
            b: round(replay_summary["median_true_dB"][b], 3) == expected
            for b, expected in displayed["median_true_dB"].items()
        },
        "paired_gain_dB_q1_median_q3": [
            round(value, 3) == expected
            for value, expected in zip(
                replay_summary["paired_gain_dB_q1_median_q3"],
                displayed["paired_gain_dB_q1_median_q3"],
            )
        ],
        "positive_gain_draws": replay_summary["positive_gain_draws"] == displayed["positive_gain_draws"],
        "feasible_designs": replay_summary["feasible_designs"] == displayed["feasible_designs"],
    }
    return {
        "matched_rows": len(replay),
        "same_lp_counts_and_feasibility": all(
            replay[key][field] == archived[key][field]
            for key in replay for field in ("lpSolves", "feasible")
        ),
        "maximum_absolute_change": {
            field: max(abs(float(replay[key][field]) - float(archived[key][field])) for key in replay)
            for field in numerical
        },
        "manuscript_displayed_values": displayed,
        "matches_manuscript_displayed_precision": display_matches,
        "archived": archive_summary,
        "replayed": replay_summary,
    }


def reconstruct_table_and_figure(destination):
    archived = read_csv(MANUSCRIPT / "mc-summary.csv")
    by_key = {
        (int(r["K"]), int(r["gammaDB"]), r["mode"], int(r["outer"]) * int(r["inner"])): r
        for r in archived
    }

    rows = []
    for users, threshold in CONDITIONS:
        for budget in BUDGETS:
            physics, fixed, learned = [by_key[users, threshold, m, budget] for m in MODES]
            fields = [
                users, threshold, budget, physics["validAO"],
                physics["hits"], fixed["hits"], learned["hits"],
                f"{float(physics['medianSeconds']):.3f}",
                f"{float(physics['medianGapDB']):.3f}",
                f"{float(physics['pairedSpeedup']):.1f}",
            ]
            rows.append(" & ".join(map(str, fields)) + r"\\")
    printed = [
        line.strip() for line in (MANUSCRIPT / "mc-rows.tex").read_text().splitlines()
        if " & " in line
    ]
    in_source = [
        line.strip() for line in (MANUSCRIPT / "main.tex").read_text().splitlines()
        if re.match(r"^[46] & (8|12) & (1|6|15) & ", line)
    ]
    (destination / "table-i-from-archive.tex").write_text("\n".join(rows) + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(8, 4.25), gridspec_kw={"height_ratios": [1.65, 1]})
    palette = ["#264d70", "#ad4f24", "#177d70"]
    colors = LinearSegmentedColormap.from_list("hit_fraction", ["#f5f7fa", "#3b738a", "#0a3b52"])
    names = ["Physics", "Fixed", "Learned"]
    for c, (users, threshold) in enumerate(CONDITIONS):
        top, bottom = axes[:, c]
        top.set(title=f"{users} users, {threshold} dB", yscale="log", ylim=(0.008, 3.5))
        top.grid(axis="y", alpha=0.3)
        hits = np.empty((len(MODES), len(BUDGETS)))
        labels = {}
        for m, (mode, marker, color) in enumerate(zip(MODES, ("o", "s", "^"), palette)):
            records = [by_key[users, threshold, mode, b] for b in BUDGETS]
            gaps = [float(r["medianGapDB"]) for r in records]
            top.plot(
                np.arange(3) + (m - 1) * 0.16, gaps, marker,
                color=color, markerfacecolor="white" if m == 1 else color,
                label=names[m], markersize=5,
            )
            for b, record in enumerate(records):
                total, count = int(record["validAO"]), int(record["hits"])
                hits[m, b] = count / total
                labels[m, b] = f"{count}/{total}"
        top.axhline(0.1, color="0.25", linestyle="--", linewidth=0.9, label="0.1 dB target")
        bottom.imshow(hits, cmap=colors, vmin=0, vmax=1, aspect="auto")
        for (m, b), label in labels.items():
            bottom.text(b, m, label, color="white" if hits[m, b] >= 0.43 else "black",
                        ha="center", va="center", fontsize=8)
        top.set(xlim=(-0.5, 2.5))
        bottom.set_yticks(range(3), names if c == 0 else ["", "", ""], fontsize=8)
        bottom.tick_params(axis="y", length=0)
        for axis in (top, bottom):
            axis.set_xticks(range(3), BUDGETS, fontsize=8)
        if c == 0:
            top.set_ylabel("Median AO gap (dB)", fontsize=9)
        else:
            top.tick_params(axis="y", labelleft=False)
        bottom.set_xlabel("LP budget", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4, frameon=False, fontsize=8)
    fig.subplots_adjust(top=0.87, bottom=0.19, hspace=0.38, wspace=0.28, left=0.12, right=0.97)
    fig.text(0.5, 0.06, "Cells show quality hits per feasible AO reference.", ha="center", fontsize=8)
    fig.savefig(destination / "fig1-from-archived-aggregates.pdf")
    plt.close(fig)
    return {
        "source": "first-isac-paper/manuscript/mc-summary.csv",
        "archived_summary_rows": len(archived),
        "table_i_rows": len(rows),
        "matches_mc_rows_tex": rows == printed,
        "matches_main_tex_table_i": rows == in_source,
        "figure_1_archived_data_points": len(CONDITIONS) * len(MODES) * len(BUDGETS),
        "source_matlab_optimization_replayed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tutorial", type=Path)
    parser.add_argument("destination", type=Path, help="Fresh output directory")
    args = parser.parse_args()
    tutorial = args.tutorial.resolve()
    destination = args.destination.resolve()
    if git(tutorial, "rev-parse", "HEAD") != TUTORIAL_REVISION:
        parser.error(f"Expected tutorial revision {TUTORIAL_REVISION}")
    fresh = destination / "heldout-budget-replayed.csv"
    command = [sys.executable, EXPERIMENTS / "heldout_budget.py", tutorial, "--csv", fresh]
    manifest = environment(tutorial, command)
    if manifest["covariance_sha256"] != COV_SHA256:
        parser.error("Benchmark covariance file hash differs from PROTOCOL.md")
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "run-environment.json", manifest)
    manifest["sampled_child_process_thread_counts"] = run_untouched(command)
    manifest["completed_utc"] = utc_now()
    write_json(destination / "run-environment.json", manifest)

    write_json(destination / "budget-comparison.json", compare_budget(fresh))
    write_json(destination / "archive-reconstruction.json", reconstruct_table_and_figure(destination))
    subprocess.run(
        [sys.executable, EXPERIMENTS / "plot_budget.py", fresh, destination / "fig3-replayed.pdf"],
        check=True, cwd=ROOT,
    )


if __name__ == "__main__":
    main()
