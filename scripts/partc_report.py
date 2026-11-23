"""Generate Part C distributions, paired tables and vector scientific figures."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator
import numpy as np
from scipy.stats import binomtest

from partc_run import BUDGETS, CONDITIONS, CORE, METHODS, QUALITY_DB, SEEDS

TRUE_DB = 0.5609881611015304
NAMES = {"gaussian": "AO Gaussian", "zf": "AO ZF", "cooper": "AO Charnes–Cooper",
         "tone": "Per-tone SDP", "lp": "Priced LP", "cold": "Fresh dictionary and basis",
         "dictionary": "Reuse dictionary", "basis": "Reuse dictionary and basis"}
PALETTE = ["#32506f", "#bd7028", "#866399", "#54868a", "#21473b"]
BUDGET_MARKERS = ["o", "s", "D", "^", "P"]


def read(path):
    return [json.loads(l)["design"] for l in path.read_text().splitlines()]


def valid(row):
    return row["status"] == "solved" and row.get("feasible", False)


def by_key(rows, fields):
    keyed = {tuple(r[f] for f in fields): r for r in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"Duplicate result keys for {fields}")
    return keyed


def ci_rate(n, total):
    if total == 0:
        return None
    ci = binomtest(n, total).proportion_ci(confidence_level=.95, method="exact")
    return dict(hits=n, n=total, fraction=n / total, ci95=[ci.low, ci.high])


def stats(values, rng=None):
    a = np.asarray(values)
    if not len(a):
        return None
    est = dict(n=len(a), q1=float(np.quantile(a, .25)), median=float(np.median(a)),
               q3=float(np.quantile(a, .75)))
    if rng is not None:
        sample = rng.integers(len(a), size=(10000, len(a)))
        est["ci95"] = np.quantile(np.median(a[sample], axis=1), [.025, .975]).tolist()
    return est


def static_summary(data, rng):
    reference = {(r["seed"], r["users"], r["gamma"]): r for r in data if r["method"] == "reference"}
    groups = defaultdict(list)
    for row in data:
        ref = reference[row["seed"], row["users"], row["gamma"]]
        if valid(row) and valid(ref):
            row["design_gap_db"] = ref["design_streams_db"] - row["design_streams_db"]
            row["true_gap_db"] = ref["true_streams_db"] - row["true_streams_db"]
            row["paired_speedup"] = ref["seconds"] / row["seconds"]
        groups[row["users"], row["gamma"], row["method"], row["outer"], row["inner"]].append(row)

    summary = []
    for key, rows in sorted(groups.items()):
        if sorted(r["seed"] for r in rows) != list(SEEDS):
            raise ValueError(f"Incomplete group {key}")
        paired = [r for r in rows if "design_gap_db" in r]
        users, gamma, method, outer, inner = key
        hits_design = sum(r["design_gap_db"] <= QUALITY_DB for r in paired)
        hits_true = sum(r["true_gap_db"] <= TRUE_DB for r in paired)
        rec = dict(users=users, gamma=gamma, method=method, outer=outer, inner=inner,
                   n_attempted=len(rows), n_paired=len(paired),
                   feasibility=ci_rate(sum(valid(r) for r in rows), len(rows)),
                   design_hits=ci_rate(hits_design, len(rows)),
                   true_hits=ci_rate(hits_true, len(rows)),
                   design_hits_paired=ci_rate(hits_design, len(paired)),
                   true_hits_paired=ci_rate(hits_true, len(paired)))
        for field in ("seconds", "design_gap_db", "true_gap_db", "paired_speedup"):
            rec[field] = stats([r[field] for r in paired], rng)
        rec["counts"] = {field: stats([r[field] for r in paired if field in r])
                         for field in ("outer_steps", "sdp_solves", "lp_solves", "price_evals")}
        summary.append(rec)
    if len(reference) != 96 or len(groups) != 34:
        raise ValueError(f"Unexpected static coverage: {len(reference)} ref and {len(groups)} groups")
    coords = [r for r in summary if (r["users"], r["gamma"]) == (4, 8) and r["n_paired"] > 0]
    for a in coords:
        point = (a["seconds"]["median"], a["design_gap_db"]["median"])
        a["pareto"] = not any(
            b["seconds"]["median"] <= point[0] and b["design_gap_db"]["median"] <= point[1]
            and (b["seconds"]["median"], b["design_gap_db"]["median"]) != point for b in coords
        )
    return summary


def clusters(pairs, rng):
    """One resampling unit is a whole independent trajectory."""
    matrix = np.full((32, 3), np.nan)
    for (chain, frame), value in pairs.items():
        matrix[chain - 21101, frame - 1] = value
    if not pairs:
        return None
    out = stats(list(pairs.values()))
    indices = rng.integers(32, size=(10000, 32))
    samples = matrix[indices].reshape(10000, -1)
    nonempty = np.any(np.isfinite(samples), axis=1)
    out["ci95"] = np.quantile(np.nanmedian(samples[nonempty], axis=1), [.025, .975]).tolist()
    return out


def warm_summary(data, rng):
    patterns = by_key([r for r in data if r["frame"] > 0], ("rho", "method", "chain", "frame"))
    keys = [(chain, f) for chain in range(21101, 21133) for f in range(1, 4)]
    if len(patterns) != 768 or len([r for r in data if r["frame"] == 0]) != 32:
        raise ValueError("Incomplete correlated-chain cohort")
    result = []
    ratios = []
    for rho in (.9, .99):
        cohorts = {m: {k: patterns[rho, m, *k] for k in keys}
                   for m in ("reference", "cold", "dictionary", "basis")}
        for m in ("cold", "dictionary", "basis"):
            rows = cohorts[m]
            hits = {k: r["seconds"] for k, r in rows.items() if r["hit"]}
            chains_all = sum(all((c, f) in hits for f in range(1, 4)) for c in range(21101, 21133))
            row = dict(rho=rho, method=m, n_attempted=len(rows), n_hit=len(hits),
                       n_ref_feasible=sum(valid(r) for r in cohorts["reference"].values()),
                       all_three_hit=ci_rate(chains_all, 32),
                       by_frame=[ci_rate(sum(k[1] == f for k in hits), 32) for f in range(1, 4)],
                       hit_seconds=clusters(hits, rng),
                       first_simplex=stats([r["first_simplex_iterations"] for r in rows.values()
                                            if r["first_simplex_iterations"] is not None]),
                       kept_columns=stats([r["retained_columns"] for r in rows.values()]),
                       basis_accepted=sum(r["basis_status"] == 0 for r in rows.values()),
                       basis_requested=sum(r["basis_requested"] for r in rows.values()))
            result.append(row)
        for base, compare in (("cold", "dictionary"), ("cold", "basis"), ("dictionary", "basis")):
            matched = {k: cohorts[base][k]["seconds"] / cohorts[compare][k]["seconds"] for k in keys
                       if cohorts[base][k]["hit"] and cohorts[compare][k]["hit"]}
            ratios.append(dict(rho=rho, base=base, compare=compare, ratio=clusters(matched, rng)))
    return result, ratios


def scaling_summary(data, rng):
    results = []
    for size in (8, 16, 32, 64):
        for method in ("zf", "tone", "lp"):
            rows = [r for r in data if r["array"] == size and r["method"] == method]
            if len(rows) != 32 and not (method == "zf" and not rows):
                raise ValueError(f"Incomplete scaling cohort for {size}, {method}")
            solved = [r for r in rows if valid(r)]
            results.append(dict(array=size, method=method, attempted=len(rows), solved=len(solved),
                                timeouts=sum(r["status"] == "timeout" for r in rows),
                                seconds=stats([r["seconds"] for r in solved], rng)))
    return results


def ecdf(ax, values, color, label=None, total=None, linestyle="-"):
    a = np.sort(values)
    if a.size:
        y = np.arange(1, a.size + 1) / (total if total is not None else a.size)
        ax.step(np.r_[a[0], a], np.r_[0, y], where="post", lw=1.65, color=color,
                label=label, linestyle=linestyle)


def plot_core(data, target):
    fig, axes = plt.subplots(3, 3, figsize=(9.2, 6.2), layout="constrained")
    for j, (users, gamma) in enumerate(CONDITIONS):
        axes[0, j].set_title(f"{users} users, {gamma} dB")
        subset = [r for r in data if (r["users"], r["gamma"]) == (users, gamma)
                  and r["method"] == "lp" and "design_gap_db" in r]
        views = (("seconds", "Design time (s)"),
                 ("design_gap_db", "Estimated-kernel gap (dB)"),
                 ("true_gap_db", "True-SCNR gap (dB)"))
        for k, (field, label) in enumerate(views):
            ax = axes[k, j]
            for (outer, inner), color in zip(CORE, PALETTE):
                equal = [r[field] for r in subset if (r["outer"], r["inner"]) == (outer, inner)]
                ecdf(ax, equal, color, f"{outer}×{inner}")
            ax.set(xlabel=label, ylim=(0, 1.02))
            ax.grid(alpha=.2)
            if j == 0:
                ax.set_ylabel("Empirical CDF")
            if field != "seconds":
                ax.axvline(0, lw=.6, color=".5")
                ax.axvline(QUALITY_DB if field == "design_gap_db" else TRUE_DB,
                           lw=.8, linestyle=":", color=".3")
    axes[0, 2].legend(loc="lower right", fontsize=8)
    fig.savefig(target)
    plt.close(fig)


def plot_pareto(summary, target):
    fig, ax = plt.subplots(figsize=(7, 4.3), layout="constrained")
    groups = [r for r in summary if (r["users"], r["gamma"]) == (4, 8)]
    for method, color in zip(METHODS, PALETTE):
        values = {(r["outer"], r["inner"]): r for r in groups
                  if r["method"] == method and r["n_paired"] > 0}
        coords = [(r["seconds"]["median"], r["design_gap_db"]["median"])
                  for b in BUDGETS if (r := values.get(b))]
        ax.plot(*zip(*coords), label=NAMES[method], linewidth=1.25, color=color)
        for b, marker in zip(BUDGETS, BUDGET_MARKERS):
            if r := values.get(b):
                ax.scatter(r["seconds"]["median"], r["design_gap_db"]["median"], marker=marker, s=30,
                           facecolor=color if r["pareto"] else "white", edgecolor=color, zorder=3)
    ref = next(r for r in groups if r["method"] == "reference")
    ax.scatter(ref["seconds"]["median"], 0., marker="*", color="black", s=65, label="Uncapped AO reference")
    ax.axhline(QUALITY_DB, lw=.85, linestyle=":", color=".35")
    ax.axhline(0, lw=.6, color=".5")
    ax.set(xscale="log", xlabel="Median design wall time (s)", ylabel="Median paired design gap to AO (dB)")
    ax.grid(alpha=.2)
    legend = ax.legend(loc="upper right", fontsize=8)
    ax.add_artist(legend)
    handles = [plt.Line2D([], [], marker=m, color=".4", markersize=4, linestyle="none", label=f"{o}×{i}")
               for (o, i), m in zip(BUDGETS, BUDGET_MARKERS)]
    ax.legend(handles=handles, loc="lower left", ncol=len(handles), fontsize=7.3)
    fig.savefig(target)
    plt.close(fig)


def plot_tracking(warm, sizes, target):
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.6), layout="constrained")
    for mode, color in zip(("cold", "dictionary", "basis"), PALETTE):
        for rho, ls in ((.9, "--"), (.99, "-")):
            rows = [r for r in warm if r["rho"] == rho and r["method"] == mode]
            ecdf(left, [r["seconds"] for r in rows if r["hit"]], color,
                 label=NAMES[mode] if rho == .99 else None, total=len(rows), linestyle=ls)
    left.set(xlabel=f"Time to {QUALITY_DB:.3f} dB design gap (s)", ylabel="Fraction of all scored frames",
             ylim=(0, 1.02), title="Correlated-channel tracking")
    handles, labels = left.get_legend_handles_labels()
    for ls in ("--", "-"):
        handles.append(plt.Line2D([], [], color=".45", linestyle=ls))
    left.legend(handles, [*labels, "ρ = 0.90", "ρ = 0.99"], fontsize=7.1, loc="lower right")
    for mode, label, color in (("zf", "Joint SDP", PALETTE[0]), ("tone", "Per-tone SDP", PALETTE[1]),
                               ("lp", "Priced LP", PALETTE[-1])):
        points = [r for r in sizes if r["method"] == mode and r["seconds"]]
        med = np.array([r["seconds"]["median"] for r in points])
        q1 = np.array([r["seconds"]["q1"] for r in points])
        q3 = np.array([r["seconds"]["q3"] for r in points])
        right.errorbar([r["array"] for r in points], med, yerr=[med - q1, q3 - med], color=color,
                       marker="o", linewidth=1.3, capsize=2.5, label=label)
    right.set(xscale="log", yscale="log", xlabel="Antennas per array", ylabel="Design time (s)",
              xticks=[8, 16, 32, 64], xticklabels=["8", "16", "32", "64"], title="Array scaling, 1×1 cap")
    right.xaxis.set_minor_locator(NullLocator())
    right.xaxis.set_minor_formatter(NullFormatter())
    right.legend(fontsize=8)
    for ax in (left, right):
        ax.grid(alpha=.23)
    fig.savefig(target)
    plt.close(fig)


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("x", newline="") as stream:
        out = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        out.writeheader()
        out.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    options = p.parse_args()
    static = [r for path in sorted(options.source.glob("static-*/designs.jsonl")) for r in read(path)
              if not r["warmup"]]
    warm = [r for r in read(options.source / "warm/designs.jsonl") if not r["warmup"]]
    sized = [r for r in read(options.source / "scaling/designs.jsonl") if not r["warmup"]]
    rng = np.random.default_rng(495001)
    stats_s = static_summary(static, rng)
    stats_w, ratios = warm_summary(warm, rng)
    stats_a = scaling_summary(sized, rng)
    options.destination.mkdir(parents=True, exist_ok=False)
    with (options.destination / "summary.json").open("x") as log:
        json.dump(dict(static=stats_s, warm=stats_w, warm_ratios=ratios, scaling=stats_a), log,
                  indent=2, allow_nan=False)
        log.write("\n")
    write_csv(options.destination / "static-designs.csv", static)
    write_csv(options.destination / "correlated-frames.csv", warm)
    write_csv(options.destination / "scaling-designs.csv", sized)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    plot_core(static, options.destination / "core-ecdf.pdf")
    plot_pareto(stats_s, options.destination / "pareto.pdf")
    plot_tracking(warm, stats_a, options.destination / "warm-start-scaling.pdf")


if __name__ == "__main__":
    main()
