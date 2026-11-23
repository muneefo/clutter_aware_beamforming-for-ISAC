"""Summarize fixed-receiver certification, power use and archived paired controls."""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from partc_run import CONDITIONS, CORE, SEEDS


def read_jsonl(path):
    with path.open() as source:
        return [json.loads(l) for l in source]


def stats(values, rng):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return None
    bootstrap = rng.integers(a.size, size=(10000, a.size))
    q1, med, q3 = np.quantile(a, [.25, .5, .75]).tolist()
    return dict(n=a.size, min=float(a.min()), max=float(a.max()), q1=q1, median=med, q3=q3,
                ci95=np.quantile(np.median(a[bootstrap], axis=1), [.025, .975]).tolist())


def key(r):
    return tuple(r[k] for k in ("seed", "users", "gamma", "outer", "inner"))


def verify_atoms(source, entry):
    by_run = defaultdict(lambda: dict(added=0, active=0, ever=0, total=0., priced=0., rows=0))
    for path in sorted(source.glob("cert-*/terminal-atoms.csv.gz")):
        with gzip.open(path, "rt", newline="") as raw:
            for row in csv.DictReader(raw):
                sample = by_run[tuple(int(row[k]) for k in ("seed", "users", "gamma", "outer", "inner"))]
                p = float(row["power_w"])
                sample["total"] += p
                sample["rows"] += 1
                if row["added_outer"] != "0":
                    sample["added"] += 1
                    sample["priced"] += p
                    sample["active"] += row["active"] == "True"
                    sample["ever"] += row["ever_active"] == "True"
    for r in entry:
        if r["status"] == "error":
            continue
        s = by_run[key(r)]
        if (s["added"], s["active"], s["ever"]) != (r["added_atoms"], r["priced_active"], r["priced_ever_active"]):
            raise ValueError(f"Atom provenance totals differ at {key(r)}")
        if not all(np.isclose(s[a], r[b], atol=3e-10, rtol=0)
                   for a, b in (("total", "total_power_w"), ("priced", "priced_power_w"))):
            raise ValueError(f"Decoded atom powers differ at {key(r)}")


def cohort(rows, rng):
    output = []
    for users, gamma in CONDITIONS:
        for outer, inner in CORE:
            subset = [r for r in rows if (r["users"], r["gamma"], r["outer"], r["inner"]) == (users, gamma, outer, inner)]
            converged = [r for r in subset if r["status"] == "converged"]
            active = [r for r in subset if r.get("added_atoms", 0) > 0]
            ratio = [r["nominal_bound_db"] / r["observed_gap_db"] for r in converged
                     if r["observed_gap_db"] >= .001]
            table = dict(users=users, gamma=gamma, outer=outer, inner=inner,
                         attempted=len(subset), converged=len(converged),
                         nominal_violations=sum(r["nominal_violation"] for r in converged),
                         audited_violations=sum(r["audited_violation"] for r in converged),
                         nominal_below_reference_upper=sum(r["nominal_bound_db"] + 1e-7 < r["reference_gap_upper_db"]
                                                           for r in converged),
                         reference_solves=stats([r["reference_solves"] for r in converged], rng),
                         passes=stats([r["pricing_rounds"] for r in converged], rng),
                         terminal_candidates=stats([r["terminal_candidates"] for r in converged], rng),
                         initial_power_share=stats([r["initial_power_w"] / r["total_power_w"] for r in converged], rng),
                         bound_ratio=stats(ratio, rng),
                         slack_db=stats([r["nominal_bound_db"] - r["observed_gap_db"] for r in converged], rng),
                         terminal_active_fraction=stats([r["priced_active"] / r["added_atoms"] for r in active], rng),
                         ever_active_fraction=stats([r["priced_ever_active"] / r["added_atoms"] for r in active], rng),
                         pooled_terminal_active=(sum(r["priced_active"] for r in active) / sum(r["added_atoms"] for r in active)
                                                 if active else None),
                         pooled_ever_active=(sum(r["priced_ever_active"] for r in active) / sum(r["added_atoms"] for r in active)
                                             if active else None))
            for k in ("nominal_bound_db", "observed_gap_db", "reference_residual_db", "reference_gap_upper_db",
                      "t", "added_atoms", "priced_power_share", "fixed_to_priced_db", "random_to_priced_db"):
                table[k] = stats([r[k] for r in converged], rng)
            output.append(table)
    return output


def historical(part_b, rng):
    rows = [r["design"] for r in read_jsonl(part_b / "t1/designs.jsonl") if not r["design"]["warmup"]]
    schedules = {(r["seed"], r["outer"], r["inner"]): r["labels"]
                 for r in read_jsonl(part_b / "t1/schedules.jsonl")}
    stats_b, pairs = [], []
    for cap in ((3, 2), (5, 3)):
        found = {(r["seed"], r["arm"]): r for r in rows
                 if (r["outer"], r["inner"]) == cap and r["arm"] in ("B6", "B7", "B8")}
        for arm in ("B6", "B7"):
            paired = []
            for seed in range(16001, 16033):
                a, b = found[seed, arm], found[seed, "B8"]
                if not a["feasible"] or not b["feasible"]:
                    continue
                plan = schedules[(seed, *cap)]
                planned = sum(len(owners) for p in plan for owners in p)
                if len(plan) != cap[0] * cap[1] - 1 or b["atoms"] != 448 + planned:
                    raise ValueError("Saved B8 schedule does not reproduce the final atom count")
                if a["atoms"] != (b["atoms"] if arm == "B7" else 448):
                    raise ValueError("Historical comparison is not matched by the declared bank rule")
                r = dict(seed=seed, baseline=arm, outer=cap[0], inner=cap[1],
                         baseline_atoms=a["atoms"], priced_atoms=b["atoms"],
                         design_gain_db=b["design_streams_db"] - a["design_streams_db"],
                         true_gain_db=b["true_streams_db"] - a["true_streams_db"])
                pairs.append(r)
                paired.append(r)
            stats_b.append(dict(baseline=arm, outer=cap[0], inner=cap[1], n=len(paired),
                                design_gain_db=stats([r["design_gain_db"] for r in paired], rng),
                                true_gain_db=stats([r["true_gain_db"] for r in paired], rng)))
    return stats_b, pairs


def draw_plot(rows, destination):
    r = [r for r in rows if r["status"] == "converged"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
    for (o, i), marker, color in zip(CORE, ["o", "s", "^"], ["#365674", "#bd7028", "#28766d"]):
        points = [p for p in r if (p["outer"], p["inner"]) == (o, i)]
        ax.scatter([p["observed_gap_db"] for p in points], [p["nominal_bound_db"] for p in points],
                   color=color, s=16, alpha=.77, marker=marker, label=f"{o}×{i}")
    maximum = max(p["observed_gap_db"] for p in r)
    grid = np.geomspace(1e-5, 1.15 * maximum, 180)
    ax.plot(grid, grid, linestyle="--", color=".4", linewidth=.9, label="Bound = observed gap")
    ax.set(xlabel="Observed fixed-receiver covariance gap (dB)", ylabel="Certified bound (dB)")
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlim(min(-1e-5, min(p["observed_gap_db"] for p in r)), 1.2 * maximum)
    ax.grid(alpha=.22)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(destination)
    plt.close(fig)


def write_csv(path, entries):
    fields = list(dict.fromkeys(k for row in entries for k in row))
    with path.open("x", newline="") as stream:
        out = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        out.writeheader()
        out.writerows(entries)


def interval(s, decimals=4, factor=1.):
    if not s:
        return "—"
    lo, hi = s["ci95"]
    return f"{factor*s['median']:.{decimals}f} [{factor*lo:.{decimals}f}, {factor*hi:.{decimals}f}]"


def table(out, cols, rows):
    for row in (cols, ["---"] * len(cols), *rows):
        print("| " + " | ".join(map(str, row)) + " |", file=out)


def tables(out, summary, ablation):
    print("# Part D tables\n", file=out)
    print("All brackets are 95% percentile bootstrap intervals for the median. The ratio column "
          "uses observed gaps ≥0.001 dB and gives the eligible count. Power fractions use the "
          "registered positivity threshold for activity and full decoded watts for power shares.\n", file=out)
    table(out, ("Users", "SINR", "Budget", "References", "Bound (dB)", "Observed gap (dB)",
                "Bound / gap", "Ratio pairs", "Violations"), [
        (r["users"], r["gamma"], f"{r['outer']}×{r['inner']}", f"{r['converged']}/{r['attempted']}",
         interval(r["nominal_bound_db"]), interval(r["observed_gap_db"]),
         interval(r["bound_ratio"], 1), (r["bound_ratio"] or {}).get("n", 0), r["nominal_violations"])
        for r in summary
    ])
    print("\n## Atom use and fixed-receiver controls\n", file=out)
    table(out, ("Users", "SINR", "Budget", "Pricing passes", "Added atoms", "Terminal active (%)",
                "Ever active (%)", "Priced power share (%)", "Initial power share (%)",
                "Gain over fixed (dB)", "Gain over random (dB)"), [
        (r["users"], r["gamma"], f"{r['outer']}×{r['inner']}", int(r["passes"]["median"]),
         interval(r["added_atoms"], 0), interval(r["terminal_active_fraction"], 1, 100),
         interval(r["ever_active_fraction"], 1, 100), interval(r["priced_power_share"], 3, 100),
         interval(r["initial_power_share"], 3, 100), interval(r["fixed_to_priced_db"], 3),
         interval(r["random_to_priced_db"], 3))
        for r in summary
    ])
    print("\n## Archived Part B end-to-end controls\n", file=out)
    table(out, ("Baseline", "Budget", "Draw pairs", "Priced design gain (dB)", "Priced true gain (dB)"), [
        (r["baseline"], f"{r['outer']}×{r['inner']}", r["n"], interval(r["design_gain_db"], 3),
         interval(r["true_gain_db"], 3)) for r in ablation
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("--part-b", type=Path, default=Path("results/partb-2026-11-22"))
    args = p.parse_args()
    raw = [r for s in sorted(args.source.glob("cert-*/bounds.jsonl")) for r in read_jsonl(s)]
    rows = [r["result"] for r in raw]
    if len({key(r) for r in rows}) != len(rows) or len(rows) != len(SEEDS) * len(CONDITIONS) * len(CORE):
        raise ValueError("The registered Part D set is incomplete or has duplicates")
    verify_atoms(args.source, rows)
    rechecked = [r for s in sorted(args.source.glob("recheck-*/restricted.jsonl")) for r in read_jsonl(s)]
    if len(rechecked) != len(rows) or {key(r) for r in rechecked} != {key(r) for r in rows}:
        raise ValueError("The independent terminal LP cross-check is incomplete")
    rng = np.random.default_rng(596001)
    summary = cohort(rows, rng)
    ablation, pairs = historical(args.part_b, rng)
    solved = [r for r in rows if r["status"] == "converged"]
    audit = dict(attempted=len(rows), converged=len(solved), feasible_returned=sum(r.get("feasible", False) for r in rows),
                 violations_nominal=sum(r["nominal_violation"] for r in solved),
                 violations_audited=sum(r["audited_violation"] for r in solved),
                 maximum_reference_residual_db=max(r["reference_residual_db"] for r in solved),
                 max_psi_dual_disagreement=max(abs(r["psi"] - r["psi_dual"]) for r in solved),
                 max_numerical_correction_db=max(abs(r["audited_bound_db"] - r["nominal_bound_db"]) for r in solved),
                 min_clutter_eigenvalue=min(r["min_clutter_eigenvalue"] for r in solved),
                 **{f"max_{k}": max(r[k] for r in solved) for k in (
                     "eq_violation", "inequality_violation", "variable_violation", "dual_column_violation",
                     "decoded_relative_qos_shortfall", "decoded_power_violation", "witness_mix",
                     "scale_rc_violation", "multiplier_clip")})
    cross_check = dict(rechecked=len(rechecked), nominal_violations=sum(r["nominal_violation"] for r in rechecked),
                       max_restricted_bracket_db=max(r["bracket_db"] for r in rechecked),
                       max_raw_objective_disagreement_db=max(abs(r["raw_minus_ipm_db"]) for r in rechecked),
                       max_decoded_relative_qos_shortfall=max(r["decoded_relative_qos_shortfall"] for r in rechecked),
                       max_gap_interval_width_db=max(r["gap_upper_db"] - r["gap_lower_db"] for r in rechecked))
    args.destination.mkdir(parents=True, exist_ok=False)
    write_csv(args.destination / "terminal.csv", rows)
    write_csv(args.destination / "partb-paired.csv", pairs)
    write_csv(args.destination / "restricted-recheck.csv", rechecked)
    with (args.destination / "summary.json").open("x") as out:
        json.dump(dict(audit=audit, cross_check=cross_check, cohorts=summary, archived_ablation=ablation),
                  out, indent=2, allow_nan=False)
        out.write("\n")
    with (args.destination / "tables.md").open("x") as out:
        tables(out, summary, ablation)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    draw_plot(rows, args.destination / "certificate-vs-gap.pdf")


if __name__ == "__main__":
    main()
