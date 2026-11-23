"""Aggregate Part E teacher coverage, original-checkpoint status and paired endpoints."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from parte_evaluate import ARMS, BUDGETS
from parte_teacher import HELD_OUT, ORIGINAL, TRAIN

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def paired(values, rng):
    a = np.asarray(values)
    if not a.size:
        return None
    b = rng.integers(len(a), size=(10000, len(a)))
    lo, hi = np.quantile(np.median(a[b], axis=1), [.025, .975])
    q1, med, q3 = np.quantile(a, [.25, .5, .75])
    return dict(n=len(a), median=float(med), ci95=[float(lo), float(hi)],
                q1=float(q1), q3=float(q3), minimum=float(a.min()), maximum=float(a.max()))


def summarize_teacher(rows, label_hash):
    if len(rows) != len(TRAIN) or {r["scene_id"] for r in rows} != set(TRAIN):
        raise ValueError("The teacher attempt IDs do not match the preregistration")
    good = [r for r in rows if r["status"] == "converged"]
    if {r["scene_id"] for r in good} != {r["scene_id"] for r in label_hash}:
        raise ValueError("The weights used a different set of teacher labels")
    return dict(attempted=len(rows), converged=len(good),
                failures=[r for r in rows if r["status"] != "converged"],
                cold_rechecks=sum(bool(r["failures"]) for r in rows),
                ipm_retries=sum(s["ipm_retries"] for r in good for s in r["rows"]),
                dense_price_fallbacks=sum(s["dense_prices"] for r in good for s in r["rows"]),
                max_covariance_gap_db=max(r["rows"][-1]["gap_db"] for r in good),
                max_outer_upper_gain_db=max(r["rows"][-1]["outer_upper_gain_db"] for r in good),
                max_witness_mix=max(s["max_witness_mix"] for r in good for s in r["rows"]),
                lp_solves=sum(r["lp_solves"] for r in good),
                lp_solves_median=float(np.median([r["lp_solves"] for r in good])),
                outer_steps_median=float(np.median([r["outer_steps"] for r in good])),
                max_outer_steps=max(r["outer_steps"] for r in good),
                teacher_u_angle_deg=paired([r["teacher_deployment_u_angle_deg"] for r in label_hash],
                                           np.random.default_rng(71008)))


def compare(eval_rows):
    grouped = defaultdict(dict)
    arms = (*ARMS, "spsa") if any(r["arm"] == "spsa" for r in eval_rows) else ARMS
    expected = {(i, o, k) for i in (*HELD_OUT, *ORIGINAL) for o, k in BUDGETS}
    for row in eval_rows:
        coord = row["scene_id"], row["outer"], row["inner"]
        if row["arm"] in grouped[coord]:
            raise ValueError(f"Duplicate evaluation endpoint {coord} {row['arm']}")
        grouped[coord][row["arm"]] = row
    if set(grouped) != expected or any(set(v) != set(arms) for v in grouped.values()):
        raise ValueError("Missing matched evaluation rows")
    for (sid, o, k), arm_results in grouped.items():
        solves = o*k
        solved = [r for r in arm_results.values() if r["status"] == "solved"]
        for row in solved:
            per_tone = 3 * row["users"] + 2
            added = 32 * (row["users"] + 1) * (solves - 1)
            if (row["atoms_added"], row["atoms"], row["lp_solves"], row["pricing_passes"]) != (
                    added, 32 * per_tone + added, solves, solves):
                raise ValueError(f"Unmatched compute/atom budgets at {(sid, o, k)}")
        if solves == 1 and len(solved) > 1:
            for score in ("design_streams_db", "true_streams_db"):
                if np.ptp([r[score] for r in solved]) > 1e-12:
                    raise ValueError("The 1×1 no-proposal scores differ")
    rng = np.random.default_rng(71007)
    results, deltas = [], []
    for cohort, identifiers in (("held-out maps", HELD_OUT), ("original benchmark", ORIGINAL)):
        for outer, inner in BUDGETS:
            for arm in arms:
                pairs = []
                coverage = 0
                for seed in identifiers:
                    bucket = grouped[seed, outer, inner]
                    r, p = bucket[arm], bucket["physics"]
                    coverage += bool(r.get("feasible", False))
                    if not (r.get("feasible", False) and p.get("feasible", False)):
                        continue
                    denom = r["first_user_proposals"]
                    d = dict(scene_id=seed, cohort=cohort, outer=outer, inner=inner, arm=arm,
                             design_delta_db=r["design_streams_db"] - p["design_streams_db"],
                             true_delta_db=r["true_streams_db"] - p["true_streams_db"],
                             elapsed_delta_ms=1000 * (r["seconds"] - p["seconds"]),
                             time_ratio=r["seconds"] / p["seconds"],
                             seconds=r["seconds"], policy_ms=r["inference_seconds"] * 1000,
                             ever_active_share=r["first_user_proposals_ever_active"] / denom if denom else None,
                             first_user_terminal_watts=r["first_user_proposals_terminal_watts"])
                    pairs.append(d)
                    deltas.append(d)
                s = dict(cohort=cohort, outer=outer, inner=inner, arm=arm, feasible=coverage, total=len(identifiers))
                for f in ("design_delta_db", "true_delta_db", "elapsed_delta_ms", "time_ratio", "seconds",
                          "policy_ms", "ever_active_share", "first_user_terminal_watts"):
                    s[f] = paired([r[f] for r in pairs if r[f] is not None], rng)
                if all(r["design_delta_db"] == r["true_delta_db"] for r in pairs):
                    s["true_delta_db"] = s["design_delta_db"]
                results.append(s)
    direct = []
    for cohort, identifiers in (("held-out maps", HELD_OUT), ("original benchmark", ORIGINAL)):
        for outer, inner in BUDGETS:
            pairs = [(grouped[seed, outer, inner]["envelope"], grouped[seed, outer, inner]["reduced-cost"])
                     for seed in identifiers]
            pairs = [(e, r) for e, r in pairs if e.get("feasible") and r.get("feasible")]
            design = [e["design_streams_db"] - r["design_streams_db"] for e, r in pairs]
            true = [e["true_streams_db"] - r["true_streams_db"] for e, r in pairs]
            design_stats, true_stats = paired(design, rng), paired(true, rng)
            direct.append(dict(cohort=cohort, outer=outer, inner=inner, design_delta_db=design_stats,
                               true_delta_db=design_stats if design == true else true_stats))
    return results, deltas, direct


def table(rows, direct, path):
    def text(s, digits=3):
        if s is None:
            return "—"
        med, lo, hi = (f"{v:.2e}" if 0 < abs(v) < .5 * 10**-digits else f"{v:.{digits}f}"
                       for v in (s["median"], *s["ci95"]))
        return f"{med} [{lo}, {hi}]"
    lines = ["# Part E paired evaluation", "",
             "Effects are method minus physics, paired by scene. Brackets are 95% paired bootstrap intervals.", "",
             "| Cohort | Budget | Arm | Feasible | Design SCNR (dB) | True SCNR (dB) | Time ratio | Time (ms) |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        s = r["seconds"]
        median_time = 1e3*s["median"] if s else float("nan")
        lines.append("| " + " | ".join(map(str, (r["cohort"], f"{r['outer']}×{r['inner']}", r["arm"],
                    f"{r['feasible']}/{r['total']}", text(r["design_delta_db"]), text(r["true_delta_db"]),
                    text(r["time_ratio"]), f"{median_time:.2f}"))) + " |")
    lines += ["", "## Envelope supervision relative to reduced cost alone", "",
              "| Cohort | Budget | Design SCNR (dB) | True SCNR (dB) |",
              "| --- | --- | ---: | ---: |"]
    for r in direct:
        cells = r["cohort"], f"{r['outer']}×{r['inner']}", text(r["design_delta_db"]), text(r["true_delta_db"])
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def plot(results, path):
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.1), layout="constrained")
    markers = {"random": ("o", "#3675a5"), "gradient": ("s", "#b46d2d"),
               "reduced-cost": ("^", "#2b8165"), "envelope": ("D", "#a54665"), "spsa": ("P", "#6e5294")}
    for row, cohort in zip(axes, ("held-out maps", "original benchmark")):
        for ax, budget in zip(row, ((2, 1), (3, 2))):
            ax.axvline(1., color="0.66", linewidth=.65)
            ax.axhline(0., color="0.66", linewidth=.65)
            ax.scatter(1., 0., marker="+", color="black", s=24, label="Physics pricing")
            for r in results:
                if (r["cohort"] != cohort or (r["outer"], r["inner"]) != budget
                        or r["arm"] == "physics" or r["time_ratio"] is None):
                    continue
                x, y = r["time_ratio"], r["design_delta_db"]
                marker, color = markers[r["arm"]]
                ax.errorbar(x["median"], y["median"],
                            xerr=[[x["median"]-x["ci95"][0]], [x["ci95"][1]-x["median"]]],
                            yerr=[[y["median"]-y["ci95"][0]], [y["ci95"][1]-y["median"]]],
                            marker=marker, color=color, linewidth=.85, capsize=2, linestyle="none",
                            label=r["arm"].replace("-", " ").capitalize())
            ax.set_title(f"{cohort.capitalize()} · {budget[0]}×{budget[1]}")
            ax.grid(alpha=.20)
        row[0].set_ylabel("Paired design SCNR change (dB)")
    for ax in axes[-1]:
        ax.set_xlabel("Paired design time ratio to physics")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(handles), fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def write_csv(path, rows):
    with path.open("x", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    roots = args.results
    teacher = [row for p in sorted(roots.glob("teacher-*/teacher.jsonl")) for row in jsonl(p)]
    eval_rows = [row for p in sorted(roots.glob("evaluation-*/designs.jsonl")) for row in jsonl(p)]
    labels = jsonl(roots / "training/training-labels.jsonl")
    audit = summarize_teacher(teacher, labels)
    result, delta, direct = compare(eval_rows)
    archive = json.loads(sorted(roots.glob("evaluation-*/archive.json"))[0].read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "paired-results.csv", delta)
    write_csv(args.output / "designs.csv", eval_rows)
    write_csv(args.output / "teacher-coverage.csv", [
        {k: v for k, v in r.items() if k not in ("rows", "choi_check")} for r in teacher])
    (args.output / "summary.json").write_text(json.dumps(
        dict(teacher=audit, archive=archive, evaluation=result, envelope_vs_reduced=direct),
        indent=2, allow_nan=False) + "\n")
    table(result, direct, args.output / "tables.md")
    plot(result, args.output / "proposal-ablation.pdf")


if __name__ == "__main__":
    main()
