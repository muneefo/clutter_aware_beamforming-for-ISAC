"""Summarize attained Part A runs, preserving coverage and feasibility."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

METRICS = ("design_cov_db", "true_cov_db", "design_streams_db", "true_streams_db")
ARMS = ("base", "cap_5x3", "outer_0.1", "inner_0.01", "redraw_distances",
        "range_100_500", "exponent_3.1", "power_cap_199.53")


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summary(values, rng=None):
    if not values:
        return None
    array = np.array(values)
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    answer = dict(n=len(array), q1=float(q1), median=float(median), q3=float(q3))
    if rng is not None:
        sample = rng.integers(len(array), size=(10_000, len(array)))
        answer["med_ci95"] = np.quantile(np.median(array[sample], axis=1), [0.025, 0.975]).tolist()
    return answer


def mean_linear_db(values):
    return float(10 * np.log10(np.mean(10 ** (np.array(values) / 10)))) if values else None


def bootstrap_mean(values, rng):
    if not values:
        return None
    values = np.array(values)
    sample = rng.integers(len(values), size=(10_000, len(values)))
    return np.quantile(values[sample].mean(axis=1), [0.025, 0.975]).tolist()


def csi_report(rows):
    rng = np.random.default_rng(390001)
    results = []
    for gamma in (8, 12):
        threshold = [r for r in rows if r["gamma_db"] == gamma]
        zero = {r["seed"]: r for r in threshold if r["delta_margin_db"] == 0 and r["status"] == "solved"}
        for setting in ("legacy", 0, 0.25, 0.5, 1):
            attempted = [r for r in threshold if r["delta_margin_db"] == setting]
            solved = [r for r in attempted if r["status"] == "solved"]
            feasible = [r for r in solved if r["physically_feasible"]]
            contrast = [zero[r["seed"]]["true_db"] - r["true_db"] for r in feasible
                        if r["seed"] in zero and zero[r["seed"]]["physically_feasible"]]
            answer = dict(gamma_db=gamma, margin_setting=setting, attempted=len(attempted),
                          nominal_feasible=len(feasible), achieved=summary([r["min_sinr_margin_db"] for r in solved]),
                          true_cost_db_vs_no_tightening=summary(contrast, rng))
            for i, level in enumerate((0.01, 0.03, 0.05)):
                conditional = [r[f"feasible_csi_{i}"] / 100 for r in solved]
                stable = sum(v == 1 for v in conditional)
                ci = binomtest(stable, len(attempted)).proportion_ci() if attempted else None
                answer[f"relative_error_{level}"] = dict(
                    mean_over_designs=sum(conditional) / len(attempted) if attempted else None,
                    mean_ci95=bootstrap_mean(conditional + [0] * (len(attempted) - len(solved)), rng),
                    per_design_fraction=summary(conditional, rng), feasible_in_all_100=stable,
                    all_100_exact_ci95=list(ci) if ci else None)
            results.append(answer)
    return results


def ao_report(rows, curves, lp_rows):
    rng = np.random.default_rng(390001)
    conditions = []
    for gamma in (8, 12):
        baseline = {r["seed"]: r for r in rows if r["gamma_db"] == gamma and
                    r["start_index"] == 0 and r["arm"] == "base" and r["status"] == "solved"}
        curve = float(curves[4, "estimated", gamma]["digitized_scnr_db"])
        for arm in ARMS:
            attempted = [r for r in rows if r["gamma_db"] == gamma and r["arm"] == arm and r["start_index"] == 0]
            solved = [r for r in attempted if r["status"] == "solved"]
            matched = [r for r in solved if r["seed"] in baseline]
            true_cov = mean_linear_db([r["true_cov_db"] for r in solved])
            report = dict(gamma_db=gamma, arm=arm, attempted=len(attempted), solved=len(solved),
                          nominal_feasible=sum(r["physically_feasible"] for r in solved),
                          digitized_true_db=curve, true_cov_linear_mean_db=true_cov,
                          digitized_minus_replay_db=(curve - true_cov if true_cov is not None else None),
                          design_cov_summary=summary([r["design_cov_db"] for r in solved], rng),
                          true_cov_change_vs_baseline_db=summary(
                              [r["true_cov_db"] - baseline[r["seed"]]["true_cov_db"] for r in matched], rng))
            if arm == "base":
                paired_lp = [
                    (baseline[r["seed"]], r) for r in lp_rows
                    if r["gamma_db"] == gamma and r["seed"] in baseline
                    and r["delta_margin_db"] == "legacy" and r.get("physically_feasible")
                    and baseline[r["seed"]]["physically_feasible"]
                ]
                report.update(
                    full_covariance_reconstruction_change_db=summary(
                        [r["true_full_db"] - r["true_cov_db"] for r in solved], rng),
                    truncation_change_db=summary([r["true_streams_db"] - r["true_full_db"] for r in solved], rng),
                    material_truncation_true=sum(
                        abs(r["true_streams_db"] - r["true_full_db"]) >= 0.001 for r in solved),
                    material_truncation_design=sum(
                        abs(r["design_streams_db"] - r["design_full_db"]) >= 0.001 for r in solved),
                    designs_with_rank_excess=sum(r["rank_excess_tones"] > 0 for r in solved),
                    target_coefficient_only_true_cov_mean_db=mean_linear_db(
                        [r["target_coefficient_only_true_cov_db"] for r in solved]),
                    paired_true_stream_gap_db=summary(
                        [a["true_streams_db"] - b["true_db"] for a, b in paired_lp], rng),
                    paired_design_cov_to_lp_stream_gap_db=summary(
                        [a["design_cov_db"] - b["design_db"] for a, b in paired_lp], rng),
                    paired_design_stream_gap_db=summary(
                        [a["design_streams_db"] - b["design_db"] for a, b in paired_lp], rng),
                )
            conditions.append(report)
    spread = {}
    for seed in (13001, 13002, 13003):
        attempted = [r for r in rows if r["seed"] == seed and r["gamma_db"] == 8 and r["arm"] == "base"]
        solved = [r for r in attempted if r["status"] == "solved"]
        spread[seed] = dict(attempted=len(attempted), solved=len(solved))
        if len(solved) == 20:
            spread[seed].update({metric: float(np.ptp([r[metric] for r in solved])) for metric in METRICS})
    threshold = {}
    if all(r["solved"] == 20 for r in spread.values()):
        for metric in METRICS:
            threshold[metric] = max(0.1, 2 * float(np.median([r[metric] for r in spread.values()])))
    return {"conditions": conditions, "ao_spread": spread, "quality_threshold_db": threshold}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--design-csv", type=Path)
    args = parser.parse_args()
    ao = [j["design"] for path in sorted(args.data.glob("ao-*/ao.jsonl")) for j in jsonl(path)]
    csi = [row for path in sorted(args.data.glob("csi*/csi-designs.jsonl")) for row in jsonl(path)]
    with (args.data / "fig12b-digitized.csv").open(newline="") as stream:
        curves = {(int(r["users"]), r["design_kernel"], int(r["gamma_db"])): r for r in csv.DictReader(stream)}
    keys = {(r["seed"], r["gamma_db"], r["arm"], r["start_index"]) for r in ao}
    if len(keys) != len(ao):
        raise ValueError("Duplicate AO jobs in results")
    if args.design_csv:
        columns = list(dict.fromkeys(field for row in ao for field in row))
        with args.design_csv.open("x", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(ao)
    with args.output.open("x") as out:
        json.dump(dict(csi=csi_report(csi), ao=ao_report(ao, curves, csi)), out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
