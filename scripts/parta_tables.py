"""Render the Part A reconciliation, spread and robustness tables from summary.json."""
import argparse
import json
from pathlib import Path


def table(out, headings, rows):
    def emit(values):
        print("| " + " | ".join(map(str, values)) + " |", file=out)

    emit(headings)
    emit(["---"] * len(headings))
    for row in rows:
        emit(row)


def interval(stat, precision=3):
    if stat is None:
        return "—"
    lo, hi = stat["med_ci95"]
    return f"{stat['median']:.{precision}f} [{lo:.{precision}f}, {hi:.{precision}f}]"


def reconciliation(out, result):
    ao = result["ao"]
    a = {(row["gamma_db"], row["arm"]): row for row in ao["conditions"]}
    names = {
        "base": "Python AO, pinned settings",
        "cap_5x3": "AO caps 5×3",
        "outer_0.1": "Outer tolerance 0.1 dB",
        "inner_0.01": "Inner relative tolerance 0.01",
        "redraw_distances": "Resampled distances",
        "range_100_500": "Distance range 100–500 m",
        "exponent_3.1": "Path-loss exponent 3.1",
        "power_cap_199.53": "Power cap 199.526 W",
    }
    rows = [["Digitized Fig. 12(b), estimated K=4",
             f"{a[8, 'base']['digitized_true_db']:.2f}", "—",
             f"{a[12, 'base']['digitized_true_db']:.2f}", "—"]]
    for arm, name in names.items():
        line = [name]
        for gamma in (8, 12):
            case = a[gamma, arm]
            mean = case["true_cov_linear_mean_db"]
            line.extend((f"{mean:.3f} ({case['solved']}/{case['attempted']})" if mean is not None else "—",
                         interval(case["true_cov_change_vs_baseline_db"])))
        rows.append(line)
    target_row = ["Target coefficient only, ×1.995262"]
    for gamma in (8, 12):
        case = a[gamma, "base"]
        value = case.get("target_coefficient_only_true_cov_mean_db")
        target_row.extend((f"{value:.3f}" if value is not None else "—", "+3.000 by normalization"))
    rows.append(target_row)
    print("## A1 · Reconciliation", file=out)
    print("\nTrue covariance SCNR uses the ideal kernel and the returned receiver. "
          "The run averages in linear units before converting to dB. "
          "Parentheses give successfully solved draws out of attempts. "
          "Changes are paired medians in dB with 95% bootstrap intervals.\n", file=out)
    table(out, ("Factor", "8 dB mean SCNR", "8 dB paired change", "12 dB mean SCNR",
                "12 dB paired change"), rows)


def initialization(out, result):
    print("\n## A2 · Initialization sensitivity\n", file=out)
    ao = result["ao"]
    fields = ("design_cov_db", "true_streams_db")
    threshold = ao["quality_threshold_db"]
    rows = [
        (seed, f"{values['solved']}/{values['attempted']}",
         *(f"{values[k]:.3f}" if k in values else "—" for k in fields))
        for seed, values in ao["ao_spread"].items()
    ]
    if threshold:
        rows.append(("Quality tolerance", "—", *(f"{threshold[k]:.3f}" for k in fields)))
    table(out, ("Channel seed", "AO solves", "Design spread (dB)", "True SCNR spread (dB)"), rows)


def stream_scoring(out, result):
    print("\n## A3 · Stream recovery and paired gaps", file=out)
    print("\nGaps are AO minus the 5×3 priced LP for physically feasible pairs. "
          "Estimates are paired medians in dB with 95% bootstrap intervals. "
          "A material truncation change is at least 0.001 dB in absolute value.\n", file=out)
    rows = []
    for row in result["ao"]["conditions"]:
        if row["arm"] == "base":
            rows.append((
                row["gamma_db"], row["solved"], row.get("designs_with_rank_excess"),
                row.get("material_truncation_design"), row.get("material_truncation_true"),
                interval(row.get("paired_design_stream_gap_db")),
                interval(row.get("paired_true_stream_gap_db")),
            ))
    table(out, ("SINR (dB)", "AO solves", "Rank-excess designs", "Design changes", "True changes",
                "Design gap", "True gap"), rows)


def feasibility(out, result):
    print("\n## A4 · Margin and CSI errors", file=out)
    print("\nThe three percentages are the mean across all draws of the fraction of "
          "100 perturbed channels that satisfy every nominal SINR requirement. "
          "The SCNR cost is the paired median versus zero explicit tightening, with its "
          "95% bootstrap interval.\n", file=out)
    rows = []
    for row in result["csi"]:
        rows.append((
            row["gamma_db"], row["margin_setting"], f"{row['achieved']['median']:.6f}",
            interval(row["true_cost_db_vs_no_tightening"], precision=4),
            *(f"{100 * row[f'relative_error_{error}']['mean_over_designs']:.3f}%"
              for error in (0.01, 0.03, 0.05)),
        ))
    table(out, ("SINR (dB)", "Tightening (dB)", "Achieved median (dB)", "SCNR cost (dB)",
                "1% CSI error", "3% CSI error", "5% CSI error"), rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = json.loads(args.summary.read_text())
    with args.output.open("x") as output:
        for section in (reconciliation, initialization, stream_scoring, feasibility):
            section(output, result)


if __name__ == "__main__":
    main()
