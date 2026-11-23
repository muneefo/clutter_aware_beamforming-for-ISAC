"""Render the three registered Part C result tables from summary.json."""
import argparse
import json
from pathlib import Path


def table(out, names, rows):
    for row in (names, ["---"] * len(names), *rows):
        print("| " + " | ".join(map(str, row)) + " |", file=out)


def spread(stat, digits=3):
    if not stat:
        return "—"
    return f"{stat['median']:.{digits}f} [{stat['q1']:.{digits}f}, {stat['q3']:.{digits}f}]"


def confidence(stat, digits=3):
    if not stat:
        return "—"
    left, right = stat["ci95"]
    return f"{stat['median']:.{digits}f} [{left:.{digits}f}, {right:.{digits}f}]"


def count(rate):
    if not rate:
        return "—"
    left, right = rate["ci95"]
    return f"{rate['hits']}/{rate['n']} [{100 * left:.1f}%, {100 * right:.1f}%]"


def render(out, data):
    print("# Part C tables\n", file=out)
    print("## C1–C2 · Core cohort\n", file=out)
    print("Med [IQR] reports the median with both quartiles. Bootstrap brackets give 95% "
          "intervals. Hit brackets give exact Clopper–Pearson 95% intervals over all "
          "attempted channels. The signed gap is reference AO minus the priced LP.\n", file=out)
    core = { (1, 1), (3, 2), (5, 3) }
    rows = []
    for r in data["static"]:
        if r["method"] == "lp" and (r["outer"], r["inner"]) in core:
            rows.append((r["users"], r["gamma"], f"{r['outer']}×{r['inner']}", r["n_paired"],
                         spread(r["seconds"]), confidence(r["seconds"]),
                         spread(r["design_gap_db"]), confidence(r["design_gap_db"]),
                         spread(r["true_gap_db"]), confidence(r["true_gap_db"]),
                         count(r["design_hits"]), count(r["true_hits"])))
    table(out, ("K", "SINR (dB)", "Budget", "Pairs", "Seconds med [IQR]", "Seconds bootstrap",
                "Design dB med [IQR]", "Design dB bootstrap", "True dB med [IQR]",
                "True dB bootstrap", "Design hits", "True hits"), rows)
    print("\n## C3 · Pareto endpoints at four users and 8 dB\n", file=out)
    rows = []
    for r in data["static"]:
        if (r["users"], r["gamma"]) == (4, 8):
            rows.append((r["method"], f"{r['outer']}×{r['inner']}", r["n_paired"],
                         confidence(r["seconds"]), confidence(r["design_gap_db"]),
                         (r["counts"]["sdp_solves"] or {}).get("median", "—"),
                         (r["counts"]["lp_solves"] or {}).get("median", "—"),
                         "yes" if r.get("pareto") else ""))
    table(out, ("Arm", "Budget label", "Pairs", "Seconds [bootstrap CI]", "Design gap dB [bootstrap CI]",
                "Median SDP solves", "Median LP solves", "Nondominated"), rows)

    print("\n## C4 · Correlated tracking and array scaling\n", file=out)
    table(out, ("ρ", "Mode", "Frames within target", "Time to target (s) [cluster bootstrap CI]",
                "Chains hitting all three frames [exact CI]", "Initial simplex pivots med [IQR]",
                "Basis accepted / requested"), [
        (f"{r['rho']:.2f}", r["method"], f"{r['n_hit']}/{r['n_attempted']}",
         confidence(r["hit_seconds"]), count(r["all_three_hit"]), spread(r["first_simplex"], 1),
         f"{r['basis_accepted']}/{r['basis_requested']}") for r in data["warm"]
    ])
    print("\nRatios use frames where both methods reached the target. The bootstrap resamples chains.\n", file=out)
    table(out, ("ρ", "Baseline", "Compare", "Paired frames", "Median time ratio [cluster bootstrap CI]"), [
        (f"{r['rho']:.2f}", r["base"], r["compare"], (r["ratio"] or {}).get("n", 0),
         confidence(r["ratio"])) for r in data["warm_ratios"]
    ])
    print(file=out)
    table(out, ("Antennas per array", "Method", "Feasible / attempted", "Timeouts", "Seconds med [IQR]",
                "Seconds bootstrap"), [
        (r["array"], r["method"], f"{r['solved']}/{r['attempted']}", r["timeouts"],
         spread(r["seconds"]), confidence(r["seconds"])) for r in data["scaling"]
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    with args.destination.open("x") as out:
        render(out, json.loads(args.summary.read_text()))


if __name__ == "__main__":
    main()
