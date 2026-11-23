"""Produce paired Part B tables, flattened experimental rows and timing metadata."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

DESIGN_TOLERANCE = 0.8671285631736225
TRUE_TOLERANCE = 0.5609881611015304
STAGES = ("build", "solve", "price", "mvdr", "kernel", "recovery", "parallel_tone_wall", "other")
COMPARISONS = [
    ("Warm start", "B1_30x30", "B2_30x30"),
    ("Cap 5×3", "B2_30x30", "B3_5x3"),
    ("Cap 3×2", "B2_30x30", "B3_3x2"),
    ("Charnes–Cooper, 5 outer", "B3_5x3", "B4_5x1"),
    ("Charnes–Cooper, 3 outer", "B3_3x2", "B4_3x1"),
    ("Tone decomposition", "B3_3x2", "B5_3x2"),
    ("Random additions, 5×3", "B6_5x3", "B7_5x3"),
    ("Analytic in place of random, 5×3", "B7_5x3", "B8_5x3"),
    ("Analytic pricing versus fixed, 5×3", "B6_5x3", "B8_5x3"),
    ("Random additions, 3×2", "B6_3x2", "B7_3x2"),
    ("Analytic in place of random, 3×2", "B7_3x2", "B8_3x2"),
    ("Analytic pricing versus fixed, 3×2", "B6_3x2", "B8_3x2"),
    ("End-to-end, descriptive, 5×3", "B1_30x30", "B8_5x3"),
    ("End-to-end, descriptive, 3×2", "B1_30x30", "B8_3x2"),
]


def arm(row):
    return f"{row['arm']}_{row['outer']}x{row['inner']}"


def eligible(row):
    return row["status"] == "solved" and row.get("feasible") is True


def quantiles(values):
    if not len(values):
        return None
    return dict(zip(("q1", "median", "q3"), np.quantile(values, [.25, .5, .75]).tolist()))


def bootstrap(values, rng):
    estimates = quantiles(values)
    if estimates:
        a = np.asarray(values)
        idx = rng.integers(a.size, size=(10000, a.size))
        estimates["ci95"] = np.quantile(np.median(a[idx], axis=1), [.025, .975]).tolist()
    return estimates


def paired(a, b, rng):
    keys = a.keys() & b.keys()
    pairs = [(a[k], b[k]) for k in sorted(keys) if eligible(a[k]) and eligible(b[k])]
    ratio = [x["seconds"] / y["seconds"] for x, y in pairs]
    design = [y["design_streams_db"] - x["design_streams_db"] for x, y in pairs]
    truth = [y["true_streams_db"] - x["true_streams_db"] for x, y in pairs]
    within = [r for r, d, t in zip(ratio, design, truth) if d >= -DESIGN_TOLERANCE and t >= -TRUE_TOLERANCE]
    ci = binomtest(len(within), len(keys)).proportion_ci() if keys else None
    return dict(attempted_pairs=len(keys), feasible_pairs=len(pairs),
                matched_quality_pairs=len(within), matched_quality_ci95=list(ci) if ci else None,
                time_ratio=bootstrap(ratio, rng), design_delta_db=bootstrap(design, rng),
                true_delta_db=bootstrap(truth, rng), matched_quality_time_ratio=bootstrap(within, rng))


def arm_summary(rows):
    accepted = [r for r in rows if eligible(r)]
    clocks = {name: quantiles([r["timings"][name] for r in accepted]) for name in STAGES}
    worker_seconds = {
        name: quantiles([r["tone_worker_seconds"][name] for r in accepted if "tone_worker_seconds" in r])
        for name in ("build", "solve", "price")
    }
    counts = {k: quantiles([r[k] for r in accepted if k in r]) for k in (
        "outer_steps", "lp_solves", "sdp_solves", "dinkelbach_solves", "price_evals",
        "bisection_steps", "local_masters", "atoms")}
    return dict(attempted=len(rows), eligible=len(accepted), statuses=dict(Counter(r["status"] for r in rows)),
                seconds=quantiles([r["seconds"] for r in accepted]),
                design_db=quantiles([r["design_streams_db"] for r in accepted]),
                true_db=quantiles([r["true_streams_db"] for r in accepted]),
                timings=clocks, worker_seconds=worker_seconds, counts=counts)


def collect(source):
    expected = set(range(16001, 16033))
    blocks, flat = {}, []
    for thread in (1, 2):
        run = source / f"t{thread}"
        rows = [json.loads(line)["design"] for line in (run / "designs.jsonl").read_text().splitlines()]
        data = {}
        for row in rows:
            if row["warmup"]:
                continue
            group = data.setdefault(arm(row), {})
            if row["seed"] in group:
                raise ValueError(f"Duplicate row for {thread} {arm(row)} {row['seed']}")
            group[row["seed"]] = row
            flat.append({**{k: v for k, v in row.items() if k not in ("timings", "tone_worker_seconds")},
                         **{"seconds_" + k: v for k, v in row.get("timings", {}).items()},
                         **{"worker_seconds_" + k: v for k, v in row.get("tone_worker_seconds", {}).items()}})
        if len(data) != 13 or any(set(g) != expected for g in data.values()):
            raise ValueError(f"Incomplete preregistered job set in t{thread}")
        blocks[thread] = data
    return blocks, flat


def distribution(stat, digits=3):
    if stat is None:
        return "—"
    return "{:.{d}f} [{:.{d}f}, {:.{d}f}]".format(stat["median"], *stat["ci95"], d=digits)


def median(stat, digits=3):
    if stat is None:
        return "—"
    return f"{stat['median']:.{digits}f}"


def table(out, header, rows):
    def print_row(row):
        print("| " + " | ".join(map(str, row)) + " |", file=out)
    print_row(header)
    print_row(["---"] * len(header))
    for row in rows:
        print_row(row)


def render(computed, out):
    print("# Part B timing and quality tables\n", file=out)
    print("All SCNRs score the recovered streams and returned receiver. The design score uses "
          "the estimated kernel and true SCNR uses the ideal kernel. Times are medians in seconds. "
          "The numerator of each time ratio is the first named arm. Brackets in paired tables give "
          "draw-level percentile bootstrap 95% intervals.\n", file=out)
    for t, groups in computed["arms"].items():
        print(f"## {t} thread setting\n", file=out)
        table(out, ("Arm", "Eligible / attempted", "Design SCNR", "True SCNR", "Median time",
                    "SDPs", "LPs", "Price evals", "Bisections", "Restricted masters"), (
            (name, f"{g['eligible']}/{g['attempted']}", median(g["design_db"]), median(g["true_db"]),
             median(g["seconds"]), *(median(g["counts"][k], 1) for k in (
                 "sdp_solves", "lp_solves", "price_evals", "bisection_steps", "local_masters")))
            for name, g in groups.items()
        ))
        print(f"\n### Runtime breakdown for {t} thread setting\n", file=out)
        table(out, ("Arm", *STAGES), (
            (name, *(median(g["timings"][k], 6) for k in STAGES)) for name, g in groups.items()
        ))
        print("\nIn the parallel B5 row the tone worker stages overlap and are represented by the "
              "parallel tone wall time. The JSON includes their separate summed process-stage times. "
              "Stage medians summarize the draws individually.\n", file=out)
        print("### Paired effects\n", file=out)
        table(out, ("Contrast", "Feasible pairs", "Time ratio", "Design Δ dB", "True Δ dB",
                    "Within both quality limits", "Matched-quality time ratio"), (
            (c["label"], f"{c['feasible_pairs']}/{c['attempted_pairs']}", distribution(c["time_ratio"], 2),
             distribution(c["design_delta_db"]), distribution(c["true_delta_db"]), c["matched_quality_pairs"],
             distribution(c["matched_quality_time_ratio"], 2))
            for c in computed["contrasts"][t]
        ))
        print(file=out)
    print("## Single-to-two-thread effects\n", file=out)
    table(out, ("Arm", "Feasible pairs", "Single / two-thread time ratio", "Design Δ dB", "True Δ dB"), (
        (name, f"{c['feasible_pairs']}/{c['attempted_pairs']}", distribution(c["time_ratio"], 2),
         distribution(c["design_delta_db"]), distribution(c["true_delta_db"]))
        for name, c in computed["thread_comparison"].items()
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    blocks, flat = collect(args.runs)
    args.destination.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(395001)
    names = list(dict.fromkeys(k for _, a, b in COMPARISONS for k in (a, b)))
    computed = dict(arms={str(t): {k: arm_summary(d[k].values()) for k in names} for t, d in blocks.items()},
                    contrasts={str(t): [dict(label=label, before=a, after=b, **paired(d[a], d[b], rng))
                                        for label, a, b in COMPARISONS] for t, d in blocks.items()},
                    thread_comparison={k: paired(blocks[1][k], blocks[2][k], rng) for k in names},
                    quality_tolerances=dict(design=DESIGN_TOLERANCE, true=TRUE_TOLERANCE))
    with (args.destination / "designs.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in flat for k in r)),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(flat, key=lambda r: (r["threads"], r["seed"], arm(r))))
    with (args.destination / "summary.json").open("x") as stream:
        json.dump(computed, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (args.destination / "tables.md").open("x") as stream:
        render(computed, stream)


if __name__ == "__main__":
    main()
