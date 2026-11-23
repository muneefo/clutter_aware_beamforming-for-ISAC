"""Format the Part F review tables and Letter-size figures from measured results."""
import argparse
import hashlib
import json
from pathlib import Path

import partc_report
import partd_report
from matplotlib import pyplot as plt


def summary(source, relative):
    return json.loads((source / relative / "summary.json").read_text())


def cell(stats, digits=3, interval="ci95"):
    med = stats["median"]
    low, high = stats[interval]
    return f"${med:.{digits}f}$ [${low:.{digits}f}$, ${high:.{digits}f}$]"


def rows(path, values):
    with path.open("x") as out:
        out.write("% Generated from the pinned results by scripts/partf_assets.py\n")
        for fields in values:
            out.write(" & ".join(map(str, fields)) + r" \\" + "\n")


def reconciliation(a, path):
    selected = {gamma: {r["arm"]: r for r in a["conditions"] if r["gamma_db"] == gamma}
                for gamma in (8, 12)}
    base = [selected[g]["base"] for g in (8, 12)]
    controls = (("cap_5x3", r"AO capped at $5\times3$"), ("outer_0.1", "Outer tolerance 0.1 dB"),
                ("range_100_500", "User range 100--500 m"), ("power_cap_199.53", "Power cap 199.526 W"))
    table = [
        ("Fig. 12(b), digitized", *[f"{r['digitized_true_db']:.2f}" for r in base]),
        ("Pinned Python AO", *[f"{r['true_cov_linear_mean_db']:.3f}" for r in base]),
        ("Figure minus replay", *[f"{r['digitized_minus_replay_db']:.3f}" for r in base]),
    ]
    table += [(label, *[f"{selected[g][arm]['true_cov_linear_mean_db']:.3f}" for g in (8, 12)])
              for arm, label in controls]
    rows(path, table)


def ladder(b, path):
    registry = (
        ("B1_30x30", "B1", "Gaussian AO--SDP"), ("B2_30x30", "B2", "Structured AO--SDP"),
        ("B3_3x2", "B3", r"Structured, $3\times2$"),
        ("B4_3x1", "B4", "Charnes--Cooper SDP"), ("B5_3x2", "B5", "Per-tone SDP"),
        ("B6_3x2", "B6", "Fixed dictionary"), ("B7_3x2", "B7", "Matched random LP"),
        ("B8_3x2", "B8", "Analytic pricing LP"),
    )
    table = []
    for key, arm, label in registry:
        r = b["arms"]["1"][key]
        table.append((arm, label, f"{r['seconds']['median']:.3f}", f"{r['true_db']['median']:.3f}",
                      f"{r['eligible']}/{r['attempted']}"))
    rows(path, table)


def components(b, path):
    arms = (("B1_30x30", "B1"), ("B3_3x2", r"B3, $3\times2$"),
            ("B4_3x1", "B4, three outer"), ("B5_3x2", "B5, serial"),
            ("B8_3x2", r"B8, $3\times2$"))
    fields = ("build", "solve", "price", "mvdr", "kernel", "recovery")
    rows(path, [(label, *[f"{b['arms']['1'][arm]['timings'][key]['median']:.6f}" for key in fields])
                for arm, label in arms])


def learning(e, path, held_out=True):
    cohort = "held-out maps" if held_out else "original benchmark"
    arms = {"random": "Random", "gradient": "Gradient", "reduced-cost": "Reduced cost", "envelope": "Envelope"}
    table = []
    for o, i in ((2, 1), (3, 2)):
        for arm, label in arms.items():
            r = next(r for r in e["evaluation"] if (r["cohort"], r["outer"], r["inner"], r["arm"])
                     == (cohort, o, i, arm))
            metrics = [cell(r["design_delta_db"])]
            if not held_out:
                metrics.append(cell(r["true_delta_db"]))
            table.append((fr"${o}\times{i}$", label, *metrics, f"{r['time_ratio']['median']:.3f}"))
    rows(path, table)


def core(c, path):
    table = []
    for r in c["static"]:
        if r["method"] != "lp" or (r["outer"], r["inner"]) not in ((1, 1), (3, 2), (5, 3)):
            continue
        rate = []
        for k in ("design_hits", "true_hits"):
            hit = r[k]
            lo, hi = (100 * p for p in hit["ci95"])
            rate.append(f"{hit['hits']}/{hit['n']} [{lo:.1f}, {hi:.1f}]")
        table.append((r["users"], r["gamma"], fr"${r['outer']}\times{r['inner']}$",
                      cell(r["seconds"]), cell(r["design_gap_db"]), cell(r["true_gap_db"]), *rate))
    rows(path, table)


def csi(a, path):
    table = []
    for r in a:
        rates = []
        for error in ("0.01", "0.03", "0.05"):
            rates.append(f"{100 * r[f'relative_error_{error}']['mean_over_designs']:.2f}")
        table.append((r["gamma_db"], r["margin_setting"], f"{r['achieved']['median']:.6f}",
                      cell(r["true_cost_db_vs_no_tightening"], 4, "med_ci95"), *rates))
    rows(path, table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    names = {k: f"part{k}-2026-11-{23 if k == 'e' else 22}" for k in "abcde"}
    paths = {k: name if k == "a" else f"{name}/summary" for k, name in names.items()}
    a, b, c, d, e = (summary(args.results, paths[k]) for k in "abcde")
    warm_path = args.results / names["c"] / "warm/designs.jsonl"
    warm = [r for r in partc_report.read(warm_path) if not r["warmup"]]
    shards = sorted((args.results / names["d"]).glob("cert-*/bounds.jsonl"))
    endpoints = [r["result"] for s in shards for r in partd_report.read_jsonl(s)]
    if len(endpoints) != d["audit"]["converged"] or any(r["status"] != "converged" for r in endpoints):
        raise ValueError("Part F expects the complete certified Part D cohort")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    reconciliation(a["ao"], out / "reconciliation.tex")
    ladder(b, out / "ladder.tex")
    components(b, out / "components.tex")
    learning(e, out / "learning-heldout.tex")
    learning(e, out / "learning-benchmark.tex", held_out=False)
    core(c, out / "core.tex")
    csi(a["csi"], out / "csi.tex")

    plt.rcParams.update({"font.size": 7.4, "pdf.fonttype": 42, "axes.titlesize": 8})
    partc_report.plot_pareto(c["static"], out / "pareto.pdf", letter=True)
    partd_report.draw_plot(endpoints, out / "certificate.pdf", letter=True)
    partc_report.plot_tracking(warm, c["scaling"], out / "tracking-scaling.pdf", letter=True)
    inputs = [args.results / paths[k] / "summary.json" for k in "abcde"] + [warm_path, *shards]
    manifest = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs]
    (out / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
