"""Collect results/<exp>/ summaries into results/experiment_table.{md,json} (plan §10, §23)."""
import argparse
import json
from pathlib import Path

ASPECTS = ("F", "P", "C")
NAMES = {"F": "MF", "P": "BP", "C": "CC"}


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=["baseline", "hierarchical", "taxonomy", "combined"])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/experiment_table")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--sweep", nargs="*", default=[])
    ap.add_argument("--sweep-budget", type=int, default=64)
    args = ap.parse_args()
    root = Path(args.results_dir)
    rows = []
    for exp in args.experiments:
        d = root / exp
        ev = load(d / f"eval_{exp}_val.json") or {}
        for a in ASPECTS:
            tr = load(d / f"{a}_train_summary.json")
            if tr is None:
                continue
            vm = tr["val_metrics_at_best"]
            m = ev.get("aspects", {}).get(a, {})
            off = ev.get("cafaeval", {}).get("aspects", {}).get(a, {})
            rows.append({
                "experiment": exp, "aspect": a, "K": tr["K"], "best_epoch": tr["best_epoch"],
                "epochs_run": tr["epochs_run"],
                "model_sel_fmax": vm["val_fmax"],
                "fmax": m.get("fmax"), "threshold": m.get("threshold"),
                "precision": m.get("precision"), "recall": m.get("recall"),
                "wfmax": m.get("wfmax"),
                "cafaeval_fmax_subset": off.get("fmax"), "cafaeval_wfmax_subset": off.get("wfmax"),
                "val_loss": vm["val_loss"], "violation_rate": vm["val_violation_rate"],
                "violation_rate_gt001": vm.get("val_violation_rate_gt001"),
                "mean_violation_mag": vm["val_mean_violation_mag"],
                "train_time_sec": tr["train_time_total_sec"], "peak_gpu_gb": tr["peak_gpu_mem_gb"],
                "config_hash": tr["config_hash"], "vocab_hash": tr["vocab_hash"],
            })
    Path(args.out + ".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def f(v, n=4):
        return "—" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))

    lines = [
        "| Exp | Aspect | K | best ep | Fmax | thr | P | R | wFmax (IA) | cafaeval Fmax / wFmax (subset) | viol. rate (>0) | viol. rate (>0.01) | mean viol. | train s | config |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['experiment']} | {NAMES[r['aspect']]} | {r['K']} | {r['best_epoch']}/{r['epochs_run']} | "
            f"{f(r['fmax'])} | {f(r['threshold'], 2)} | {f(r['precision'])} | {f(r['recall'])} | {f(r['wfmax'])} | "
            f"{f(r['cafaeval_fmax_subset'])} / {f(r['cafaeval_wfmax_subset'])} | {f(r['violation_rate'])} | "
            f"{f(r['violation_rate_gt001'])} | {f(r['mean_violation_mag'], 6)} | {f(r['train_time_sec'], 0)} | `{r['config_hash']}` |"
        )
    # mean over aspects
    lines += ["", "| Exp | mean Fmax | mean wFmax |", "|---|---|---|"]
    for exp in args.experiments:
        ev = load(root / exp / f"eval_{exp}_val.json")
        if ev:
            lines.append(f"| {exp} | {f(ev.get('mean_fmax'))} | {f(ev.get('mean_wfmax'))} |")
    Path(args.out + ".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    extra_sections(args)



def seed_summary(experiments, seeds, root: Path):
    """mean ± std over seeds of full-protocol Fmax / wFmax (seed 42 = plain name)."""
    import numpy as np

    out = {}
    for exp in experiments:
        per = {a: {"fmax": [], "wfmax": []} for a in ASPECTS}
        means = {"fmax": [], "wfmax": []}
        for s in seeds:
            name = exp if s == 42 else f"{exp}_s{s}"
            ev = load(root / name / f"eval_{name}_val.json")
            if not ev:
                continue
            for a in ASPECTS:
                for k in ("fmax", "wfmax"):
                    per[a][k].append(ev["aspects"][a][k])
            means["fmax"].append(ev["mean_fmax"])
            means["wfmax"].append(ev["mean_wfmax"])
        out[exp] = {"n_seeds": len(means["fmax"]),
                    **{f"{a}_{k}": (float(np.mean(v)), float(np.std(v))) for a in ASPECTS for k, v in per[a].items() if v},
                    **{f"mean_{k}": (float(np.mean(v)), float(np.std(v))) for k, v in means.items() if v}}
    return out


def sweep_table(names, root: Path, budget: int):
    """Best model-selection Fmax within the first ``budget`` epochs (matched training budget)."""
    rows = []
    for name in names:
        rec = {"name": name}
        for a in ASPECTS:
            h = load(root / name / f"{a}_history.json")
            if h:
                rec[a] = max(x["val_fmax"] for x in h[:budget])
                rec[f"{a}_epochs"] = len(h)
        rows.append(rec)
    return rows


def extra_sections(args):
    root = Path(args.results_dir)
    lines = []
    if args.seeds:
        ss = seed_summary(args.experiments, args.seeds, root)
        lines += ["", f"### Mean ± std over seeds {args.seeds} (internal validation, full truth)", "",
                  "| Exp | n | MF Fmax | BP Fmax | CC Fmax | mean Fmax | MF wFmax | BP wFmax | CC wFmax | mean wFmax |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for exp, r in ss.items():
            if not r["n_seeds"]:
                continue
            c = lambda k: f"{r[k][0]:.4f} ± {r[k][1]:.4f}" if k in r else "—"
            lines.append(f"| {exp} | {r['n_seeds']} | {c('F_fmax')} | {c('P_fmax')} | {c('C_fmax')} | {c('mean_fmax')} | "
                         f"{c('F_wfmax')} | {c('P_wfmax')} | {c('C_wfmax')} | {c('mean_wfmax')} |")
        Path(args.out + "_seeds.json").write_text(json.dumps(ss, indent=2), encoding="utf-8")
    if args.sweep:
        rows = sweep_table(args.sweep, root, args.sweep_budget)
        lines += ["", f"### λ sweep — best model-selection Fmax within the first {args.sweep_budget} epochs (matched budget)", "",
                  "| run | MF | BP | CC | mean |", "|---|---|---|---|---|"]
        for r in rows:
            vals = [r.get(a) for a in ASPECTS]
            mean = sum(vals) / 3 if all(v is not None for v in vals) else None
            lines.append(f"| {r['name']} | " + " | ".join("—" if v is None else f"{v:.4f}" for v in vals)
                         + f" | {'—' if mean is None else f'{mean:.4f}'} |")
    if lines:
        with open(args.out + ".md", "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
