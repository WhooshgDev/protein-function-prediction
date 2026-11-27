"""Hierarchy-violation diagnostics from saved checkpoints on the validation split (plan §8).

For each experiment/aspect: fraction of (protein, parent->child edge) pairs with s_child > s_parent
(any margin, and margin > 0.01 / > 0.1), mean violation magnitude, and mean magnitude given a
violation. Raw model scores (no post-hoc propagation). Writes results/hierarchy_diagnostics.json.
"""
import argparse
import json
from pathlib import Path

import torch

from _common import ROOT_DIR  # noqa: F401  (sets sys.path)
from src.data.artifacts import ProcessedData
from src.evaluation.predict import load_model_from_checkpoint
from src.utils.config import load_config


@torch.no_grad()
def diagnose(model, x, tax, edges, device, bs=2048):
    e = torch.as_tensor(edges, dtype=torch.long, device=device)
    tot = {"pairs": 0, "gt0": 0.0, "gt001": 0.0, "gt01": 0.0, "mag": 0.0}
    for s in range(0, len(x), bs):
        xb = torch.as_tensor(x[s : s + bs], device=device)
        tb = None if tax is None else torch.as_tensor(tax[s : s + bs], device=device)
        p = torch.sigmoid(model(xb, tb) if (tb is not None and model.taxonomy_enabled) else model(xb))
        d = p[:, e[:, 1]] - p[:, e[:, 0]]
        tot["pairs"] += d.numel()
        tot["gt0"] += float((d > 0).sum())
        tot["gt001"] += float((d > 0.01).sum())
        tot["gt01"] += float((d > 0.1).sum())
        tot["mag"] += float(torch.relu(d).sum())
    n = max(tot["pairs"], 1)
    return {"violation_rate": tot["gt0"] / n, "violation_rate_gt001": tot["gt001"] / n,
            "violation_rate_gt01": tot["gt01"] / n, "mean_violation_magnitude": tot["mag"] / n,
            "mean_magnitude_given_violation": tot["mag"] / max(tot["gt0"], 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/baseline.yaml", help="Any config (for data paths).")
    ap.add_argument("--names", nargs="+", required=True, help="Experiment names (checkpoint prefixes).")
    ap.add_argument("--out", default="results/hierarchy_diagnostics.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pdata = ProcessedData(cfg["data"]["processed_dir"], cfg["data"]["embedding_dir"])
    valid_ids = pdata.split_ids("valid")
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in args.names:
        for a in "FPC":
            ck = Path("checkpoints") / f"{name}_{a}_best.pt"
            if not ck.exists():
                continue
            model, meta = load_model_from_checkpoint(ck, device)
            y = pdata.labels_for(a, valid_ids)
            keep = [p for p, k in zip(valid_ids, y.getnnz(axis=1) > 0) if k]
            x, _ = pdata.embeddings("valid", keep)
            tax = pdata.taxonomy_indices("valid", keep) if model.taxonomy_enabled else None
            r = diagnose(model, x, tax, pdata.edges(a), device)
            r["lambda"] = meta["config"]["loss"].get("hierarchical_lambda", 0.0)
            results.setdefault(name, {})[a] = r
            print(name, a, {k: f"{v:.3g}" for k, v in r.items()})
    out_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
