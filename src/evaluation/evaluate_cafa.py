"""File-based evaluation (plan §12).

Two clearly separated protocols:
  * ``internal_validation`` — our GPU implementation of CAFA-style protein-centric Fmax and
    IA-weighted Fmax on the held-out split. Ground truth = full propagated validation annotations
    (terms outside the training vocabulary count as misses). Predictions are max-propagated up the
    DAG first, as the official evaluator does (``prop='max'``).
  * ``cafaeval`` — the official CAFA evaluator package (``cafaeval``, BioComputingUP), run on the
    same files (optionally a protein subset) as a cross-check / smoke test.

Neither is an official CAFA 5 leaderboard score: that requires the temporal hold-out ground truth.
"""
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from src.evaluation.metrics import FmaxAccumulator

ASPECTS = ("F", "P", "C")


def load_ground_truth(annotations_tsv: Union[str, Path], ids: List[str]) -> Dict[str, Dict[str, Set[str]]]:
    """Read ``propagated_annotations.tsv`` restricted to ``ids`` -> {aspect: {pid: {GO}}}."""
    keep = set(ids)
    df = pd.read_csv(annotations_tsv, sep="\t")
    df = df[df["EntryID"].isin(keep)]
    gt: Dict[str, Dict[str, Set[str]]] = {a: {} for a in ASPECTS}
    for (pid, aspect), grp in df.groupby(["EntryID", "aspect"]):
        gt[aspect][pid] = set(grp["term"])
    return gt


def read_predictions(pred_path: Union[str, Path], chunksize: int = 5_000_000) -> pd.DataFrame:
    """Read a headerless ``pid<TAB>GO<TAB>score`` file with categorical columns (memory-light)."""
    parts = []
    for chunk in pd.read_csv(pred_path, sep="\t", header=None, names=["pid", "go", "score"],
                             dtype={"pid": "category", "go": "category", "score": np.float32},
                             chunksize=chunksize):
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    for c in ("pid", "go"):
        df[c] = df[c].astype(str).astype("category")
    return df


def evaluate_prediction_file(
    pred_path: Union[str, Path],
    gt: Dict[str, Dict[str, Set[str]]],
    ontology,
    ia: Optional[Dict[str, float]] = None,
    thresholds: Optional[np.ndarray] = None,
    propagate: bool = True,
    device: Optional[str] = None,
    chunk_rows: int = 2048,
    preds: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Internal-validation metrics per aspect from a prediction file."""
    from src.models.hierarchical_loss import compute_hierarchy_diagnostics  # noqa: F401
    from src.evaluation.predict import propagate_scores_max

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = np.linspace(0.01, 0.99, 99) if thresholds is None else thresholds
    t0 = time.time()
    df = preds if preds is not None else read_predictions(pred_path)
    term_aspect = df["go"].cat.categories.map(
        lambda g: ontology.terms[g].aspect if g in ontology.terms else "?")
    results: Dict[str, Any] = {"protocol": "internal_validation", "propagate_predictions": propagate,
                               "weighted": ia is not None, "aspects": {}}
    for aspect in ASPECTS:
        truth = gt.get(aspect, {})
        if not truth:
            continue
        prot = sorted(truth)
        prot_pos = {p: i for i, p in enumerate(prot)}
        pred_terms = set(df["go"].cat.categories[np.asarray(term_aspect == aspect)])
        terms = sorted(pred_terms | set().union(*truth.values()))
        term_pos = {g: i for i, g in enumerate(terms)}

        go_codes = df["go"].cat.codes.to_numpy()
        aspect_mask = np.asarray(term_aspect == aspect)[go_codes]
        sub = df[aspect_mask]
        rows = sub["pid"].astype(str).map(prot_pos)
        keep = rows.notna().to_numpy()
        r = rows[keep].astype(np.int64).to_numpy()
        c = sub["go"].astype(str).map(term_pos).to_numpy()[keep].astype(np.int64)
        s = sub["score"].to_numpy()[keep]
        S = sparse.csr_matrix((s, (r, c)), shape=(len(prot), len(terms)), dtype=np.float32)
        tr, tc = [], []
        for p, gos in truth.items():
            for g in gos:
                tr.append(prot_pos[p])
                tc.append(term_pos[g])
        Y = sparse.csr_matrix((np.ones(len(tr), np.float32), (tr, tc)), shape=S.shape)

        edges = ontology.build_go_edges(term_pos, aspect) if propagate else None
        e_t = torch.as_tensor(edges, dtype=torch.long, device=device) if edges is not None else None
        w = np.array([ia.get(g, 0.0) for g in terms], dtype=np.float32) if ia is not None else None
        acc = FmaxAccumulator(thresholds, device=device)
        acc_w = FmaxAccumulator(thresholds, ia_weights=w, device=device) if w is not None else None
        n_pred_prot = 0
        for i in range(0, len(prot), chunk_rows):
            sc = torch.as_tensor(S[i : i + chunk_rows].toarray(), device=device)
            if e_t is not None:
                sc = propagate_scores_max(sc, e_t)
            yt = torch.as_tensor(Y[i : i + chunk_rows].toarray(), device=device)
            n_pred_prot += int((sc > 0).any(dim=1).sum())
            acc.update(sc, yt)
            if acc_w is not None:
                acc_w.update(sc, yt)
        res = acc.compute()
        out = {
            "fmax": res["fmax"], "threshold": res["best_threshold"],
            "precision": res["precision_at_fmax"], "recall": res["recall_at_fmax"],
            "coverage_at_fmax": res["coverage_at_fmax"],
            "num_gt_proteins": len(prot), "num_gt_proteins_with_predictions": n_pred_prot,
            "num_terms_evaluated": len(terms), "curve": res["curve"],
        }
        if acc_w is not None:
            rw = acc_w.compute()
            out.update({"wfmax": rw["fmax"], "wfmax_threshold": rw["best_threshold"],
                        "wprecision": rw["precision_at_fmax"], "wrecall": rw["recall_at_fmax"],
                        "wcurve": rw["curve"]})
        results["aspects"][aspect] = out
    a = results["aspects"]
    if a:
        results["mean_fmax"] = float(np.mean([v["fmax"] for v in a.values()]))
        if ia is not None:
            results["mean_wfmax"] = float(np.mean([v["wfmax"] for v in a.values()]))
    results["eval_time_sec"] = time.time() - t0
    return results


def run_official_cafaeval(
    pred_path: Union[str, Path],
    gt: Dict[str, Dict[str, Set[str]]],
    obo_path: Union[str, Path],
    ia_path: Union[str, Path, None] = None,
    protein_subset: Optional[List[str]] = None,
    preds: Optional[pd.DataFrame] = None,
    n_cpu: int = 4,
) -> Dict[str, Any]:
    """Run the official ``cafaeval`` package on (a subset of) the predictions and ground truth."""
    import cafaeval
    from cafaeval.evaluation import cafa_eval

    keep = set(protein_subset) if protein_subset is not None else None
    df = preds if preds is not None else read_predictions(pred_path)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pred_dir = tmp / "pred"
        pred_dir.mkdir()
        sub = df if keep is None else df[df["pid"].astype(str).isin(keep)]
        sub.to_csv(pred_dir / "model.tsv", sep="\t", header=False, index=False)
        gt_file = tmp / "gt.tsv"
        with open(gt_file, "w", encoding="utf-8") as f:
            for aspect in ASPECTS:
                for pid, gos in gt.get(aspect, {}).items():
                    if keep is None or pid in keep:
                        for g in gos:
                            f.write(f"{pid}\t{g}\n")
        t0 = time.time()
        df_all, dfs_best = cafa_eval(str(obo_path), str(pred_dir), str(gt_file),
                                     ia=str(ia_path) if ia_path else None, prop="max", norm="cafa",
                                     th_step=0.01, n_cpu=n_cpu)
    ns_map = {"molecular_function": "F", "biological_process": "P", "cellular_component": "C"}
    out: Dict[str, Any] = {"protocol": "cafaeval", "cafaeval_version": getattr(cafaeval, "__version__", "unknown"),
                           "num_proteins": len(keep) if keep is not None else None,
                           "eval_time_sec": time.time() - t0, "aspects": {}}
    for metric, col, tau_col in (("f", "f", "tau"), ("f_w", "f_w", "tau")):
        if metric not in dfs_best:
            continue
        best = dfs_best[metric].reset_index()
        for _, row in best.iterrows():
            a = ns_map.get(row["ns"], row["ns"])
            rec = out["aspects"].setdefault(a, {})
            key = "fmax" if metric == "f" else "wfmax"
            rec[key] = float(row[col])
            rec[f"{key}_threshold"] = float(row[tau_col])
            rec[f"{key}_coverage"] = float(row.get("cov" if metric == "f" else "cov_w", np.nan))
    return out


def run_cafa_evaluator_smoke_test(
    predictions_path: Union[str, Path],
    ground_truth_path: Union[str, Path],
    obo_path: Union[str, Path],
    ia_path: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """Smoke test: run cafaeval on a prediction file and a 2-column ``pid<TAB>GO`` ground-truth file."""
    from cafaeval.evaluation import cafa_eval

    pred_file = Path(predictions_path)
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")
    with tempfile.TemporaryDirectory() as tmp:
        pred_dir = Path(tmp) / "pred"
        pred_dir.mkdir()
        (pred_dir / pred_file.name).write_bytes(pred_file.read_bytes())
        _, dfs_best = cafa_eval(str(obo_path), str(pred_dir), str(ground_truth_path),
                                ia=str(ia_path) if ia_path else None, prop="max", norm="cafa", th_step=0.01)
    best = dfs_best["f"].reset_index()
    return {
        "protocol": "evaluator_smoke_test",
        "predictions_path": str(pred_file),
        "status": "ok",
        "fmax_by_namespace": {r["ns"]: float(r["f"]) for _, r in best.iterrows()},
    }
