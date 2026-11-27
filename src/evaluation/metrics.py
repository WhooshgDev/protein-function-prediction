from typing import Any, Dict, Optional
import numpy as np


def compute_fmax(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    ia_weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Sweep thresholds in [0, 1] and compute protein-centric Precision, Recall, F1, and Fmax.

    Args:
        scores: Predicted probabilities of shape (N, K) in [0, 1]
        labels: Binary ground-truth array of shape (N, K) in {0, 1}
        thresholds: 1D array of thresholds to evaluate
        ia_weights: Optional Information Accretion weights of shape (K,)
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32) > 0.5

    if ia_weights is not None:
        w = np.asarray(ia_weights, dtype=np.float32).reshape(1, -1)
    else:
        w = np.ones((1, labels.shape[1]), dtype=np.float32)

    true_protein_weight = (labels * w).sum(axis=1)
    valid_proteins = true_protein_weight > 0
    n_total_valid = int(valid_proteins.sum())

    if n_total_valid == 0:
        return {
            "fmax": 0.0,
            "best_threshold": float(thresholds[0]),
            "precision_at_fmax": 0.0,
            "recall_at_fmax": 0.0,
            "curve": [],
        }

    best_f1 = -1.0
    best_thresh = float(thresholds[0])
    best_p = 0.0
    best_r = 0.0
    curve = []

    labels_valid = labels[valid_proteins]
    scores_valid = scores[valid_proteins]
    true_w_valid = true_protein_weight[valid_proteins]

    for t in thresholds:
        preds = scores_valid >= t
        pred_w = (preds * w).sum(axis=1)
        tp_w = ((preds & labels_valid) * w).sum(axis=1)

        has_pred = pred_w > 0
        n_pred = int(has_pred.sum())

        if n_pred > 0:
            precision = float((tp_w[has_pred] / pred_w[has_pred]).mean())
        else:
            precision = 0.0

        recall = float((tp_w / true_w_valid).mean())

        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        curve.append(
            {
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
            best_p = precision
            best_r = recall

    return {
        "fmax": float(best_f1),
        "best_threshold": best_thresh,
        "precision_at_fmax": best_p,
        "recall_at_fmax": best_r,
        "num_evaluated_proteins": n_total_valid,
        "curve": curve,
    }


def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    aspect: str = "F",
    ia: Optional[np.ndarray] = None,
    protocol: str = "internal_validation",
) -> Dict[str, Any]:
    """Contract entrypoint: Evaluate predictions and return metric dictionary + protocol metadata."""
    res = compute_fmax(scores=scores, labels=labels, thresholds=thresholds, ia_weights=ia)
    res["aspect"] = aspect
    res["protocol"] = protocol
    res["weighted_ia"] = ia is not None
    return res


class FmaxAccumulator:
    """Streaming, GPU-friendly protein-centric P/R/F over a threshold grid (same definition as
    ``compute_fmax``): precision averaged over proteins with >=1 prediction at t, recall averaged
    over proteins with >=1 true term; optional IA weights turn this into weighted Fmax.

    ``update`` may receive ``extra_true_weight`` (B,) — the weight of true terms that are not among
    the K scored columns (e.g. validation terms outside the training vocabulary) — so they still
    count as misses in recall.
    """

    def __init__(self, thresholds, ia_weights=None, device=None):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.t = torch.as_tensor(np.asarray(thresholds, dtype=np.float32), device=self.device)
        self.w = None if ia_weights is None else torch.as_tensor(
            np.asarray(ia_weights, dtype=np.float32), device=self.device)
        self.tp, self.pred, self.true = [], [], []

    def update(self, scores, labels, extra_true_weight=None) -> None:
        torch = self.torch
        s = torch.as_tensor(scores, device=self.device, dtype=torch.float32)
        y = torch.as_tensor(labels, device=self.device, dtype=torch.float32) > 0.5
        w = self.w if self.w is not None else torch.ones(s.shape[1], device=self.device)
        yw = y.float() * w
        true_w = yw.sum(dim=1)
        if extra_true_weight is not None:
            true_w = true_w + torch.as_tensor(extra_true_weight, device=self.device, dtype=torch.float32)
        tp = torch.empty((s.shape[0], self.t.numel()), device=self.device)
        pr = torch.empty_like(tp)
        for j, th in enumerate(self.t):
            p = (s >= th).float()
            pr[:, j] = p @ w
            tp[:, j] = (p * yw).sum(dim=1)
        self.tp.append(tp.cpu())
        self.pred.append(pr.cpu())
        self.true.append(true_w.cpu())

    def compute(self) -> Dict[str, Any]:
        torch = self.torch
        thresholds = self.t.cpu().numpy()
        if not self.true:
            return {"fmax": 0.0, "best_threshold": float(thresholds[0]), "precision_at_fmax": 0.0,
                    "recall_at_fmax": 0.0, "num_evaluated_proteins": 0, "curve": []}
        tp = torch.cat(self.tp).double()
        pr = torch.cat(self.pred).double()
        tw = torch.cat(self.true).double()
        valid = tw > 0
        tp, pr, tw = tp[valid], pr[valid], tw[valid]
        n = int(valid.sum())
        has_pred = pr > 0
        n_pred = has_pred.sum(dim=0)
        prec = torch.where(has_pred, tp / pr.clamp(min=1e-12), torch.zeros_like(tp)).sum(dim=0) / n_pred.clamp(min=1)
        rec = (tp / tw.unsqueeze(1)).sum(dim=0) / max(n, 1)
        f1 = torch.where(prec + rec > 0, 2 * prec * rec / (prec + rec).clamp(min=1e-12), torch.zeros_like(prec))
        best = int(torch.argmax(f1)) if n else 0
        curve = [
            {"threshold": float(thresholds[j]), "precision": float(prec[j]), "recall": float(rec[j]),
             "f1": float(f1[j]), "coverage": float(n_pred[j]) / max(n, 1)}
            for j in range(len(thresholds))
        ]
        return {
            "fmax": float(f1[best]) if n else 0.0,
            "best_threshold": float(thresholds[best]),
            "precision_at_fmax": float(prec[best]) if n else 0.0,
            "recall_at_fmax": float(rec[best]) if n else 0.0,
            "coverage_at_fmax": curve[best]["coverage"] if n else 0.0,
            "num_evaluated_proteins": n,
            "curve": curve,
        }


def threshold_grid(eval_cfg: Dict[str, Any]) -> np.ndarray:
    return np.linspace(float(eval_cfg.get("threshold_min", 0.01)), float(eval_cfg.get("threshold_max", 0.99)),
                       int(eval_cfg.get("threshold_steps", 99)))
