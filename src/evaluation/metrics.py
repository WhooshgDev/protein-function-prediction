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
