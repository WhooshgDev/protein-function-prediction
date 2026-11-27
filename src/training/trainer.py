"""Ontology-agnostic training loop (plan §7, §15 Trainer contract).

The trainer only sees embeddings, label tensors and (optionally) parent->child edge indices; it
never needs to know how GO IDs were parsed.
"""
import copy
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.metrics import FmaxAccumulator, threshold_grid
from src.models.hierarchical_loss import HierarchicalLoss


def _forward(model: nn.Module, batch: Dict[str, Any], device: str) -> torch.Tensor:
    emb = batch["embedding"].to(device, non_blocking=True)
    tax_idx = batch.get("taxonomy_idx")
    if tax_idx is not None and getattr(model, "taxonomy_enabled", True):
        return model(emb, tax_idx.to(device, non_blocking=True))
    return model(emb)


class Trainer:
    """Trainer supporting BCE + optional GO hierarchical penalty, early stopping on validation Fmax."""

    def __init__(
        self,
        device: Optional[str] = None,
        parent_child_edges: Optional[torch.Tensor] = None,
        ia_weights: Optional[np.ndarray] = None,
        logger=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.parent_child_edges = (
            parent_child_edges.to(self.device) if parent_child_edges is not None else None
        )
        self.ia_weights = ia_weights
        self.logger = logger

    @torch.no_grad()
    def evaluate(
        self, model: nn.Module, loader: Iterable, criterion: HierarchicalLoss, thresholds: np.ndarray,
        lambda_hier: float,
    ) -> Dict[str, Any]:
        model.eval()
        acc = FmaxAccumulator(thresholds, device=self.device)
        acc_w = FmaxAccumulator(thresholds, ia_weights=self.ia_weights, device=self.device) \
            if self.ia_weights is not None else None
        loss_sum, n_batches, n_rows = 0.0, 0, 0
        viol_count, viol_count_m, viol_mag, edge_total = 0.0, 0.0, 0.0, 0
        edges = self.parent_child_edges
        for batch in loader:
            labels = batch["labels"].to(self.device)
            logits = _forward(model, batch, self.device)
            loss, _ = criterion(logits, labels, parent_child_edges=edges, lambda_hier=lambda_hier)
            loss_sum += float(loss)
            n_batches += 1
            probs = torch.sigmoid(logits)
            acc.update(probs, labels)
            if acc_w is not None:
                acc_w.update(probs, labels)
            if edges is not None and edges.numel():
                diff = probs[:, edges[:, 1]] - probs[:, edges[:, 0]]
                viol_count += float((diff > 0).sum())
                viol_count_m += float((diff > 0.01).sum())
                viol_mag += float(torch.relu(diff).sum())
                edge_total += diff.numel()
            n_rows += labels.shape[0]
        res = acc.compute()
        out = {
            "val_loss": loss_sum / max(1, n_batches),
            "val_fmax": res["fmax"],
            "val_precision": res["precision_at_fmax"],
            "val_recall": res["recall_at_fmax"],
            "val_best_threshold": res["best_threshold"],
            "val_violation_rate": viol_count / max(1, edge_total),
            "val_violation_rate_gt001": viol_count_m / max(1, edge_total),
            "val_mean_violation_mag": viol_mag / max(1, edge_total),
            "_curve": res["curve"],
        }
        if acc_w is not None:
            rw = acc_w.compute()
            out.update({"val_wfmax": rw["fmax"], "val_wfmax_threshold": rw["best_threshold"]})
        return out

    def fit(
        self,
        model: nn.Module,
        train_loader: Iterable,
        valid_loader: Iterable,
        config: Dict[str, Any],
        checkpoint_path: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Contract entrypoint: train, track validation Fmax, save best checkpoint.

        Returns (best_checkpoint_dict, per-epoch history).
        """
        model = model.to(self.device)
        train_cfg = config.get("training", {})
        loss_cfg = config.get("loss", {})
        thresholds = threshold_grid(config.get("evaluation", {}))

        lr = float(train_cfg.get("learning_rate", 1e-3))
        weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        epochs = int(train_cfg.get("epochs", 20))
        patience = int(train_cfg.get("early_stopping_patience", 4))
        lambda_hier = float(loss_cfg.get("hierarchical_lambda", 0.0))

        criterion = HierarchicalLoss(lambda_hier=lambda_hier)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_fmax, best_epoch, epochs_no_improve = -1.0, -1, 0
        best_checkpoint: Dict[str, Any] = {}
        history: List[Dict[str, Any]] = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_start = time.time()

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            sums = {"loss_total": 0.0, "loss_bce": 0.0, "loss_hier": 0.0}
            n_batches = 0
            for batch in train_loader:
                labels = batch["labels"].to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = _forward(model, batch, self.device)
                loss, parts = criterion(logits, labels, parent_child_edges=self.parent_child_edges,
                                        lambda_hier=lambda_hier)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {parts}")
                loss.backward()
                optimizer.step()
                for k in sums:
                    sums[k] += parts[k]
                n_batches += 1
            train_time = time.time() - t0

            val = self.evaluate(model, valid_loader, criterion, thresholds, lambda_hier)
            curve = val.pop("_curve")
            record = {
                "epoch": epoch,
                "train_loss": sums["loss_total"] / max(1, n_batches),
                "train_bce_loss": sums["loss_bce"] / max(1, n_batches),
                "train_hier_loss": sums["loss_hier"] / max(1, n_batches),
                **val,
                "train_time_sec": train_time,
                "epoch_time_sec": time.time() - t0,
            }
            history.append(record)
            if self.logger:
                self.logger.info(
                    f"epoch {epoch:02d} | train {record['train_loss']:.5f} (hier {record['train_hier_loss']:.2e})"
                    f" | val {record['val_loss']:.5f} | Fmax {record['val_fmax']:.4f}@{record['val_best_threshold']:.2f}"
                    f" | wFmax {record.get('val_wfmax', float('nan')):.4f} | viol {record['val_violation_rate']:.4f}"
                    f" | {record['epoch_time_sec']:.1f}s"
                )

            if record["val_fmax"] > best_fmax:
                best_fmax, best_epoch, epochs_no_improve = record["val_fmax"], epoch, 0
                best_checkpoint = {
                    "epoch": epoch,
                    "best_fmax": best_fmax,
                    "best_threshold": record["val_best_threshold"],
                    "val_metrics": {k: v for k, v in record.items()},
                    "val_curve": curve,
                    "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                    "config": config,
                    "metadata": metadata or {},
                }
                if checkpoint_path is not None:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = checkpoint_path.with_suffix(".pt.tmp")
                    torch.save(best_checkpoint, tmp)
                    tmp.replace(checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if self.logger:
                        self.logger.info(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                    break

        best_checkpoint["best_epoch"] = best_epoch
        best_checkpoint["train_time_total_sec"] = time.time() - t_start
        best_checkpoint["peak_gpu_mem_gb"] = (
            torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        )
        return best_checkpoint, history
