import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_fmax
from src.models.hierarchical_loss import HierarchicalLoss, compute_hierarchy_diagnostics


class Trainer:
    """Ontology-agnostic trainer supporting BCE + optional GO hierarchical penalty and early stopping."""

    def __init__(
        self,
        device: Optional[str] = None,
        parent_child_edges: Optional[torch.Tensor] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.parent_child_edges = (
            parent_child_edges.to(self.device) if parent_child_edges is not None else None
        )

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        config: Dict[str, Any],
        checkpoint_path: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Contract entrypoint: Train model, track validation Fmax, and save best checkpoint."""
        model = model.to(self.device)
        train_cfg = config.get("training", {})
        loss_cfg = config.get("loss", {})
        eval_cfg = config.get("evaluation", {})

        lr = float(train_cfg.get("learning_rate", 1e-3))
        weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        epochs = int(train_cfg.get("epochs", 20))
        patience = int(train_cfg.get("early_stopping_patience", 4))
        lambda_hier = float(loss_cfg.get("hierarchical_lambda", 0.0))

        t_min = float(eval_cfg.get("threshold_min", 0.01))
        t_max = float(eval_cfg.get("threshold_max", 0.99))
        t_steps = int(eval_cfg.get("threshold_steps", 99))
        thresholds = np.linspace(t_min, t_max, t_steps)

        criterion = HierarchicalLoss(lambda_hier=lambda_hier)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_fmax = -1.0
        best_epoch = -1
        epochs_no_improve = 0
        best_checkpoint: Dict[str, Any] = {}
        history: List[Dict[str, Any]] = []

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            train_loss_sum = 0.0
            train_bce_sum = 0.0
            train_hier_sum = 0.0
            n_batches = 0

            for batch in train_loader:
                emb = batch["embedding"].to(self.device)
                labels = batch["labels"].to(self.device)
                tax_idx = batch.get("taxonomy_idx")

                optimizer.zero_grad()
                if tax_idx is not None:
                    logits = model(emb, tax_idx.to(self.device))
                else:
                    logits = model(emb)

                loss, loss_Parts = criterion(
                    logits=logits,
                    targets=labels,
                    parent_child_edges=self.parent_child_edges,
                    lambda_hier=lambda_hier,
                )
                loss.backward()
                optimizer.step()

                train_loss_sum += loss_Parts["loss_total"]
                train_bce_sum += loss_Parts["loss_bce"]
                train_hier_sum += loss_Parts["loss_hier"]
                n_batches += 1

            # Validation pass
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            all_probs = []
            all_targets = []

            with torch.no_grad():
                for batch in valid_loader:
                    emb = batch["embedding"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    tax_idx = batch.get("taxonomy_idx")

                    if tax_idx is not None:
                        logits = model(emb, tax_idx.to(self.device))
                    else:
                        logits = model(emb)

                    v_loss, _ = criterion(
                        logits=logits,
                        targets=labels,
                        parent_child_edges=self.parent_child_edges,
                        lambda_hier=lambda_hier,
                    )
                    val_loss_sum += float(v_loss.item())
                    val_batches += 1

                    probs = torch.sigmoid(logits)
                    all_probs.append(probs.cpu().numpy())
                    all_targets.append(labels.cpu().numpy())

            val_probs_np = np.vstack(all_probs) if all_probs else np.zeros((0, 0), dtype=np.float32)
            val_targets_np = np.vstack(all_targets) if all_targets else np.zeros((0, 0), dtype=np.float32)

            val_metrics = compute_fmax(
                scores=val_probs_np,
                labels=val_targets_np,
                thresholds=thresholds,
            )
            hier_diag = {"violation_rate": 0.0, "mean_violation_magnitude": 0.0}
            if self.parent_child_edges is not None and val_probs_np.size > 0:
                hier_diag = compute_hierarchy_diagnostics(
                    torch.from_numpy(val_probs_np).to(self.device),
                    self.parent_child_edges,
                )

            epoch_time = time.time() - t0
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss_sum / max(1, n_batches),
                "train_bce_loss": train_bce_sum / max(1, n_batches),
                "train_hier_loss": train_hier_sum / max(1, n_batches),
                "val_loss": val_loss_sum / max(1, val_batches),
                "val_fmax": val_metrics["fmax"],
                "val_precision": val_metrics["precision_at_fmax"],
                "val_recall": val_metrics["recall_at_fmax"],
                "val_best_threshold": val_metrics["best_threshold"],
                "val_violation_rate": hier_diag["violation_rate"],
                "val_mean_violation_mag": hier_diag["mean_violation_magnitude"],
                "epoch_time_sec": epoch_time,
            }
            history.append(epoch_record)

            if val_metrics["fmax"] > best_fmax:
                best_fmax = val_metrics["fmax"]
                best_epoch = epoch
                epochs_no_improve = 0
                best_checkpoint = {
                    "epoch": epoch,
                    "best_fmax": best_fmax,
                    "best_threshold": val_metrics["best_threshold"],
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                    "config": config,
                    "metadata": metadata or {},
                }
                if checkpoint_path is not None:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_checkpoint, checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        best_checkpoint["best_epoch"] = best_epoch
        return best_checkpoint, history
