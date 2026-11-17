from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalLoss(nn.Module):
    """Combined BCEWithLogitsLoss and GO hierarchy violation penalty:
    L_total = L_BCE(logits, targets) + lambda_hier * mean(max(0, s_child - s_parent)^2)
    where s = sigmoid(logits) and edges[:, 0] = parent_idx, edges[:, 1] = child_idx.
    """

    def __init__(
        self,
        lambda_hier: float = 0.0,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.lambda_hier = float(lambda_hier)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        parent_child_edges: Optional[torch.Tensor] = None,
        lambda_hier: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        lam = self.lambda_hier if lambda_hier is None else float(lambda_hier)
        bce_loss = self.bce(logits, targets)

        if parent_child_edges is None or parent_child_edges.numel() == 0 or lam == 0.0:
            hier_loss = torch.zeros((), dtype=logits.dtype, device=logits.device)
        else:
            probs = torch.sigmoid(logits)
            parent_idx = parent_child_edges[:, 0].long()
            child_idx = parent_child_edges[:, 1].long()
            s_parent = probs[:, parent_idx]
            s_child = probs[:, child_idx]
            violations = F.relu(s_child - s_parent)
            hier_loss = torch.mean(violations ** 2)

        total_loss = bce_loss + lam * hier_loss
        metrics = {
            "loss_total": float(total_loss.detach().item()),
            "loss_bce": float(bce_loss.detach().item()),
            "loss_hier": float(hier_loss.detach().item()),
        }
        return total_loss, metrics


@torch.no_grad()
def compute_hierarchy_diagnostics(
    probs: torch.Tensor,
    parent_child_edges: torch.Tensor,
    margin: float = 0.0,
) -> Dict[str, float]:
    """Compute hierarchy violation rate and mean violation magnitude over (parent, child) edges."""
    if parent_child_edges is None or parent_child_edges.numel() == 0:
        return {"violation_rate": 0.0, "mean_violation_magnitude": 0.0}

    parent_idx = parent_child_edges[:, 0].long()
    child_idx = parent_child_edges[:, 1].long()
    diff = probs[:, child_idx] - probs[:, parent_idx]
    violated = diff > margin
    violation_rate = float(violated.float().mean().item())
    mean_mag = float(F.relu(diff).mean().item())
    return {
        "violation_rate": violation_rate,
        "mean_violation_magnitude": mean_mag,
    }
