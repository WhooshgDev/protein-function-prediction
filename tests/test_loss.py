import torch
import torch.nn as nn
from src.models.hierarchical_loss import HierarchicalLoss


def test_hierarchical_loss_zero_when_consistent_and_positive_when_violated():
    # Edge: parent = 0, child = 1
    edges = torch.tensor([[0, 1]], dtype=torch.long)
    targets = torch.tensor([[1.0, 0.0]])
    loss_fn = HierarchicalLoss(lambda_hier=1.0)

    # Case 1: parent logit (5.0) > child logit (-5.0) -> no hierarchy violation
    logits_consistent = torch.tensor([[5.0, -5.0]], requires_grad=True)
    _, metrics_ok = loss_fn(logits_consistent, targets, parent_child_edges=edges)
    assert metrics_ok["loss_hier"] == 0.0

    # Case 2: child logit (4.0) > parent logit (-4.0) -> positive hierarchy penalty
    logits_violated = torch.tensor([[-4.0, 4.0]], requires_grad=True)
    total_loss, metrics_bad = loss_fn(logits_violated, targets, parent_child_edges=edges)
    assert metrics_bad["loss_hier"] > 0.0
    total_loss.backward()
    assert logits_violated.grad is not None
    assert torch.all(logits_violated.grad != 0)


def test_hierarchical_loss_lambda_zero_matches_bce():
    edges = torch.tensor([[0, 1]], dtype=torch.long)
    logits = torch.randn(4, 5)
    targets = torch.randint(0, 2, (4, 5)).float()

    loss_fn = HierarchicalLoss(lambda_hier=0.0)
    total_loss, _ = loss_fn(logits, targets, parent_child_edges=edges)
    bce_ref = nn.BCEWithLogitsLoss()(logits, targets)
    assert torch.allclose(total_loss, bce_ref)


def test_hierarchical_loss_edge_direction_matters():
    # Same logits, reversed edge direction -> only one direction is a violation
    logits = torch.tensor([[3.0, -3.0]])  # node0 high, node1 low
    targets = torch.zeros(1, 2)
    loss_fn = HierarchicalLoss(lambda_hier=1.0)
    _, m_ok = loss_fn(logits, targets, parent_child_edges=torch.tensor([[0, 1]]))  # parent 0, child 1
    _, m_bad = loss_fn(logits, targets, parent_child_edges=torch.tensor([[1, 0]]))  # parent 1, child 0
    assert m_ok["loss_hier"] == 0.0 and m_bad["loss_hier"] > 0.0


def test_hierarchy_diagnostics_and_score_propagation():
    from src.evaluation.predict import propagate_scores_max
    from src.models.hierarchical_loss import compute_hierarchy_diagnostics

    edges = torch.tensor([[0, 1], [1, 2]])  # 0 -> 1 -> 2 (parent -> child)
    probs = torch.tensor([[0.1, 0.5, 0.9]])
    diag = compute_hierarchy_diagnostics(probs, edges)
    assert diag["violation_rate"] == 1.0
    fixed = propagate_scores_max(probs, edges)
    assert torch.allclose(fixed, torch.tensor([[0.9, 0.9, 0.9]]))
    assert compute_hierarchy_diagnostics(fixed, edges)["violation_rate"] == 0.0
