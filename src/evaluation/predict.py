from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[str] = None,
) -> np.ndarray:
    """Run inference over DataLoader, apply sigmoid to logits, and return (N, K) probabilities in [0, 1]."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    preds = []
    for batch in data_loader:
        emb = batch["embedding"].to(device)
        tax_idx = batch.get("taxonomy_idx")
        if tax_idx is not None:
            tax_idx = tax_idx.to(device)
            logits = model(emb, tax_idx)
        else:
            logits = model(emb)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)

    if not preds:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(preds)
