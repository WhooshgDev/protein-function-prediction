from typing import Optional
import numpy as np
from scipy import sparse
import torch
from torch.utils.data import Dataset


class ProteinFunctionDataset(Dataset):
    """PyTorch Dataset pairing precomputed ESM-2 embeddings, optional taxonomy indices, and sparse GO labels."""

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: Optional[sparse.csr_matrix] = None,
        taxonomy_indices: Optional[np.ndarray] = None,
    ):
        if labels is not None and embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Embeddings count ({embeddings.shape[0]}) != labels count ({labels.shape[0]})"
            )
        if taxonomy_indices is not None and embeddings.shape[0] != len(taxonomy_indices):
            raise ValueError(
                f"Embeddings count ({embeddings.shape[0]}) != taxonomy count ({len(taxonomy_indices)})"
            )
        self.embeddings = embeddings
        self.labels = labels
        self.taxonomy_indices = taxonomy_indices

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> dict:
        item = {
            "embedding": torch.as_tensor(self.embeddings[idx], dtype=torch.float32),
        }
        if self.taxonomy_indices is not None:
            item["taxonomy_idx"] = torch.tensor(int(self.taxonomy_indices[idx]), dtype=torch.long)
        if self.labels is not None:
            dense_label = self.labels[idx].toarray().squeeze(0)
            item["labels"] = torch.as_tensor(dense_label, dtype=torch.float32)
        return item
