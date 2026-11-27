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


class SparseBatchLoader:
    """Fast minibatch iterator over (embeddings, optional taxonomy idx, CSR labels).

    Slices whole CSR row blocks at once and densifies only the minibatch on the target device
    (plan §4/§7: sparse on disk, dense per minibatch). Yields the same dict keys as
    ``ProteinFunctionDataset`` (embedding / taxonomy_idx / labels) plus ``index``.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: Optional[sparse.csr_matrix] = None,
        taxonomy_indices: Optional[np.ndarray] = None,
        batch_size: int = 256,
        shuffle: bool = False,
        seed: int = 42,
        device: Optional[str] = None,
        drop_last: bool = False,
    ):
        # Reuse ProteinFunctionDataset's alignment checks
        ProteinFunctionDataset(embeddings, labels, taxonomy_indices)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.emb = torch.as_tensor(np.ascontiguousarray(embeddings), dtype=torch.float32).to(self.device)
        self.tax = None if taxonomy_indices is None else torch.as_tensor(
            np.asarray(taxonomy_indices), dtype=torch.long).to(self.device)
        self.labels = None if labels is None else labels.tocsr().astype(np.float32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = self.emb.shape[0]
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def _dense_labels(self, idx: np.ndarray) -> torch.Tensor:
        sub = self.labels[idx]
        coo = sub.tocoo()
        out = torch.zeros(sub.shape, dtype=torch.float32, device=self.device)
        if coo.nnz:
            r = torch.as_tensor(coo.row, dtype=torch.long, device=self.device)
            c = torch.as_tensor(coo.col, dtype=torch.long, device=self.device)
            out[r, c] = torch.as_tensor(coo.data, device=self.device)
        return out

    def __iter__(self):
        n = self.emb.shape[0]
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for s in range(0, n, self.batch_size):
            idx = order[s : s + self.batch_size]
            if self.drop_last and len(idx) < self.batch_size:
                break
            t_idx = torch.as_tensor(idx, device=self.device)
            batch = {"embedding": self.emb[t_idx], "index": idx}
            if self.tax is not None:
                batch["taxonomy_idx"] = self.tax[t_idx]
            if self.labels is not None:
                batch["labels"] = self._dense_labels(idx)
            yield batch
