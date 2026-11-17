import torch
import torch.nn as nn


class TaxonomyEncoder(nn.Module):
    """Learned embedding table for NCBI taxonomy indices (with index 0 reserved for UNK/missing)."""

    def __init__(self, num_taxa: int, embedding_dim: int = 64, unk_idx: int = 0):
        super().__init__()
        self.num_taxa = num_taxa
        self.embedding_dim = embedding_dim
        self.unk_idx = unk_idx
        self.embedding = nn.Embedding(num_embeddings=max(1, num_taxa), embedding_dim=embedding_dim)

    def encode_taxonomy(self, taxonomy_ids: torch.Tensor) -> torch.Tensor:
        """Map taxonomy indices [batch_size] to dense representations [batch_size, d_tax]."""
        safe_ids = taxonomy_ids.clone().long()
        out_of_bounds = (safe_ids < 0) | (safe_ids >= self.num_taxa)
        safe_ids[out_of_bounds] = self.unk_idx
        return self.embedding(safe_ids)

    def forward(self, taxonomy_ids: torch.Tensor) -> torch.Tensor:
        return self.encode_taxonomy(taxonomy_ids)
