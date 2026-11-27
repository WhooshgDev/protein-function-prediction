from typing import Optional
import torch
import torch.nn as nn
from src.models.classifier import MLPClassifier
from src.models.taxonomy_encoder import TaxonomyEncoder


class SequenceTaxonomyFusionModel(nn.Module):
    """Fuses sequence embedding h_seq (640d) with optional taxonomy embedding h_tax (d_tax) before MLP."""

    def __init__(
        self,
        seq_dim: int = 640,
        num_classes: int = 100,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        taxonomy_enabled: bool = False,
        num_taxa: int = 1,
        tax_dim: int = 64,
        tax_unk_dropout: float = 0.0,
    ):
        super().__init__()
        self.taxonomy_enabled = taxonomy_enabled
        # Training-time probability of replacing a taxon with UNK (index 0). Without it the UNK row
        # never gets gradients (every train taxon is in the vocab) and unseen taxa map to a random vector.
        self.tax_unk_dropout = float(tax_unk_dropout)
        if taxonomy_enabled:
            self.tax_encoder = TaxonomyEncoder(num_taxa=num_taxa, embedding_dim=tax_dim)
            in_dim = seq_dim + tax_dim
        else:
            self.tax_encoder = None
            in_dim = seq_dim

        self.classifier = MLPClassifier(
            input_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        seq_embedding: torch.Tensor,
        taxonomy_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.taxonomy_enabled:
            if taxonomy_idx is None:
                taxonomy_idx = torch.zeros(
                    seq_embedding.size(0), dtype=torch.long, device=seq_embedding.device
                )
            if self.training and self.tax_unk_dropout > 0:
                drop = torch.rand(taxonomy_idx.shape, device=taxonomy_idx.device) < self.tax_unk_dropout
                taxonomy_idx = taxonomy_idx.masked_fill(drop, self.tax_encoder.unk_idx)
            tax_emb = self.tax_encoder(taxonomy_idx)
            x = torch.cat([seq_embedding, tax_emb], dim=-1)
        else:
            x = seq_embedding
        return self.classifier(x)
