import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """Baseline MLP Classifier: Linear(in_dim, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, num_classes).
    Returns raw logits (no sigmoid in forward) for numerical stability with BCEWithLogitsLoss.
    """

    def __init__(
        self,
        input_dim: int = 640,
        hidden_dim: int = 512,
        num_classes: int = 100,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
