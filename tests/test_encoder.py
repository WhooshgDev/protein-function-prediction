import torch
from src.models.classifier import MLPClassifier
from src.models.esm_encoder import mean_pool_residues


def test_mean_pool_residues_excludes_bos_eos_pad():
    # Batch size 1, sequence length 5: [BOS, res1, res2, EOS, PAD]
    hidden = torch.tensor(
        [[[100.0, 100.0], [2.0, 4.0], [4.0, 8.0], [200.0, 200.0], [999.0, 999.0]]]
    )
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    special_tokens_mask = torch.tensor([[1, 0, 0, 1, 1]])

    pooled = mean_pool_residues(hidden, attention_mask, special_tokens_mask)
    assert pooled.shape == (1, 2)
    assert torch.allclose(pooled, torch.tensor([[3.0, 6.0]]))


def test_mlp_classifier_output_shape():
    model = MLPClassifier(input_dim=640, hidden_dim=512, num_classes=25, dropout=0.3)
    x = torch.randn(4, 640)
    logits = model(x)
    assert logits.shape == (4, 25)
    assert torch.isfinite(logits).all()
