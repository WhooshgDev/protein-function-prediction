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


def test_mean_pool_without_special_mask_uses_positions():
    hidden = torch.tensor([[[100.0], [2.0], [4.0], [200.0], [999.0]],
                           [[100.0], [6.0], [200.0], [999.0], [999.0]]])
    attn = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    pooled = mean_pool_residues(hidden, attn)
    assert torch.allclose(pooled, torch.tensor([[3.0], [6.0]]))


def test_make_windows_chunk_covers_sequence_and_truncate_is_explicit():
    from src.models.esm_encoder import make_windows

    seq = "".join("ACDEFGHIKL"[i % 10] for i in range(2500))
    wins = make_windows(seq, max_length=1022, policy="chunk", overlap=256)
    assert all(len(w) <= 1022 for w in wins)
    assert wins[0] == seq[:1022] and seq.endswith(wins[-1])
    covered = set()
    start = 0
    for w in wins:
        covered.update(range(start, start + len(w)))
        start += 1022 - 256
    assert covered == set(range(len(seq)))
    assert make_windows(seq, 1022, "truncate", 0) == [seq[:1022]]
    assert make_windows("ACD", 1022, "chunk", 256) == ["ACD"]


def test_token_batches_respect_budget_and_cover_all():
    import numpy as np
    from src.models.esm_encoder import _token_batches

    lengths = np.array([5, 100, 50, 1000, 300, 7, 999])
    seen = []
    for idx in _token_batches(lengths, max_tokens=1100):
        assert len(idx) == 1 or (lengths[idx].max() + 2) * len(idx) <= 1100
        seen.extend(idx.tolist())
    assert sorted(seen) == list(range(len(lengths)))


def test_sequence_key_dedups_identical_sequences():
    from src.models.esm_encoder import sequence_key

    assert sequence_key("MKV") == sequence_key("MKV") != sequence_key("MKA")


def test_taxonomy_unk_dropout_trains_unk_row_only_in_train_mode():
    from src.models.fusion import SequenceTaxonomyFusionModel

    torch.manual_seed(0)
    m = SequenceTaxonomyFusionModel(seq_dim=4, num_classes=3, hidden_dim=8, taxonomy_enabled=True,
                                    num_taxa=5, tax_dim=2, tax_unk_dropout=0.5)
    x, t = torch.randn(64, 4), torch.full((64,), 3, dtype=torch.long)
    m.train()
    m(x, t).sum().backward()
    assert m.tax_encoder.embedding.weight.grad[0].abs().sum() > 0  # UNK row receives gradient
    m.eval()
    m.zero_grad()
    out1, out2 = m(x, t), m(x, t)
    assert torch.equal(out1, out2)  # no dropout at inference
    out1.sum().backward()
    assert m.tax_encoder.embedding.weight.grad[0].abs().sum() == 0
