import numpy as np
from src.evaluation.metrics import compute_fmax


def test_compute_fmax_perfect_and_imperfect():
    labels = np.array([[1, 1, 0], [1, 0, 1]], dtype=np.float32)
    scores = np.array([[0.9, 0.8, 0.1], [0.85, 0.2, 0.95]], dtype=np.float32)

    res = compute_fmax(scores, labels, thresholds=np.linspace(0.1, 0.9, 9))
    assert np.isclose(res["fmax"], 1.0)
    assert np.isclose(res["precision_at_fmax"], 1.0)
    assert np.isclose(res["recall_at_fmax"], 1.0)


def test_streaming_accumulator_matches_reference_with_and_without_ia():
    from src.evaluation.metrics import FmaxAccumulator

    rng = np.random.default_rng(1)
    s = rng.random((200, 30)).astype(np.float32)
    y = (rng.random((200, 30)) < 0.15).astype(np.float32)
    w = rng.random(30).astype(np.float32)
    t = np.linspace(0.01, 0.99, 99)
    for ia in (None, w):
        ref = compute_fmax(s, y, t, ia)
        acc = FmaxAccumulator(t, ia_weights=ia, device="cpu")
        acc.update(s[:77], y[:77])
        acc.update(s[77:], y[77:])
        got = acc.compute()
        assert np.isclose(ref["fmax"], got["fmax"], atol=1e-6)
        assert np.isclose(ref["best_threshold"], got["best_threshold"], atol=1e-6)


def test_out_of_vocab_truth_lowers_recall():
    from src.evaluation.metrics import FmaxAccumulator

    s = np.array([[0.9, 0.1]], dtype=np.float32)
    y = np.array([[1.0, 0.0]], dtype=np.float32)
    acc = FmaxAccumulator(np.array([0.5]), device="cpu")
    acc.update(s, y, extra_true_weight=np.array([1.0]))  # one more true term outside the vocab
    res = acc.compute()
    assert np.isclose(res["recall_at_fmax"], 0.5) and np.isclose(res["precision_at_fmax"], 1.0)


def test_fmax_no_positive_proteins_and_threshold_monotonic_recall():
    res = compute_fmax(np.zeros((2, 3)), np.zeros((2, 3)))
    assert res["fmax"] == 0.0
    rng = np.random.default_rng(2)
    s, y = rng.random((50, 10)), (rng.random((50, 10)) < 0.3).astype(float)
    rec = [c["recall"] for c in compute_fmax(s, y)["curve"]]
    assert all(a >= b - 1e-12 for a, b in zip(rec, rec[1:]))
