from src.data.split import audit_split_leakage, create_split


def test_split_no_leakage():
    pids = [f"P{i}" for i in range(20)]
    # P0 and P1 share an identical sequence
    seqs = ["SEQ_SHARED", "SEQ_SHARED"] + [f"SEQ_{i}" for i in range(2, 20)]
    train_ids, valid_ids, manifest = create_split(pids, sequences=seqs, val_ratio=0.2, seed=42)

    id_to_seq = dict(zip(pids, seqs))
    audit = audit_split_leakage(train_ids, valid_ids, id_to_seq)
    assert audit["id_overlap"] == 0
    assert audit["exact_sequence_overlap"] == 0
    assert manifest["num_train"] + manifest["num_valid"] == 20


def test_minhash_groups_near_duplicates_and_split_keeps_clusters_together():
    import numpy as np
    from src.data.split import minhash_clusters

    rng = np.random.default_rng(0)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    base = "".join(rng.choice(aa, 300))
    near = base[:150] + ("A" if base[150] != "A" else "C") + base[151:]  # 1 substitution
    others = ["".join(rng.choice(aa, 300)) for _ in range(30)]
    seqs = [base, near] + others
    cids, info = minhash_clusters(seqs, k=5, num_hashes=64, bands=16)
    assert cids[0] == cids[1]
    assert len(set(cids[2:].tolist())) == 30  # unrelated random sequences stay apart
    pids = [f"P{i}" for i in range(len(seqs))]
    clusters = {p: str(c) for p, c in zip(pids, cids)}
    tr, va, man = create_split(pids, sequences=seqs, clusters=clusters, val_ratio=0.3, seed=1, method="minhash_cluster")
    audit = audit_split_leakage(tr, va, dict(zip(pids, seqs)), clusters)
    assert audit["shared_clusters"] == 0 and audit["id_overlap"] == 0
    assert man["method"] == "minhash_cluster"


def test_split_is_deterministic():
    pids = [f"P{i}" for i in range(50)]
    seqs = [f"SEQ{i}" for i in range(50)]
    a = create_split(pids, sequences=seqs, seed=7)
    b = create_split(pids, sequences=seqs, seed=7)
    assert a[0] == b[0] and a[1] == b[1]
