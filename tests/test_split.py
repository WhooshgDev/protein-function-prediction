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
