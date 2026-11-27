from pathlib import Path
from src.data.preprocessing import normalize_protein_id, parse_annotations, parse_fasta
from src.data.taxonomy import UNK_TAXON_INDEX, build_taxonomy_vocabulary, encode_taxonomy_ids


def test_normalize_protein_id():
    assert normalize_protein_id(">sp|P12345|PROT_HUMAN Some desc") == "P12345"
    assert normalize_protein_id(">P99999 description") == "P99999"


def test_parse_fasta_and_deduplication(tmp_path: Path):
    fasta_file = tmp_path / "sample.fasta"
    fasta_file.write_text(
        ">sp|P00001|A\nACDEFG\n>sp|P00001|A_dup\nACDEFG\n>P00002\n\n>P00003\nMKWVTFISLL\n",
        encoding="utf-8",
    )
    pids, seqs, stats = parse_fasta(fasta_file)
    assert pids == ["P00001", "P00003"]
    assert seqs == ["ACDEFG", "MKWVTFISLL"]
    assert stats["duplicate_ids"] == 1
    assert stats["empty_records"] == 1


def test_parse_annotations_deduplicates(tmp_path: Path):
    tsv_file = tmp_path / "terms.tsv"
    tsv_file.write_text(
        "EntryID\tterm\taspect\nP00001\tGO:0003674\tF\nP00001\tGO:0003674\tF\nP00002\tGO:0008150\tP\n",
        encoding="utf-8",
    )
    records = parse_annotations(tsv_file)
    assert len(records) == 2
    assert records[0] == ("P00001", "GO:0003674", "F")


def test_taxonomy_unk_handling():
    vocab = build_taxonomy_vocabulary([9606, 10090, 9606])
    assert vocab[-1] == UNK_TAXON_INDEX
    raw_ids, mapped = encode_taxonomy_ids(
        ["P1", "P2", "P3"],
        {"P1": 9606, "P2": 7227},  # P2 is unseen (7227), P3 is missing
        tax_vocab=vocab,
    )
    assert mapped[0] == vocab[9606]
    assert mapped[1] == UNK_TAXON_INDEX
    assert mapped[2] == UNK_TAXON_INDEX


def test_fasta_header_field_and_sparse_batch_loader(tmp_path: Path):
    import numpy as np
    from scipy import sparse
    from src.data.dataset import SparseBatchLoader
    from src.data.preprocessing import parse_fasta_header_field

    fa = tmp_path / "t.fasta"
    fa.write_text(">Q1\t9606\nMKV\n>Q2\t10090\nMKA\n", encoding="utf-8")
    assert parse_fasta_header_field(fa) == {"Q1": "9606", "Q2": "10090"}

    rng = np.random.default_rng(0)
    y = sparse.random(37, 11, density=0.2, format="csr", random_state=0)
    y.data[:] = 1.0
    x = rng.random((37, 4)).astype(np.float32)
    loader = SparseBatchLoader(x, y, np.arange(37), batch_size=8, shuffle=True, device="cpu")
    seen = []
    for b in loader:
        idx = b["index"]
        assert np.allclose(b["labels"].numpy(), y[idx].toarray())
        assert np.allclose(b["embedding"].numpy(), x[idx])
        assert (b["taxonomy_idx"].numpy() == idx).all()
        seen.extend(idx.tolist())
    assert sorted(seen) == list(range(37)) and len(loader) == 5


def test_annotation_parser_header_detection(tmp_path: Path):
    tsv = tmp_path / "t.tsv"
    tsv.write_text("P1\tGO:0003674\tMFO\n", encoding="utf-8")  # no header row
    assert parse_annotations(tsv) == [("P1", "GO:0003674", "MFO")]
