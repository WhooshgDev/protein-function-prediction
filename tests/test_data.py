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
