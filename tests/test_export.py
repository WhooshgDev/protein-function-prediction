from pathlib import Path
import numpy as np
import pytest
from src.evaluation.export_cafa import export_predictions


def test_export_predictions_valid_schema(tmp_path: Path):
    out_file = tmp_path / "sub.tsv"
    pids = ["P00001", "P00002"]
    vocab = {"GO:0003674": 0, "GO:0005488": 1}
    scores = np.array([[0.95, 0.40], [0.80, 0.0001]], dtype=np.float32)

    summary = export_predictions(pids, vocab, scores, out_file, schema_config={"min_score": 0.01})
    assert summary["num_target_proteins"] == 2
    assert summary["num_covered_proteins"] == 2
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("P00001\tGO:0003674\t0.9500")


def test_export_predictions_rejects_invalid_scores(tmp_path: Path):
    out_file = tmp_path / "bad.tsv"
    with pytest.raises(ValueError):
        export_predictions(
            ["P1"],
            {"GO:0003674": 0},
            np.array([[1.5]], dtype=np.float32),
            out_file,
        )


def test_streaming_writer_and_validator_roundtrip(tmp_path: Path):
    from src.evaluation.export_cafa import StreamingCafaWriter, validate_submission

    out = tmp_path / "sub.tsv"
    w = StreamingCafaWriter(out)
    w.write_protein("P1", ["GO:0003674", "GO:0005488"], np.array([0.9, 0.2]))
    w.write_protein("P2", ["GO:0008150"], np.array([1.0]))
    summary = w.close()
    assert summary["num_rows_written"] == 3
    assert out.read_text().splitlines()[0] == "P1\tGO:0003674\t0.900"  # sample_submission format
    ok = validate_submission(out, expected_ids=["P1", "P2"], max_terms_per_protein=1500)
    assert ok["valid"], ok["errors"]
    bad = validate_submission(out, expected_ids=["P1", "P2", "P3"])
    assert not bad["valid"] and bad["missing_targets"] == 1


def test_streaming_writer_rejects_bad_rows(tmp_path: Path):
    from src.evaluation.export_cafa import StreamingCafaWriter

    w = StreamingCafaWriter(tmp_path / "x.tsv")
    with pytest.raises(ValueError):
        w.write_protein("P1", ["GO:0003674", "GO:0003674"], np.array([0.5, 0.4]))
    with pytest.raises(ValueError):
        w.write_protein("P1", ["GO:0003674"], np.array([np.nan]))
    w.write_protein("P1", ["GO:0003674"], np.array([0.5]))
    with pytest.raises(ValueError):
        w.write_protein("P1", ["GO:0005488"], np.array([0.5]))
    w.close()
