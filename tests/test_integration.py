"""Integration smoke test (plan §18): ~300 synthetic proteins through
prepare -> ontology -> labels -> split -> (synthetic) embeddings -> train 1 epoch -> predict -> evaluate -> export.

ESM-2 is replaced by synthetic label-correlated embeddings so the test runs offline on CPU;
the real encoder is covered by tests/test_encoder.py and the extraction sanity checks.
"""
from pathlib import Path

import numpy as np
import pytest

OBO = """format-version: 1.2
data-version: test/2026-01-01

[Term]
id: GO:0003674
name: molecular_function
namespace: molecular_function

[Term]
id: GO:0005488
name: binding
namespace: molecular_function
is_a: GO:0003674

[Term]
id: GO:0005515
name: protein binding
namespace: molecular_function
is_a: GO:0005488

[Term]
id: GO:0003824
name: catalytic activity
namespace: molecular_function
is_a: GO:0003674

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process

[Term]
id: GO:0008152
name: metabolic process
namespace: biological_process
is_a: GO:0008150

[Term]
id: GO:0006412
name: translation
namespace: biological_process
is_a: GO:0008152

[Term]
id: GO:0005575
name: cellular_component
namespace: cellular_component

[Term]
id: GO:0005634
name: nucleus
namespace: cellular_component
relationship: part_of GO:0005575

[Term]
id: GO:0005737
name: cytoplasm
namespace: cellular_component
is_a: GO:0005575

[Term]
id: GO:0099999
name: obsolete thing
namespace: molecular_function
is_obsolete: true
replaced_by: GO:0005515
"""

LEAVES = {"F": ["GO:0005515", "GO:0003824"], "P": ["GO:0006412"], "C": ["GO:0005634", "GO:0005737"]}
ANC = {
    "GO:0005515": ["GO:0005488", "GO:0003674"], "GO:0003824": ["GO:0003674"],
    "GO:0006412": ["GO:0008152", "GO:0008150"], "GO:0005634": ["GO:0005575"], "GO:0005737": ["GO:0005575"],
}
CODES = {"F": "MFO", "P": "BPO", "C": "CCO"}


def _make_raw(raw: Path, n: int = 300, n_test: int = 40):
    rng = np.random.default_rng(0)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    (raw / "Train").mkdir(parents=True)
    (raw / "Test").mkdir()
    (raw / "Train" / "go-basic.obo").write_text(OBO, encoding="utf-8")
    pids = [f"P{i:05d}" for i in range(n)]
    seqs = ["".join(rng.choice(aa, rng.integers(30, 120))) for _ in range(n)]
    seqs[1] = seqs[0]  # exact duplicate must stay on one side of the split
    with open(raw / "Train" / "train_sequences.fasta", "w") as f:
        for p, s in zip(pids, seqs):
            f.write(f">{p} sp|{p}|X_HUMAN desc\n{s}\n")
    with open(raw / "Train" / "train_terms.tsv", "w") as f:
        f.write("EntryID\tterm\taspect\n")
        for i, p in enumerate(pids):
            for a, leaves in LEAVES.items():
                if rng.random() < 0.8:
                    leaf = leaves[i % len(leaves)]
                    for g in [leaf] + ANC[leaf]:
                        f.write(f"{p}\t{g}\t{CODES[a]}\n")
            if i == 5:
                f.write(f"{p}\tGO:0099999\tMFO\n")  # obsolete -> remapped
    with open(raw / "Train" / "train_taxonomy.tsv", "w") as f:
        f.write("EntryID\ttaxonomyID\n")
        for i, p in enumerate(pids):
            f.write(f"{p}\t{[9606, 10090, 7227][i % 3]}\n")
    with open(raw / "Test" / "testsuperset.fasta", "w") as f:
        for j in range(n_test):
            f.write(f">T{j:04d}\t{[9606, 4932][j % 2]}\n{''.join(rng.choice(aa, 60))}\n")
    with open(raw / "IA.txt", "w") as f:
        for g in ["GO:0003674", "GO:0005488", "GO:0005515", "GO:0003824", "GO:0008150", "GO:0008152",
                  "GO:0006412", "GO:0005575", "GO:0005634", "GO:0005737"]:
            f.write(f"{g}\t{0.0 if g in ('GO:0003674', 'GO:0008150', 'GO:0005575') else 1.5}\n")
    return pids


@pytest.mark.filterwarnings("ignore")
def test_end_to_end_mini_pipeline(tmp_path: Path):
    from src.data.artifacts import ProcessedData, load_ia
    from src.data.ontology import GOOntology
    from src.data.pipeline import prepare_data
    from src.evaluation.evaluate_cafa import evaluate_prediction_file, load_ground_truth
    from src.evaluation.export_cafa import validate_submission
    from src.evaluation.predict import predict_and_export
    from src.training.train import checkpoint_path_for, run_training
    from src.utils.io import save_json_atomic

    raw = tmp_path / "raw"
    _make_raw(raw)
    config = {
        "experiment_name": "itest",
        "seed": 1,
        "data": {
            "raw_dir": str(raw),
            "raw_files": {"test_sequences": "Test/testsuperset.fasta", "test_taxon_list": "Test/none.tsv",
                          "sample_submission": "none.tsv"},
            "processed_dir": str(tmp_path / "processed"),
            "embedding_dir": str(tmp_path / "emb"),
            "split": "cluster", "val_ratio": 0.2,
        },
        "model": {"esm_name": "synthetic", "hidden_dim": 32, "dropout": 0.0},
        "training": {"learning_rate": 0.01, "weight_decay": 0.0, "batch_size": 32, "epochs": 1,
                     "early_stopping_patience": 1},
        "loss": {"hierarchical_lambda": 0.1},
        "taxonomy": {"enabled": True, "embedding_dim": 8},
        "evaluation": {"aspects": ["F", "P", "C"], "threshold_min": 0.01, "threshold_max": 0.99, "threshold_steps": 99},
        "export": {"min_score": 0.01, "max_terms_per_protein": 1500},
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "results_dir": str(tmp_path / "results"),
    }

    # 1. prepare
    out = prepare_data(config)
    qc = out["qc"]
    assert qc["annotations"]["pairs_remapped_obsolete_or_alt"] == 1
    assert out["split_manifest"]["leakage_audit"]["exact_sequence_overlap"] == 0
    assert out["manifest"]["vocab_sizes"] == {"F": 4, "P": 3, "C": 3}
    for a in "FPC":
        assert qc["labels"][a]["label_hierarchy_violations"] == 0
    pdata = ProcessedData(config["data"]["processed_dir"], config["data"]["embedding_dir"])

    # 2. synthetic embeddings carrying label signal, written in the extractor's file layout
    rng = np.random.default_rng(0)
    emb_dir = Path(config["data"]["embedding_dir"])
    emb_dir.mkdir()
    for split in ("train", "valid", "test"):
        ids = pdata.split_ids(split)
        if split == "test":
            x = rng.normal(size=(len(ids), 16)).astype(np.float32)
        else:
            y = np.hstack([pdata.labels_for(a, ids).toarray() for a in "FPC"])
            x = np.hstack([y, rng.normal(scale=0.1, size=(len(ids), 16 - y.shape[1]))]).astype(np.float32)
        np.save(emb_dir / f"embeddings_{split}.npy", x)
        save_json_atomic(ids, emb_dir / f"embeddings_{split}_ids.json")

    # 3. train 1 epoch per aspect
    for a in "FPC":
        summary = run_training(config, a)
        assert summary["epochs_run"] == 1 and checkpoint_path_for(config, a).exists()
        assert np.isfinite(summary["val_metrics_at_best"]["val_loss"])

    # 4. predict + export (validation and test)
    ckpts = {a: checkpoint_path_for(config, a) for a in "FPC"}
    for split in ("valid", "test"):
        x, ids = pdata.embeddings(split)
        tax = pdata.taxonomy_indices(split, ids)
        path = tmp_path / f"pred_{split}.tsv"
        s = predict_and_export(ckpts, x, ids, path, taxonomy_indices=tax, export_cfg=config["export"],
                               expected_vocab_hashes=pdata.manifest["vocab_hashes"], device="cpu")
        assert s["num_covered_proteins"] == len(ids)
        check = validate_submission(path, expected_ids=ids)
        assert check["valid"], check["errors"]

    # 5. evaluate
    onto = GOOntology()
    onto.parse_obo(raw / "Train" / "go-basic.obo")
    gt = load_ground_truth(pdata.proc / "propagated_annotations.tsv", pdata.split_ids("valid"))
    res = evaluate_prediction_file(tmp_path / "pred_valid.tsv", gt, onto, ia=load_ia(str(raw / "IA.txt")), device="cpu")
    for a in "FPC":
        m = res["aspects"][a]
        assert 0.0 <= m["fmax"] <= 1.0 and 0.0 <= m["wfmax"] <= 1.0
        assert m["num_gt_proteins"] > 0
