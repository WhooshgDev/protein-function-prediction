# PFP-CAFA5: Protein Function Prediction Pipeline

An end-to-end machine learning pipeline for **Protein Function Prediction (PFP)** on the **CAFA 5** dataset. Given an amino acid sequence (and optional NCBI taxonomy ID), the system predicts Gene Ontology (GO) terms across all three ontology aspects:
- **MF** — Molecular Function (`F`)
- **BP** — Biological Process (`P`)
- **CC** — Cellular Component (`C`)

---

## Experiment Matrix
| Experiment ID | Model Variant | Description | Config File |
| :--- | :--- | :--- | :--- |
| **E0** | Baseline | Frozen ESM-2 (150M) + Mean Residue Pooling + MLP + BCE | `configs/baseline.yaml` |
| **E1** | Hierarchical Loss | Baseline + GO DAG Parent–Child Violation Penalty | `configs/hierarchical.yaml` |
| **E2** | Taxonomy Embedding | Baseline + Learned NCBI Taxonomy Embedding Fusion | `configs/taxonomy.yaml` |
| **E3** | Full Combined | Baseline + Hierarchical Loss + Taxonomy Embedding | `configs/combined.yaml` |

---

## Repository Structure
```text
PFP-CAFA5/
├── configs/
│   ├── baseline.yaml           # E0: Baseline configuration
│   ├── hierarchical.yaml       # E1: GO hierarchical penalty enabled
│   ├── taxonomy.yaml           # E2: Taxonomy embedding enabled
│   └── combined.yaml           # E3: Full combined model
├── data/
│   ├── raw/                    # Raw CAFA 5 dataset files (ignored by Git)
│   ├── processed/              # Cleaned IDs, GO vocabularies, sparse CSR label matrices, split manifest
│   └── embeddings/             # Cached ESM-2 embeddings (.npy / shards)
├── src/
│   ├── data/
│   │   ├── preprocessing.py    # FASTA & annotation TSV parsers, canonical ID normalization, QC stats
│   │   ├── ontology.py         # GO OBO parser, obsolete term resolution, ancestor propagation, CSR matrices
│   │   ├── split.py            # Sequence/cluster-based train/val split & leakage auditing
│   │   ├── dataset.py          # PyTorch Dataset with on-the-fly sparse-to-dense minibatch label loading
│   │   └── taxonomy.py         # NCBI Taxonomy ID parser, deterministic vocabulary & UNK handling
│   ├── models/
│   │   ├── esm_encoder.py      # Frozen ESM-2 (150M) residue-only mean pooling (excluding BOS/EOS/PAD)
│   │   ├── classifier.py       # Baseline MLP head returning raw logits
│   │   ├── hierarchical_loss.py# BCEWithLogitsLoss + GO hierarchy violation penalty & diagnostics
│   │   ├── taxonomy_encoder.py # Learned taxonomy embedding with reserved UNK index
│   │   └── fusion.py           # Sequence + Taxonomy feature concatenation/fusion module
│   ├── training/
│   │   ├── trainer.py          # Training loop, AdamW, early stopping on validation Fmax, checkpointing
│   │   └── train.py            # High-level training runner
│   ├── evaluation/
│   │   ├── metrics.py          # Threshold sweep for protein-centric Precision, Recall, F1, and Fmax
│   │   ├── predict.py          # Batched inference with sigmoid probability output
│   │   ├── export_cafa.py      # CAFA submission formatter & schema validator
│   │   └── evaluate_cafa.py    # Official CAFA evaluator wrapper & smoke test runner
│   └── utils/
│       ├── config.py           # YAML configuration loader & SHA-256 config fingerprinting
│       ├── seed.py             # Deterministic seeding across Python, NumPy, and PyTorch
│       ├── logger.py           # Standardized console & file logging
│       └── io.py               # Atomic JSON writes & file/list SHA-256 hashing
├── scripts/
│   ├── prepare_data.py         # Step 1: Data ingestion, ontology processing, label matrix & split creation
│   ├── extract_embeddings.py   # Step 2: Frozen ESM-2 embedding extraction & caching
│   ├── train.py                # Step 3: Model training per GO aspect (F, P, C)
│   ├── predict.py              # Step 4: Checkpoint inference & prediction export
│   ├── evaluate.py             # Step 5: Internal validation Fmax & CAFA evaluator execution
│   └── run_pipeline.py         # Step 6: End-to-end pipeline execution
├── tests/                      # Unit and integration tests
├── checkpoints/                # Saved model checkpoints (ignored by Git)
├── predictions/                # Exported prediction files (ignored by Git)
├── results/                    # Evaluation metrics, learning curves, and audit reports
├── logs/                       # Execution logs
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Setup & Installation

```bash
pip install -r requirements.txt
```

### Dataset Preparation (`data/raw/`)
Download the official **CAFA 5** dataset and place the raw files into `data/raw/`:
- `train_sequences.fasta`
- `train_terms.tsv`
- `train_taxonomy.tsv`
- `go-basic.obo`
- `testsuperset.fasta`
- `IA.txt`
- `sample_submission.csv` (or `.tsv`)

> **Note:** Large raw data files, cached embeddings, checkpoints, and prediction outputs are excluded from Git via `.gitignore`.

---

## Command-Line Workflow

```bash
# 1. Prepare dataset, ontology vocabularies, propagated labels, and train/val split
python scripts/prepare_data.py --config configs/baseline.yaml

# 2. Extract frozen ESM-2 residue-pooled embeddings (640d)
python scripts/extract_embeddings.py --config configs/baseline.yaml --split all

# 3. Train model (run per aspect: F, P, C)
python scripts/train.py --config configs/baseline.yaml --aspect F

# 4. Run inference & export predictions
python scripts/predict.py --config configs/baseline.yaml --checkpoint checkpoints/baseline_F_best.pt --input testsuperset

# 5. Evaluate predictions
python scripts/evaluate.py --config configs/baseline.yaml --predictions predictions/baseline_val.csv --split validation

# 6. Run full end-to-end pipeline
python scripts/run_pipeline.py --config configs/baseline.yaml
```

---

## Running Tests

```bash
pytest tests/
```
