# PFP-CAFA5 — Progress Log

Running log of what was done, what was discovered, and decisions made while implementing the plan in
`PFP CAFA 5 – Kế Hoạch Triển Khai Toàn Diện Dự Án (3 Thành Viên · 7 Ngày).docx`.
Newest step at the bottom.

---

## Step 0 — Starting point (2026-11-27)

**State of the repo:** one commit (`0f868ff`) with the skeleton. Library modules (parsers, ontology,
split, losses, metrics, exporter) were partly written; `prepare_data()` raised `NotImplementedError`,
`train.py` was a stub, and **all six `scripts/*.py` only loaded the config and logged a line** — no
pipeline step actually ran.

**Hardware:** NVIDIA RTX 5080 16 GB (Blackwell, `sm_120`), Windows 11. No system Python on PATH.

## Step 1 — Environment with uv

- `uv init --bare --python 3.12` → `pyproject.toml`; dependencies taken from `requirements.txt`.
- **Discovery:** Blackwell GPUs need CUDA ≥ 12.8 PyTorch wheels. The default PyPI torch would not use the GPU,
  so `torch` is pinned to the `pytorch-cu128` index via `[tool.uv.sources]`.
- `uv sync` → Python 3.12, `torch 2.11.0+cu128`, `transformers 5.x`. Verified: `torch.cuda.is_available() == True`,
  matmul on the RTX 5080 works.
- Exact versions are locked in `uv.lock` (plan §14 "pin versions").

Usage from now on: `uv run python scripts/<step>.py ...` and `uv run pytest`.

## Step 2 — Reading the plan & inventorying the raw data

The docx is the Vietnamese spec (24 sections): data → ontology → split → ESM-2 embeddings → MLP baseline (E0)
→ hierarchical loss (E1) → taxonomy embedding (E2) → combined (E3) → predict/export/evaluate, plus audits,
tests, reproducibility and a results report.

**Actual files in `data/raw/` (plan §3 says "don't guess paths/columns"):**

| File | Real format | Notes |
|---|---|---|
| `Train/train_sequences.fasta` | 142,246 records, header `>P20536 sp\|P20536\|UNG_VACCC ... OX=10249` | first token is the ID |
| `Train/train_terms.tsv` | header `EntryID  term  aspect`, ~5.3 M rows | **aspect is `BPO`/`MFO`/`CCO`, not `P`/`F`/`C`** |
| `Train/train_taxonomy.tsv` | header `EntryID  taxonomyID` | |
| `Train/go-basic.obo` | GO OBO | |
| `Test (Targets)/testsuperset.fasta` | 141,865 records, header `>Q9CQV8<TAB>10090` | **taxon ID is in the header**; folder name has spaces |
| `Test (Targets)/testsuperset-taxon-list.tsv` | `ID  Species` | list of target species |
| `IA.txt` | `GO:xxxxxxx<TAB>weight`, no header | information-accretion weights |
| `sample_submission.tsv` | **no header**, tab-separated, `protein  GO  score` (3 decimals) | `.tsv`, not `.csv` |

The README paths (`data/raw/train_sequences.fasta`, …) don't match this layout, so the code resolves raw
files through a configurable `data.raw_files` block instead of hard-coding names.

## Step 3 — Data preparation (`scripts/prepare_data.py`) ✅

Implemented `src/data/pipeline.py` (called by the `prepare_data()` contract in `preprocessing.py`).
Ran on the full data in ~25 s; every artifact from plan §3–§5 is written to `data/processed/`
(full list at the top of `pipeline.py`) with config/raw-file/vocab/split hashes in `data_manifest.json`.

**Discoveries**

- `train_terms.tsv` is **already propagated**: raw pairs == propagated pairs in every aspect.
  Propagation (is_a + part_of, same aspect) is still applied and verified idempotent, so the pipeline
  stays correct for un-propagated inputs.
- All 31,466 train GO terms exist in `go-basic.obo` (release `2023-01-01`), none are obsolete or
  alt_ids, and all have IA weights. Obsolete/alt_id handling is implemented anyway (4,169 obsolete
  terms, 3,646 alt_ids in the OBO). The OBO has no cycles under is_a/part_of and no cross-aspect edges.
- OBO also has `regulates` / `positively_regulates` / `negatively_regulates` — **not** propagated
  (matches the CAFA evaluator convention of is_a + part_of).
- 142,246 train proteins, all annotated, 0 missing taxa, 3,322 duplicate sequences,
  **15,271 sequences > 1022 residues** (max 35,375) → long-sequence policy matters.
- Test superset: 141,864 proteins (1 duplicate ID dropped), 90 taxa, 10,293 sequences > 1022.
  **73,653 test proteins are identical (same ID + sequence) to training proteins** — CAFA 5 targets
  include already-annotated proteins (temporal evaluation of *new* annotations).
- No MMseqs2 / CD-HIT on this machine → implemented **MinHash-LSH clustering** (5-mers, 64 hashes,
  16 bands × 4 rows, union-find; ~50% collision at Jaccard 0.5 ≈ high identity). Benchmarked on all
  142k sequences (20 s):

  | k, hashes, bands | clusters | largest | note |
  |---|---|---|---|
  | **5, 64, 16** | **98,554** | **84** | chosen |
  | 5, 64, 32 | 57,208 | 38,782 | transitive chaining → giant component |
  | 3, 64, 32 | 37 | 142,196 | useless |

  This is an approximation of a ~high-identity cluster split, **not** a 30%-identity MMseqs2 split;
  homologs below that similarity can still sit on both sides. If `mmseqs` is on PATH, `split: cluster`
  uses it automatically.

**Results**

| | F (MF) | P (BP) | C (CC) |
|---|---|---|---|
| K (terms seen in train split) | 6,835 | 20,584 | 2,872 |
| parent→child edges | 8,689 | 40,561 | 4,719 |
| terms with 1 train positive | 1,638 | 2,989 | 369 |
| valid terms outside vocab | 389 | 701 | 85 |
| label hierarchy violations | 0 | 0 | 0 |

Split: 113,797 train / 28,449 valid; 0 ID overlap, 0 exact-sequence overlap, 0 shared clusters.
Taxonomy vocab (train split only): 2,795 taxa + UNK; unseen-taxon rate 1.4% (valid), 1.3% (test).

Design decisions: vocabularies are fit on the **train split only** (plan §5 "fit preprocessing on
train only"); no rare terms dropped (`min_term_count: 1`); internal evaluation uses the *full*
propagated validation truth, so terms outside the vocabulary still count as misses.

## Step 4 — ESM-2 embeddings (`scripts/extract_embeddings.py`)

- Frozen `facebook/esm2_t30_150M_UR50D`, fp16, 640-d residue mean pooling (BOS/EOS/PAD excluded).
- **Long sequences: `max_length_policy: chunk`** — 1022-residue windows with 256 overlap, pooled per
  window, aggregated by residue-weighted mean. (`truncate` also available.) Recorded in the manifest.
- **Cache keyed by sequence hash** (not protein ID): duplicates and the 73,653 train/test overlaps are
  embedded once. Shards are written atomically (`.npy` then `.keys.json`) and resumed on rerun.
- **Discovery:** the HF `EsmTokenizer` is pure Python. Replaced with a byte lookup table — verified to
  produce identical IDs/masks.
- **Benchmark** (500 random train seqs, `results/embedding_benchmark.json`): ~180 seq/s
  (~104k residues/s) at 8k tokens/batch, peak VRAM 0.5 GB. Larger budgets were *slower* (more padding;
  the GPU is already compute-bound at ~30 TFLOPs fp16), so `max_tokens_per_batch: 8000`.
- Sanity checks run before every extraction: repeatability, padding contamination (embedding of a
  sequence alone vs. padded in a batch), finiteness on the longest sequence.

## Step 5 — Training, prediction, evaluation & export code

All six scripts now do real work (`scripts/_common.py` adds `--set key=value` config overrides, used
for the λ sweep and smoke runs). New/rewritten modules:

| Module | What it does |
|---|---|
| `src/data/artifacts.py` | Loads processed artifacts; **asserts embedding ID order == split ID order** before use |
| `src/data/dataset.py` `SparseBatchLoader` | Vectorized CSR row slicing, densified on GPU per minibatch |
| `src/training/trainer.py` | AdamW, BCE(+hier), early stopping on val Fmax, streaming val Fmax + IA-wFmax, hierarchy violation rate, time/VRAM |
| `src/training/train.py` | One model per aspect; checkpoint stores weights, optimizer, epoch, best metric, config, **vocab list + hash**, split hashes, ESM id, seed, software versions; writes history JSON + learning-curve PNG |
| `src/evaluation/metrics.py` `FmaxAccumulator` | Streaming GPU P/R/F over 99 thresholds — identical to `compute_fmax` (tested), never holds the N×K matrix |
| `src/evaluation/predict.py` | Streams predictions for all aspects in 2,048-protein blocks (full test score matrix would be ~17 GB); optional max-propagation of scores up the DAG |
| `src/evaluation/export_cafa.py` | `StreamingCafaWriter` (sample_submission format) + `validate_submission` read-back check |
| `src/evaluation/evaluate_cafa.py` | Internal file-based Fmax/wFmax vs. the **full** validation truth + official `cafaeval` run on a subset as cross-check |
| `scripts/compare_experiments.py` | Builds `results/experiment_table.{md,json}` |

**Decisions**

- *Positive–unlabelled policy:* each aspect model trains/validates only on proteins that have ≥1
  annotation in that aspect; their unannotated terms are BCE negatives (benchmark convention, plan §2).
- *Export:* scores ≥ 0.01, at most 450 MF / 750 BP / 300 CC terms per protein (= CAFA's 1,500 limit);
  every target protein gets ≥1 row; raw sigmoid scores (3 decimals), no calibration.
- *Evaluation:* predictions are max-propagated before scoring, like `cafaeval` (`prop='max'`).
  `cafaeval==1.3.0` (the official CAFA evaluator package) added as a dependency; it uses
  is_a+part_of and τ = 0.01…0.99, the same conventions as the internal metric.

**Tests:** 31 passing (`uv run pytest`), including `tests/test_integration.py` — 300 synthetic
proteins through prepare → ontology → labels → split → embeddings → 1-epoch training (with taxonomy +
hierarchical loss) → predict → export → validate → evaluate.

## Step 6 — Embedding extraction finished ✅

`results/embedding_benchmark.json`, `data/embeddings/embedding_manifest.json`:

| split | proteins | new unique seqs embedded | reused from cache | > 1022 aa (chunked) | time |
|---|---|---|---|---|---|
| train | 113,797 | 111,085 | 2,712 | 12,254 | 9.2 min |
| valid | 28,449 | 27,839 | 610 | 3,017 | 2.4 min |
| test | 141,864 | 66,191 | **75,673** | 10,293 | 3.9 min |

~110–117k residues/s, peak VRAM 0.52 GB. Sanity checks: repeat diff 0.0, padding diff 1.4e-3
(fp16 noise), longest sequence (35,375 aa) finite, dim 640. All arrays finite, ID order verified.

## Step 7 — E0 baseline trained, evaluated, exported ✅

**Discovery:** with the template budget (20 epochs) every aspect was **still improving at epoch 20**
(1–3 s/epoch on cached embeddings). Raised `training.epochs` to 100 in *all four* configs (same
budget for every experiment); early stopping (patience 4) stopped MF at 78, BP at 84, CC at 51.
Model-selection Fmax went MF 0.702→0.740, BP 0.404→0.433, CC 0.703→0.714.

**Evaluator cross-check:** on the same 3,000 random validation proteins, the internal evaluator and
the official `cafaeval` agree to ~1e-9 for MF and CC and ~1e-4 for BP (both Fmax and IA-weighted).
So the internal full-validation numbers are computed exactly like the CAFA evaluator would.

E0 internal validation (full 28k validation split): see the table in Step 8.

Test superset export `predictions/baseline_test.tsv`: 141,864 / 141,864 proteins, 67.4 M rows, max 1,143
terms per protein, read-back validation passed (format identical to `sample_submission.tsv`).

## Step 8 — E1 hierarchical-loss λ sweep (in progress) and E2 taxonomy ✅

**Discovery — the penalty is ~1000× smaller than BCE.** At epoch 20 the train BCE is ~3e-3 but
`mean(max(0, s_c − s_p)²)` is ~1e-6, because the mean runs over every (protein, edge) pair and almost
all violations are between near-zero probabilities. With λ ∈ {0.01, 0.1, 1.0} (the plan's starting
points) the runs are near-identical to E0. Extended the grid to λ ∈ {10, 100, 1000}.

**Discovery — the baseline is already nearly hierarchy-consistent** (`scripts/hierarchy_diagnostics.py`,
raw scores on validation, `results/hierarchy_diagnostics.json`):

| | viol. rate (any) | viol. rate (Δ > 0.01) | viol. rate (Δ > 0.1) | mean magnitude given violation |
|---|---|---|---|---|
| E0 MF | 21.2% | 0.054% | 0.011% | 2.0e-4 |
| E0 BP | 15.6% | 0.066% | 0.006% | 2.5e-4 |
| E0 CC | 8.3% | 0.093% | 0.010% | 6.5e-4 |

The headline "violation rate" counts numerical noise; meaningful violations are < 0.1% of pairs.
Training on fully propagated labels already teaches the hierarchy, so the penalty has little to fix.

**E2 taxonomy embedding** (full validation, internal protocol):

| | MF Fmax | BP Fmax | CC Fmax | mean Fmax | mean wFmax |
|---|---|---|---|---|---|
| E0 baseline | 0.7495 | 0.4457 | 0.7172 | 0.6375 | 0.5351 |
| E2 taxonomy | 0.7515 | 0.4691 | 0.7437 | **0.6548** | **0.5552** |

Gains are concentrated in BP (+2.3) and CC (+2.7). Caveat (plan §9): 98.6% of validation proteins
come from taxa seen in training, so part of this may be species-level label priors (shortcut).

**E0 error analysis** (`results/baseline/error_analysis.json`):
- Term-centric micro-AP collapses for rare terms — BP: 0.009 (1 train positive) → 0.041 (2–10) →
  0.119 (101–1000) → 0.456 (>1000); MF 0.124 → 0.805; CC 0.014 → 0.784.
- Sequences > 1022 aa (chunked) score 3–5 points lower than 201–1022 aa in every aspect.
- Unseen-taxon proteins (n ≈ 100–300) are too few for a reliable comparison.

## Step 9 — λ selection, reproducibility checks

**λ sweep result** (full validation, internal protocol, mean over MF/BP/CC):

| λ | 0 (E0) | 0.01 | 0.1 | 1.0 | 10 | 100 | 1000 |
|---|---|---|---|---|---|---|---|
| mean Fmax | 0.6375 | 0.6331 | 0.6327 | 0.6320 | 0.6276 | see final table | see final table |

λ ≤ 1 are within 0.001 of each other (noise); λ ≥ 10 lowers Fmax. **Selection rule (validation
only): the largest λ within 0.002 of the best λ>0 → λ = 1.0**, the only small value that measurably
lowers violations (~20% smaller magnitude) without costing Fmax. Set in `hierarchical.yaml` and
`combined.yaml`. That none of the λ>0 runs beat E0 is a result, not a bug; seeds (Step 10) measure
how much of the gap is noise.

**Determinism:** retraining E1 with λ=1.0 reproduced the sweep run `hier_l1.0` exactly (MF: same best
epoch 76, Fmax 0.7403).

**Clean-checkout reproducibility (plan day 7):** copied only `src/ scripts/ configs/ tests/
pyproject.toml uv.lock` into an empty folder → `uv sync --frozen` → `uv run pytest` (31 passed) →
`scripts/run_pipeline.py --config configs/combined.yaml --set data.subset=2000 --set training.epochs=3`
ran all six steps unattended in 2 min 40 s, with schema-valid validation and test exports.

**Gotcha:** editing a bash script while bash is running it corrupts the run (bash reads the file
incrementally). `scripts/run_experiments.sh` hit this once; the finished runs were unaffected.

## Step 10 — Early-stopping artefact found → fair rerun of the whole matrix

**Discovery.** Every λ>0 run, from λ=0.01 to λ=100, stopped BP at exactly epoch 64 (best 60), while
both λ=0 runs ran to 84–90. Identical stopping epochs across a 10,000× range of λ can't be the
penalty's doing. Per-epoch BP validation Fmax shows the curves are **nearly identical**, and λ=1 is
slightly *ahead* of E0 up to epoch 60:

| epoch | 20 | 40 | 60 | 62 | 64 |
|---|---|---|---|---|---|
| E0 (λ=0) | 0.4039 | 0.4193 | 0.4280 | **0.4281** | 0.4275 |
| λ=1.0 | 0.4046 | 0.4200 | 0.4282 | 0.4278 | 0.4273 |
| λ=100 | 0.4016 | 0.4130 | 0.4206 | 0.4201 | 0.4201 |

All runs reach a noisy plateau near epoch 60, where gains are ~1e-4 per epoch. E0 gained +0.0001 at
epoch 62, which reset its patience, and it kept creeping up to 0.4329 by epoch 80; the λ>0 runs
just missed and stopped. **With patience 4, the "E1 < E0" gap was produced by early stopping.**

**Fix (applied to all four configs identically):** `early_stopping_patience: 10`, `epochs: 150`.
Re-running E0–E3 × seeds {42, 43, 44} with full validation + export for each
(`scripts/run_matrix.sh <seed>`, three seeds in parallel).

**λ sweep at a matched budget** (best model-selection Fmax within the first 64 epochs; MF and BP
runs all lasted ≥ 64 epochs, so they're comparable; CC runs with λ ≤ 1 stopped at 32 under the old
patience and aren't):

| run | MF | BP |
|---|---|---|
| λ=0 | 0.7370 | 0.4281 |
| λ=0.01 | 0.7367 | 0.4279 |
| λ=0.1 | 0.7364 | 0.4279 |
| λ=1.0 | **0.7376** | **0.4282** |
| λ=10 | 0.7336 | 0.4266 |
| λ=100 | 0.7243 | 0.4205 |
| λ=1000 | 0.7110 | — |

λ ≤ 1 ≈ no effect on Fmax; λ ≥ 10 hurts monotonically. λ = 1.0 kept for E1/E3.

## Step 11 — Matrix results so far + taxonomy UNK bug found

Patience-10 reruns, mean Fmax over MF/BP/CC (internal validation, full truth):

| | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| E0 baseline | 0.6424 | 0.6415 | 0.6418 |
| E1 hierarchical (λ=1) | 0.6400 | 0.6406 | 0.6409 |
| E2 taxonomy | 0.6610 | 0.6617 | 0.6586 |

Seed spread is ~0.001, so E2's +1.9 points is real and E1's −0.15 is small but consistent.

**Discovery — E2 collapses on unseen taxa** (`results/*/error_analysis.json`, protein-centric Fmax):

| | MF | BP | CC |
|---|---|---|---|
| E0, unseen taxa (n=309/203/110) | 0.721 | 0.522 | 0.743 |
| **E2, unseen taxa** | **0.580** | **0.296** | **0.655** |
| E2, seen taxa | 0.759 | 0.480 | 0.750 |

Root cause: every training protein's taxon is in the vocabulary, so **the UNK row (index 0) never
receives a gradient** and stays at its random initialization. Unseen-taxon proteins (1.4% of
validation, 1.3% of the test superset) get random taxonomy features, which is the risk plan §9
warns about. Rare-term AP also *dropped* for MF singletons (0.121 → 0.056), consistent with the
model leaning on species priors.

**Fix:** `taxonomy.unk_dropout` — during training each taxon index is replaced by UNK with
probability 0.1, so the UNK row learns a taxon-agnostic embedding (inference unchanged; unit test
added). Run as variants **E2b `taxonomy_unkdrop`** and **E3b `combined_unkdrop`** (configs identical
to E2/E3 except that one line), so E2/E3 stay exactly as the plan specifies.
The four main configs were left byte-identical to what their checkpoints were trained with (config
hashes verified).

## Step 12 — Full matrix done (3 seeds); UNK-dropout fix verified

Mean ± std over seeds 42/43/44, internal validation (full truth):

| ID | Experiment | mean Fmax | mean wFmax |
|---|---|---|---|
| E0 | baseline | 0.6419 ± 0.0004 | 0.5402 ± 0.0005 |
| E1 | + hierarchical (λ=1) | 0.6405 ± 0.0003 | 0.5382 ± 0.0009 |
| E2 | + taxonomy | 0.6605 ± 0.0013 | 0.5629 ± 0.0015 |
| E3 | + both | 0.6595 ± 0.0005 | 0.5612 ± 0.0007 |
| E2b | E2 + UNK-dropout 0.1 | 0.6623 (0.6620 / 0.6619 / 0.6630) | 0.5655 |

**UNK-dropout repairs unseen taxa** (seed 42, protein-centric Fmax, MF / BP / CC):

| | seen taxa | unseen taxa |
|---|---|---|
| E0 | 0.755 / 0.451 / 0.721 | 0.721 / 0.522 / 0.743 |
| E2 | 0.759 / 0.480 / 0.750 | 0.580 / 0.296 / 0.655 |
| **E2b** | 0.757 / 0.480 / 0.751 | **0.695 / 0.481 / 0.704** |

Seen-taxon performance is unchanged, unseen recovers most of the way to the baseline, and overall
Fmax is higher than plain E2 on every seed.

**Hierarchy diagnostics (final models):** violations > 0.01 in BP are 0.070% (E0) → 0.055% (E1, λ=1)
→ 0.035% (λ=10) → 0.008% (λ=100). A larger λ does enforce consistency, but it costs Fmax, and E0's
violations are already tiny.

**Evaluator agreement:** for all four seed-42 experiments, `cafaeval` and the internal evaluator
agree to ≤ 6e-4 on the 3,000-protein subset, for both Fmax and wFmax in every aspect.

Test-superset exports (seed 42) are all schema-valid, with 141,864 / 141,864 targets covered.

## Step 13 — Final results and deliverables ✅

| ID | mean Fmax (3 seeds) | mean wFmax |
|---|---|---|
| E0 baseline | 0.6419 ± 0.0004 | 0.5402 ± 0.0005 |
| E1 hierarchical | 0.6405 ± 0.0003 | 0.5382 ± 0.0009 |
| E2 taxonomy | 0.6605 ± 0.0013 | 0.5629 ± 0.0015 |
| E3 combined | 0.6595 ± 0.0005 | 0.5612 ± 0.0007 |
| **E2b taxonomy + UNK-dropout** | **0.6623 ± 0.0005** | **0.5655 ± 0.0006** |
| E3b combined + UNK-dropout | 0.6609 ± 0.0006 | 0.5633 ± 0.0009 |

E3b unseen taxa (MF / BP / CC): 0.693 / 0.486 / 0.694, the same recovery as E2b.
All six test-superset exports are schema-valid (141,864 targets each). The full write-up is in
`results/REPORT.md`; tables are in `results/experiment_table.md`.
**Recommended submission: `predictions/taxonomy_unkdrop_test.tsv`.**

Not done / out of scope: an official CAFA 5 score (needs the temporal ground truth), an MMseqs2
30%-identity split (binary not installed), and git commits (left for you to review; the plan
asks for feature branches + PRs instead of committing to `main`).
