# PFP-CAFA5 — Results Report

All numbers come from **internal validation** on the held-out 20% split (28,449 proteins,
MinHash-cluster split) unless marked otherwise. They are **not** official CAFA 5 scores: those
require the temporal ground truth collected after the CAFA 5 deadline. The step-by-step log of
how these results were reached is in [`PROGRESS.md`](../PROGRESS.md).

## 1. Setup (plan §23 experiment record)

| Item | Value |
|---|---|
| Data | CAFA 5 Kaggle release; GO `releases/2023-01-01`; 142,246 train proteins, 141,864 test-superset targets |
| Split | MinHash-LSH cluster split (5-mers, 64 hashes, 16×4 bands) + exact-duplicate grouping, seed 42; 113,797 train / 28,449 valid; split hashes in `data/processed/split_manifest.json` |
| Labels | is_a + part_of propagation (input already propagated); K = 6,835 MF / 20,584 BP / 2,872 CC (all terms seen in the train split) |
| Encoder | `facebook/esm2_t30_150M_UR50D`, frozen, fp16, residue-mean pooling, 1022-residue windows with 256 overlap for longer sequences |
| Head | Linear(640[+64], 512) → ReLU → Dropout 0.3 → Linear(512, K); one model per aspect |
| Training | AdamW lr 1e-3, wd 1e-4, batch 256, ≤150 epochs, early stopping on val Fmax (patience 10) |
| Seeds | 42, 43, 44 (mean ± std reported) |
| Hardware | RTX 5080 16 GB, torch 2.11 + CUDA 12.8 |
| Evaluator | internal GPU Fmax / IA-weighted Fmax (τ = 0.01…0.99, predictions max-propagated) + official `cafaeval 1.3.0` on 3,000 random validation proteins |

## 2. Main results — mean ± std over 3 seeds

Protein-centric Fmax and IA-weighted Fmax (wFmax, using `IA.txt`), full validation truth (terms
outside the training vocabulary count as misses).

| ID | Experiment | MF Fmax | BP Fmax | CC Fmax | **mean Fmax** | MF wFmax | BP wFmax | CC wFmax | **mean wFmax** |
|---|---|---|---|---|---|---|---|---|---|
| E0 | baseline | 0.7532 ± 0.0005 | 0.4511 ± 0.0009 | 0.7214 ± 0.0003 | 0.6419 ± 0.0004 | 0.6591 ± 0.0005 | 0.3871 ± 0.0014 | 0.5744 ± 0.0002 | 0.5402 ± 0.0005 |
| E1 | + hierarchical loss (λ=1) | 0.7513 ± 0.0010 | 0.4494 ± 0.0007 | 0.7207 ± 0.0008 | 0.6405 ± 0.0003 | 0.6562 ± 0.0020 | 0.3853 ± 0.0009 | 0.5731 ± 0.0012 | 0.5382 ± 0.0009 |
| E2 | + taxonomy embedding | 0.7544 ± 0.0020 | 0.4784 ± 0.0009 | 0.7485 ± 0.0012 | 0.6605 ± 0.0013 | 0.6579 ± 0.0017 | 0.4157 ± 0.0009 | 0.6152 ± 0.0023 | 0.5629 ± 0.0015 |
| E3 | + both | 0.7525 ± 0.0019 | 0.4770 ± 0.0001 | 0.7490 ± 0.0005 | 0.6595 ± 0.0005 | 0.6544 ± 0.0024 | 0.4141 ± 0.0003 | 0.6150 ± 0.0007 | 0.5612 ± 0.0007 |
| E2b | E2 + UNK-dropout 0.1 | 0.7582 ± 0.0016 | 0.4793 ± 0.0008 | 0.7496 ± 0.0007 | **0.6623 ± 0.0005** | 0.6620 ± 0.0023 | 0.4177 ± 0.0004 | 0.6168 ± 0.0014 | **0.5655 ± 0.0006** |
| E3b | E3 + UNK-dropout 0.1 | 0.7567 ± 0.0020 | 0.4773 ± 0.0010 | 0.7486 ± 0.0003 | **0.6609 ± 0.0006** | 0.6596 ± 0.0030 | 0.4153 ± 0.0004 | 0.6150 ± 0.0006 | **0.5633 ± 0.0009** |

Per-run details (precision/recall at Fmax, thresholds, best epoch, training time, violation rates,
config hashes) are in [`experiment_table.md`](experiment_table.md).

**Official evaluator cross-check (seed 42, 3,000 random validation proteins).** `cafaeval` and the
internal evaluator agree to ≤ 6e-4 on every experiment and aspect, for both Fmax and wFmax
(e.g. E0 MF 0.7488 / 0.7488, BP 0.4486 / 0.4486, CC 0.7273 / 0.7273).

## 3. Findings

1. **Taxonomy helps BP and CC, not MF.** E2 adds +2.7 Fmax in BP and +2.7 in CC (≈ 10–20× the seed
   std), but MF is flat (+0.1) and MF wFmax slightly drops. Cellular location and process
   annotations are strongly species-dependent; molecular function is mostly encoded in sequence.
2. **But plain E2 breaks on unseen taxa.** For the ~1.4% of validation proteins whose taxon is absent
   from training, E2 is far worse than the baseline (BP 0.296 vs 0.522, MF 0.580 vs 0.721). The UNK
   embedding row never receives a gradient. UNK-dropout (E2b/E3b) is the fix (§4): E2b is the best model overall (mean Fmax 0.6623, wFmax 0.5655) and recovers unseen-taxon BP to 0.481.
3. **The hierarchical penalty has nothing to fix.** Training on fully propagated labels already
   yields nearly consistent scores: only 0.05–0.09% of (protein, edge) pairs violate by more than
   0.01. With `mean(max(0, s_c − s_p)²)` over all pairs, the penalty is ~1e-6 versus BCE ~3e-3, so
   λ ≤ 1 has no effect and λ ≥ 10 lowers Fmax monotonically (λ sweep in `experiment_table.md`).
   E1 vs E0 is −0.14 mean Fmax, small but consistent across seeds.
4. **The training budget matters more than the λ choice.** The template's 20 epochs / patience 4
   under-trains (E0 mean Fmax 0.638 → 0.642 with patience 10), and with patience 4 the E1-vs-E0
   comparison was decided by early-stopping luck on a 1e-4/epoch plateau (PROGRESS.md step 10).
5. **Rare terms are the main weakness.** Term-centric micro-AP (E0) falls from 0.46–0.81 for terms
   with > 1,000 training positives to 0.01–0.12 for singletons; 389 / 701 / 85 validation terms do
   not occur in training at all. Taxonomy improves rare BP/CC terms but hurts MF singletons.
6. **Long sequences** (> 1022 aa, window-averaged) score 3–5 Fmax points below 201–1022 aa.

## 4. Taxonomy unseen-taxon analysis

Protein-centric Fmax, validation split, seed 42 (`results/*/error_analysis.json`):

| | MF seen | MF unseen | BP seen | BP unseen | CC seen | CC unseen |
|---|---|---|---|---|---|---|
| n proteins | 15,527 | 309 | 18,203 | 203 | 18,485 | 110 |
| E0 baseline | 0.755 | 0.721 | 0.451 | 0.522 | 0.721 | 0.743 |
| E2 taxonomy | 0.759 | **0.580** | 0.480 | **0.296** | 0.750 | **0.655** |
| E3 combined | 0.757 | **0.588** | 0.478 | **0.303** | 0.750 | **0.653** |
| E2b taxonomy + UNK-dropout | 0.757 | 0.695 | 0.480 | 0.481 | 0.751 | 0.704 |
| E3b combined + UNK-dropout | 0.755 | 0.693 | 0.478 | 0.486 | 0.749 | 0.694 |

Unseen-taxon sets are small (110–309 proteins), so read differences of a few points with caution.
Test superset: 1.3% of targets come from taxa unseen in training.

## 5. Hierarchy diagnostics (raw scores, validation, seed 42)

| model | λ | aspect | viol. rate (any Δ>0) | Δ > 0.01 | Δ > 0.1 | mean Δ given violation |
|---|---|---|---|---|---|---|
| baseline | 0 | MF | 0.305 | 0.0500% | 0.01129% | 1.38e-04 |
| baseline | 0 | BP | 0.184 | 0.0698% | 0.00761% | 2.30e-04 |
| baseline | 0 | CC | 0.108 | 0.1000% | 0.01472% | 6.01e-04 |
| hier_l0.01 | 0.01 | MF | 0.211 | 0.0540% | 0.01049% | 2.00e-04 |
| hier_l0.01 | 0.01 | BP | 0.151 | 0.0641% | 0.00489% | 2.47e-04 |
| hier_l0.01 | 0.01 | CC | 0.087 | 0.0819% | 0.00800% | 5.45e-04 |
| hier_l0.1 | 0.1 | MF | 0.214 | 0.0534% | 0.00999% | 1.91e-04 |
| hier_l0.1 | 0.1 | BP | 0.150 | 0.0635% | 0.00473% | 2.46e-04 |
| hier_l0.1 | 0.1 | CC | 0.088 | 0.0807% | 0.00787% | 5.27e-04 |
| hierarchical | 1 | MF | 0.258 | 0.0503% | 0.00884% | 1.40e-04 |
| hierarchical | 1 | BP | 0.171 | 0.0551% | 0.00496% | 1.85e-04 |
| hierarchical | 1 | CC | 0.098 | 0.0915% | 0.01104% | 5.56e-04 |
| hier_l10 | 10 | MF | 0.194 | 0.0285% | 0.00253% | 8.28e-05 |
| hier_l10 | 10 | BP | 0.145 | 0.0347% | 0.00104% | 1.37e-04 |
| hier_l10 | 10 | CC | 0.080 | 0.0507% | 0.00252% | 3.21e-04 |
| hier_l100 | 100 | MF | 0.170 | 0.0072% | 0.00013% | 2.49e-05 |
| hier_l100 | 100 | BP | 0.136 | 0.0083% | 0.00005% | 5.45e-05 |
| hier_l100 | 100 | CC | 0.072 | 0.0114% | 0.00012% | 1.00e-04 |
| taxonomy | 0 | MF | 0.304 | 0.0465% | 0.01129% | 1.37e-04 |
| taxonomy | 0 | BP | 0.239 | 0.0818% | 0.01080% | 2.17e-04 |
| taxonomy | 0 | CC | 0.123 | 0.1085% | 0.01774% | 6.01e-04 |
| combined | 1 | MF | 0.278 | 0.0466% | 0.00902% | 1.27e-04 |
| combined | 1 | BP | 0.238 | 0.0729% | 0.00833% | 1.82e-04 |
| combined | 1 | CC | 0.122 | 0.0987% | 0.01380% | 5.03e-04 |
| taxonomy_unkdrop | 0 | MF | 0.244 | 0.0518% | 0.01126% | 1.77e-04 |
| taxonomy_unkdrop | 0 | BP | 0.223 | 0.0816% | 0.01058% | 2.30e-04 |
| taxonomy_unkdrop | 0 | CC | 0.132 | 0.1059% | 0.01930% | 5.78e-04 |
| combined_unkdrop | 1 | MF | 0.244 | 0.0473% | 0.00851% | 1.42e-04 |
| combined_unkdrop | 1 | BP | 0.214 | 0.0697% | 0.00750% | 1.89e-04 |
| combined_unkdrop | 1 | CC | 0.107 | 0.1005% | 0.01323% | 5.68e-04 |

`hierarchical` = λ 1.0 with patience 10; `hier_l*` = sweep runs (patience 4).
Larger λ removes violations (λ=100: BP violations > 0.01 fall ~90%) but costs Fmax (§3).

## 6. Error analysis — E0 term-centric micro-AP by training frequency

| train positives | MF | BP | CC |
|---|---|---|---|
| 1 | 0.121 | 0.009 | 0.008 |
| 2–10 | 0.324 | 0.044 | 0.070 |
| 11–100 | 0.400 | 0.066 | 0.174 |
| 101–1000 | 0.499 | 0.126 | 0.254 |
| > 1000 | 0.808 | 0.461 | 0.787 |

By sequence length (E0 Fmax MF / BP / CC): ≤200 aa 0.749 / 0.406 / 0.730; 201–500 0.757 / 0.469 / 0.738;
501–1022 0.761 / 0.460 / 0.711; > 1022 0.727 / 0.429 / 0.672.

## 7. Cost

| | MF | BP | CC |
|---|---|---|---|
| E0 training time (seed 42, 3 jobs sharing the GPU) | 662 s | 1,451 s | 191 s |
| E0 peak GPU memory | 0.55 GB | 1.45 GB | 0.42 GB |

ESM-2 embedding extraction: 15.5 min total for 283k proteins (205k unique new sequences),
~110k residues/s, 0.52 GB peak VRAM. Test-superset inference + export: ~40 s per experiment
(≈ 47–59 M rows, 1.2–1.6 GB TSV).

## 8. Deliverables

| | Path |
|---|---|
| Checkpoints (3 aspects × experiments × seeds) | `checkpoints/<exp>[_s43|_s44]_<F|P|C>_best.pt` (vocab + hashes inside) |
| Validation predictions | `predictions/<exp>_val.tsv` |
| Test-superset submissions (seed 42) | `predictions/{baseline,hierarchical,taxonomy,combined,taxonomy_unkdrop,combined_unkdrop}_test.tsv` |
| Per-experiment results | `results/<exp>/` (histories, learning curves, threshold curves, eval JSON incl. cafaeval, error analysis) |
| Comparison tables | `results/experiment_table.{md,json}`, `results/experiment_table_seeds.json` |
| Hierarchy diagnostics | `results/hierarchy_diagnostics.json` |
| Data QC / manifests | `data/processed/qc_report.json`, `data_manifest.json`, `split_manifest.json`, `data/embeddings/embedding_manifest.json` |

**Recommended submission:** `predictions/taxonomy_unkdrop_test.tsv` (E2b). It has the best mean Fmax and wFmax on every seed, is robust on unseen taxa, and does not use the hierarchical penalty, which gave no benefit.

## 9. Limitations

- Positive–unlabelled labels: unannotated terms are trained as negatives.
- The MinHash split approximates high-identity clustering only, so remote homologs can cross the
  split and scores are optimistic compared with a 30%-identity split or CAFA's temporal split.
- 98.6% of validation proteins come from taxa seen in training, so taxonomy gains partly reflect
  species-level label priors, and the unseen-taxon estimates rest on 110–309 proteins.
- Long-sequence window averaging; a frozen 150M encoder; one split; three seeds.
- Internal validation and the cafaeval subset run are not an official CAFA 5 evaluation.
