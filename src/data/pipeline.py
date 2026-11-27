"""Step 1 orchestration: raw CAFA 5 files -> processed, versioned artifacts (plan §3-§5, §15).

Outputs (all in ``data.processed_dir``):
  protein_ids.json, sequences.jsonl, taxonomy_ids.npy          canonical train-set table (FASTA order)
  test_ids.json, test_sequences.jsonl, test_taxonomy_ids.npy   test superset table (FASTA order)
  raw_annotations.tsv                                           deduplicated, resolved input labels
  propagated_annotations.tsv                                    labels after ancestor propagation
  vocab_{F,P,C}.json                                            GO-ID -> column index (sorted)
  go_edges_{F,P,C}.npy / go_edge_relations_{F,P,C}.json         parent->child edges + relation types
  labels_{F,P,C}.npz / labels_raw_{F,P,C}.npz                   CSR (N, K) propagated / raw labels
  train_ids.txt, valid_ids.txt, split_manifest.json            split + manifest
  taxonomy_vocab.json                                           NCBI taxon -> index (train split only)
  qc_report.json, data_manifest.json                            QC stats, audits, fingerprints
"""
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse

from src.data.ontology import GOOntology, build_sparse_label_matrix
from src.data.preprocessing import parse_annotations, parse_fasta, parse_fasta_header_field
from src.data.split import audit_split_leakage, create_split, minhash_clusters, mmseqs_clusters
from src.data.taxonomy import (
    MISSING_TAXON_ID,
    UNK_TAXON_INDEX,
    build_taxonomy_vocabulary,
    encode_taxonomy_ids,
    parse_taxonomy_tsv,
)
from src.utils.config import compute_config_hash
from src.utils.io import compute_file_hash, compute_list_hash, save_json_atomic
from src.utils.logger import get_logger

ASPECTS = ("F", "P", "C")
TERMS_ASPECT_CODES = {"MFO": "F", "BPO": "P", "CCO": "C", "F": "F", "P": "P", "C": "C"}

DEFAULT_RAW_FILES = {
    "train_sequences": "Train/train_sequences.fasta",
    "train_terms": "Train/train_terms.tsv",
    "train_taxonomy": "Train/train_taxonomy.tsv",
    "obo": "Train/go-basic.obo",
    "test_sequences": "Test (Targets)/testsuperset.fasta",
    "test_taxon_list": "Test (Targets)/testsuperset-taxon-list.tsv",
    "ia": "IA.txt",
    "sample_submission": "sample_submission.tsv",
}


def resolve_raw_files(config: Dict[str, Any]) -> Dict[str, Path]:
    data_cfg = config.get("data", {})
    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    names = {**DEFAULT_RAW_FILES, **(data_cfg.get("raw_files") or {})}
    return {k: raw_dir / v for k, v in names.items()}


def _write_jsonl_sequences(path: Path, ids: List[str], seqs: List[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for pid, seq in zip(ids, seqs):
            f.write(json.dumps({"id": pid, "length": len(seq), "sequence": seq}) + "\n")
    tmp.replace(path)


def read_jsonl_sequences(path: Path) -> Tuple[List[str], List[str]]:
    ids, seqs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ids.append(rec["id"])
            seqs.append(rec["sequence"])
    return ids, seqs


def _length_histogram(lengths: List[int]) -> Dict[str, Any]:
    arr = np.asarray(lengths)
    if arr.size == 0:
        return {}
    edges = [0, 100, 200, 400, 600, 800, 1022, 2000, 5000, 10**9]
    counts, _ = np.histogram(arr, bins=edges)
    return {
        "percentiles": {str(p): float(np.percentile(arr, p)) for p in (1, 5, 25, 50, 75, 95, 99)},
        "bins": {f"{edges[i]}-{edges[i + 1]}": int(c) for i, c in enumerate(counts)},
        "num_longer_than_1022": int((arr > 1022).sum()),
    }


def _save_csr(path: Path, mat: sparse.csr_matrix) -> None:
    sparse.save_npz(path, mat.tocsr(), compressed=True)


def prepare_data(config: Dict[str, Any], logger=None) -> Dict[str, Any]:
    """Run data ingestion, ontology processing, labels, taxonomy and split; write artifacts."""
    logger = logger or get_logger("prepare_data")
    data_cfg = config.get("data", {})
    seed = int(config.get("seed", 42))
    out_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = resolve_raw_files(config)
    for key in ("train_sequences", "train_terms", "train_taxonomy", "obo"):
        if not raw[key].exists():
            raise FileNotFoundError(f"Required raw file missing ({key}): {raw[key]}")

    subset: Optional[int] = data_cfg.get("subset")
    qc: Dict[str, Any] = {"created_at": datetime.now(timezone.utc).isoformat()}

    # ---------------- Sequences ----------------
    logger.info(f"Parsing train FASTA {raw['train_sequences']}")
    protein_ids, sequences, seq_qc = parse_fasta(raw["train_sequences"])
    if subset:
        protein_ids, sequences = protein_ids[: int(subset)], sequences[: int(subset)]
        logger.warning(f"SUBSET MODE: keeping first {len(protein_ids)} train proteins")
    seq_qc["length_distribution"] = _length_histogram([len(s) for s in sequences])
    qc["train_sequences"] = seq_qc
    pid_set = set(protein_ids)

    # ---------------- Ontology ----------------
    relations = tuple(data_cfg.get("ontology", {}).get("relations", ["is_a", "part_of"]))
    logger.info(f"Parsing GO ontology {raw['obo']} (propagation relations: {relations})")
    onto = GOOntology(allowed_relations=relations)
    onto.parse_obo(raw["obo"])
    n_cycle_nodes = onto.find_cycles()
    qc["ontology"] = {
        "data_version": onto.data_version,
        "num_terms": len(onto.terms),
        "num_obsolete": sum(t.is_obsolete for t in onto.terms.values()),
        "num_alt_ids": len(onto.alt_id_map),
        "terms_per_aspect": dict(Counter(t.aspect for t in onto.terms.values() if not t.is_obsolete)),
        "propagation_relations": list(relations),
        "cross_aspect_edges_ignored": int(sum(
            1 for t in onto.terms.values() for p, _ in t.parents
            if p in onto.terms and onto.terms[p].aspect != t.aspect
        )),
        "cycle_nodes": n_cycle_nodes,
    }
    if n_cycle_nodes:
        raise ValueError(f"GO graph has {n_cycle_nodes} nodes in cycles under {relations}")

    # ---------------- Annotations ----------------
    logger.info(f"Parsing annotations {raw['train_terms']}")
    records = parse_annotations(raw["train_terms"])
    ann_qc = Counter()
    raw_ann: Dict[str, Dict[str, Set[str]]] = {a: defaultdict(set) for a in ASPECTS}
    affected_obsolete: List[Tuple[str, str]] = []
    for pid, go_id, aspect_code in records:
        ann_qc["input_pairs"] += 1
        if pid not in pid_set:
            ann_qc["pairs_protein_not_in_fasta"] += 1
            continue
        resolved = onto.resolve_term(go_id)
        if resolved is None:
            if go_id in onto.terms:
                ann_qc["pairs_obsolete_dropped"] += 1
                affected_obsolete.append((pid, go_id))
            else:
                ann_qc["pairs_unknown_go_dropped"] += 1
            continue
        if resolved != go_id:
            ann_qc["pairs_remapped_obsolete_or_alt"] += 1
        term_aspect = onto.terms[resolved].aspect
        declared = TERMS_ASPECT_CODES.get(aspect_code, "")
        if declared and declared != term_aspect:
            ann_qc["pairs_aspect_mismatch"] += 1
        raw_ann[term_aspect][pid].add(resolved)

    prop_ann: Dict[str, Dict[str, Set[str]]] = {a: {} for a in ASPECTS}
    for a in ASPECTS:
        for pid, terms in raw_ann[a].items():
            prop_ann[a][pid] = onto.propagate(terms)
    # Idempotency check on a sample
    for a in ASPECTS:
        for pid in list(prop_ann[a])[:200]:
            assert onto.propagate(prop_ann[a][pid]) == prop_ann[a][pid], "propagation not idempotent"

    annotated = set().union(*[set(raw_ann[a]) for a in ASPECTS])
    ann_qc_d = dict(ann_qc)
    ann_qc_d.update({
        "proteins_with_any_annotation": len(annotated),
        "proteins_without_annotation": len(pid_set - annotated),
        "per_aspect": {
            a: {
                "proteins": len(raw_ann[a]),
                "raw_pairs": int(sum(len(v) for v in raw_ann[a].values())),
                "propagated_pairs": int(sum(len(v) for v in prop_ann[a].values())),
                "unique_terms_raw": len(set().union(*raw_ann[a].values())) if raw_ann[a] else 0,
                "unique_terms_propagated": len(set().union(*prop_ann[a].values())) if prop_ann[a] else 0,
            }
            for a in ASPECTS
        },
        "obsolete_dropped_examples": affected_obsolete[:20],
    })
    qc["annotations"] = ann_qc_d

    with open(out_dir / "raw_annotations.tsv", "w", encoding="utf-8") as f:
        f.write("EntryID\tterm\taspect\n")
        for a in ASPECTS:
            for pid in sorted(raw_ann[a]):
                for go in sorted(raw_ann[a][pid]):
                    f.write(f"{pid}\t{go}\t{a}\n")
    with open(out_dir / "propagated_annotations.tsv", "w", encoding="utf-8") as f:
        f.write("EntryID\tterm\taspect\n")
        for a in ASPECTS:
            for pid in sorted(prop_ann[a]):
                for go in sorted(prop_ann[a][pid]):
                    f.write(f"{pid}\t{go}\t{a}\n")

    # ---------------- Taxonomy (raw) ----------------
    pid_to_taxon = parse_taxonomy_tsv(raw["train_taxonomy"])
    raw_tax, _ = encode_taxonomy_ids(protein_ids, pid_to_taxon)

    # ---------------- Split ----------------
    split_method = data_cfg.get("split", "cluster")
    val_ratio = float(data_cfg.get("val_ratio", 0.2))
    cl_cfg = data_cfg.get("clustering", {}) or {}
    clusters: Optional[Dict[str, str]] = None
    cl_info: Dict[str, Any] = {}
    if split_method in ("cluster", "mmseqs", "minhash"):
        use_mmseqs = split_method == "mmseqs"
        if split_method == "cluster":
            import shutil
            use_mmseqs = shutil.which("mmseqs") is not None
        if use_mmseqs:
            logger.info("Clustering with MMseqs2 easy-cluster")
            clusters, cl_info = mmseqs_clusters(
                protein_ids, sequences,
                min_seq_id=float(cl_cfg.get("mmseqs_min_seq_id", 0.3)),
                coverage=float(cl_cfg.get("mmseqs_coverage", 0.8)),
            )
        else:
            logger.info("MMseqs2 not found -> MinHash-LSH k-mer clustering")
            cids, cl_info = minhash_clusters(
                sequences,
                k=int(cl_cfg.get("kmer_size", 5)),
                num_hashes=int(cl_cfg.get("num_hashes", 64)),
                bands=int(cl_cfg.get("bands", 16)),
                seed=seed,
            )
            clusters = {pid: f"c{int(c)}" for pid, c in zip(protein_ids, cids)}
        # Exact duplicates must always share a group, even if the clusterer separated them.
        seq_to_cluster: Dict[str, str] = {}
        for pid, seq in zip(protein_ids, sequences):
            seq_to_cluster.setdefault(seq, clusters[pid])
        merged = 0
        for pid, seq in zip(protein_ids, sequences):
            if clusters[pid] != seq_to_cluster[seq]:
                clusters[pid] = seq_to_cluster[seq]
                merged += 1
        cl_info["exact_duplicate_reassignments"] = merged
        effective = "mmseqs_cluster" if use_mmseqs else "minhash_cluster"
        train_ids, valid_ids, manifest = create_split(
            protein_ids, sequences=sequences, clusters=clusters, val_ratio=val_ratio,
            seed=seed, method=effective, clustering_info=cl_info,
        )
    elif split_method == "sequence":
        train_ids, valid_ids, manifest = create_split(
            protein_ids, sequences=sequences, val_ratio=val_ratio, seed=seed)
    elif split_method == "random":
        train_ids, valid_ids, manifest = create_split(protein_ids, val_ratio=val_ratio, seed=seed)
    else:
        raise ValueError(f"Unknown split method: {split_method}")

    id_to_seq = dict(zip(protein_ids, sequences))
    leakage = audit_split_leakage(train_ids, valid_ids, id_to_seq, clusters)
    if leakage["id_overlap"] or leakage["exact_sequence_overlap"]:
        raise AssertionError(f"Split leakage detected: {leakage}")
    manifest["leakage_audit"] = leakage
    (out_dir / "train_ids.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (out_dir / "valid_ids.txt").write_text("\n".join(valid_ids) + "\n", encoding="utf-8")

    # ---------------- Vocabularies (fit on train split only) ----------------
    train_set = set(train_ids)
    valid_set = set(valid_ids)
    min_count = int(data_cfg.get("min_term_count", 1))
    vocabs: Dict[str, Dict[str, int]] = {}
    vocab_hashes: Dict[str, str] = {}
    label_qc: Dict[str, Any] = {}
    for a in ASPECTS:
        counts = Counter(go for pid, terms in prop_ann[a].items() if pid in train_set for go in terms)
        observed = {go for go, c in counts.items() if c >= min_count}
        vocab = onto.build_aspect_vocabulary(a, observed_terms=observed)
        vocabs[a] = vocab
        vocab_hashes[a] = compute_list_hash(list(vocab.keys()))
        save_json_atomic(vocab, out_dir / f"vocab_{a}.json")

        edges, rels = onto.build_go_edges(vocab, a, return_relations=True)
        np.save(out_dir / f"go_edges_{a}.npy", edges)
        save_json_atomic({"vocab_hash": vocab_hashes[a], "relations": rels,
                          "relation_counts": dict(Counter(rels))},
                         out_dir / f"go_edge_relations_{a}.json")

        y = build_sparse_label_matrix(protein_ids, prop_ann[a], vocab)
        y_raw = build_sparse_label_matrix(protein_ids, raw_ann[a], vocab)
        _save_csr(out_dir / f"labels_{a}.npz", y)
        _save_csr(out_dir / f"labels_raw_{a}.npz", y_raw)

        # ---- checks (plan §4) ----
        assert all(onto.terms[g].aspect == a for g in vocab), "cross-aspect term in vocab"
        if edges.size:
            assert edges.max() < len(vocab) and (edges[:, 0] != edges[:, 1]).all()
        dense_ok = int(y.sum()) == sum(
            len([g for g in prop_ann[a].get(p, ()) if g in vocab]) for p in protein_ids)
        assert dense_ok, "sparse label matrix lost entries"
        # Hierarchy consistency of the label matrix: child=1 must imply parent=1 (all edges)
        if edges.size:
            yc = y.tocsc()
            diff = yc[:, edges[:, 1]] - yc[:, edges[:, 0]]
            label_hier_viol = int((diff > 0).sum())
        else:
            label_hier_viol = 0
        term_freq = np.asarray(y[[i for i, p in enumerate(protein_ids) if p in train_set]].sum(axis=0)).ravel()
        valid_terms_all = {g for p, t in prop_ann[a].items() if p in valid_set for g in t}
        label_qc[a] = {
            "K": len(vocab),
            "num_edges": int(len(edges)),
            "edge_relation_counts": dict(Counter(rels)),
            "train_positive_pairs": int(term_freq.sum()),
            "label_hierarchy_violations": label_hier_viol,
            "term_frequency": {
                "max": int(term_freq.max()) if term_freq.size else 0,
                "median": float(np.median(term_freq)) if term_freq.size else 0.0,
                "terms_with_1": int((term_freq == 1).sum()),
                "terms_le_10": int((term_freq <= 10).sum()),
                "terms_ge_100": int((term_freq >= 100).sum()),
            },
            "valid_proteins_annotated": int(sum(1 for p in prop_ann[a] if p in valid_set)),
            "valid_terms_outside_vocab": len(valid_terms_all - set(vocab)),
            "vocab_hash": vocab_hashes[a],
        }
        logger.info(f"[{a}] K={len(vocab)} edges={len(edges)} train_pos={int(term_freq.sum())}")
    qc["labels"] = label_qc

    # ---------------- Taxonomy vocab (train split only) ----------------
    train_tax = [int(t) for p, t in zip(protein_ids, raw_tax) if p in train_set]
    tax_vocab = build_taxonomy_vocabulary(train_tax, min_count=int(config.get("taxonomy", {}).get("min_count", 1)))
    save_json_atomic({str(k): v for k, v in tax_vocab.items()}, out_dir / "taxonomy_vocab.json")
    np.save(out_dir / "taxonomy_ids.npy", raw_tax)
    _, train_tax_idx = encode_taxonomy_ids(protein_ids, pid_to_taxon, tax_vocab)
    valid_mask = np.array([p in valid_set for p in protein_ids])
    tax_qc = {
        "num_taxa_in_train_split": len(tax_vocab) - 1,
        "missing_taxon_train_set": int((raw_tax == MISSING_TAXON_ID).sum()),
        "valid_unseen_taxon_rate": float((train_tax_idx[valid_mask] == UNK_TAXON_INDEX).mean()) if valid_mask.any() else 0.0,
    }

    # ---------------- Test superset ----------------
    test_qc: Dict[str, Any] = {}
    n_test = 0
    if raw["test_sequences"].exists():
        logger.info(f"Parsing test superset {raw['test_sequences']}")
        test_ids, test_seqs, t_qc = parse_fasta(raw["test_sequences"])
        header_tax = parse_fasta_header_field(raw["test_sequences"], field_index=1)
        test_tax = np.array([int(header_tax.get(p, MISSING_TAXON_ID)) for p in test_ids], dtype=np.int64)
        if subset:
            test_ids, test_seqs, test_tax = test_ids[: int(subset)], test_seqs[: int(subset)], test_tax[: int(subset)]
        t_qc["length_distribution"] = _length_histogram([len(s) for s in test_seqs])
        save_json_atomic(test_ids, out_dir / "test_ids.json")
        _write_jsonl_sequences(out_dir / "test_sequences.jsonl", test_ids, test_seqs)
        np.save(out_dir / "test_taxonomy_ids.npy", test_tax)
        test_idx = np.array([tax_vocab.get(int(t), UNK_TAXON_INDEX) for t in test_tax])
        tax_qc["test_missing_taxon"] = int((test_tax == MISSING_TAXON_ID).sum())
        tax_qc["test_unseen_taxon_rate"] = float((test_idx == UNK_TAXON_INDEX).mean()) if len(test_idx) else 0.0
        tax_qc["test_num_distinct_taxa"] = int(len(set(test_tax.tolist())))
        test_qc = t_qc
        n_test = len(test_ids)
        test_qc["overlap_ids_with_train"] = len(set(test_ids) & pid_set)
    qc["taxonomy"] = tax_qc
    qc["test_sequences"] = test_qc

    # ---------------- Canonical tables ----------------
    save_json_atomic(protein_ids, out_dir / "protein_ids.json")
    _write_jsonl_sequences(out_dir / "sequences.jsonl", protein_ids, sequences)

    manifest["vocab_hashes"] = vocab_hashes
    save_json_atomic(manifest, out_dir / "split_manifest.json")
    save_json_atomic(qc, out_dir / "qc_report.json")

    raw_hashes = {k: compute_file_hash(p) for k, p in raw.items() if p.exists()}
    data_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_config_hash": compute_config_hash({"data": data_cfg, "seed": seed}),
        "raw_file_hashes": raw_hashes,
        "go_data_version": onto.data_version,
        "propagation_relations": list(relations),
        "propagation_policy": "positive labels closed under same-aspect ancestors via allowed relations",
        "obsolete_policy": "remap via single non-obsolete replaced_by (or alt_id); otherwise drop + log",
        "protein_ids_hash": compute_list_hash(protein_ids),
        "num_proteins": len(protein_ids),
        "num_test_proteins": n_test,
        "vocab_hashes": vocab_hashes,
        "vocab_sizes": {a: len(v) for a, v in vocabs.items()},
        "split_method": manifest["method"],
        "train_ids_hash": manifest["train_ids_hash"],
        "valid_ids_hash": manifest["valid_ids_hash"],
        "subset": subset,
    }
    save_json_atomic(data_manifest, out_dir / "data_manifest.json")
    logger.info(f"Data preparation complete -> {out_dir}")
    return {"qc": qc, "manifest": data_manifest, "split_manifest": manifest}
