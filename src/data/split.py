"""Train/validation splitting with leakage control.

Methods (plan §5):
  * ``mmseqs``   – MMseqs2 ``easy-cluster`` (used when the binary is on PATH).
  * ``minhash``  – MinHash-LSH clustering on amino-acid k-mer sets + union-find. Approximates a
                   high-identity sequence clustering without external tools. Default here.
  * ``sequence`` – group only exact duplicate sequences (weaker fallback).
  * ``random``   – plain random split (weakest; for debugging only).

Every method keeps whole groups on one side of the split so exact duplicates (and, for the cluster
methods, near-duplicates) never straddle train/validation.
"""
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.io import compute_list_hash

_MERSENNE = np.uint64((1 << 61) - 1)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return int(root)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def _kmer_codes(seq: str, k: int) -> np.ndarray:
    """Encode all k-mers of a sequence as unique uint64 integers (base-32 packing)."""
    arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8).astype(np.uint64) - np.uint64(64)
    if arr.size < k:
        arr = np.pad(arr, (0, k - arr.size))
    codes = np.zeros(arr.size - k + 1, dtype=np.uint64)
    for j in range(k):
        codes = (codes << np.uint64(5)) | arr[j : arr.size - k + 1 + j]
    return np.unique(codes)


def minhash_clusters(
    sequences: List[str],
    k: int = 5,
    num_hashes: int = 64,
    bands: int = 16,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Cluster sequences by MinHash-LSH on k-mer sets; returns (cluster_id per sequence, info).

    Two sequences land in the same LSH bucket (and are unioned) when one of ``bands`` bands of
    ``num_hashes // bands`` MinHash rows matches exactly. Collision probability for Jaccard J is
    ``1 - (1 - J^r)^b``; the 50% point is roughly ``(1/b)^(1/r)``.
    """
    if num_hashes % bands != 0:
        raise ValueError("num_hashes must be divisible by bands")
    rows = num_hashes // bands
    rng = np.random.default_rng(seed)
    a = rng.integers(1, int(_MERSENNE), size=num_hashes, dtype=np.uint64)
    b = rng.integers(0, int(_MERSENNE), size=num_hashes, dtype=np.uint64)

    n = len(sequences)
    sigs = np.empty((n, num_hashes), dtype=np.uint64)
    for i, seq in enumerate(sequences):
        codes = _kmer_codes(seq, k)
        # Universal hashing (wrap-around uint64 arithmetic is deterministic and fine for MinHash).
        hashed = (codes[:, None] * a[None, :] + b[None, :]) % _MERSENNE
        sigs[i] = hashed.min(axis=0)

    uf = _UnionFind(n)
    for band in range(bands):
        block = np.ascontiguousarray(sigs[:, band * rows : (band + 1) * rows])
        keys = block.view(np.dtype((np.void, block.dtype.itemsize * rows))).ravel()
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        same = np.nonzero(sorted_keys[1:] == sorted_keys[:-1])[0]
        for pos in same:
            uf.union(int(order[pos]), int(order[pos + 1]))

    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int64)
    _, cluster_ids = np.unique(roots, return_inverse=True)
    sizes = np.bincount(cluster_ids)
    info = {
        "algorithm": "minhash_lsh_union_find",
        "kmer_size": k,
        "num_hashes": num_hashes,
        "bands": bands,
        "rows_per_band": rows,
        "approx_jaccard_threshold_50pct": float((1.0 / bands) ** (1.0 / rows)),
        "num_clusters": int(sizes.size),
        "num_singletons": int((sizes == 1).sum()),
        "largest_cluster": int(sizes.max()) if sizes.size else 0,
        "mean_cluster_size": float(sizes.mean()) if sizes.size else 0.0,
    }
    return cluster_ids, info


def mmseqs_clusters(
    protein_ids: List[str],
    sequences: List[str],
    min_seq_id: float = 0.3,
    coverage: float = 0.8,
    threads: int = 8,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Run MMseqs2 easy-cluster; returns (protein_id -> representative id, info)."""
    exe = shutil.which("mmseqs")
    if exe is None:
        raise FileNotFoundError("mmseqs binary not found on PATH")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fasta = tmp_path / "in.fasta"
        with open(fasta, "w", encoding="utf-8") as f:
            for pid, seq in zip(protein_ids, sequences):
                f.write(f">{pid}\n{seq}\n")
        prefix = tmp_path / "clu"
        subprocess.run(
            [exe, "easy-cluster", str(fasta), str(prefix), str(tmp_path / "work"),
             "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--threads", str(threads)],
            check=True, capture_output=True,
        )
        mapping: Dict[str, str] = {}
        with open(f"{prefix}_cluster.tsv", "r", encoding="utf-8") as f:
            for line in f:
                rep, member = line.rstrip("\n").split("\t")
                mapping[member] = rep
    info = {
        "algorithm": "mmseqs2_easy_cluster",
        "min_seq_id": min_seq_id,
        "coverage": coverage,
        "num_clusters": len(set(mapping.values())),
    }
    return mapping, info


def create_split(
    protein_ids: List[str],
    sequences: Optional[List[str]] = None,
    clusters: Optional[Dict[str, str]] = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    method: str = "cluster",
    clustering_info: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Create train/validation split and return (train_ids, valid_ids, split_manifest).

    Groups are formed by cluster ID when ``clusters`` is given, otherwise by exact sequence (when
    ``sequences`` is given), otherwise by protein ID. Whole groups are assigned to one side.
    Output ID lists preserve the canonical input order.
    """
    rng = np.random.default_rng(seed)
    n_total = len(protein_ids)

    group_to_idx: Dict[str, List[int]] = {}
    for idx, pid in enumerate(protein_ids):
        if clusters is not None and pid in clusters:
            group_key = f"cluster:{clusters[pid]}"
        elif sequences is not None:
            group_key = f"seq:{sequences[idx]}"
        else:
            group_key = f"pid:{pid}"
        group_to_idx.setdefault(group_key, []).append(idx)

    unique_groups = sorted(group_to_idx.keys())
    rng.shuffle(unique_groups)

    target_val = int(round(n_total * val_ratio))
    is_valid = np.zeros(n_total, dtype=bool)
    n_valid = 0
    for g in unique_groups:
        members = group_to_idx[g]
        # Skip groups that would overshoot the target by a lot (keeps giant clusters in train).
        if n_valid >= target_val:
            break
        if n_valid + len(members) > target_val * 1.02 and len(members) > 1:
            continue
        is_valid[members] = True
        n_valid += len(members)

    train_ids = [pid for pid, v in zip(protein_ids, is_valid) if not v]
    valid_ids = [pid for pid, v in zip(protein_ids, is_valid) if v]

    if clusters is not None:
        effective_method = method
    elif sequences is not None:
        effective_method = "sequence_grouped"
    else:
        effective_method = "random"

    manifest = {
        "method": effective_method,
        "seed": seed,
        "val_ratio": val_ratio,
        "num_total": n_total,
        "num_groups": len(unique_groups),
        "num_train": len(train_ids),
        "num_valid": len(valid_ids),
        "input_ids_hash": compute_list_hash(protein_ids),
        "train_ids_hash": compute_list_hash(train_ids),
        "valid_ids_hash": compute_list_hash(valid_ids),
        "clustering": clustering_info or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return train_ids, valid_ids, manifest


def audit_split_leakage(
    train_ids: List[str],
    valid_ids: List[str],
    id_to_seq: Optional[Dict[str, str]] = None,
    clusters: Optional[Dict[str, str]] = None,
) -> Dict[str, int]:
    """Check ID overlap, exact sequence overlap and (optionally) shared clusters between splits."""
    train_set = set(train_ids)
    valid_set = set(valid_ids)
    result = {"id_overlap": len(train_set & valid_set), "exact_sequence_overlap": 0}

    if id_to_seq is not None:
        train_seqs = {id_to_seq[pid] for pid in train_ids if pid in id_to_seq}
        valid_seqs = {id_to_seq[pid] for pid in valid_ids if pid in id_to_seq}
        result["exact_sequence_overlap"] = len(train_seqs & valid_seqs)

    if clusters is not None:
        train_cl = {clusters[p] for p in train_ids if p in clusters}
        valid_cl = {clusters[p] for p in valid_ids if p in clusters}
        result["shared_clusters"] = len(train_cl & valid_cl)
    return result
