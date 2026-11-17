from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from src.utils.io import compute_list_hash


def create_split(
    protein_ids: List[str],
    sequences: Optional[List[str]] = None,
    clusters: Optional[Dict[str, str]] = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    method: str = "cluster",
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Create train/validation split and return (train_ids, valid_ids, split_manifest).
    Ensures exact sequence deduplication across splits when sequences are provided.
    """
    rng = np.random.default_rng(seed)
    n_total = len(protein_ids)

    # Group by cluster ID or exact sequence to prevent leakage across splits
    group_to_pids: Dict[str, List[str]] = {}
    for idx, pid in enumerate(protein_ids):
        if clusters is not None and pid in clusters:
            group_key = f"cluster:{clusters[pid]}"
        elif sequences is not None:
            group_key = f"seq:{sequences[idx]}"
        else:
            group_key = f"pid:{pid}"
        group_to_pids.setdefault(group_key, []).append(pid)

    unique_groups = sorted(group_to_pids.keys())
    rng.shuffle(unique_groups)

    train_ids: List[str] = []
    valid_ids: List[str] = []
    target_val = int(round(n_total * val_ratio))

    for g in unique_groups:
        pids = group_to_pids[g]
        if len(valid_ids) < target_val:
            valid_ids.extend(pids)
        else:
            train_ids.extend(pids)

    manifest = {
        "method": method if clusters is not None else ("sequence_grouped" if sequences is not None else "random"),
        "seed": seed,
        "val_ratio": val_ratio,
        "num_total": n_total,
        "num_train": len(train_ids),
        "num_valid": len(valid_ids),
        "input_ids_hash": compute_list_hash(protein_ids),
        "train_ids_hash": compute_list_hash(train_ids),
        "valid_ids_hash": compute_list_hash(valid_ids),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return train_ids, valid_ids, manifest


def audit_split_leakage(
    train_ids: List[str],
    valid_ids: List[str],
    id_to_seq: Optional[Dict[str, str]] = None,
) -> Dict[str, int]:
    """Check for ID overlap and exact sequence overlap between train and validation sets."""
    train_set = set(train_ids)
    valid_set = set(valid_ids)
    id_overlap = len(train_set.intersection(valid_set))

    seq_overlap = 0
    if id_to_seq is not None:
        train_seqs = {id_to_seq[pid] for pid in train_ids if pid in id_to_seq}
        valid_seqs = {id_to_seq[pid] for pid in valid_ids if pid in id_to_seq}
        seq_overlap = len(train_seqs.intersection(valid_seqs))

    return {
        "id_overlap": id_overlap,
        "exact_sequence_overlap": seq_overlap,
    }
