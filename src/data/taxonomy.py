from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

UNK_TAXON_INDEX = 0
MISSING_TAXON_ID = -1


def parse_taxonomy_tsv(tsv_path: Union[str, Path]) -> Dict[str, int]:
    """Parse protein_id -> NCBI taxonomy ID mapping from TSV."""
    mapping: Dict[str, int] = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            if idx == 0 and not parts[1].strip().isdigit():
                continue
            pid = parts[0].strip()
            try:
                tax_id = int(parts[1].strip())
            except ValueError:
                tax_id = MISSING_TAXON_ID
            mapping[pid] = tax_id
    return mapping


def build_taxonomy_vocabulary(train_taxon_ids: List[int], min_count: int = 1) -> Dict[int, int]:
    """Build deterministic taxonomy vocabulary from training set only.
    Index 0 is reserved for UNK / missing / unseen taxa.
    """
    counts: Dict[int, int] = {}
    for tid in train_taxon_ids:
        if tid != MISSING_TAXON_ID and tid >= 0:
            counts[tid] = counts.get(tid, 0) + 1
    valid_taxa = sorted(tid for tid, c in counts.items() if c >= min_count)
    vocab = {MISSING_TAXON_ID: UNK_TAXON_INDEX}
    for idx, tid in enumerate(valid_taxa, start=1):
        vocab[tid] = idx
    return vocab


def encode_taxonomy_ids(
    protein_ids: List[str],
    pid_to_taxon: Dict[str, int],
    tax_vocab: Optional[Dict[int, int]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Align taxonomy IDs to canonical protein_ids order and optionally map to vocabulary indices."""
    raw_ids = np.array([pid_to_taxon.get(pid, MISSING_TAXON_ID) for pid in protein_ids], dtype=np.int64)
    if tax_vocab is None:
        return raw_ids, None
    mapped_indices = np.array([tax_vocab.get(int(tid), UNK_TAXON_INDEX) for tid in raw_ids], dtype=np.int64)
    return raw_ids, mapped_indices
