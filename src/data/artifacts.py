"""Load processed artifacts with explicit ID alignment checks (plan §11: never trust row order)."""
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

from src.data.taxonomy import UNK_TAXON_INDEX
from src.utils.io import compute_list_hash, load_json


class ProcessedData:
    def __init__(self, processed_dir, embedding_dir):
        self.proc = Path(processed_dir)
        self.emb_dir = Path(embedding_dir)
        self.protein_ids: List[str] = load_json(self.proc / "protein_ids.json")
        self.row_of = {p: i for i, p in enumerate(self.protein_ids)}
        self.manifest = load_json(self.proc / "data_manifest.json")
        self.split_manifest = load_json(self.proc / "split_manifest.json")
        self._cache: Dict[str, object] = {}

    # ---- ids ----
    def split_ids(self, split: str) -> List[str]:
        if split == "test":
            return load_json(self.proc / "test_ids.json")
        name = {"train": "train_ids.txt", "valid": "valid_ids.txt", "validation": "valid_ids.txt"}[split]
        return (self.proc / name).read_text(encoding="utf-8").split()

    # ---- GO ----
    def vocab(self, aspect: str) -> Dict[str, int]:
        key = f"vocab_{aspect}"
        if key not in self._cache:
            self._cache[key] = load_json(self.proc / f"vocab_{aspect}.json")
        return self._cache[key]

    def vocab_list(self, aspect: str) -> List[str]:
        v = self.vocab(aspect)
        out = [""] * len(v)
        for g, i in v.items():
            out[i] = g
        return out

    def edges(self, aspect: str) -> np.ndarray:
        return np.load(self.proc / f"go_edges_{aspect}.npy")

    def labels(self, aspect: str) -> sparse.csr_matrix:
        key = f"labels_{aspect}"
        if key not in self._cache:
            self._cache[key] = sparse.load_npz(self.proc / f"labels_{aspect}.npz").tocsr()
        return self._cache[key]

    def labels_for(self, aspect: str, ids: List[str]) -> sparse.csr_matrix:
        rows = np.array([self.row_of[p] for p in ids], dtype=np.int64)
        return self.labels(aspect)[rows]

    # ---- taxonomy ----
    def taxonomy_vocab(self) -> Dict[int, int]:
        return {int(k): v for k, v in load_json(self.proc / "taxonomy_vocab.json").items()}

    def taxonomy_indices(self, split: str, ids: List[str]) -> np.ndarray:
        vocab = self.taxonomy_vocab()
        if split == "test":
            raw = np.load(self.proc / "test_taxonomy_ids.npy")
            test_ids = self.split_ids("test")
            assert test_ids == ids, "test ids order mismatch"
        else:
            all_raw = np.load(self.proc / "taxonomy_ids.npy")
            raw = all_raw[[self.row_of[p] for p in ids]]
        return np.array([vocab.get(int(t), UNK_TAXON_INDEX) for t in raw], dtype=np.int64)

    # ---- embeddings ----
    def embeddings(self, split: str, ids: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        split = "valid" if split == "validation" else split
        emb = np.load(self.emb_dir / f"embeddings_{split}.npy", mmap_mode="r")
        emb_ids: List[str] = load_json(self.emb_dir / f"embeddings_{split}_ids.json")
        expected = self.split_ids(split)
        if emb_ids != expected:
            raise AssertionError(
                f"Embedding ids for '{split}' do not match the processed split "
                f"({compute_list_hash(emb_ids)} vs {compute_list_hash(expected)}); re-run extraction."
            )
        if emb.shape[0] != len(emb_ids):
            raise AssertionError("embedding rows != ids")
        if ids is None:
            return np.asarray(emb), emb_ids
        pos = {p: i for i, p in enumerate(emb_ids)}
        return np.asarray(emb[[pos[p] for p in ids]]), list(ids)


@lru_cache(maxsize=4)
def load_ia(ia_path: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with open(ia_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                out[parts[0]] = float(parts[1])
    return out


def ia_vector(vocab_list: List[str], ia: Dict[str, float]) -> np.ndarray:
    return np.array([ia.get(g, 0.0) for g in vocab_list], dtype=np.float32)
