"""Frozen ESM-2 feature extraction (plan §6).

* Pooling: mean over residue positions only (BOS/EOS/PAD excluded).
* Long sequences: ``max_length_policy='chunk'`` splits sequences longer than ``max_length`` residues
  into overlapping windows (``chunk_overlap`` residues), pools each window, and aggregates by a
  residue-count-weighted mean. ``'truncate'`` keeps the first ``max_length`` residues. Both are
  recorded in the embedding manifest; nothing is truncated silently.
* Batching: windows are sorted by length and packed under a token budget.
* Caching: shards keyed by sequence hash are written atomically; reruns skip finished sequences,
  and identical sequences (duplicates, train/test overlap) are embedded only once.
"""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch


def mean_pool_residues(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean-pool only valid amino acid residue positions (excluding BOS, EOS, and padding).

    Args:
        hidden_states: Tensor of shape (B, L, D)
        attention_mask: Tensor of shape (B, L) where 1 = non-pad, 0 = pad
        special_tokens_mask: Optional Tensor of shape (B, L) where 1 = special token (BOS/EOS/PAD), 0 = residue
    Returns:
        Pooled embeddings of shape (B, D)
    """
    if special_tokens_mask is not None:
        valid_mask = (attention_mask > 0) & (special_tokens_mask == 0)
    else:
        # Standard ESM layout: index 0 is BOS (<cls>), last non-pad position is EOS (<eos>)
        L = attention_mask.shape[1]
        lengths = attention_mask.sum(dim=1, keepdim=True)  # includes BOS+EOS
        pos = torch.arange(L, device=attention_mask.device).unsqueeze(0)
        valid_mask = (pos >= 1) & (pos < lengths - 1)

    mask_float = valid_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask_float).sum(dim=1)
    counts = mask_float.sum(dim=1).clamp(min=1.0)
    return summed / counts


def make_windows(seq: str, max_length: int, policy: str, overlap: int) -> List[str]:
    """Split one sequence into the residue windows that will be embedded."""
    if len(seq) <= max_length:
        return [seq]
    if policy == "truncate":
        return [seq[:max_length]]
    if policy != "chunk":
        raise ValueError(f"Unsupported max_length_policy: {policy}")
    stride = max_length - overlap
    if stride <= 0:
        raise ValueError("chunk_overlap must be smaller than max_length")
    windows = []
    start = 0
    while True:
        windows.append(seq[start : start + max_length])
        if start + max_length >= len(seq):
            break
        start += stride
    return windows


def _token_batches(lengths: np.ndarray, max_tokens: int, max_batch: int = 512) -> Iterator[np.ndarray]:
    """Yield index arrays (into ``lengths``) sorted by length and packed under a token budget."""
    order = np.argsort(-lengths, kind="stable")
    batch: List[int] = []
    cur_max = 0
    for idx in order:
        l = int(lengths[idx]) + 2
        new_max = max(cur_max, l)
        if batch and (new_max * (len(batch) + 1) > max_tokens or len(batch) >= max_batch):
            yield np.array(batch)
            batch, new_max = [], l
        batch.append(int(idx))
        cur_max = new_max
    if batch:
        yield np.array(batch)


class ESMEmbedder:
    """Frozen ESM-2 wrapper producing residue-mean-pooled embeddings."""

    def __init__(
        self,
        model_name: str = "facebook/esm2_t30_150M_UR50D",
        device: Optional[str] = None,
        dtype: str = "float16",
        max_length: int = 1022,
        max_length_policy: str = "chunk",
        chunk_overlap: int = 256,
        max_tokens_per_batch: int = 24000,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype) if self.device != "cpu" else torch.float32
        self.max_length = max_length
        self.policy = max_length_policy
        self.overlap = chunk_overlap
        self.max_tokens = max_tokens_per_batch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self.device, dtype=self.dtype)
        self.hidden_dim = int(self.model.config.hidden_size)
        # Fast vectorized tokenizer: ESM tokenizes one residue per token, so a byte lookup table
        # reproduces EsmTokenizer output exactly (verified in tests) at a fraction of the cost.
        vocab = self.tokenizer.get_vocab()
        self._lut = np.full(256, self.tokenizer.unk_token_id, dtype=np.int64)
        for tok, idx in vocab.items():
            if len(tok) == 1:
                self._lut[ord(tok)] = idx
        self._cls = self.tokenizer.cls_token_id
        self._eos = self.tokenizer.eos_token_id
        self._pad = self.tokenizer.pad_token_id

    def tokenize(self, windows: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        L = max(len(w) for w in windows) + 2
        ids = np.full((len(windows), L), self._pad, dtype=np.int64)
        attn = np.zeros((len(windows), L), dtype=np.int64)
        for i, w in enumerate(windows):
            n = len(w)
            ids[i, 0] = self._cls
            ids[i, 1 : n + 1] = self._lut[np.frombuffer(w.encode("ascii", errors="replace"), dtype=np.uint8)]
            ids[i, n + 1] = self._eos
            attn[i, : n + 2] = 1
        return torch.from_numpy(ids), torch.from_numpy(attn)

    def info(self) -> Dict[str, Any]:
        import transformers

        return {
            "esm_name": self.model_name,
            "hidden_dim": self.hidden_dim,
            "dtype": str(self.dtype).replace("torch.", ""),
            "device": self.device,
            "device_name": torch.cuda.get_device_name(0) if self.device.startswith("cuda") else "cpu",
            "max_length": self.max_length,
            "max_length_policy": self.policy,
            "chunk_overlap": self.overlap if self.policy == "chunk" else None,
            "chunk_aggregation": "residue-count-weighted mean of window means" if self.policy == "chunk" else None,
            "pooling": "mean over residue tokens (BOS/EOS/PAD excluded)",
            "max_tokens_per_batch": self.max_tokens,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "tokenizer_class": type(self.tokenizer).__name__,
        }

    @torch.no_grad()
    def _embed_windows(self, windows: List[str]) -> np.ndarray:
        ids, attn = self.tokenize(windows)
        input_ids = ids.to(self.device, non_blocking=True)
        attn = attn.to(self.device, non_blocking=True)
        out = self.model(input_ids=input_ids, attention_mask=attn).last_hidden_state
        pooled = mean_pool_residues(out.float(), attn)  # BOS/EOS/PAD excluded via positions
        return pooled.cpu().numpy()

    def embed(self, sequences: List[str], progress: bool = True) -> np.ndarray:
        """Embed a list of sequences; returns float32 array (N, D) in input order."""
        win_seq: List[str] = []
        win_owner: List[int] = []
        for i, s in enumerate(sequences):
            for w in make_windows(s, self.max_length, self.policy, self.overlap):
                win_seq.append(w)
                win_owner.append(i)
        lengths = np.array([len(w) for w in win_seq])
        win_emb = np.zeros((len(win_seq), self.hidden_dim), dtype=np.float32)
        batches = list(_token_batches(lengths, self.max_tokens))
        iterator = batches
        if progress:
            from tqdm import tqdm

            iterator = tqdm(batches, desc="ESM batches", leave=False)
        for idx in iterator:
            win_emb[idx] = self._embed_windows([win_seq[j] for j in idx])

        out = np.zeros((len(sequences), self.hidden_dim), dtype=np.float64)
        weight = np.zeros(len(sequences), dtype=np.float64)
        owner = np.asarray(win_owner)
        np.add.at(out, owner, win_emb * lengths[:, None])
        np.add.at(weight, owner, lengths)
        out = (out / np.maximum(weight, 1)[:, None]).astype(np.float32)
        if not np.isfinite(out).all():
            raise FloatingPointError("Non-finite values detected in ESM embeddings.")
        return out


# ---------------------------------------------------------------------------
# Resumable shard cache
# ---------------------------------------------------------------------------

def sequence_key(seq: str) -> str:
    """Cache key: hash of the sequence, so identical sequences (duplicates, train/test overlap)
    are embedded once."""
    import hashlib

    return hashlib.sha1(seq.encode("ascii", errors="replace")).hexdigest()


def _shard_dir(cache_dir: Path, set_name: str) -> Path:
    d = cache_dir / "shards" / set_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached_keys(cache_dir: Path, set_name: str) -> Dict[str, Tuple[Path, int]]:
    """Map sequence key -> (shard npy path, row) for all completed shards."""
    done: Dict[str, Tuple[Path, int]] = {}
    for keys_file in sorted(_shard_dir(cache_dir, set_name).glob("shard_*.keys.json")):
        npy = keys_file.with_name(keys_file.name.replace(".keys.json", ".npy"))
        if not npy.exists():
            continue
        for row, key in enumerate(json.loads(keys_file.read_text(encoding="utf-8"))):
            done[key] = (npy, row)
    return done


def extract_to_cache(
    embedder: ESMEmbedder,
    protein_ids: List[str],
    sequences: List[str],
    cache_dir: Path,
    set_name: str,
    shard_size: int = 10000,
    logger=None,
) -> Dict[str, Any]:
    """Embed every unique sequence not already cached; write shards atomically.

    A shard's ``.npy`` is written before its ``.keys.json``; a shard counts as complete only
    when both exist, so an interrupted run resumes cleanly.
    """
    done = cached_keys(cache_dir, set_name)
    key_to_seq: Dict[str, str] = {}
    for seq in sequences:
        k = sequence_key(seq)
        if k not in done:
            key_to_seq.setdefault(k, seq)
    todo = sorted(key_to_seq, key=lambda k: len(key_to_seq[k]))
    sdir = _shard_dir(cache_dir, set_name)
    existing = [int(f.name.split("_")[1].split(".")[0]) for f in sdir.glob("shard_*.npy")]
    next_shard = max(existing) + 1 if existing else 0
    stats = {
        "num_ids": len(protein_ids),
        "unique_sequences_to_embed": len(todo),
        "cached_or_duplicate": len(protein_ids) - len(todo),
        "seconds": 0.0,
    }
    if logger:
        logger.info(f"[{set_name}] {len(protein_ids)} ids -> {len(todo)} new unique sequences to embed")
    if not todo:
        return stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n_res = 0
    for s in range(0, len(todo), shard_size):
        keys = todo[s : s + shard_size]
        seqs = [key_to_seq[k] for k in keys]
        emb = embedder.embed(seqs, progress=False)
        n_res += sum(len(x) for x in seqs)
        base = sdir / f"shard_{next_shard:05d}"
        tmp_npy = sdir / f"shard_{next_shard:05d}.npy.tmp"
        with open(tmp_npy, "wb") as f:
            np.save(f, emb)
        os.replace(tmp_npy, base.with_suffix(".npy"))
        tmp_keys = sdir / f"shard_{next_shard:05d}.keys.json.tmp"
        tmp_keys.write_text(json.dumps(keys), encoding="utf-8")
        os.replace(tmp_keys, sdir / f"shard_{next_shard:05d}.keys.json")
        next_shard += 1
        elapsed = time.time() - t0
        n_done = s + len(keys)
        rate = n_done / max(elapsed, 1e-9)
        if logger:
            logger.info(
                f"[{set_name}] {n_done}/{len(todo)} | {rate:.1f} seq/s | {n_res / elapsed:.0f} res/s"
                f" | ETA(by count) {(len(todo) - n_done) / rate / 60:.1f} min"
            )
    stats["seconds"] = time.time() - t0
    stats["sequences_per_sec"] = len(todo) / max(stats["seconds"], 1e-9)
    stats["residues_per_sec"] = n_res / max(stats["seconds"], 1e-9)
    if torch.cuda.is_available():
        stats["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return stats


def assemble_from_cache(
    cache_dir: Path, set_name: str, protein_ids: List[str], sequences: List[str]
) -> np.ndarray:
    """Gather cached embeddings into an array aligned exactly with ``protein_ids``."""
    done = cached_keys(cache_dir, set_name)
    keys = [sequence_key(s) for s in sequences]
    missing = [p for p, k in zip(protein_ids, keys) if k not in done]
    if missing:
        raise KeyError(f"{len(missing)} ids missing from cache '{set_name}' (e.g. {missing[:3]})")
    by_shard: Dict[Path, List[Tuple[int, int]]] = {}
    for out_row, k in enumerate(keys):
        path, row = done[k]
        by_shard.setdefault(path, []).append((out_row, row))
    dim = np.load(next(iter(by_shard)), mmap_mode="r").shape[1]
    result = np.zeros((len(protein_ids), dim), dtype=np.float32)
    for path, pairs in by_shard.items():
        arr = np.load(path, mmap_mode="r")
        out_rows = np.array([p[0] for p in pairs])
        src_rows = np.array([p[1] for p in pairs])
        result[out_rows] = arr[src_rows]
    return result


def encode_sequences(
    sequences: List[str],
    protein_ids: List[str],
    model_name: str = "facebook/esm2_t30_150M_UR50D",
    batch_size: int = 16,
    max_length: int = 1022,
    max_length_policy: str = "chunk",
    device: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    set_name: str = "adhoc",
    chunk_overlap: int = 256,
) -> np.ndarray:
    """Contract entrypoint: frozen ESM-2 residue mean-pooled embeddings (N, D), aligned with IDs.

    ``batch_size`` is converted into a token budget (batch_size * (max_length + 2)).
    With ``cache_dir`` the result is cached by ID and reused on later calls.
    """
    if len(sequences) != len(protein_ids):
        raise ValueError("sequences and protein_ids must have identical length.")
    embedder = ESMEmbedder(
        model_name=model_name, device=device, max_length=max_length,
        max_length_policy=max_length_policy, chunk_overlap=chunk_overlap,
        max_tokens_per_batch=batch_size * (max_length + 2),
    )
    if cache_dir is None:
        emb = embedder.embed(sequences)
    else:
        extract_to_cache(embedder, protein_ids, sequences, Path(cache_dir), set_name)
        emb = assemble_from_cache(Path(cache_dir), set_name, protein_ids, sequences)
    if emb.shape != (len(protein_ids), embedder.hidden_dim):
        raise AssertionError(f"Unexpected embedding shape {emb.shape}")
    return emb
