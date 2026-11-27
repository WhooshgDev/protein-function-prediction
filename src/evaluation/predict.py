from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[str] = None,
) -> np.ndarray:
    """Run inference over DataLoader, apply sigmoid to logits, and return (N, K) probabilities in [0, 1]."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    preds = []
    for batch in data_loader:
        emb = batch["embedding"].to(device)
        tax_idx = batch.get("taxonomy_idx")
        if tax_idx is not None:
            tax_idx = tax_idx.to(device)
            logits = model(emb, tax_idx)
        else:
            logits = model(emb)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)

    if not preds:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(preds)


# ---------------------------------------------------------------------------
# Checkpoint loading + streaming multi-aspect prediction/export
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(path, device: Optional[str] = None):
    """Rebuild the model saved by ``src.training.train`` and return (model, checkpoint)."""
    from src.training.train import build_model

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["metadata"]
    config = ckpt["config"]
    first_w = ckpt["model_state_dict"]["classifier.net.0.weight"]
    tax_dim = int(config.get("taxonomy", {}).get("embedding_dim", 64)) if config.get("taxonomy", {}).get("enabled") else 0
    model = build_model(config, num_classes=len(meta["vocab"]), num_taxa=int(meta.get("num_taxa", 1)),
                        seq_dim=first_w.shape[1] - tax_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return model, ckpt


def propagate_scores_max(probs: torch.Tensor, edges: torch.Tensor, max_iter: int = 64) -> torch.Tensor:
    """Make scores hierarchy-consistent: s_parent = max(s_parent, s_child) until a fixed point."""
    if edges is None or edges.numel() == 0:
        return probs
    parent, child = edges[:, 0], edges[:, 1]
    idx = parent.unsqueeze(0).expand(probs.shape[0], -1)
    for _ in range(max_iter):
        new = probs.scatter_reduce(1, idx, probs[:, child], reduce="amax", include_self=True)
        if torch.equal(new, probs):
            break
        probs = new
    return probs


@torch.no_grad()
def predict_and_export(
    checkpoints: dict,
    embeddings: np.ndarray,
    protein_ids: list,
    output_path,
    taxonomy_indices: Optional[np.ndarray] = None,
    export_cfg: Optional[dict] = None,
    propagate: bool = False,
    edges: Optional[dict] = None,
    expected_vocab_hashes: Optional[dict] = None,
    batch_size: int = 2048,
    device: Optional[str] = None,
    logger=None,
) -> dict:
    """Score all aspects for ``protein_ids`` and stream a CAFA submission.

    Per protein and aspect, keeps scores >= ``min_score`` and at most
    ``max_terms_per_aspect[aspect]`` terms (default split of the CAFA 1500-term limit). Raw sigmoid
    scores are written (no calibration).
    """
    import time

    from src.evaluation.export_cafa import GO_ID_PATTERN, StreamingCafaWriter

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = export_cfg or {}
    min_score = float(cfg.get("min_score", 0.01))
    total_cap = int(cfg.get("max_terms_per_protein", 1500))
    caps = cfg.get("max_terms_per_aspect") or {"F": total_cap * 3 // 10, "P": total_cap // 2, "C": total_cap // 5}
    models = {}
    for aspect, path in checkpoints.items():
        model, ckpt = load_model_from_checkpoint(path, device)
        vocab = ckpt["metadata"]["vocab"]
        bad = [g for g in vocab if not GO_ID_PATTERN.match(g)]
        if bad:
            raise ValueError(f"Invalid GO ids in checkpoint vocab: {bad[:3]}")
        if expected_vocab_hashes and ckpt["metadata"]["vocab_hash"] != expected_vocab_hashes[aspect]:
            raise ValueError(f"Checkpoint {path} vocab does not match processed vocab for {aspect}")
        e = None
        if propagate and edges is not None:
            e = torch.as_tensor(edges[aspect], dtype=torch.long, device=device)
        models[aspect] = (model, np.array(vocab), e, min(int(caps[aspect]), len(vocab)))
    writer = StreamingCafaWriter(output_path, sep=cfg.get("sep", "\t"), float_format=int(cfg.get("float_format", 3)))
    emb_t = torch.as_tensor(np.ascontiguousarray(embeddings), dtype=torch.float32)
    tax_t = None if taxonomy_indices is None else torch.as_tensor(taxonomy_indices, dtype=torch.long)
    t0 = time.time()
    for s in range(0, len(protein_ids), batch_size):
        x = emb_t[s : s + batch_size].to(device)
        tx = None if tax_t is None else tax_t[s : s + batch_size].to(device)
        per_aspect = []
        for aspect, (model, vocab, e, cap) in models.items():
            logits = model(x, tx) if (tx is not None and model.taxonomy_enabled) else model(x)
            probs = torch.sigmoid(logits)
            if e is not None:
                probs = propagate_scores_max(probs, e)
            top_s, top_i = torch.topk(probs, k=cap, dim=1)
            per_aspect.append((vocab, top_s.cpu().numpy(), top_i.cpu().numpy()))
        for r, pid in enumerate(protein_ids[s : s + batch_size]):
            gos, scs = [], []
            best_go, best_s = None, -1.0
            for vocab, top_s, top_i in per_aspect:
                keep = top_s[r] >= min_score
                gos.extend(vocab[top_i[r][keep]].tolist())
                scs.extend(top_s[r][keep].tolist())
                if top_s[r][0] > best_s:
                    best_go, best_s = vocab[top_i[r][0]], float(top_s[r][0])
            if not gos:  # every target protein gets at least its top-scoring term
                gos, scs = [best_go], [best_s]
            writer.write_protein(pid, gos, np.round(np.asarray(scs), 6).clip(0, 1))
        if logger and (s // batch_size) % 10 == 0:
            logger.info(f"predicted {min(s + batch_size, len(protein_ids))}/{len(protein_ids)}")
    summary = writer.close()
    summary.update({"num_target_proteins": len(protein_ids), "inference_time_sec": time.time() - t0,
                    "min_score": min_score, "max_terms_per_aspect": caps, "propagated_scores": propagate,
                    "checkpoints": {a: str(p) for a, p in checkpoints.items()}})
    return summary
