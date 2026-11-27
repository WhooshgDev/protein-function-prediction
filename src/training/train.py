"""High-level training runner (one model per GO aspect — plan §7 "separate models per aspect")."""
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from src.data.artifacts import ProcessedData, ia_vector, load_ia
from src.data.dataset import SparseBatchLoader
from src.data.pipeline import resolve_raw_files
from src.models.fusion import SequenceTaxonomyFusionModel
from src.training.trainer import Trainer
from src.utils.config import compute_config_hash, load_config
from src.utils.io import compute_list_hash, save_json_atomic
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def experiment_name(config: Dict[str, Any]) -> str:
    return str(config.get("experiment_name", "experiment"))


def checkpoint_path_for(config: Dict[str, Any], aspect: str) -> Path:
    return Path(config.get("checkpoint_dir", "checkpoints")) / f"{experiment_name(config)}_{aspect}_best.pt"


def results_dir_for(config: Dict[str, Any]) -> Path:
    d = Path(config.get("results_dir", "results")) / experiment_name(config)
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_model(config: Dict[str, Any], num_classes: int, num_taxa: int, seq_dim: int = 640):
    m, tax = config.get("model", {}), config.get("taxonomy", {})
    return SequenceTaxonomyFusionModel(
        seq_dim=seq_dim,
        num_classes=num_classes,
        hidden_dim=int(m.get("hidden_dim", 512)),
        dropout=float(m.get("dropout", 0.3)),
        taxonomy_enabled=bool(tax.get("enabled", False)),
        num_taxa=num_taxa,
        tax_dim=int(tax.get("embedding_dim", 64)),
        tax_unk_dropout=float(tax.get("unk_dropout", 0.0)),
    )


def _plot_curves(history, out_png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="valid")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("Loss")
    ax[1].plot(ep, [h["val_fmax"] for h in history], label="Fmax")
    if "val_wfmax" in history[0]:
        ax[1].plot(ep, [h["val_wfmax"] for h in history], label="IA-weighted Fmax")
    ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].set_title("Validation")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def run_training(config: Dict[str, Any], aspect: str, logger=None) -> Dict[str, Any]:
    """Train one aspect model from processed artifacts + cached embeddings."""
    exp = experiment_name(config)
    logger = logger or get_logger(f"train.{exp}.{aspect}", log_file=f"logs/train_{exp}_{aspect}.log")
    seed = int(config.get("seed", 42))
    set_seed(seed)
    data_cfg = config["data"]
    pdata = ProcessedData(data_cfg["processed_dir"], data_cfg["embedding_dir"])

    train_ids_all = pdata.split_ids("train")
    valid_ids_all = pdata.split_ids("valid")
    y_train_all = pdata.labels_for(aspect, train_ids_all)
    y_valid_all = pdata.labels_for(aspect, valid_ids_all)
    # Positive-unlabelled policy: train/validate only on proteins with >=1 annotation in this aspect;
    # unannotated terms of those proteins are treated as negatives (BCE convention, see README).
    tr_keep = np.asarray(y_train_all.getnnz(axis=1) > 0)
    va_keep = np.asarray(y_valid_all.getnnz(axis=1) > 0)
    train_ids = [p for p, k in zip(train_ids_all, tr_keep) if k]
    valid_ids = [p for p, k in zip(valid_ids_all, va_keep) if k]
    y_train, y_valid = y_train_all[tr_keep], y_valid_all[va_keep]

    x_train, _ = pdata.embeddings("train", train_ids)
    x_valid, _ = pdata.embeddings("valid", valid_ids)
    tax_enabled = bool(config.get("taxonomy", {}).get("enabled", False))
    t_train = pdata.taxonomy_indices("train", train_ids) if tax_enabled else None
    t_valid = pdata.taxonomy_indices("valid", valid_ids) if tax_enabled else None
    num_taxa = len(pdata.taxonomy_vocab())

    vocab_list = pdata.vocab_list(aspect)
    K = len(vocab_list)
    edges = torch.as_tensor(pdata.edges(aspect), dtype=torch.long)
    raw = resolve_raw_files(config)
    ia_w = ia_vector(vocab_list, load_ia(str(raw["ia"]))) if raw["ia"].exists() else None
    logger.info(f"[{exp}/{aspect}] train={len(train_ids)} valid={len(valid_ids)} K={K} edges={len(edges)}"
                f" taxonomy={tax_enabled} lambda={config['loss'].get('hierarchical_lambda', 0.0)}")

    bs = int(config["training"].get("batch_size", 256))
    train_loader = SparseBatchLoader(x_train, y_train, t_train, batch_size=bs, shuffle=True, seed=seed)
    valid_loader = SparseBatchLoader(x_valid, y_valid, t_valid, batch_size=1024, shuffle=False)

    model = build_model(config, K, num_taxa, seq_dim=x_train.shape[1])
    metadata = {
        "experiment": exp,
        "aspect": aspect,
        "vocab": vocab_list,
        "vocab_hash": compute_list_hash(vocab_list),
        "expected_vocab_hash": pdata.manifest["vocab_hashes"][aspect],
        "split_hash": {"train": pdata.split_manifest["train_ids_hash"], "valid": pdata.split_manifest["valid_ids_hash"]},
        "train_proteins_hash": compute_list_hash(train_ids),
        "esm_model": config["model"]["esm_name"],
        "config_hash": compute_config_hash(config),
        "seed": seed,
        "num_taxa": num_taxa,
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "cuda": torch.version.cuda},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    assert metadata["vocab_hash"] == metadata["expected_vocab_hash"], "vocab hash mismatch with data manifest"

    trainer = Trainer(parent_child_edges=edges, ia_weights=ia_w, logger=logger)
    ckpt_path = checkpoint_path_for(config, aspect)
    best, history = trainer.fit(model, train_loader, valid_loader, config, checkpoint_path=ckpt_path,
                                metadata=metadata)
    # Persist final bookkeeping into the checkpoint too
    best["metadata"]["train_time_total_sec"] = best["train_time_total_sec"]
    best["metadata"]["peak_gpu_mem_gb"] = best["peak_gpu_mem_gb"]
    best["metadata"]["best_epoch"] = best["best_epoch"]
    torch.save(best, ckpt_path)

    out_dir = results_dir_for(config)
    save_json_atomic(history, out_dir / f"{aspect}_history.json")
    _plot_curves(history, out_dir / f"{aspect}_learning_curve.png", f"{exp} / {aspect}")
    summary = {
        "experiment": exp,
        "aspect": aspect,
        "best_epoch": best["best_epoch"],
        "epochs_run": len(history),
        "val_metrics_at_best": best["val_metrics"],
        "train_time_total_sec": best["train_time_total_sec"],
        "peak_gpu_mem_gb": best["peak_gpu_mem_gb"],
        "num_train": len(train_ids),
        "num_valid": len(valid_ids),
        "K": K,
        "checkpoint": str(ckpt_path),
        "config_hash": metadata["config_hash"],
        "vocab_hash": metadata["vocab_hash"],
        "protocol": "internal_validation (vocab-restricted, model selection)",
    }
    save_json_atomic(summary, out_dir / f"{aspect}_train_summary.json")
    save_json_atomic(best["val_curve"], out_dir / f"{aspect}_val_threshold_curve.json")
    logger.info(f"[{exp}/{aspect}] best epoch {best['best_epoch']} Fmax={best['best_fmax']:.4f} -> {ckpt_path}")
    return summary


def run_training_from_config(config_path: str, aspect: str = "F") -> Dict[str, Any]:
    """Backwards-compatible helper invoked by scripts/train.py."""
    return run_training(load_config(config_path), aspect)
