import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from _common import add_common_args, load_cli_config

from src.data.pipeline import read_jsonl_sequences
from src.models.esm_encoder import ESMEmbedder, assemble_from_cache, extract_to_cache
from src.utils.io import compute_list_hash, load_json, save_json_atomic
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def cache_set_name(config) -> str:
    m = config["model"]
    short = m["esm_name"].split("/")[-1]
    return f"{short}_{m.get('max_length_policy', 'chunk')}{m.get('max_length', 1022)}"


def build_embedder(config, max_tokens=None) -> ESMEmbedder:
    m, e = config["model"], config.get("embedding", {})
    return ESMEmbedder(
        model_name=m["esm_name"],
        dtype=e.get("dtype", "float16"),
        max_length=int(m.get("max_length", 1022)),
        max_length_policy=m.get("max_length_policy", "chunk"),
        chunk_overlap=int(m.get("chunk_overlap", 256)),
        max_tokens_per_batch=int(max_tokens or e.get("max_tokens_per_batch", 8000)),
    )


def run_benchmark(config, ids, seqs, n, logger) -> None:
    """Plan §6: measure throughput + peak VRAM on a representative sample; pick token budget."""
    rng = np.random.default_rng(0)
    pick = rng.choice(len(seqs), size=min(n, len(seqs)), replace=False)
    sample = [seqs[i] for i in pick]
    results = []
    embedder = build_embedder(config)
    embedder.embed(sample[:20], progress=False)  # warm-up
    for budget in (8000, 16000, 24000, 32000, 48000):
        embedder.max_tokens = budget
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            embedder.embed(sample, progress=False)
            torch.cuda.synchronize()
            dt = time.time() - t0
            rec = {"max_tokens_per_batch": budget, "seq_per_sec": len(sample) / dt,
                   "res_per_sec": sum(map(len, sample)) / dt,
                   "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
        except torch.cuda.OutOfMemoryError:
            rec = {"max_tokens_per_batch": budget, "oom": True}
        logger.info(f"benchmark {rec}")
        results.append(rec)
    out = {"sample_size": len(sample), "mean_length": float(np.mean([len(s) for s in sample])),
           "embedder": embedder.info(), "results": results,
           "created_at": datetime.now(timezone.utc).isoformat()}
    save_json_atomic(out, "results/embedding_benchmark.json")


def run_sanity_checks(embedder: ESMEmbedder, seqs, logger) -> dict:
    """Repeatability, padding contamination, and long-sequence handling checks (plan §6)."""
    short = sorted(seqs[:2000], key=len)
    a, b = short[10], short[-10]
    solo = embedder.embed([a], progress=False)
    padded = embedder.embed([a, b], progress=False)[:1]  # 'a' gets padded next to 'b'
    again = embedder.embed([a], progress=False)
    long_seq = max(seqs, key=len)
    long_emb = embedder.embed([long_seq], progress=False)
    checks = {
        "repeat_max_abs_diff": float(np.abs(solo - again).max()),
        "padding_max_abs_diff": float(np.abs(solo - padded).max()),
        "long_sequence_length": len(long_seq),
        "long_sequence_finite": bool(np.isfinite(long_emb).all()),
        "dim": int(solo.shape[1]),
    }
    logger.info(f"sanity checks: {checks}")
    if checks["padding_max_abs_diff"] > 5e-2:
        raise AssertionError(f"Padding contaminates pooled embedding: {checks}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="2. Extract frozen ESM-2 residue-pooled embeddings.")
    add_common_args(parser)
    parser.add_argument("--split", type=str, default="all", choices=["train", "valid", "test", "all"],
                        help="Dataset split to extract embeddings for.")
    parser.add_argument("--benchmark", type=int, default=0, metavar="N",
                        help="Only benchmark token budgets on N random train sequences.")
    args = parser.parse_args()

    config = load_cli_config(args)
    set_seed(int(config.get("seed", 42)))
    logger = get_logger("extract_embeddings", log_file="logs/extract_embeddings.log")
    proc = Path(config["data"]["processed_dir"])
    emb_dir = Path(config["data"]["embedding_dir"])
    emb_dir.mkdir(parents=True, exist_ok=True)

    all_ids, all_seqs = read_jsonl_sequences(proc / "sequences.jsonl")
    if args.benchmark:
        run_benchmark(config, all_ids, all_seqs, args.benchmark, logger)
        return

    seq_of = dict(zip(all_ids, all_seqs))
    sets = {}
    if args.split in ("train", "all"):
        ids = (proc / "train_ids.txt").read_text(encoding="utf-8").split()
        sets["train"] = (ids, [seq_of[i] for i in ids])
    if args.split in ("valid", "all"):
        ids = (proc / "valid_ids.txt").read_text(encoding="utf-8").split()
        sets["valid"] = (ids, [seq_of[i] for i in ids])
    if args.split in ("test", "all") and (proc / "test_sequences.jsonl").exists():
        sets["test"] = read_jsonl_sequences(proc / "test_sequences.jsonl")

    set_name = cache_set_name(config)
    embedder = build_embedder(config)
    logger.info(f"Embedder: {embedder.info()}")
    sanity = run_sanity_checks(embedder, all_seqs, logger)

    manifest_path = emb_dir / "embedding_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {"splits": {}}
    manifest.update({"embedder": embedder.info(), "cache_set": set_name, "sanity_checks": sanity,
                     "updated_at": datetime.now(timezone.utc).isoformat()})
    for split, (ids, seqs) in sets.items():
        stats = extract_to_cache(embedder, ids, seqs, emb_dir, set_name,
                                 shard_size=int(config.get("embedding", {}).get("shard_size", 10000)),
                                 logger=logger)
        emb = assemble_from_cache(emb_dir, set_name, ids, seqs)
        assert emb.shape == (len(ids), embedder.hidden_dim), emb.shape
        assert np.isfinite(emb).all(), "non-finite embeddings"
        np.save(emb_dir / f"embeddings_{split}.npy", emb)
        save_json_atomic(ids, emb_dir / f"embeddings_{split}_ids.json")
        n_long = sum(len(s) > int(config["model"].get("max_length", 1022)) for s in seqs)
        manifest["splits"][split] = {
            "num": len(ids), "shape": list(emb.shape), "ids_hash": compute_list_hash(ids),
            "num_longer_than_max_length": n_long, "extraction": stats,
        }
        save_json_atomic(manifest, manifest_path)
        logger.info(f"[{split}] saved {emb.shape} -> {emb_dir / f'embeddings_{split}.npy'}")


if __name__ == "__main__":
    main()
