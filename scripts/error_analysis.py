"""Error analysis on validation predictions (plan §16 day 6).

* term-centric micro average precision by training-frequency bin of the GO term
* protein-centric Fmax by sequence-length bin and by taxon seen / unseen in the train split
Writes results/<exp>/error_analysis.json.
"""
import argparse
from collections import Counter

import numpy as np
from _common import add_common_args, load_cli_config

from src.data.artifacts import ProcessedData, load_ia
from src.data.ontology import GOOntology
from src.data.pipeline import read_jsonl_sequences, resolve_raw_files
from src.data.taxonomy import UNK_TAXON_INDEX
from src.evaluation.evaluate_cafa import evaluate_prediction_file, load_ground_truth, read_predictions
from src.training.train import experiment_name, results_dir_for
from src.utils.io import save_json_atomic
from src.utils.logger import get_logger

FREQ_BINS = [(1, 1), (2, 10), (11, 100), (101, 1000), (1001, 10**9)]
LEN_BINS = [(0, 200), (201, 500), (501, 1022), (1023, 10**9)]


def micro_ap(pairs_scores: np.ndarray, pairs_labels: np.ndarray, n_pos: int) -> float:
    """AP over predicted (protein, term) pairs; unpredicted positives count as never retrieved."""
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-pairs_scores, kind="stable")
    lab = pairs_labels[order]
    tp = np.cumsum(lab)
    prec = tp / np.arange(1, len(lab) + 1)
    return float((prec * lab).sum() / n_pos)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    config = load_cli_config(args)
    if args.name:
        config["experiment_name"] = args.name
    exp = experiment_name(config)
    logger = get_logger(f"error_analysis.{exp}")
    raw = resolve_raw_files(config)
    pdata = ProcessedData(config["data"]["processed_dir"], config["data"]["embedding_dir"])
    valid_ids = pdata.split_ids("valid")
    train_ids = pdata.split_ids("train")
    gt = load_ground_truth(pdata.proc / "propagated_annotations.tsv", valid_ids)
    preds = read_predictions(args.predictions)
    onto = GOOntology()
    onto.parse_obo(raw["obo"])
    ia = load_ia(str(raw["ia"]))
    out = {"experiment": exp, "predictions": args.predictions, "term_frequency_bins": {}, "length_bins": {},
           "taxon_seen": {}}

    # --- term-centric by frequency ---
    for a in ("F", "P", "C"):
        freq = np.asarray(pdata.labels_for(a, train_ids).sum(axis=0)).ravel()
        vocab = pdata.vocab_list(a)
        term_freq = dict(zip(vocab, freq.astype(int)))
        truth_pairs = {(p, g) for p, gos in gt[a].items() for g in gos}
        gt_prot = set(gt[a])
        sub = preds[preds["pid"].astype(str).isin(gt_prot) & preds["go"].astype(str).isin(term_freq)]
        pid_s, go_s, sc = sub["pid"].astype(str).to_numpy(), sub["go"].astype(str).to_numpy(), sub["score"].to_numpy()
        is_pos = np.fromiter(((p, g) in truth_pairs for p, g in zip(pid_s, go_s)), dtype=bool, count=len(pid_s))
        f_of = np.array([term_freq[g] for g in go_s])
        pos_freq = Counter()
        for p, g in truth_pairs:
            if g in term_freq:
                pos_freq[g] += 1
        res = {}
        for lo, hi in FREQ_BINS:
            terms_in = [g for g, f in term_freq.items() if lo <= f <= hi]
            n_pos = sum(pos_freq[g] for g in terms_in)
            m = (f_of >= lo) & (f_of <= hi)
            res[f"{lo}-{hi if hi < 10**9 else 'inf'}"] = {
                "num_terms": len(terms_in), "valid_positive_pairs": n_pos,
                "micro_ap": micro_ap(sc[m], is_pos[m], n_pos),
            }
        n_oov = sum(1 for p, g in truth_pairs if g not in term_freq)
        res["outside_vocab"] = {"valid_positive_pairs": n_oov, "micro_ap": 0.0}
        out["term_frequency_bins"][a] = res
        logger.info(f"[{a}] AP by train frequency: { {k: round(v['micro_ap'], 3) for k, v in res.items()} }")

    # --- protein-centric by length / taxon ---
    ids, seqs = read_jsonl_sequences(pdata.proc / "sequences.jsonl")
    length = dict(zip(ids, map(len, seqs)))
    tax_idx = dict(zip(valid_ids, pdata.taxonomy_indices("valid", valid_ids)))
    groups = {f"len_{lo}-{hi if hi < 10**9 else 'inf'}": (lambda p, lo=lo, hi=hi: lo <= length[p] <= hi)
              for lo, hi in LEN_BINS}
    groups.update({"taxon_seen": lambda p: tax_idx[p] != UNK_TAXON_INDEX,
                   "taxon_unseen": lambda p: tax_idx[p] == UNK_TAXON_INDEX})
    for gname, fn in groups.items():
        gt_sub = {a: {p: g for p, g in d.items() if fn(p)} for a, d in gt.items()}
        r = evaluate_prediction_file(args.predictions, gt_sub, onto, ia=ia, preds=preds)
        rec = {a: {"fmax": v["fmax"], "wfmax": v["wfmax"], "n": v["num_gt_proteins"]} for a, v in r["aspects"].items()}
        (out["taxon_seen"] if gname.startswith("taxon") else out["length_bins"])[gname] = rec
        logger.info(f"{gname}: { {a: (round(v['fmax'], 3), v['n']) for a, v in rec.items()} }")
    save_json_atomic(out, results_dir_for(config) / "error_analysis.json")


if __name__ == "__main__":
    main()
