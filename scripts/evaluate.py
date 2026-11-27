import argparse
import json
from pathlib import Path

import numpy as np
from _common import add_common_args, load_cli_config

from src.data.artifacts import ProcessedData, load_ia
from src.data.ontology import GOOntology
from src.data.pipeline import resolve_raw_files
from src.evaluation.evaluate_cafa import (
    evaluate_prediction_file, load_ground_truth, read_predictions, run_official_cafaeval,
)
from src.evaluation.metrics import threshold_grid
from src.training.train import experiment_name, results_dir_for
from src.utils.io import compute_file_hash, save_json_atomic
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="5. Evaluate predictions on validation or via CAFA evaluator.")
    add_common_args(parser)
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions file.")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "valid"],
                        help="Split to evaluate against (only the held-out validation split has labels).")
    parser.add_argument("--cafaeval-proteins", type=int, default=3000,
                        help="Run official cafaeval on N random validation proteins (0 = skip, -1 = all).")
    parser.add_argument("--name", type=str, default=None, help="Override experiment_name.")
    args = parser.parse_args()

    config = load_cli_config(args)
    if args.name:
        config["experiment_name"] = args.name
    seed = int(config.get("seed", 42))
    set_seed(seed)
    exp = experiment_name(config)
    logger = get_logger(f"evaluate.{exp}", log_file=f"logs/evaluate_{exp}.log")
    raw = resolve_raw_files(config)
    pdata = ProcessedData(config["data"]["processed_dir"], config["data"]["embedding_dir"])
    valid_ids = pdata.split_ids("valid")
    gt = load_ground_truth(pdata.proc / "propagated_annotations.tsv", valid_ids)
    onto = GOOntology(allowed_relations=tuple(config["data"].get("ontology", {}).get("relations", ["is_a", "part_of"])))
    onto.parse_obo(raw["obo"])
    ia = load_ia(str(raw["ia"]))
    preds = read_predictions(args.predictions)
    logger.info(f"Loaded {len(preds)} prediction rows for {preds['pid'].nunique()} proteins")

    res = evaluate_prediction_file(args.predictions, gt, onto, ia=ia,
                                   thresholds=threshold_grid(config.get("evaluation", {})), preds=preds)
    res["predictions_path"] = args.predictions
    res["predictions_hash"] = compute_file_hash(args.predictions)
    res["experiment"] = exp
    for a, m in res["aspects"].items():
        logger.info(f"[internal_validation] {a}: Fmax={m['fmax']:.4f}@{m['threshold']:.2f} "
                    f"P={m['precision']:.4f} R={m['recall']:.4f} | wFmax={m['wfmax']:.4f}@{m['wfmax_threshold']:.2f}"
                    f" | n={m['num_gt_proteins']}")

    if args.cafaeval_proteins:
        rng = np.random.default_rng(seed)
        subset = None
        if args.cafaeval_proteins > 0:
            annotated = sorted(set().union(*[set(gt[a]) for a in gt]))
            subset = sorted(rng.choice(annotated, size=min(args.cafaeval_proteins, len(annotated)), replace=False).tolist())
        logger.info(f"Running official cafaeval on {len(subset) if subset else 'all'} proteins ...")
        off = run_official_cafaeval(args.predictions, gt, raw["obo"], raw["ia"], protein_subset=subset, preds=preds)
        if subset is not None:
            gt_sub = {a: {p: g for p, g in d.items() if p in set(subset)} for a, d in gt.items()}
            internal_sub = evaluate_prediction_file(args.predictions, gt_sub, onto, ia=ia, preds=preds)
            off["internal_on_same_subset"] = {a: {"fmax": v["fmax"], "wfmax": v["wfmax"]}
                                              for a, v in internal_sub["aspects"].items()}
        for a, m in off["aspects"].items():
            logger.info(f"[cafaeval] {a}: {m} | internal same subset: {off.get('internal_on_same_subset', {}).get(a)}")
        res["cafaeval"] = off

    stem = Path(args.predictions).stem
    out = results_dir_for(config) / f"eval_{stem}.json"
    save_json_atomic(res, out)
    logger.info(f"mean Fmax={res.get('mean_fmax', 0):.4f} mean wFmax={res.get('mean_wfmax', 0):.4f} -> {out}")


if __name__ == "__main__":
    main()
