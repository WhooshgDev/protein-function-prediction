import argparse
from pathlib import Path

from _common import add_common_args, load_cli_config

from src.data.artifacts import ProcessedData
from src.evaluation.export_cafa import validate_submission
from src.evaluation.predict import load_model_from_checkpoint, predict_and_export
from src.training.train import checkpoint_path_for, experiment_name, results_dir_for
from src.utils.io import save_json_atomic
from src.utils.logger import get_logger
from src.utils.seed import set_seed

INPUT_TO_SPLIT = {"validation": "valid", "valid": "valid", "testsuperset": "test", "test": "test"}


def main() -> None:
    parser = argparse.ArgumentParser(description="4. Run inference and export predictions.")
    add_common_args(parser)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single-aspect checkpoint (.pt). Default: all of checkpoints/<exp>_{F,P,C}_best.pt.")
    parser.add_argument("--input", type=str, default="testsuperset", choices=sorted(INPUT_TO_SPLIT),
                        help="Target protein set.")
    parser.add_argument("--output", type=str, default=None, help="Optional output file path.")
    parser.add_argument("--propagate", action="store_true",
                        help="Max-propagate scores up the GO DAG before export (hierarchy-consistent output).")
    parser.add_argument("--name", type=str, default=None, help="Override experiment_name.")
    args = parser.parse_args()

    config = load_cli_config(args)
    if args.name:
        config["experiment_name"] = args.name
    set_seed(int(config.get("seed", 42)))
    exp = experiment_name(config)
    logger = get_logger(f"predict.{exp}", log_file=f"logs/predict_{exp}.log")
    split = INPUT_TO_SPLIT[args.input]

    if args.checkpoint:
        _, ck = load_model_from_checkpoint(args.checkpoint, device="cpu")
        checkpoints = {ck["metadata"]["aspect"]: Path(args.checkpoint)}
    else:
        checkpoints = {a: checkpoint_path_for(config, a) for a in config["evaluation"].get("aspects", ["F", "P", "C"])}
    missing = [str(p) for p in checkpoints.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    pdata = ProcessedData(config["data"]["processed_dir"], config["data"]["embedding_dir"])
    x, ids = pdata.embeddings(split)
    tax = pdata.taxonomy_indices(split, ids) if config.get("taxonomy", {}).get("enabled") else None
    suffix = "val" if split == "valid" else "test"
    out = Path(args.output) if args.output else Path("predictions") / f"{exp}_{suffix}{'_prop' if args.propagate else ''}.tsv"
    edges = {a: pdata.edges(a) for a in checkpoints} if args.propagate else None
    logger.info(f"Predicting {len(ids)} {split} proteins with {list(checkpoints)} -> {out}")
    summary = predict_and_export(
        checkpoints, x, ids, out, taxonomy_indices=tax, export_cfg=config.get("export", {}),
        propagate=args.propagate, edges=edges, expected_vocab_hashes=pdata.manifest["vocab_hashes"],
        logger=logger,
    )
    logger.info(f"Wrote {summary['num_rows_written']} rows in {summary['inference_time_sec']:.1f}s; validating...")
    valid_go = set().union(*[set(pdata.vocab(a)) for a in checkpoints])
    check = validate_submission(out, expected_ids=ids, valid_go_ids=valid_go,
                                max_terms_per_protein=int(config.get("export", {}).get("max_terms_per_protein", 1500)))
    summary["validation"] = check
    logger.info(f"Schema validation: valid={check['valid']} rows={check['num_rows']} proteins={check['num_proteins']}"
                f" missing={check['missing_targets']} max_terms={check['max_terms_per_protein_seen']}")
    save_json_atomic(summary, results_dir_for(config) / f"predict_{suffix}{'_prop' if args.propagate else ''}.json")
    if not check["valid"]:
        raise SystemExit(f"Submission failed validation: {check['errors'][:5]}")


if __name__ == "__main__":
    main()
