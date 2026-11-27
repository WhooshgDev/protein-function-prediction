"""6. End-to-end pipeline: prepare -> embed -> train F/P/C -> predict valid -> evaluate -> predict test.

Each step is skipped when its outputs already exist (use --force to redo). Steps run as
subprocesses of the per-step scripts so this file is exactly the documented CLI workflow.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from _common import ROOT_DIR, add_common_args, load_cli_config

from src.training.train import checkpoint_path_for, experiment_name
from src.utils.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="6. Run full PFP-CAFA5 end-to-end pipeline.")
    add_common_args(parser)
    parser.add_argument("--force", action="store_true", help="Re-run steps even if outputs exist.")
    parser.add_argument("--skip-test", action="store_true", help="Do not predict the test superset.")
    args = parser.parse_args()

    config = load_cli_config(args)
    exp = experiment_name(config)
    logger = get_logger("run_pipeline", log_file=f"logs/run_pipeline_{exp}.log")
    proc = Path(config["data"]["processed_dir"])
    emb = Path(config["data"]["embedding_dir"])
    common = ["--config", args.config] + [x for o in args.overrides for x in ("--set", o)]

    def step(name, script, extra, outputs):
        if not args.force and outputs and all(Path(o).exists() for o in outputs):
            logger.info(f"[skip] {name}: outputs exist")
            return
        cmd = [sys.executable, str(ROOT_DIR / "scripts" / script)] + common + extra
        logger.info(f"[run] {name}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=ROOT_DIR)

    step("prepare_data", "prepare_data.py", [], [proc / "data_manifest.json"])
    step("extract_embeddings", "extract_embeddings.py", ["--split", "all"],
         [emb / f"embeddings_{s}.npy" for s in ("train", "valid", "test")])
    for a in config["evaluation"].get("aspects", ["F", "P", "C"]):
        step(f"train_{a}", "train.py", ["--aspect", a], [checkpoint_path_for(config, a)])
    val_pred = Path("predictions") / f"{exp}_val.tsv"
    step("predict_valid", "predict.py", ["--input", "validation"], [val_pred])
    step("evaluate_valid", "evaluate.py", ["--predictions", str(val_pred)],
         [Path("results") / exp / f"eval_{val_pred.stem}.json"])
    if not args.skip_test:
        step("predict_test", "predict.py", ["--input", "testsuperset"], [Path("predictions") / f"{exp}_test.tsv"])
    logger.info(f"Pipeline finished for experiment '{exp}'")


if __name__ == "__main__":
    main()
