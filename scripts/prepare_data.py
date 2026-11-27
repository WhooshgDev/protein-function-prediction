import argparse

from _common import add_common_args, load_cli_config

from src.data.pipeline import prepare_data
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="1. Prepare CAFA 5 dataset, GO ontology, labels, and split.")
    add_common_args(parser)
    args = parser.parse_args()

    config = load_cli_config(args)
    set_seed(int(config.get("seed", 42)))
    logger = get_logger("prepare_data", log_file="logs/prepare_data.log")
    logger.info(f"Loaded config from {args.config}")
    result = prepare_data(config, logger=logger)
    m = result["manifest"]
    logger.info(f"proteins={m['num_proteins']} test={m['num_test_proteins']} K={m['vocab_sizes']} split={m['split_method']}")


if __name__ == "__main__":
    main()
