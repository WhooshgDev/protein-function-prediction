import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="2. Extract frozen ESM-2 residue-pooled embeddings.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "valid", "test", "all"],
        help="Dataset split to extract embeddings for.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    logger = get_logger("extract_embeddings")
    logger.info(f"Loaded config from {args.config} | split={args.split}")


if __name__ == "__main__":
    main()
