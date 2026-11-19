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
    parser = argparse.ArgumentParser(description="5. Evaluate predictions on validation or via CAFA evaluator.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions file.")
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Split to evaluate against (default: validation).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    logger = get_logger("evaluate")
    logger.info(f"Loaded config from {args.config} | predictions={args.predictions} | split={args.split}")


if __name__ == "__main__":
    main()
