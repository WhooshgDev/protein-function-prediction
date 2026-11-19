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
    parser = argparse.ArgumentParser(description="4. Run inference and export predictions.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument(
        "--input",
        type=str,
        default="testsuperset",
        help="Input split or target protein set (e.g. validation, testsuperset).",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional output file path.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    logger = get_logger("predict")
    logger.info(f"Loaded config from {args.config} | checkpoint={args.checkpoint} | input={args.input}")


if __name__ == "__main__":
    main()
