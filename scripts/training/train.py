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
    parser = argparse.ArgumentParser(description="3. Train PFP model for a target GO aspect.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--aspect",
        type=str,
        required=True,
        choices=["F", "P", "C"],
        help="GO aspect to train (F=MF, P=BP, C=CC).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    logger = get_logger("train")
    logger.info(f"Loaded config from {args.config} | aspect={args.aspect}")


if __name__ == "__main__":
    main()
