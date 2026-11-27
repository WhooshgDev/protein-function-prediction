"""Shared CLI helpers for pipeline scripts."""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.config import load_config_with_overrides  # noqa: E402


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Override a config value, e.g. --set loss.hierarchical_lambda=0.01 (repeatable).",
    )


def load_cli_config(args):
    return load_config_with_overrides(args.config, args.overrides)
