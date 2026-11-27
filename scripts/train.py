import argparse

from _common import add_common_args, load_cli_config

from src.training.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="3. Train PFP model for a target GO aspect.")
    add_common_args(parser)
    parser.add_argument("--aspect", type=str, required=True, choices=["F", "P", "C", "all"],
                        help="GO aspect to train (F=MF, P=BP, C=CC, all=each in turn).")
    parser.add_argument("--name", type=str, default=None, help="Override experiment_name (e.g. for sweeps).")
    args = parser.parse_args()

    config = load_cli_config(args)
    if args.name:
        config["experiment_name"] = args.name
    aspects = ["F", "P", "C"] if args.aspect == "all" else [args.aspect]
    for a in aspects:
        run_training(config, a)


if __name__ == "__main__":
    main()
