from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import prepare, train_and_evaluate
from .plots import plot_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily electricity-demand forecasting")
    parser.add_argument("command", choices=["prepare", "train", "run"])
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command in {"prepare", "run"}:
        data = prepare(config)
        print(f"Prepared {len(data):,} daily records")
    if args.command in {"train", "run"}:
        results = train_and_evaluate(config)
        plot_comparison(results, config.artifacts_dir / "model_comparison_test.png")
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
