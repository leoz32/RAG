from __future__ import annotations

import argparse

from src.config.settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG hallucination evaluation platform entrypoint")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--experiment-id", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("show-config")
    args = parser.parse_args()

    settings = load_settings(args.config, experiment_id=args.experiment_id)
    if args.command == "show-config":
        print(settings.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
