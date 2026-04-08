from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import load_settings


def _run(script_name: str, config_path: str, experiment_id: str) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / script_name),
        "--config",
        config_path,
        "--experiment-id",
        experiment_id,
    ]
    subprocess.run(command, check=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    settings = load_settings(args.config)
    experiment_id = settings.experiment_id

    _run("build_index.py", args.config, experiment_id)
    _run("run_generation.py", args.config, experiment_id)
    _run("run_evaluation.py", args.config, experiment_id)


if __name__ == "__main__":
    main()
