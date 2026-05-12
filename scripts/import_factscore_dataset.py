from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import AppSettings, load_settings
from src.evaluation.dataset_source import load_question_dataframe
from src.utils.io import ensure_dir, write_csv, write_json


def import_factscore_dataset(settings: AppSettings) -> dict:
    if not settings.factscore_source_dir.exists():
        raise FileNotFoundError(f"FActScore source directory not found: {settings.factscore_source_dir}")

    manifest_rows: list[dict] = []
    output_root = settings.imported_factscore_dir
    ensure_dir(output_root)

    requested_split = settings.dataset.factscore_split
    requested_models = set(settings.dataset.factscore_models)
    for split_dir in sorted((settings.factscore_source_dir / "data").glob("*")):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        if requested_split and split != requested_split:
            continue
        for source_file in sorted(split_dir.glob("*.jsonl")):
            if requested_models and source_file.stem not in requested_models:
                continue
            derived_settings = settings.model_copy(
                update={
                    "dataset": settings.dataset.model_copy(
                        update={
                            "kind": "factscore_jsonl",
                            "qa_dataset_path": str(source_file.relative_to(settings.project_root)),
                        }
                    )
                }
            )
            frame = load_question_dataframe(derived_settings)
            destination = output_root / split / f"{source_file.stem}.csv"
            write_csv(destination, frame)
            manifest_rows.append(
                {
                    "split": split,
                    "source_model": source_file.stem,
                    "source_path": str(source_file),
                    "output_csv_path": str(destination),
                    "row_count": int(len(frame)),
                    "has_annotations": bool(frame["factscore_annotations"].notna().any()),
                }
            )

    manifest = {
        "experiment_id": settings.experiment_id,
        "factscore_source_dir": str(settings.factscore_source_dir),
        "imports": manifest_rows,
    }
    write_json(settings.factscore_manifest_path, manifest)
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--experiment-id", default=None)
    args = parser.parse_args()

    settings = load_settings(args.config, experiment_id=args.experiment_id)
    manifest = import_factscore_dataset(settings)
    print(pd.DataFrame(manifest["imports"]).to_string(index=False))


if __name__ == "__main__":
    main()
