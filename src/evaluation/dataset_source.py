from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.config.settings import AppSettings
from src.utils.io import read_csv, read_jsonl


FACTSCORE_REQUIRED_COLUMNS = {"input", "output", "topic"}


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._-") or "unknown"


def _normalize_factscore_question(text: str) -> str:
    normalized = str(text).strip()
    normalized = re.sub(r"^\s*Question:\s*", "", normalized, flags=re.IGNORECASE)
    return normalized.strip()


def _load_factscore_jsonl(path: Path) -> pd.DataFrame:
    rows = read_jsonl(path)
    if not rows:
        return pd.DataFrame()
    missing = FACTSCORE_REQUIRED_COLUMNS - set(rows[0].keys())
    if missing:
        raise ValueError(f"FActScore file missing required keys {sorted(missing)}: {path}")

    split = path.parent.name
    source_model = path.stem
    records: list[dict] = []
    for index, row in enumerate(rows, start=1):
        annotations = row.get("annotations")
        records.append(
            {
                "question_id": f"factscore_{split}_{_slugify(source_model)}_{index:04d}",
                "question": _normalize_factscore_question(row["input"]),
                "ground_truth": str(row["output"]).strip(),
                "question_type": "factscore_bio",
                "topic": str(row["topic"]).strip(),
                "source_split": split,
                "source_model": source_model,
                "reference_output": str(row["output"]).strip(),
                "factscore_annotations": None if annotations is None else json.dumps(annotations, ensure_ascii=False),
            }
        )
    return pd.DataFrame(records)


def load_question_dataframe(settings: AppSettings, limit: int | None = None) -> pd.DataFrame:
    if settings.dataset.kind == "qa_csv":
        df = read_csv(settings.qa_dataset_path)
    elif settings.dataset.kind == "factscore_jsonl":
        df = _load_factscore_jsonl(settings.qa_dataset_path)
    else:
        raise ValueError(f"Unsupported dataset kind: {settings.dataset.kind}")

    effective_limit = limit if limit is not None else settings.dataset.sample_limit
    if effective_limit is not None:
        df = df.head(effective_limit).reset_index(drop=True)
    return df
