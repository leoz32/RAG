from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.io import ensure_dir, read_csv


def _serialize_csv_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def append_records_csv(path: Path, records: Iterable[dict]) -> None:
    rows = list(records)
    if not rows:
        return

    ensure_dir(path.parent)
    serialized_rows = [{key: _serialize_csv_value(value) for key, value in row.items()} for row in rows]
    fieldnames = list(serialized_rows[0].keys())
    file_exists = path.exists() and path.stat().st_size > 0

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(serialized_rows)


def load_completed_keys(path: Path, key_columns: list[str]) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()

    df = read_csv(path)
    missing = [column for column in key_columns if column not in df.columns]
    if missing:
        return set()

    completed: set[tuple[str, ...]] = set()
    for row in df[key_columns].itertuples(index=False, name=None):
        completed.add(tuple("" if pd.isna(value) else str(value) for value in row))
    return completed


def finalize_csv(path: Path, key_columns: list[str], sort_columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = read_csv(path)
    if key_columns:
        df = df.drop_duplicates(subset=key_columns, keep="last")
    if sort_columns:
        existing_sort_columns = [column for column in sort_columns if column in df.columns]
        if existing_sort_columns:
            df = df.sort_values(existing_sort_columns).reset_index(drop=True)
    df.to_csv(path, index=False)
    return df
