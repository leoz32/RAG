from __future__ import annotations

import pandas as pd

from src.config.settings import AppSettings
from src.utils.io import write_csv, write_json


METRIC_COLUMNS = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]


def build_metric_summaries(df: pd.DataFrame, settings: AppSettings) -> dict:
    overall = df[METRIC_COLUMNS].mean(numeric_only=True).to_dict()
    pipeline_summary = df.groupby("pipeline_type")[METRIC_COLUMNS].mean(numeric_only=True).reset_index()
    question_type_summary = (
        df.groupby(["pipeline_type", "question_type"])[METRIC_COLUMNS]
        .mean(numeric_only=True)
        .reset_index()
    )

    write_csv(settings.metrics_summary_csv_path, pd.DataFrame([overall]))
    write_csv(settings.pipeline_summary_csv_path, pipeline_summary)
    write_csv(settings.question_type_summary_csv_path, question_type_summary)

    payload = {
        "experiment_id": settings.experiment_id,
        "overall": overall,
        "pipeline_summary": pipeline_summary.to_dict(orient="records"),
        "question_type_summary": question_type_summary.to_dict(orient="records"),
    }
    write_json(settings.metrics_summary_json_path, payload)
    return payload
