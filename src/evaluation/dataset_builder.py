from __future__ import annotations

import pandas as pd

from src.config.schemas import EvaluationRecord
from src.config.settings import AppSettings
from src.utils.enums import PipelineType
from src.utils.io import read_csv


def build_evaluation_dataframe(settings: AppSettings) -> pd.DataFrame:
    generation_df = read_csv(settings.generation_results_path)
    qa_df = read_csv(settings.qa_dataset_path)

    required_columns = {"question_id", "question", "ground_truth", "question_type"}
    missing = required_columns - set(qa_df.columns)
    if missing:
        raise ValueError(f"qa_dataset.csv missing required columns: {sorted(missing)}")

    merged = generation_df.merge(
        qa_df[["question_id", "ground_truth", "question_type"]],
        on="question_id",
        how="left",
        suffixes=("", "_qa"),
    )

    rag_context_lookup = (
        generation_df[generation_df["pipeline_type"] == PipelineType.RAG.value]
        .set_index("question_id")["contexts"]
        .to_dict()
    )

    evaluation_contexts: list[str] = []
    for _, row in merged.iterrows():
        if row["pipeline_type"] == PipelineType.RAG.value:
            evaluation_contexts.append(row["contexts"])
        else:
            # This is evaluation-only reference context and is never used during baseline generation.
            evaluation_contexts.append(rag_context_lookup.get(row["question_id"], "[]"))

    merged["evaluation_contexts"] = evaluation_contexts
    if "ground_truth_qa" in merged.columns:
        merged["ground_truth"] = merged["ground_truth"].fillna(merged["ground_truth_qa"])
    if "question_type_qa" in merged.columns:
        merged["question_type"] = merged["question_type"].fillna(merged["question_type_qa"])

    desired_order = EvaluationRecord.model_fields.keys()
    for column in desired_order:
        if column not in merged.columns:
            merged[column] = None
    return merged
