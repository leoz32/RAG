from __future__ import annotations

import ast

import pandas as pd

from src.config.schemas import EvaluationRecord
from src.config.settings import AppSettings
from src.evaluation.dataset_source import load_question_dataframe
from src.retrieval.retriever import Retriever
from src.utils.enums import PipelineType


def _deserialize_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [text]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    return [str(value)]


def build_evaluation_dataframe(settings: AppSettings) -> pd.DataFrame:
    from src.utils.io import read_csv

    generation_df = read_csv(settings.generation_results_path)
    if "generation_status" in generation_df.columns:
        generation_df = generation_df[
            generation_df["generation_status"].fillna("success").astype(str).str.lower() != "failed"
        ].copy()
    qa_df = load_question_dataframe(settings)

    required_columns = {"question_id", "question", "ground_truth", "question_type"}
    missing = required_columns - set(qa_df.columns)
    if missing:
        raise ValueError(f"qa_dataset.csv missing required columns: {sorted(missing)}")

    merged = generation_df.merge(
        qa_df[
            [
                column
                for column in [
                    "question_id",
                    "ground_truth",
                    "question_type",
                    "topic",
                    "source_split",
                    "source_model",
                    "reference_output",
                    "factscore_annotations",
                ]
                if column in qa_df.columns
            ]
        ],
        on="question_id",
        how="left",
        suffixes=("", "_qa"),
    )

    rag_context_lookup = {}
    rag_rows = generation_df[generation_df["pipeline_type"] == PipelineType.RAG.value]
    for _, row in rag_rows.iterrows():
        rag_context_lookup[str(row["question_id"])] = _deserialize_list(row.get("contexts"))

    reference_context_lookup: dict[str, list[str]] = {}
    reference_context_source_lookup: dict[str, str] = {}
    strategy = settings.evaluation.reference_context_strategy
    if strategy == "retrieve_once_per_question":
        retriever = Retriever(settings)
        unique_questions = (
            qa_df[["question_id", "question"]]
            .drop_duplicates(subset=["question_id"])
            .itertuples(index=False, name=None)
        )
        for question_id, question in unique_questions:
            retrievals = retriever.retrieve(str(question), str(question_id))
            reference_context_lookup[str(question_id)] = [record.text for record in retrievals]
            reference_context_source_lookup[str(question_id)] = "evaluation_retriever"
    else:
        for question_id, contexts in rag_context_lookup.items():
            reference_context_lookup[question_id] = contexts
            reference_context_source_lookup[question_id] = "rag_generation_contexts"

    evaluation_contexts: list[list[str]] = []
    evaluation_context_sources: list[str] = []
    for _, row in merged.iterrows():
        question_id = str(row["question_id"])
        if row["pipeline_type"] == PipelineType.RAG.value:
            contexts = _deserialize_list(row.get("contexts"))
            if contexts:
                evaluation_contexts.append(contexts)
                evaluation_context_sources.append("rag_generation_contexts")
            else:
                evaluation_contexts.append(reference_context_lookup.get(question_id, []))
                evaluation_context_sources.append(reference_context_source_lookup.get(question_id, "missing"))
        else:
            evaluation_contexts.append(reference_context_lookup.get(question_id, []))
            evaluation_context_sources.append(reference_context_source_lookup.get(question_id, "missing"))

    merged["evaluation_contexts"] = evaluation_contexts
    merged["evaluation_context_source"] = evaluation_context_sources
    if "ground_truth_qa" in merged.columns:
        merged["ground_truth"] = merged["ground_truth"].fillna(merged["ground_truth_qa"])
    if "question_type_qa" in merged.columns:
        merged["question_type"] = merged["question_type"].fillna(merged["question_type_qa"])
    for column in ["topic", "source_split", "source_model", "reference_output", "factscore_annotations"]:
        qa_column = f"{column}_qa"
        if qa_column in merged.columns:
            merged[column] = merged[column].fillna(merged[qa_column])
    if "evaluation_source" not in merged.columns:
        merged["evaluation_source"] = "rag_project_generation"

    desired_order = EvaluationRecord.model_fields.keys()
    for column in desired_order:
        if column not in merged.columns:
            merged[column] = None
    return merged
