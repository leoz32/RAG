from __future__ import annotations

import ast

import pandas as pd

from src.config.settings import AppSettings
from src.ingestion.embedder import EmbeddingClient
from src.utils.checkpoint import append_records_csv, finalize_csv, load_completed_keys


def _ensure_list(value) -> list[str]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (ValueError, SyntaxError):
            return [text]
    return [str(value)]


def _build_ragas_dataset(dataset_cls, row: pd.Series):
    return dataset_cls.from_dict(
        {
            "question": [row["question"]],
            "answer": [row["answer"]],
            "contexts": [_ensure_list(row["evaluation_contexts"])],
            "ground_truth": ["" if pd.isna(row["ground_truth"]) else str(row["ground_truth"])],
        }
    )


def run_ragas(df: pd.DataFrame, settings: AppSettings, logger=None) -> pd.DataFrame:
    try:
        from datasets import Dataset
        from openai import OpenAI
        from ragas import evaluate
        from ragas.embeddings.base import LangchainEmbeddingsWrapper
        from ragas.llms import llm_factory
        from ragas.metrics import context_precision, context_recall, faithfulness
        try:
            from ragas.metrics import answer_relevance
        except ImportError:
            from ragas.metrics import answer_relevancy as answer_relevance
    except ImportError as exc:
        raise ImportError("Ragas and datasets must be installed to run evaluation.") from exc

    eval_df = df.copy()
    llm_client = OpenAI(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
    )
    ragas_llm = llm_factory(
        settings.llm.model_name,
        client=llm_client,
        temperature=settings.llm.temperature,
        max_tokens=settings.evaluation.llm_max_tokens,
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(EmbeddingClient(settings).client)

    metric_columns = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
    completed_keys = load_completed_keys(settings.question_level_results_path, ["question_id", "pipeline_type"])
    if logger is not None and completed_keys:
        logger.info("Resuming evaluation with %s completed rows already on disk", len(completed_keys))

    pending_df = eval_df[
        ~eval_df.apply(
            lambda row: (str(row["question_id"]), str(row["pipeline_type"])) in completed_keys,
            axis=1,
        )
    ].reset_index(drop=True)
    if logger is not None:
        logger.info("Pending ragas evaluations: %s", len(pending_df))

    for _, row in pending_df.iterrows():
        dataset = _build_ragas_dataset(Dataset, row)
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevance, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        scores_df = result.to_pandas()
        metric_payload = {column: None for column in metric_columns}
        if not scores_df.empty:
            for column in metric_columns:
                if column in scores_df.columns:
                    metric_payload[column] = scores_df.iloc[0][column]

        record = row.to_dict()
        record["contexts"] = _ensure_list(row["evaluation_contexts"])
        record.update(metric_payload)
        append_records_csv(settings.question_level_results_path, [record])
        completed_keys.add((str(row["question_id"]), str(row["pipeline_type"])))

    return finalize_csv(
        settings.question_level_results_path,
        key_columns=["question_id", "pipeline_type"],
        sort_columns=["question_id", "pipeline_type"],
    )
