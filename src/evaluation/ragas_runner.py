from __future__ import annotations

import ast

import pandas as pd

from src.config.settings import AppSettings
from src.ingestion.embedder import EmbeddingClient


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


def run_ragas(df: pd.DataFrame, settings: AppSettings) -> pd.DataFrame:
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
    eval_df["contexts"] = eval_df["evaluation_contexts"].apply(_ensure_list)
    eval_df = eval_df.drop(
        columns=["faithfulness", "answer_relevance", "context_precision", "context_recall"],
        errors="ignore",
    )
    dataset = Dataset.from_dict(
        {
            "question": eval_df["question"].tolist(),
            "answer": eval_df["answer"].tolist(),
            "contexts": eval_df["contexts"].tolist(),
            "ground_truth": eval_df["ground_truth"].fillna("").tolist(),
        }
    )

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

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevance, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    scores_df = result.to_pandas()
    metric_columns = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
    for column in metric_columns:
        if column not in scores_df.columns:
            scores_df[column] = None

    return pd.concat([eval_df.reset_index(drop=True), scores_df[metric_columns]], axis=1)
