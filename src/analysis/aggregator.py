from __future__ import annotations

import pandas as pd

from src.utils.enums import PipelineType


def aggregate_analysis(df: pd.DataFrame) -> dict:
    baseline_df = df[df["pipeline_type"] == PipelineType.BASELINE.value]
    rag_df = df[df["pipeline_type"] == PipelineType.RAG.value]

    baseline_faithfulness = baseline_df["faithfulness"].mean()
    rag_faithfulness = rag_df["faithfulness"].mean()
    answer_relevance_delta = rag_df["answer_relevance"].mean() - baseline_df["answer_relevance"].mean()

    question_type_breakdown = (
        df.groupby(["question_type", "pipeline_type"])[["faithfulness", "answer_relevance"]]
        .mean(numeric_only=True)
        .reset_index()
        .to_dict(orient="records")
    )

    adversarial_baseline = baseline_df[baseline_df["question_type"] == "adversarial"]
    adversarial_rag = rag_df[rag_df["question_type"] == "adversarial"]

    def refusal_rate(frame: pd.DataFrame) -> float | None:
        if frame.empty:
            return None
        return frame["answer"].fillna("").str.contains("未知", regex=False).mean()

    reasoning_baseline = baseline_df[baseline_df["question_type"] == "reasoning"]["faithfulness"].mean()
    reasoning_rag = rag_df[rag_df["question_type"] == "reasoning"]["faithfulness"].mean()

    return {
        "baseline_mean_faithfulness": baseline_faithfulness,
        "rag_mean_faithfulness": rag_faithfulness,
        "faithfulness_gain": rag_faithfulness - baseline_faithfulness,
        "answer_relevance_delta": answer_relevance_delta,
        "adversarial_refusal_rate_baseline": refusal_rate(adversarial_baseline),
        "adversarial_refusal_rate_rag": refusal_rate(adversarial_rag),
        "reasoning_faithfulness_delta": reasoning_rag - reasoning_baseline,
        "question_type_breakdown": question_type_breakdown,
    }
