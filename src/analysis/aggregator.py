from __future__ import annotations

import pandas as pd

from src.utils.enums import PipelineType


REFUSAL_PATTERNS = (
    r"未知",
    r"无法(?:直接)?确认",
    r"不能确认",
    r"无法判断",
    r"无法从资料",
    r"资料(?:中)?未提及",
    r"没有足够(?:依据|信息)",
    r"信息不足",
)


def _mean_or_none(series: pd.Series) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _mean_for_column(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    return _mean_or_none(frame[column])


def _refusal_rate(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    answers = frame["answer"].fillna("").astype(str)
    pattern = "|".join(f"(?:{item})" for item in REFUSAL_PATTERNS)
    return float(answers.str.contains(pattern, regex=True).mean())


def _pairwise_faithfulness_gain(baseline_df: pd.DataFrame, rag_df: pd.DataFrame) -> float | None:
    if "faithfulness" not in baseline_df.columns or "faithfulness" not in rag_df.columns:
        return None
    baseline_scores = baseline_df.set_index("question_id")["faithfulness"]
    rag_scores = rag_df.set_index("question_id")["faithfulness"]
    common_question_ids = baseline_scores.index.intersection(rag_scores.index)
    if len(common_question_ids) == 0:
        return None
    deltas = rag_scores.loc[common_question_ids] - baseline_scores.loc[common_question_ids]
    return _mean_or_none(deltas)


def _pairwise_metric_gain(
    baseline_df: pd.DataFrame,
    rag_df: pd.DataFrame,
    metric: str,
) -> float | None:
    if metric not in baseline_df.columns or metric not in rag_df.columns:
        return None
    baseline_scores = baseline_df.set_index("question_id")[metric]
    rag_scores = rag_df.set_index("question_id")[metric]
    common_question_ids = baseline_scores.index.intersection(rag_scores.index)
    if len(common_question_ids) == 0:
        return None
    deltas = rag_scores.loc[common_question_ids] - baseline_scores.loc[common_question_ids]
    return _mean_or_none(deltas)


def aggregate_analysis(df: pd.DataFrame) -> dict:
    baseline_df = df[df["pipeline_type"] == PipelineType.BASELINE.value]
    rag_df = df[df["pipeline_type"] == PipelineType.RAG.value]

    baseline_faithfulness = _mean_for_column(baseline_df, "faithfulness")
    rag_faithfulness = _mean_for_column(rag_df, "faithfulness")
    baseline_answer_correctness = _mean_for_column(baseline_df, "answer_correctness")
    rag_answer_correctness = _mean_for_column(rag_df, "answer_correctness")
    baseline_answer_relevance = _mean_for_column(baseline_df, "answer_relevance")
    rag_answer_relevance = _mean_for_column(rag_df, "answer_relevance")
    answer_relevance_delta = (
        None
        if baseline_answer_relevance is None or rag_answer_relevance is None
        else rag_answer_relevance - baseline_answer_relevance
    )

    breakdown_columns = [
        column
        for column in [
            "faithfulness",
            "answer_correctness",
            "factscore_init_score",
            "unsupported_claim_rate",
            "answer_relevance",
        ]
        if column in df.columns
    ]
    question_type_breakdown = (
        df.groupby(["question_type", "pipeline_type"])[breakdown_columns]
        .mean(numeric_only=True)
        .reset_index()
        .to_dict(orient="records")
    )

    adversarial_baseline = baseline_df[baseline_df["question_type"] == "adversarial"]
    adversarial_rag = rag_df[rag_df["question_type"] == "adversarial"]
    reasoning_baseline = (
        baseline_df[baseline_df["question_type"] == "reasoning"]["faithfulness"]
        if "faithfulness" in baseline_df.columns
        else pd.Series(dtype=float)
    )
    reasoning_rag = (
        rag_df[rag_df["question_type"] == "reasoning"]["faithfulness"]
        if "faithfulness" in rag_df.columns
        else pd.Series(dtype=float)
    )

    rag_empty_retrieval_rate = None
    if "retrieved_context_count" in rag_df.columns:
        rag_empty_retrieval_rate = float((rag_df["retrieved_context_count"].fillna(0) <= 0).mean())

    reasoning_baseline_mean = _mean_or_none(reasoning_baseline)
    reasoning_rag_mean = _mean_or_none(reasoning_rag)

    return {
        "baseline_mean_faithfulness": baseline_faithfulness,
        "rag_mean_faithfulness": rag_faithfulness,
        "faithfulness_gain": None
        if baseline_faithfulness is None or rag_faithfulness is None
        else rag_faithfulness - baseline_faithfulness,
        "pairwise_faithfulness_gain": _pairwise_faithfulness_gain(baseline_df, rag_df),
        "answer_correctness_delta": None
        if baseline_answer_correctness is None or rag_answer_correctness is None
        else rag_answer_correctness - baseline_answer_correctness,
        "factscore_init_score_delta": _pairwise_metric_gain(baseline_df, rag_df, "factscore_init_score"),
        "unsupported_claim_rate_delta": _pairwise_metric_gain(baseline_df, rag_df, "unsupported_claim_rate"),
        "answer_relevance_delta": answer_relevance_delta,
        "factscore_score_delta": _pairwise_metric_gain(baseline_df, rag_df, "factscore_score"),
        "adversarial_refusal_rate_baseline": _refusal_rate(adversarial_baseline),
        "adversarial_refusal_rate_rag": _refusal_rate(adversarial_rag),
        "reasoning_faithfulness_delta": None
        if reasoning_baseline_mean is None or reasoning_rag_mean is None
        else reasoning_rag_mean - reasoning_baseline_mean,
        "rag_empty_retrieval_rate": rag_empty_retrieval_rate,
        "question_type_breakdown": question_type_breakdown,
    }
