from __future__ import annotations

import pandas as pd

from src.config.settings import AppSettings
from src.utils.io import write_csv, write_json


PRIMARY_METRIC_COLUMNS = [
    "faithfulness",
    "factscore_init_score",
    "answer_correctness",
    "unsupported_claim_rate",
]
SUPPORTING_METRIC_COLUMNS = [
    "factscore_score",
    "factscore_num_facts_per_response",
    "factscore_respond_ratio",
]
DIAGNOSTIC_METRIC_COLUMNS = [
    "context_precision",
    "context_recall",
    "answer_relevance",
]
METRIC_COLUMNS = [*PRIMARY_METRIC_COLUMNS, *DIAGNOSTIC_METRIC_COLUMNS]
FACTSCORE_METRIC_COLUMNS = [*PRIMARY_METRIC_COLUMNS[1:], *SUPPORTING_METRIC_COLUMNS]
ALL_SUMMARY_METRIC_COLUMNS = [*PRIMARY_METRIC_COLUMNS, *SUPPORTING_METRIC_COLUMNS, *DIAGNOSTIC_METRIC_COLUMNS]
PAIRWISE_TOLERANCE = 1e-9


def _build_pairwise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "question_id",
        "question",
        "ground_truth",
        "question_type",
        "pipeline_type",
        "answer",
        "evaluation_context_source",
        *ALL_SUMMARY_METRIC_COLUMNS,
    ]
    available_columns = [column for column in base_columns if column in df.columns]
    paired = df[available_columns].copy()
    paired = paired.drop_duplicates(subset=["question_id", "pipeline_type"], keep="last")

    baseline_df = paired[paired["pipeline_type"] == "baseline"].set_index("question_id")
    rag_df = paired[paired["pipeline_type"] == "rag"].set_index("question_id")
    common_question_ids = baseline_df.index.intersection(rag_df.index)
    if len(common_question_ids) == 0:
        return pd.DataFrame()

    baseline_df = baseline_df.loc[common_question_ids]
    rag_df = rag_df.loc[common_question_ids]

    rows: list[dict] = []
    for question_id in common_question_ids:
        baseline_row = baseline_df.loc[question_id]
        rag_row = rag_df.loc[question_id]
        row = {
            "question_id": question_id,
            "question": rag_row.get("question") or baseline_row.get("question"),
            "ground_truth": rag_row.get("ground_truth") or baseline_row.get("ground_truth"),
            "question_type": rag_row.get("question_type") or baseline_row.get("question_type"),
            "baseline_answer": baseline_row.get("answer"),
            "rag_answer": rag_row.get("answer"),
            "baseline_evaluation_context_source": baseline_row.get("evaluation_context_source"),
            "rag_evaluation_context_source": rag_row.get("evaluation_context_source"),
        }
        for metric in ALL_SUMMARY_METRIC_COLUMNS:
            baseline_metric = baseline_row.get(metric)
            rag_metric = rag_row.get(metric)
            row[f"baseline_{metric}"] = baseline_metric
            row[f"rag_{metric}"] = rag_metric
            if pd.notna(baseline_metric) and pd.notna(rag_metric):
                row[f"{metric}_delta"] = float(rag_metric) - float(baseline_metric)
            else:
                row[f"{metric}_delta"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def _win_rate(delta_series: pd.Series) -> dict[str, float]:
    clean = delta_series.dropna()
    if clean.empty:
        return {"win_rate": 0.0, "tie_rate": 0.0, "loss_rate": 0.0}
    wins = (clean > PAIRWISE_TOLERANCE).mean()
    ties = (clean.abs() <= PAIRWISE_TOLERANCE).mean()
    losses = (clean < -PAIRWISE_TOLERANCE).mean()
    return {
        "win_rate": float(wins),
        "tie_rate": float(ties),
        "loss_rate": float(losses),
    }


def _build_pairwise_summary(pairwise_df: pd.DataFrame) -> dict:
    if pairwise_df.empty:
        return {"paired_questions": 0}

    summary = {"paired_questions": int(len(pairwise_df))}
    for metric in ALL_SUMMARY_METRIC_COLUMNS:
        delta_column = f"{metric}_delta"
        deltas = pairwise_df[delta_column]
        summary[f"{metric}_mean_delta"] = float(deltas.mean()) if deltas.notna().any() else None
        summary[f"{metric}_median_delta"] = float(deltas.median()) if deltas.notna().any() else None
        summary[f"{metric}_rag_win_breakdown"] = _win_rate(deltas)
    return summary


def _build_manual_review_queue(pairwise_df: pd.DataFrame, settings: AppSettings) -> pd.DataFrame:
    if pairwise_df.empty:
        return pd.DataFrame()

    review_df = pairwise_df.copy()
    review_df["min_faithfulness"] = review_df[
        ["baseline_faithfulness", "rag_faithfulness"]
    ].min(axis=1, skipna=True)
    review_df["faithfulness_gap_abs"] = pd.to_numeric(review_df["faithfulness_delta"], errors="coerce").abs()
    review_df["answer_relevance_gap_abs"] = pd.to_numeric(
        review_df["answer_relevance_delta"],
        errors="coerce",
    ).abs()
    if "unsupported_claim_rate_delta" in review_df.columns:
        review_df["unsupported_claim_rate_gap_abs"] = pd.to_numeric(
            review_df["unsupported_claim_rate_delta"],
            errors="coerce",
        ).abs()
    else:
        review_df["unsupported_claim_rate_gap_abs"] = None
    if "factscore_score_delta" in review_df.columns:
        review_df["factscore_score_gap_abs"] = pd.to_numeric(
            review_df["factscore_score_delta"],
            errors="coerce",
        ).abs()
    else:
        review_df["factscore_score_gap_abs"] = None
    review_df = review_df.sort_values(
        by=[
            "min_faithfulness",
            "faithfulness_gap_abs",
            "unsupported_claim_rate_gap_abs",
            "factscore_score_gap_abs",
            "answer_relevance_gap_abs",
        ],
        ascending=[True, False, False, False, False],
        na_position="last",
    )
    return review_df.head(settings.evaluation.manual_review_top_n).reset_index(drop=True)


def build_metric_summaries(df: pd.DataFrame, settings: AppSettings) -> dict:
    summary_metric_columns = [
        column
        for column in ALL_SUMMARY_METRIC_COLUMNS
        if column in df.columns
    ]
    overall = df[summary_metric_columns].mean(numeric_only=True).to_dict()
    pipeline_summary = df.groupby("pipeline_type")[summary_metric_columns].mean(numeric_only=True).reset_index()
    question_type_summary = (
        df.groupby(["pipeline_type", "question_type"])[summary_metric_columns]
        .mean(numeric_only=True)
        .reset_index()
    )
    pairwise_df = _build_pairwise_dataframe(df)
    pairwise_summary = _build_pairwise_summary(pairwise_df)
    manual_review_queue = _build_manual_review_queue(pairwise_df, settings)

    write_csv(settings.metrics_summary_csv_path, pd.DataFrame([overall]))
    write_csv(settings.pipeline_summary_csv_path, pipeline_summary)
    write_csv(settings.question_type_summary_csv_path, question_type_summary)
    write_csv(settings.pairwise_summary_csv_path, pairwise_df)
    write_csv(settings.manual_review_queue_path, manual_review_queue)

    payload = {
        "experiment_id": settings.experiment_id,
        "overall": overall,
        "pipeline_summary": pipeline_summary.to_dict(orient="records"),
        "question_type_summary": question_type_summary.to_dict(orient="records"),
        "pairwise_summary": pairwise_summary,
        "manual_review_queue_size": int(len(manual_review_queue)),
        "metric_groups": {
            "primary": [column for column in PRIMARY_METRIC_COLUMNS if column in df.columns],
            "supporting": [column for column in SUPPORTING_METRIC_COLUMNS if column in df.columns],
            "diagnostic": [column for column in DIAGNOSTIC_METRIC_COLUMNS if column in df.columns],
        },
    }
    write_json(settings.metrics_summary_json_path, payload)
    write_json(settings.pairwise_summary_json_path, pairwise_summary)
    return payload
