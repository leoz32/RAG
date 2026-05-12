from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.aggregator import aggregate_analysis
from src.analysis.report_generator import generate_analysis_report
from src.config.settings import AppSettings, load_settings
from src.evaluation.dataset_builder import build_evaluation_dataframe
from src.evaluation.factscore_runner import run_factscore
from src.evaluation.metrics_summary import build_metric_summaries
from src.evaluation.ragas_runner import run_ragas
from src.utils.enums import PipelineType
from src.utils.io import write_csv
from src.utils.logger import get_logger


FACTSCORE_COLUMNS = [
    "factscore_score",
    "factscore_init_score",
    "unsupported_claim_rate",
    "factscore_respond_ratio",
    "factscore_num_facts_per_response",
]


def _drop_stale_factscore_columns(df):
    stale_suffixes = ("_x", "_y", "_existing")
    stale_columns = [
        column
        for column in df.columns
        if any(column == f"{metric}{suffix}" for metric in FACTSCORE_COLUMNS for suffix in stale_suffixes)
    ]
    if stale_columns:
        df = df.drop(columns=stale_columns)
    return df


def _merge_factscore_results(question_level_df, factscore_df):
    question_level_df = _drop_stale_factscore_columns(question_level_df)
    merged = question_level_df.merge(
        factscore_df,
        on=["question_id", "pipeline_type"],
        how="left",
        suffixes=("_existing", ""),
    )
    for column in FACTSCORE_COLUMNS:
        existing_column = f"{column}_existing"
        if existing_column in merged.columns:
            merged[column] = merged[column].combine_first(merged[existing_column])
            merged = merged.drop(columns=[existing_column])
    return _drop_stale_factscore_columns(merged)


def _ensure_derived_metrics(question_level_df):
    if (
        "unsupported_claim_rate" not in question_level_df.columns
        and "factscore_init_score" in question_level_df.columns
    ):
        question_level_df["unsupported_claim_rate"] = 1.0 - question_level_df["factscore_init_score"].fillna(0.0)
    return question_level_df


def run_evaluation(settings: AppSettings) -> None:
    logger = get_logger("run_evaluation", settings.log_file)
    eval_df = build_evaluation_dataframe(settings)
    question_level_df = run_ragas(eval_df, settings, logger=logger)

    if settings.evaluation.factscore_enabled:
        factscore_df = run_factscore(eval_df, settings, logger=logger)
        question_level_df = _merge_factscore_results(question_level_df, factscore_df)
        logger.info("Saved FActScore results to %s", settings.factscore_results_path)

    question_level_df = _ensure_derived_metrics(question_level_df)
    write_csv(settings.question_level_results_path, question_level_df)
    logger.info("Saved question-level results to %s", settings.question_level_results_path)

    baseline_df = question_level_df[question_level_df["pipeline_type"] == PipelineType.BASELINE.value]
    rag_df = question_level_df[question_level_df["pipeline_type"] == PipelineType.RAG.value]
    write_csv(settings.baseline_eval_results_path, baseline_df)
    write_csv(settings.rag_eval_results_path, rag_df)

    build_metric_summaries(question_level_df, settings)
    aggregates = aggregate_analysis(question_level_df)
    generate_analysis_report(settings, aggregates)
    logger.info("Saved evaluation reports under %s", settings.run_dir)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--experiment-id", default=None)
    args = parser.parse_args()
    settings = load_settings(args.config, experiment_id=args.experiment_id)
    run_evaluation(settings)


if __name__ == "__main__":
    main()
