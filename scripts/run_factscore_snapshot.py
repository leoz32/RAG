from __future__ import annotations

import argparse
import shutil
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
from src.utils.enums import PipelineType
from src.utils.io import read_csv, write_csv
from src.utils.logger import get_logger


FACTSCORE_COLUMNS = [
    "factscore_response_bucket",
    "factscore_cleaned_answer",
    "factscore_has_atomic_facts",
    "factscore_score",
    "factscore_init_score",
    "unsupported_claim_rate",
    "factscore_respond_ratio",
    "factscore_num_facts_per_response",
]


def _drop_existing_factscore_columns(df):
    drop_columns = [column for column in FACTSCORE_COLUMNS if column in df.columns]
    if drop_columns:
        df = df.drop(columns=drop_columns)
    stale_suffixes = ("_x", "_y", "_existing")
    stale_columns = [
        column
        for column in df.columns
        if any(
            column == f"{metric}{suffix}"
            for metric in FACTSCORE_COLUMNS
            for suffix in stale_suffixes
        )
    ]
    if stale_columns:
        df = df.drop(columns=stale_columns)
    return df


def _merge_factscore_results(question_level_df, factscore_df):
    question_level_df = _drop_existing_factscore_columns(question_level_df)
    return question_level_df.merge(
        factscore_df,
        on=["question_id", "pipeline_type"],
        how="left",
    )


def _backup_if_exists(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)
    return backup_path


def _remove_old_factscore_outputs(settings: AppSettings, logger) -> None:
    removable_paths = [
        settings.factscore_results_path,
        settings.factscore_summary_json_path,
        settings.metrics_summary_json_path,
        settings.metrics_summary_csv_path,
        settings.pipeline_summary_csv_path,
        settings.question_type_summary_csv_path,
        settings.pairwise_summary_json_path,
        settings.pairwise_summary_csv_path,
        settings.manual_review_queue_path,
        settings.analysis_report_path,
    ]
    backup_path = _backup_if_exists(settings.factscore_results_path)
    if backup_path is not None:
        logger.info("Backed up previous FActScore results to %s", backup_path)
    for path in removable_paths:
        if path.exists():
            path.unlink()
            logger.info("Removed stale output %s", path)


def _filter_eval_df_for_existing_question_level(eval_df, question_level_df):
    key_df = question_level_df[["question_id", "pipeline_type"]].drop_duplicates()
    filtered = eval_df.merge(key_df, on=["question_id", "pipeline_type"], how="inner")
    return filtered


def _build_base_question_level_df(eval_df, logger):
    question_level_df = eval_df.copy()
    question_level_df["ragas_completed"] = False
    logger.info(
        "question_level_results.csv missing; using generation/evaluation dataframe as the base table (%s rows)",
        len(question_level_df),
    )
    return question_level_df


def run_factscore_snapshot(settings: AppSettings, scope: str) -> None:
    logger = get_logger("run_factscore_snapshot", settings.log_file)

    if not settings.generation_results_path.exists():
        raise FileNotFoundError(
            f"Generation results not found for experiment {settings.experiment_id}: {settings.generation_results_path}"
        )
    if not settings.evaluation.factscore_enabled:
        raise ValueError("evaluation.factscore_enabled must be true to run FActScore snapshots.")

    logger.info("Preparing FActScore snapshot for experiment %s (scope=%s)", settings.experiment_id, scope)
    _remove_old_factscore_outputs(settings, logger)

    eval_df = build_evaluation_dataframe(settings)

    if settings.question_level_results_path.exists():
        question_level_df = read_csv(settings.question_level_results_path)
        logger.info(
            "Loaded existing question-level results with %s rows from %s",
            len(question_level_df),
            settings.question_level_results_path,
        )
    else:
        if scope == "existing-question-level":
            raise FileNotFoundError(
                "question_level_results.csv is required for scope=existing-question-level. "
                "Use --scope all-generation to build a standalone FActScore snapshot from generation results."
            )
        question_level_df = _build_base_question_level_df(eval_df, logger)

    if scope == "existing-question-level":
        scoped_eval_df = _filter_eval_df_for_existing_question_level(eval_df, question_level_df)
        logger.info(
            "Running FActScore only on rows already present in question-level results: %s/%s matched rows",
            len(scoped_eval_df),
            len(eval_df),
        )
    else:
        scoped_eval_df = eval_df
        logger.info("Running FActScore on all available generation rows: %s rows", len(scoped_eval_df))

    factscore_df = run_factscore(scoped_eval_df, settings, logger=logger)
    question_level_df = _merge_factscore_results(question_level_df, factscore_df)

    write_csv(settings.question_level_results_path, question_level_df)
    logger.info("Updated question-level results at %s", settings.question_level_results_path)

    baseline_df = question_level_df[question_level_df["pipeline_type"] == PipelineType.BASELINE.value]
    rag_df = question_level_df[question_level_df["pipeline_type"] == PipelineType.RAG.value]
    write_csv(settings.baseline_eval_results_path, baseline_df)
    write_csv(settings.rag_eval_results_path, rag_df)

    build_metric_summaries(question_level_df, settings)
    aggregates = aggregate_analysis(question_level_df)
    generate_analysis_report(settings, aggregates)
    logger.info("Wrote FActScore snapshot summaries under %s", settings.run_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--scope",
        choices=["existing-question-level", "all-generation"],
        default="existing-question-level",
        help="Choose whether to score only rows already present in question_level_results.csv or all generation rows.",
    )
    args = parser.parse_args()
    settings = load_settings(args.config, experiment_id=args.experiment_id)
    run_factscore_snapshot(settings, scope=args.scope)


if __name__ == "__main__":
    main()
