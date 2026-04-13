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
from src.evaluation.metrics_summary import build_metric_summaries
from src.evaluation.ragas_runner import run_ragas
from src.utils.enums import PipelineType
from src.utils.io import write_csv
from src.utils.logger import get_logger


def run_evaluation(settings: AppSettings) -> None:
    logger = get_logger("run_evaluation", settings.log_file)
    eval_df = build_evaluation_dataframe(settings)
    question_level_df = run_ragas(eval_df, settings, logger=logger)
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
