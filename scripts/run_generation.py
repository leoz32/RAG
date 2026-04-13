from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.schemas import QuestionRecord
from src.config.settings import AppSettings, load_settings
from src.generation.baseline_chain import BaselineChain
from src.generation.llm_client import LLMClient
from src.generation.rag_chain import RagChain
from src.retrieval.retriever import Retriever
from src.utils.checkpoint import append_records_csv, finalize_csv, load_completed_keys
from src.utils.io import read_csv, write_csv
from src.utils.logger import get_logger


def _load_questions(settings: AppSettings, limit: int | None = None) -> list[QuestionRecord]:
    qa_df = read_csv(settings.qa_dataset_path)
    required_columns = {"question_id", "question", "ground_truth", "question_type"}
    missing = required_columns - set(qa_df.columns)
    if missing:
        raise ValueError(f"qa_dataset.csv missing required columns: {sorted(missing)}")
    if limit is not None:
        qa_df = qa_df.head(limit)
    return [QuestionRecord.model_validate(row) for row in qa_df.to_dict(orient="records")]


def _append_generation_result(settings: AppSettings, payload: dict) -> None:
    append_records_csv(settings.generation_results_path, [payload])


def run_generation(settings: AppSettings, limit: int | None = None) -> pd.DataFrame:
    logger = get_logger("run_generation", settings.log_file)
    questions = _load_questions(settings, limit=limit)
    logger.info("Loaded %s evaluation questions", len(questions))
    if limit is not None:
        logger.info("Generation limit enabled: first %s questions only", limit)
    logger.info(
        "Generation inputs: qa_dataset=%s, chunks=%s, faiss_index=%s",
        settings.qa_dataset_path,
        settings.chunks_path,
        settings.faiss_index_dir,
    )
    completed_keys = load_completed_keys(settings.generation_results_path, ["question_id", "pipeline_type"])
    if completed_keys:
        logger.info("Resuming generation with %s completed records already on disk", len(completed_keys))

    llm_client = LLMClient(settings)
    baseline_chain = BaselineChain(settings, llm_client=llm_client)
    rag_chain = RagChain(settings, retriever=Retriever(settings), llm_client=llm_client)

    for question in tqdm(questions, desc="Generating answers"):
        baseline_key = (question.question_id, "baseline")
        rag_key = (question.question_id, "rag")

        if baseline_key not in completed_keys:
            baseline_payload = baseline_chain.run(question).model_dump(mode="json")
            _append_generation_result(settings, baseline_payload)
            completed_keys.add(baseline_key)

        if rag_key not in completed_keys:
            rag_payload = rag_chain.run(question).model_dump(mode="json")
            _append_generation_result(settings, rag_payload)
            completed_keys.add(rag_key)

    results_df = finalize_csv(
        settings.generation_results_path,
        key_columns=["question_id", "pipeline_type"],
        sort_columns=["question_id", "pipeline_type"],
    )
    logger.info("Saved generation results to %s", settings.generation_results_path)
    return results_df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    settings = load_settings(args.config, experiment_id=args.experiment_id)
    run_generation(settings, limit=args.limit)


if __name__ == "__main__":
    main()
