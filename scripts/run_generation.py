from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local

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


def _build_worker_state(settings: AppSettings):
    thread_state = local()

    def get_state():
        if not hasattr(thread_state, "baseline_chain"):
            llm_client = LLMClient(settings)
            thread_state.baseline_chain = BaselineChain(settings, llm_client=llm_client)
            thread_state.rag_chain = RagChain(
                settings,
                retriever=Retriever(settings),
                llm_client=llm_client,
            )
        return thread_state

    return get_state


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

    pending_questions = [
        question
        for question in questions
        if (question.question_id, "baseline") not in completed_keys
        or (question.question_id, "rag") not in completed_keys
    ]
    logger.info(
        "Pending generation questions: %s of %s total (concurrency=%s)",
        len(pending_questions),
        len(questions),
        settings.generation.concurrency,
    )

    if settings.generation.concurrency <= 1:
        llm_client = LLMClient(settings)
        baseline_chain = BaselineChain(settings, llm_client=llm_client)
        rag_chain = RagChain(settings, retriever=Retriever(settings), llm_client=llm_client)

        for question in tqdm(pending_questions, desc="Generating answers", unit="question"):
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
    else:
        get_worker_state = _build_worker_state(settings)

        def process_question(question: QuestionRecord) -> tuple[str, list[dict]]:
            worker = get_worker_state()
            payloads: list[dict] = []
            baseline_key = (question.question_id, "baseline")
            rag_key = (question.question_id, "rag")

            if baseline_key not in completed_keys:
                payloads.append(worker.baseline_chain.run(question).model_dump(mode="json"))
            if rag_key not in completed_keys:
                payloads.append(worker.rag_chain.run(question).model_dump(mode="json"))
            return question.question_id, payloads

        with ThreadPoolExecutor(max_workers=settings.generation.concurrency) as executor:
            future_map = {executor.submit(process_question, question): question for question in pending_questions}
            progress = tqdm(total=len(future_map), desc="Generating answers", unit="question")
            for index, future in enumerate(as_completed(future_map), start=1):
                question = future_map[future]
                question_id, payloads = future.result()
                for payload in payloads:
                    _append_generation_result(settings, payload)
                    completed_keys.add((payload["question_id"], payload["pipeline_type"]))
                progress.update(1)
                progress.set_postfix_str(question_id, refresh=False)
                if index % 10 == 0 or index == len(future_map):
                    logger.info(
                        "Generation progress: %s/%s questions completed (latest: %s)",
                        index,
                        len(future_map),
                        question.question_id,
                    )
            progress.close()

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
