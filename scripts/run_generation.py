from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import local

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.schemas import GenerationRecord, QuestionRecord
from src.config.settings import AppSettings, load_settings
from src.evaluation.dataset_source import load_question_dataframe
from src.generation.baseline_chain import BaselineChain
from src.generation.llm_client import LLMClient
from src.generation.prompts import build_non_rag_prompt, build_rag_prompt
from src.generation.rag_chain import RagChain
from src.retrieval.retriever import Retriever
from src.utils.checkpoint import append_records_csv, finalize_csv, load_completed_keys
from src.utils.enums import PipelineType
from src.utils.io import read_csv, write_csv
from src.utils.logger import get_logger


def _load_questions(settings: AppSettings, limit: int | None = None) -> list[QuestionRecord]:
    qa_df = load_question_dataframe(settings)
    required_columns = {"question_id", "question", "ground_truth", "question_type"}
    missing = required_columns - set(qa_df.columns)
    if missing:
        raise ValueError(f"qa_dataset.csv missing required columns: {sorted(missing)}")
    if limit is not None:
        qa_df = qa_df.head(limit)
    optional_columns = {"topic", "source_split", "source_model", "reference_output", "factscore_annotations"}
    for column in optional_columns:
        if column not in qa_df.columns:
            qa_df[column] = None
    return [QuestionRecord.model_validate(row) for row in qa_df.to_dict(orient="records")]


def _append_generation_result(settings: AppSettings, payload: dict) -> None:
    append_records_csv(settings.generation_results_path, [payload])


def _generation_failure_payload(
    settings: AppSettings,
    question: QuestionRecord,
    pipeline_type: PipelineType,
    exc: Exception,
    rag_chain: RagChain | None = None,
) -> dict:
    contexts: list[str] = []
    context_ids: list[str] = []
    context_source_files: list[str] = []
    context_scores: list[float] = []
    retrieved_context_count = 0
    retrieval_top_k = 0

    if pipeline_type == PipelineType.RAG:
        retrieval_top_k = settings.retrieval.top_k
        if rag_chain is not None:
            try:
                retrievals = rag_chain.retriever.retrieve(question.question, question.question_id)
            except Exception:
                retrievals = []
            contexts = [record.text for record in retrievals]
            context_ids = [record.chunk_id for record in retrievals]
            context_source_files = [record.source_file for record in retrievals]
            context_scores = [float(record.score) for record in retrievals]
            retrieved_context_count = len(retrievals)
        prompt = build_rag_prompt(question.question, "\n\n".join(contexts))
    else:
        prompt = build_non_rag_prompt(question.question)

    return GenerationRecord(
        experiment_id=settings.experiment_id,
        question_id=question.question_id,
        question=question.question,
        ground_truth=question.ground_truth,
        question_type=question.question_type,
        topic=question.topic,
        source_split=question.source_split,
        source_model=question.source_model,
        reference_output=question.reference_output,
        pipeline_type=pipeline_type,
        answer="",
        contexts=contexts,
        context_ids=context_ids,
        context_source_files=context_source_files,
        context_scores=context_scores,
        retrieved_context_count=retrieved_context_count,
        prompt=prompt,
        prompt_version=settings.prompt_version,
        llm_model_name=settings.llm.model_name,
        embedding_model_name=settings.embedding.model_name,
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
        retrieval_top_k=retrieval_top_k,
        dataset_version=settings.dataset.dataset_version,
        generation_status="failed",
        generation_error_type=type(exc).__name__,
        generation_error_message=str(exc),
        timestamp=datetime.now(timezone.utc),
    ).model_dump(mode="json")


def _run_pipeline_or_failure(
    settings: AppSettings,
    question: QuestionRecord,
    pipeline_type: PipelineType,
    baseline_chain: BaselineChain,
    rag_chain: RagChain,
) -> dict:
    try:
        if pipeline_type == PipelineType.BASELINE:
            return baseline_chain.run(question).model_dump(mode="json")
        return rag_chain.run(question).model_dump(mode="json")
    except Exception as exc:
        return _generation_failure_payload(
            settings=settings,
            question=question,
            pipeline_type=pipeline_type,
            exc=exc,
            rag_chain=rag_chain,
        )


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
                baseline_payload = _run_pipeline_or_failure(
                    settings,
                    question,
                    PipelineType.BASELINE,
                    baseline_chain,
                    rag_chain,
                )
                _append_generation_result(settings, baseline_payload)
                completed_keys.add(baseline_key)
                if baseline_payload.get("generation_status") == "failed":
                    logger.warning(
                        "Generation failed and skipped: question_id=%s topic=%s pipeline=baseline error_type=%s error=%s",
                        question.question_id,
                        question.topic,
                        baseline_payload.get("generation_error_type"),
                        baseline_payload.get("generation_error_message"),
                    )

            if rag_key not in completed_keys:
                rag_payload = _run_pipeline_or_failure(
                    settings,
                    question,
                    PipelineType.RAG,
                    baseline_chain,
                    rag_chain,
                )
                _append_generation_result(settings, rag_payload)
                completed_keys.add(rag_key)
                if rag_payload.get("generation_status") == "failed":
                    logger.warning(
                        "Generation failed and skipped: question_id=%s topic=%s pipeline=rag error_type=%s error=%s",
                        question.question_id,
                        question.topic,
                        rag_payload.get("generation_error_type"),
                        rag_payload.get("generation_error_message"),
                    )
    else:
        get_worker_state = _build_worker_state(settings)
        completed_keys_snapshot = completed_keys.copy()

        def process_question(question: QuestionRecord) -> tuple[str, list[dict]]:
            worker = get_worker_state()
            payloads: list[dict] = []
            baseline_key = (question.question_id, "baseline")
            rag_key = (question.question_id, "rag")

            if baseline_key not in completed_keys_snapshot:
                payloads.append(
                    _run_pipeline_or_failure(
                        settings,
                        question,
                        PipelineType.BASELINE,
                        worker.baseline_chain,
                        worker.rag_chain,
                    )
                )
            if rag_key not in completed_keys_snapshot:
                payloads.append(
                    _run_pipeline_or_failure(
                        settings,
                        question,
                        PipelineType.RAG,
                        worker.baseline_chain,
                        worker.rag_chain,
                    )
                )
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
                    if payload.get("generation_status") == "failed":
                        logger.warning(
                            "Generation failed and skipped: question_id=%s topic=%s pipeline=%s error_type=%s error=%s",
                            payload.get("question_id"),
                            question.topic,
                            payload.get("pipeline_type"),
                            payload.get("generation_error_type"),
                            payload.get("generation_error_message"),
                        )
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
