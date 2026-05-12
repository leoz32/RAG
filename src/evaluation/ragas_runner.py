from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local

import openai
import pandas as pd

from src.config.settings import AppSettings
from src.ingestion.embedder import EmbeddingClient
from src.utils.checkpoint import append_records_csv, finalize_csv


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


def _build_ragas_dataset(dataset_cls, row: pd.Series):
    reference = "" if pd.isna(row["ground_truth"]) else str(row["ground_truth"])
    return dataset_cls.from_dict(
        {
            "question": [row["question"]],
            "answer": [row["answer"]],
            "contexts": [_ensure_list(row["evaluation_contexts"])],
            "ground_truth": [reference],
            "reference": [reference],
        }
    )


def _has_reference_ground_truth(row: pd.Series | dict) -> bool:
    value = row.get("ground_truth") if isinstance(row, dict) else row["ground_truth"]
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    return bool(str(value).strip())


def _required_metrics_for_row(row: pd.Series | dict) -> list[str]:
    base_metrics = ["faithfulness", "answer_relevance"]
    if _has_reference_ground_truth(row):
        return [*base_metrics, "answer_correctness", "context_precision", "context_recall"]
    return base_metrics


def _evaluation_llm_params(settings: AppSettings) -> dict[str, str | None]:
    return {
        "model_name": settings.evaluation.llm_model_name or settings.llm.model_name,
        "api_key": settings.evaluation.api_key or settings.llm.api_key,
        "base_url": settings.evaluation.base_url or settings.llm.base_url,
    }


def run_ragas(df: pd.DataFrame, settings: AppSettings, logger=None) -> pd.DataFrame:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings.base import LangchainEmbeddingsWrapper
        from ragas.llms import llm_factory
        from ragas.metrics import answer_correctness, context_precision, context_recall, faithfulness
        try:
            from ragas.metrics import answer_relevance
        except ImportError:
            from ragas.metrics import answer_relevancy as answer_relevance
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError("Ragas and datasets must be installed to run evaluation.") from exc

    eval_df = df.copy()

    metric_columns = [
        "faithfulness",
        "answer_correctness",
        "answer_relevance",
        "context_precision",
        "context_recall",
    ]
    legacy_metric_aliases = {
        "answer_relevance": ["answer_relevance", "answer_relevancy"],
    }
    completed_keys = set()
    if settings.question_level_results_path.exists():
        existing_df = pd.read_csv(settings.question_level_results_path)
        for _, existing_row in existing_df.iterrows():
            key = (str(existing_row["question_id"]), str(existing_row["pipeline_type"]))
            if "ragas_completed" in existing_df.columns:
                row_complete = bool(existing_row.get("ragas_completed"))
            else:
                row_complete = True
                for metric in _required_metrics_for_row(existing_row):
                    aliases = legacy_metric_aliases.get(metric, [metric])
                    has_metric = False
                    for alias in aliases:
                        if alias in existing_df.columns:
                            value = existing_row.get(alias)
                            if pd.notna(value):
                                has_metric = True
                                break
                    if not has_metric:
                        row_complete = False
                        break
            if row_complete:
                completed_keys.add(key)
    if logger is not None and completed_keys:
        logger.info("Resuming evaluation with %s fully-computed rows already on disk", len(completed_keys))

    pending_df = eval_df[
        ~eval_df.apply(
            lambda row: (str(row["question_id"]), str(row["pipeline_type"])) in completed_keys,
            axis=1,
        )
    ].reset_index(drop=True)
    effective_concurrency = settings.evaluation.concurrency
    if effective_concurrency != 1:
        if logger is not None:
            logger.warning(
                "RAGAS evaluation concurrency=%s requested, but ragas metrics are not thread-safe in this setup; forcing concurrency=1",
                effective_concurrency,
            )
        effective_concurrency = 1

    if logger is not None:
        logger.info(
            "Pending ragas evaluations: %s of %s total rows (concurrency=%s)",
            len(pending_df),
            len(eval_df),
            effective_concurrency,
        )

    if len(pending_df) == 0:
        return finalize_csv(
            settings.question_level_results_path,
            key_columns=["question_id", "pipeline_type"],
            sort_columns=["question_id", "pipeline_type"],
        )

    worker_state = local()

    def get_worker_clients():
        if not hasattr(worker_state, "ragas_llm"):
            eval_llm_params = _evaluation_llm_params(settings)
            if hasattr(openai, "OpenAI"):
                from openai import OpenAI

                llm_client = OpenAI(
                    api_key=eval_llm_params["api_key"],
                    base_url=eval_llm_params["base_url"],
                )
            else:
                openai.api_key = eval_llm_params["api_key"]
                if eval_llm_params["base_url"]:
                    openai.api_base = eval_llm_params["base_url"]
                llm_client = openai
            worker_state.ragas_llm = llm_factory(
                eval_llm_params["model_name"],
                client=llm_client,
                temperature=settings.llm.temperature,
                max_tokens=settings.evaluation.llm_max_tokens,
            )
            worker_state.ragas_embeddings = LangchainEmbeddingsWrapper(EmbeddingClient(settings).client)
        return worker_state.ragas_llm, worker_state.ragas_embeddings

    def evaluate_row(row_dict: dict) -> dict:
        row = pd.Series(row_dict)
        ragas_llm, ragas_embeddings = get_worker_clients()
        dataset = _build_ragas_dataset(Dataset, row)
        metrics_to_run = [faithfulness, answer_relevance]
        if _has_reference_ground_truth(row):
            metrics_to_run.extend([answer_correctness, context_precision, context_recall])
        result = evaluate(
            dataset=dataset,
            metrics=metrics_to_run,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        scores_df = result.to_pandas()
        metric_payload = {column: None for column in metric_columns}
        if not scores_df.empty:
            for column in metric_columns:
                aliases = legacy_metric_aliases.get(column, [column])
                for alias in aliases:
                    if alias in scores_df.columns:
                        metric_payload[column] = scores_df.iloc[0][alias]
                        break

        record = row_dict.copy()
        record["evaluation_contexts"] = _ensure_list(row_dict.get("evaluation_contexts"))
        record["contexts"] = _ensure_list(row_dict.get("contexts"))
        record.update(metric_payload)
        record["ragas_completed"] = True
        return record

    pending_records = pending_df.to_dict(orient="records")
    progress = tqdm(total=len(pending_records), desc="RAGAS rows", unit="row")

    if effective_concurrency <= 1:
        for index, row_dict in enumerate(pending_records, start=1):
            record = evaluate_row(row_dict)
            append_records_csv(settings.question_level_results_path, [record])
            completed_keys.add((str(record["question_id"]), str(record["pipeline_type"])))
            progress.update(1)
            progress.set_postfix_str(f"{record['pipeline_type']} {record['question_id']}", refresh=False)
            if logger is not None and (index == len(pending_records) or index % 10 == 0):
                logger.info(
                    "RAGAS progress: %s/%s completed (latest: %s %s)",
                    index,
                    len(pending_records),
                    record["pipeline_type"],
                    record["question_id"],
                )
    else:
        with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
            future_map = {
                executor.submit(evaluate_row, row_dict): row_dict
                for row_dict in pending_records
            }
            for index, future in enumerate(as_completed(future_map), start=1):
                record = future.result()
                append_records_csv(settings.question_level_results_path, [record])
                completed_keys.add((str(record["question_id"]), str(record["pipeline_type"])))
                progress.update(1)
                progress.set_postfix_str(f"{record['pipeline_type']} {record['question_id']}", refresh=False)
                if logger is not None and (index == len(future_map) or index % 10 == 0):
                    logger.info(
                        "RAGAS progress: %s/%s completed (latest: %s %s)",
                        index,
                        len(future_map),
                        record["pipeline_type"],
                        record["question_id"],
                    )

    progress.close()

    return finalize_csv(
        settings.question_level_results_path,
        key_columns=["question_id", "pipeline_type"],
        sort_columns=["question_id", "pipeline_type"],
    )
