from __future__ import annotations

import os
import importlib
import json
import re
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from fnmatch import fnmatch
from pathlib import Path

import pandas as pd

from src.config.settings import AppSettings
from src.utils.checkpoint import append_records_csv, finalize_csv, load_completed_keys
from src.utils.io import ensure_dir, write_json
from src.evaluation.dataset_source import load_question_dataframe


FACTSCORE_COLUMNS = [
    "question_id",
    "pipeline_type",
    "factscore_response_bucket",
    "factscore_cleaned_answer",
    "factscore_has_atomic_facts",
    "factscore_score",
    "factscore_init_score",
    "unsupported_claim_rate",
    "factscore_respond_ratio",
    "factscore_num_facts_per_response",
]


_FACTSCORE_WORKER_SCORER = None
_FACTSCORE_WORKER_SETTINGS = None

_META_PARAGRAPH_PATTERNS = [
    re.compile(r"^\s*依据[:：].*$", re.IGNORECASE | re.DOTALL),
    re.compile(r"^\s*以上信息(?:均)?(?:直接)?(?:来自|来源于).*$", re.IGNORECASE | re.DOTALL),
]

_META_SENTENCE_PATTERNS = [
    re.compile(r"(?:依据)[:：].*$", re.IGNORECASE),
    re.compile(r"以上信息(?:均)?(?:直接)?(?:来自|来源于).*$", re.IGNORECASE),
]

_NO_INFO_PATTERNS = [
    re.compile(r"无法提供.{0,20}(?:简介|生平|个人简介|相关信息|详细信息)"),
    re.compile(r"(?:资料|教学资料).{0,10}(?:未提及|没有|未提供).{0,20}(?:信息|简介|生平)"),
    re.compile(r"未找到.{0,20}(?:相关信息|具体信息|资料|记录)"),
    re.compile(r"不能提供.{0,20}(?:简介|生平|确切内容)"),
    re.compile(r"无法确认其身份"),
    re.compile(r"请提供更多背景信息"),
]


def _load_factscore_class(settings: AppSettings):
    source_dir = settings.factscore_source_dir
    if not source_dir.exists():
        raise FileNotFoundError(f"Configured FActScore source path not found: {source_dir}")
    for module_name in [name for name in list(sys.modules) if name == "factscore" or name.startswith("factscore.")]:
        sys.modules.pop(module_name, None)
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    module = importlib.import_module("factscore.factscorer")
    return module.FactScorer


def _worker_cache_dir(base_cache_dir: Path, worker_label: str) -> Path:
    return base_cache_dir / worker_label


def _iter_project_corpus_texts(settings: AppSettings) -> list[str]:
    texts: list[str] = []
    for path in sorted(settings.raw_dir.glob(settings.dataset.raw_glob)):
        if path.suffix.lower() != ".md":
            continue
        relative_path = str(path.relative_to(settings.project_root))
        if any(
            fnmatch(relative_path, pattern) or fnmatch(path.name, pattern)
            for pattern in settings.dataset.exclude_globs
        ):
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            texts.append(text)
    if not texts:
        raise FileNotFoundError(
            f"No documents available to build FActScore knowledge source under {settings.raw_dir}"
        )
    return texts


def _iter_dataset_matched_records(settings: AppSettings) -> list[dict]:
    qa_df = load_question_dataframe(settings)
    required_columns = {"topic", "reference_output"}
    missing = required_columns - set(qa_df.columns)
    if missing:
        raise ValueError(
            "dataset-matched FActScore knowledge source requires columns "
            f"{sorted(required_columns)}, missing {sorted(missing)}"
        )

    records: list[dict] = []
    seen_topics: set[str] = set()
    for row in qa_df.itertuples(index=False):
        topic = "" if row.topic is None else str(row.topic).strip()
        text = "" if row.reference_output is None else str(row.reference_output).strip()
        if not topic or not text or topic in seen_topics:
            continue
        seen_topics.add(topic)
        records.append({"title": topic, "text": [f"{topic}\n\n{text}"]})
    if not records:
        raise FileNotFoundError("No dataset-matched FActScore knowledge source records could be built.")
    return records


def _build_knowledge_source(settings: AppSettings) -> None:
    ensure_dir(settings.factscore_knowledge_source_path.parent)
    with settings.factscore_knowledge_source_path.open("w", encoding="utf-8") as handle:
        if settings.dataset.kind == "factscore_jsonl":
            records = _iter_dataset_matched_records(settings)
        else:
            records = [
                {
                    "title": settings.evaluation.factscore_project_topic_name,
                    "text": _iter_project_corpus_texts(settings),
                }
            ]
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_openai_key_file(settings: AppSettings) -> None:
    if not settings.evaluation.factscore_openai_key:
        raise ValueError(
            "FActScore requires a compatible API key. "
            f"Set environment variable {settings.evaluation.factscore_openai_key_env}."
        )
    settings.factscore_openai_key_path.write_text(
        settings.evaluation.factscore_openai_key.strip(),
        encoding="utf-8",
    )


def _create_factscorer(settings: AppSettings, worker_label: str = "main"):
    FactScorer = _load_factscore_class(settings)
    cache_dir = _worker_cache_dir(settings.run_dir / ".factscore_cache", worker_label)
    ensure_dir(cache_dir)
    scorer = FactScorer(
        model_name=settings.evaluation.factscore_model_name,
        data_dir=str(settings.factscore_source_dir),
        model_dir=str(settings.run_dir),
        cache_dir=str(cache_dir),
        openai_key=str(settings.factscore_openai_key_path),
        api_base=settings.evaluation.factscore_api_base,
        chat_model_name=settings.evaluation.factscore_chat_model_name,
        instruct_model_name=settings.evaluation.factscore_instruct_model_name,
        retrieval_type=settings.evaluation.factscore_retrieval_type,
        cost_estimate="consider_cache",
    )
    scorer.register_knowledge_source(
        name=settings.evaluation.factscore_knowledge_source_name,
        data_path=str(settings.factscore_knowledge_source_path),
        db_path=str(settings.factscore_db_path),
    )
    return scorer


def _prebuild_factscore_knowledge_db(settings: AppSettings, logger=None) -> None:
    """Build the shared FActScore SQLite DB once before spawning worker processes.

    FActScore's DocDB lazily initializes the SQLite file when it sees an empty DB.
    If multiple processes do that concurrently, they race on CREATE TABLE and the
    process pool crashes. Prebuilding in the parent process avoids that.
    """
    if settings.factscore_db_path.exists() and settings.factscore_db_path.stat().st_size > 0:
        return

    if logger is not None:
        logger.info("Prebuilding FActScore knowledge DB at %s", settings.factscore_db_path)

    scorer = _create_factscorer(settings, worker_label="prebuild")
    for db in getattr(scorer, "db", {}).values():
        try:
            db.close()
        except Exception:
            pass


def _knowledge_source_title_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _db_document_count(db_path: Path) -> int | None:
    if not db_path.exists():
        return None
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
        if cursor.fetchone() is None:
            return 0
        cursor.execute("SELECT COUNT(*) FROM documents")
        return int(cursor.fetchone()[0])
    except sqlite3.Error:
        return None
    finally:
        try:
            connection.close()
        except Exception:
            pass


def _invalidate_factscore_knowledge_artifacts(settings: AppSettings) -> None:
    artifacts = [
        settings.factscore_db_path,
        settings.run_dir / ".factscore_cache" / f"retrieval-{settings.evaluation.factscore_knowledge_source_name}.json",
        settings.run_dir / ".factscore_cache" / f"retrieval-{settings.evaluation.factscore_knowledge_source_name}.pkl",
        settings.run_dir / ".factscore_cache" / f"bm25-{settings.evaluation.factscore_knowledge_source_name}.json",
        settings.run_dir / ".factscore_cache" / f"bm25-{settings.evaluation.factscore_knowledge_source_name}.pkl",
    ]
    for artifact in artifacts:
        if artifact.exists():
            artifact.unlink()


def _ensure_factscore_knowledge_source_consistency(settings: AppSettings, logger=None) -> None:
    expected_documents = _knowledge_source_title_count(settings.factscore_knowledge_source_path)
    current_documents = _db_document_count(settings.factscore_db_path)
    if current_documents is None:
        return
    if current_documents == expected_documents:
        return
    if logger is not None:
        logger.warning(
            "FActScore knowledge DB is stale or mismatched (db=%s, expected=%s). Rebuilding %s",
            current_documents,
            expected_documents,
            settings.factscore_db_path,
        )
    _invalidate_factscore_knowledge_artifacts(settings)


def _topic_for_row(row: pd.Series, settings: AppSettings) -> str:
    if settings.dataset.kind == "factscore_jsonl" and row.get("topic"):
        return str(row["topic"]).strip()
    return settings.evaluation.factscore_project_topic_name


def _serialize_settings_for_worker(settings: AppSettings) -> dict:
    return {
        "project_root": str(settings.project_root),
        "experiment_id": settings.experiment_id,
        "factscore_source_dir": str(settings.factscore_source_dir),
        "run_dir": str(settings.run_dir),
        "factscore_openai_key_path": str(settings.factscore_openai_key_path),
        "factscore_knowledge_source_name": settings.evaluation.factscore_knowledge_source_name,
        "factscore_knowledge_source_path": str(settings.factscore_knowledge_source_path),
        "factscore_db_path": str(settings.factscore_db_path),
        "factscore_model_name": settings.evaluation.factscore_model_name,
        "factscore_api_base": settings.evaluation.factscore_api_base,
        "factscore_chat_model_name": settings.evaluation.factscore_chat_model_name,
        "factscore_instruct_model_name": settings.evaluation.factscore_instruct_model_name,
        "factscore_retrieval_type": settings.evaluation.factscore_retrieval_type,
        "factscore_gamma": settings.evaluation.factscore_gamma,
        "dataset_kind": settings.dataset.kind,
        "factscore_project_topic_name": settings.evaluation.factscore_project_topic_name,
    }


def _init_factscore_worker(serialized_settings: dict) -> None:
    global _FACTSCORE_WORKER_SCORER, _FACTSCORE_WORKER_SETTINGS
    _FACTSCORE_WORKER_SETTINGS = serialized_settings

    source_dir = Path(serialized_settings["factscore_source_dir"])
    for module_name in [name for name in list(sys.modules) if name == "factscore" or name.startswith("factscore.")]:
        sys.modules.pop(module_name, None)
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    module = importlib.import_module("factscore.factscorer")
    FactScorer = module.FactScorer
    worker_label = f"worker_{os.getpid()}"
    cache_dir = _worker_cache_dir(Path(serialized_settings["run_dir"]) / ".factscore_cache", worker_label)
    ensure_dir(cache_dir)
    scorer = FactScorer(
        model_name=serialized_settings["factscore_model_name"],
        data_dir=str(source_dir),
        model_dir=serialized_settings["run_dir"],
        cache_dir=str(cache_dir),
        openai_key=serialized_settings["factscore_openai_key_path"],
        api_base=serialized_settings["factscore_api_base"],
        chat_model_name=serialized_settings["factscore_chat_model_name"],
        instruct_model_name=serialized_settings["factscore_instruct_model_name"],
        retrieval_type=serialized_settings["factscore_retrieval_type"],
        cost_estimate="consider_cache",
    )
    scorer.register_knowledge_source(
        name=serialized_settings["factscore_knowledge_source_name"],
        data_path=serialized_settings["factscore_knowledge_source_path"],
        db_path=serialized_settings["factscore_db_path"],
    )
    _FACTSCORE_WORKER_SCORER = scorer


def _worker_topic_for_row(row_dict: dict, serialized_settings: dict) -> str:
    if serialized_settings["dataset_kind"] == "factscore_jsonl" and row_dict.get("topic"):
        return str(row_dict["topic"]).strip()
    return serialized_settings["factscore_project_topic_name"]


def _remove_meta_explanation(answer: str) -> str:
    if not answer.strip():
        return ""

    paragraphs = re.split(r"\n\s*\n+", answer.strip())
    kept_paragraphs: list[str] = []
    for paragraph in paragraphs:
        text = paragraph.strip()
        if not text:
            continue
        if any(pattern.match(text) for pattern in _META_PARAGRAPH_PATTERNS):
            continue
        for pattern in _META_SENTENCE_PATTERNS:
            text = pattern.sub("", text).strip()
        if text:
            kept_paragraphs.append(text)
    return "\n\n".join(kept_paragraphs).strip()


def _classify_factscore_response_bucket(answer: str) -> str:
    normalized = re.sub(r"\s+", " ", answer).strip()
    if not normalized:
        return "empty"
    if any(pattern.search(normalized) for pattern in _NO_INFO_PATTERNS):
        return "abstain_no_info"
    return "contentful_bio"


def _prepare_factscore_answer(answer: str) -> tuple[str, str]:
    bucket = _classify_factscore_response_bucket(answer)
    cleaned = _remove_meta_explanation(answer)
    if bucket == "abstain_no_info" and not cleaned.strip():
        return answer.strip(), bucket
    if bucket != "contentful_bio":
        return "", bucket
    return cleaned, bucket


def _score_factscore_row(row_dict: dict) -> dict:
    row = pd.Series(row_dict)
    topic = _worker_topic_for_row(row_dict, _FACTSCORE_WORKER_SETTINGS)
    raw_output = "" if pd.isna(row_dict.get("answer")) else str(row_dict["answer"])
    output, bucket = _prepare_factscore_answer(raw_output)
    result = _FACTSCORE_WORKER_SCORER.get_score(
        topics=[topic],
        generations=[output],
        gamma=_FACTSCORE_WORKER_SETTINGS["factscore_gamma"],
        knowledge_source=_FACTSCORE_WORKER_SETTINGS["factscore_knowledge_source_name"],
    )
    return _build_factscore_record(row, result, output, bucket)


def _build_factscore_record(row, result: dict, cleaned_answer: str, response_bucket: str) -> dict:
    respond_ratio = result.get("respond_ratio")
    has_atomic_facts = bool(respond_ratio and float(respond_ratio) > 0)
    init_score = result.get("init_score") if has_atomic_facts else 0.0
    return {
        "question_id": str(getattr(row, "question_id")),
        "pipeline_type": str(getattr(row, "pipeline_type")),
        "factscore_response_bucket": response_bucket,
        "factscore_cleaned_answer": cleaned_answer,
        "factscore_has_atomic_facts": has_atomic_facts,
        "factscore_score": result.get("score") if has_atomic_facts else 0.0,
        "factscore_init_score": init_score,
        "unsupported_claim_rate": max(0.0, 1.0 - float(init_score or 0.0)),
        "factscore_respond_ratio": float(respond_ratio or 0.0),
        "factscore_num_facts_per_response": result.get("num_facts_per_response") if has_atomic_facts else 0.0,
    }


def _normalize_factscore_result_df(result_df: pd.DataFrame) -> pd.DataFrame:
    normalized = result_df.copy()
    if "factscore_response_bucket" not in normalized.columns:
        normalized["factscore_response_bucket"] = "unknown_legacy"
    if "factscore_cleaned_answer" not in normalized.columns:
        normalized["factscore_cleaned_answer"] = ""
    return normalized


def _write_factscore_summary(result_df: pd.DataFrame, settings: AppSettings) -> None:
    result_df = _normalize_factscore_result_df(result_df)
    summary_rows = []

    def summarize(frame: pd.DataFrame, *, pipeline_type: str, bucket: str) -> dict:
        return {
            "pipeline_type": pipeline_type,
            "factscore_response_bucket": bucket,
            "rows": int(len(frame)),
            "factscore_score": frame["factscore_score"].mean(),
            "factscore_init_score": frame["factscore_init_score"].mean(),
            "unsupported_claim_rate": frame["unsupported_claim_rate"].mean(),
            "factscore_respond_ratio": frame["factscore_respond_ratio"].mean(),
            "factscore_num_facts_per_response": frame["factscore_num_facts_per_response"].mean(),
        }

    for pipeline_type, frame in result_df.groupby("pipeline_type"):
        summary_rows.append(summarize(frame, pipeline_type=pipeline_type, bucket="all"))
        for bucket, bucket_frame in frame.groupby("factscore_response_bucket"):
            summary_rows.append(summarize(bucket_frame, pipeline_type=pipeline_type, bucket=bucket))
    write_json(
        settings.factscore_summary_json_path,
        {
            "experiment_id": settings.experiment_id,
            "knowledge_source_name": settings.evaluation.factscore_knowledge_source_name,
            "project_topic_name": settings.evaluation.factscore_project_topic_name,
            "gamma": settings.evaluation.factscore_gamma,
            "response_buckets": [
                "all",
                "contentful_bio",
                "abstain_no_info",
                "empty",
            ],
            "pipeline_summary": summary_rows,
        },
    )


def run_factscore(df: pd.DataFrame, settings: AppSettings, logger=None) -> pd.DataFrame:
    _build_knowledge_source(settings)
    _ensure_factscore_knowledge_source_consistency(settings, logger=logger)
    _write_openai_key_file(settings)

    if logger is not None:
        logger.info(
            "Running FActScore on %s rows using local source %s",
            len(df),
            settings.factscore_source_dir,
        )

    completed_keys = load_completed_keys(
        settings.factscore_results_path,
        key_columns=["question_id", "pipeline_type"],
    )
    pending_rows = [
        row
        for row in df.itertuples(index=False)
        if (str(getattr(row, "question_id")), str(getattr(row, "pipeline_type"))) not in completed_keys
    ]
    if logger is not None and completed_keys:
        logger.info("Resuming FActScore with %s completed rows already on disk", len(completed_keys))
    if logger is not None:
        logger.info(
            "Pending FActScore evaluations: %s of %s total rows (concurrency=%s)",
            len(pending_rows),
            len(df),
            settings.evaluation.factscore_concurrency,
        )

    if not pending_rows:
        result_df = finalize_csv(
            settings.factscore_results_path,
            key_columns=["question_id", "pipeline_type"],
            sort_columns=["question_id", "pipeline_type"],
        )
        if result_df.empty:
            result_df = pd.DataFrame(columns=FACTSCORE_COLUMNS)
        else:
            result_df = _normalize_factscore_result_df(result_df)
        _write_factscore_summary(result_df, settings)
        return result_df

    if settings.evaluation.factscore_concurrency <= 1:
        scorer = _create_factscorer(settings, worker_label="main")
        for index, row in enumerate(pending_rows, start=1):
            row_series = pd.Series(row._asdict())
            topic = _topic_for_row(row_series, settings)
            raw_output = "" if pd.isna(getattr(row, "answer", None)) else str(row.answer)
            output, bucket = _prepare_factscore_answer(raw_output)
            result = scorer.get_score(
                topics=[topic],
                generations=[output],
                gamma=settings.evaluation.factscore_gamma,
                knowledge_source=settings.evaluation.factscore_knowledge_source_name,
            )
            append_records_csv(
                settings.factscore_results_path,
                [_build_factscore_record(row, result, output, bucket)],
            )
            if logger is not None and (index == len(pending_rows) or index % 10 == 0):
                logger.info(
                    "FActScore progress: %s/%s completed (latest: %s %s)",
                    index,
                    len(pending_rows),
                    getattr(row, "pipeline_type"),
                    getattr(row, "question_id"),
                )
    else:
        _prebuild_factscore_knowledge_db(settings, logger=logger)
        serialized_settings = _serialize_settings_for_worker(settings)
        pending_records = [row._asdict() for row in pending_rows]
        with ProcessPoolExecutor(
            max_workers=settings.evaluation.factscore_concurrency,
            initializer=_init_factscore_worker,
            initargs=(serialized_settings,),
        ) as executor:
            future_map = {
                executor.submit(_score_factscore_row, row_dict): row_dict
                for row_dict in pending_records
            }
            for index, future in enumerate(as_completed(future_map), start=1):
                record = future.result()
                append_records_csv(settings.factscore_results_path, [record])
                if logger is not None and (index == len(future_map) or index % 10 == 0):
                    logger.info(
                        "FActScore progress: %s/%s completed (latest: %s %s)",
                        index,
                        len(future_map),
                        record["pipeline_type"],
                        record["question_id"],
                    )

    result_df = finalize_csv(
        settings.factscore_results_path,
        key_columns=["question_id", "pipeline_type"],
        sort_columns=["question_id", "pipeline_type"],
    )
    if result_df.empty:
        result_df = pd.DataFrame(columns=FACTSCORE_COLUMNS)
    else:
        result_df = _normalize_factscore_result_df(result_df)
    _write_factscore_summary(result_df, settings)
    return result_df
