from __future__ import annotations

from pathlib import Path
from fnmatch import fnmatch

from src.config.schemas import RawDocumentRecord
from src.config.settings import AppSettings
from src.evaluation.dataset_source import load_question_dataframe


def load_markdown_documents(settings: AppSettings) -> list[RawDocumentRecord]:
    documents: list[RawDocumentRecord] = []
    exclude_globs = settings.dataset.exclude_globs
    for path in sorted(settings.raw_dir.glob(settings.dataset.raw_glob)):
        if path.suffix.lower() != ".md":
            continue
        relative_path = str(path.relative_to(settings.project_root))
        if any(fnmatch(relative_path, pattern) or fnmatch(path.name, pattern) for pattern in exclude_globs):
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(
            RawDocumentRecord(
                text=text,
                metadata={
                    "source_file": relative_path,
                    "source_type": "md",
                    "logical_page": 1,
                },
            )
        )
    if not documents:
        raise FileNotFoundError(
            f"No markdown documents found under {settings.raw_dir} matching {settings.dataset.raw_glob}"
        )
    return documents


def load_factscore_reference_documents(settings: AppSettings) -> list[RawDocumentRecord]:
    qa_df = load_question_dataframe(settings)
    required_columns = {"topic", "reference_output"}
    missing = required_columns - set(qa_df.columns)
    if missing:
        raise ValueError(
            "factscore_jsonl knowledge source requires columns "
            f"{sorted(required_columns)}, missing {sorted(missing)}"
        )

    documents: list[RawDocumentRecord] = []
    seen_topics: set[str] = set()
    for row in qa_df.itertuples(index=False):
        topic = "" if row.topic is None else str(row.topic).strip()
        text = "" if row.reference_output is None else str(row.reference_output).strip()
        if not topic or not text or topic in seen_topics:
            continue
        seen_topics.add(topic)
        content = f"{topic}\n\n{text}"
        documents.append(
            RawDocumentRecord(
                text=content,
                metadata={
                    "source_file": f"factscore::{topic}",
                    "source_type": "factscore_reference_output",
                    "logical_page": 1,
                    "topic": topic,
                },
            )
        )
    if not documents:
        raise FileNotFoundError(
            "No factscore reference documents could be built from dataset topics/reference_output."
        )
    return documents


def load_documents(settings: AppSettings) -> list[RawDocumentRecord]:
    if settings.dataset.kind == "factscore_jsonl":
        return load_factscore_reference_documents(settings)
    return load_markdown_documents(settings)


def load_single_markdown(path: Path) -> RawDocumentRecord:
    return RawDocumentRecord(
        text=path.read_text(encoding="utf-8"),
        metadata={"source_file": path.name, "source_type": "md", "logical_page": 1},
    )
