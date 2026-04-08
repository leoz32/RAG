from __future__ import annotations

from pathlib import Path

from src.config.schemas import RawDocumentRecord
from src.config.settings import AppSettings


def load_markdown_documents(settings: AppSettings) -> list[RawDocumentRecord]:
    documents: list[RawDocumentRecord] = []
    for path in sorted(settings.raw_dir.glob(settings.dataset.raw_glob)):
        if path.suffix.lower() != ".md":
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(
            RawDocumentRecord(
                text=text,
                metadata={
                    "source_file": str(path.relative_to(settings.project_root)),
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


def load_single_markdown(path: Path) -> RawDocumentRecord:
    return RawDocumentRecord(
        text=path.read_text(encoding="utf-8"),
        metadata={"source_file": path.name, "source_type": "md", "logical_page": 1},
    )
