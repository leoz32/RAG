from __future__ import annotations

import re

from src.config.schemas import ChunkRecord, RawDocumentRecord
from src.config.settings import AppSettings
from src.utils.ids import make_chunk_id


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+")
WORD_RE = re.compile(r"\S+")


def _split_into_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    if len(sentences) <= 1:
        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        return paragraphs or [text]
    return sentences


def _sentence_window_chunks(text: str, settings: AppSettings) -> list[str]:
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    target = settings.chunking.target_sentences_per_chunk
    overlap = settings.chunking.sentence_overlap
    step = max(1, target - overlap)

    chunks: list[str] = []
    for start in range(0, len(sentences), step):
        window = sentences[start : start + target]
        if not window:
            continue
        chunk_text = " ".join(window).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if start + target >= len(sentences):
            break
    return chunks


def _split_long_chunk_by_words(text: str, max_words: int) -> list[str]:
    words = WORD_RE.findall(text)
    if len(words) <= max_words:
        return [text.strip()] if text.strip() else []

    parts: list[str] = []
    for start in range(0, len(words), max_words):
        piece = " ".join(words[start : start + max_words]).strip()
        if piece:
            parts.append(piece)
    return parts


def _character_chunks(text: str, settings: AppSettings) -> list[str]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
    )
    return splitter.split_text(text)


def _split_text(text: str, settings: AppSettings) -> list[str]:
    strategy = settings.chunking.strategy.strip().lower()
    if strategy == "sentence_window":
        chunks = _sentence_window_chunks(text, settings)
        bounded: list[str] = []
        for chunk in chunks:
            bounded.extend(_split_long_chunk_by_words(chunk, settings.chunking.max_words_per_chunk))
        return bounded
    if strategy == "recursive_character":
        return _character_chunks(text, settings)
    raise ValueError(f"Unsupported chunking strategy: {settings.chunking.strategy}")


def split_documents(documents: list[RawDocumentRecord], settings: AppSettings) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for document in documents:
        # Files remain the top-level chunking boundary; strategy applies within each file.
        chunks = _split_text(document.text, settings)
        for chunk_index, chunk_text in enumerate(chunks):
            source_file = str(document.metadata["source_file"])
            logical_page = int(document.metadata.get("logical_page", 1))
            records.append(
                ChunkRecord(
                    experiment_id=settings.experiment_id,
                    chunk_id=make_chunk_id(source_file, logical_page, chunk_index, chunk_text),
                    text=chunk_text,
                    source_file=source_file,
                    source_type=str(document.metadata.get("source_type", "md")),
                    logical_page=logical_page,
                    chunk_index=chunk_index,
                    chunk_size=settings.chunking.chunk_size,
                    chunk_overlap=settings.chunking.chunk_overlap,
                    embedding_model_name=settings.embedding.model_name,
                    dataset_version=settings.dataset.dataset_version,
                )
            )
    return records
