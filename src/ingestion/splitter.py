from __future__ import annotations

from src.config.schemas import ChunkRecord, RawDocumentRecord
from src.config.settings import AppSettings
from src.utils.ids import make_chunk_id


def split_documents(documents: list[RawDocumentRecord], settings: AppSettings) -> list[ChunkRecord]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
    )

    records: list[ChunkRecord] = []
    for document in documents:
        chunks = splitter.split_text(document.text)
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
