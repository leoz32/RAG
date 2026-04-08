from __future__ import annotations

from src.config.schemas import ChunkRecord
from src.config.settings import AppSettings
from src.ingestion.embedder import EmbeddingClient


def build_faiss_index(chunks: list[ChunkRecord], settings: AppSettings) -> None:
    if not chunks:
        raise ValueError("No chunks provided for index build")

    try:
        from langchain_core.documents import Document
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:
        raise ImportError("langchain-community is required to build the FAISS index") from exc

    embedding_client = EmbeddingClient(settings).client
    documents = [
        Document(
            page_content=chunk.text,
            metadata={
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "logical_page": chunk.logical_page,
            },
        )
        for chunk in chunks
    ]
    vector_store = FAISS.from_documents(documents, embedding_client)
    vector_store.save_local(str(settings.faiss_index_dir))


def load_faiss_index(settings: AppSettings):
    if not settings.faiss_index_dir.exists():
        raise FileNotFoundError(f"FAISS index directory not found: {settings.faiss_index_dir}")

    try:
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:
        raise ImportError("langchain-community is required to load the FAISS index") from exc

    embedding_client = EmbeddingClient(settings).client
    return FAISS.load_local(
        str(settings.faiss_index_dir),
        embedding_client,
        allow_dangerous_deserialization=True,
    )
