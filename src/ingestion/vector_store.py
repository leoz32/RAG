from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from src.config.schemas import ChunkRecord
from src.config.settings import AppSettings
from src.ingestion.embedder import EmbeddingClient
from src.utils.io import ensure_dir, read_jsonl, write_json


def _chunks_signature(chunks: list[ChunkRecord]) -> str:
    digest = hashlib.sha1()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
    return digest.hexdigest()


def _load_index_state(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _expected_index_meta(settings: AppSettings) -> dict:
    return {
        "course_name": settings.dataset.course_name,
        "dataset_version": settings.dataset.dataset_version,
        "raw_dir": str(settings.raw_dir),
        "qa_dataset_path": str(settings.qa_dataset_path),
        "embedding_model_name": settings.embedding.model_name,
        "embedding_dimension": settings.embedding.dimension,
        "raw_glob": settings.dataset.raw_glob,
        "exclude_globs": settings.dataset.exclude_globs,
        "chunk_size": settings.chunking.chunk_size,
        "chunk_overlap": settings.chunking.chunk_overlap,
        "chunking_strategy": settings.chunking.strategy,
        "target_sentences_per_chunk": settings.chunking.target_sentences_per_chunk,
        "sentence_overlap": settings.chunking.sentence_overlap,
        "max_words_per_chunk": settings.chunking.max_words_per_chunk,
    }


def _validate_index_meta(settings: AppSettings, index_meta: dict) -> None:
    expected = _expected_index_meta(settings)
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = index_meta.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected={expected_value!r}, actual={actual_value!r}")
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "FAISS index metadata does not match the current experiment configuration. "
            f"Index dir: {settings.faiss_index_dir}. {mismatch_text}"
        )


def _append_embedding_cache(path: Path, payloads: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_embedding_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {str(row["chunk_id"]): row for row in read_jsonl(path) if "chunk_id" in row}


def _expected_embedding_cache_meta(settings: AppSettings) -> dict:
    return {
        "embedding_provider": settings.embedding.provider,
        "embedding_model_name": settings.embedding.model_name,
        "embedding_base_url": settings.embedding.base_url,
        "embedding_dimension": settings.embedding.dimension,
        "embedding_max_input_tokens": settings.embedding.max_input_tokens,
        "embedding_request_batch_size": settings.embedding.request_batch_size,
    }


def _cache_meta_matches(settings: AppSettings) -> bool:
    cache_meta = _load_index_state(settings.embedding_cache_meta_path)
    expected = _expected_embedding_cache_meta(settings)
    return all(cache_meta.get(key) == value for key, value in expected.items())


def _average_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("Cannot average zero embedding vectors")
    size = len(vectors[0])
    averaged = [0.0] * size
    for vector in vectors:
        if len(vector) != size:
            raise ValueError("Embedding vectors have inconsistent dimensions")
        for index, value in enumerate(vector):
            averaged[index] += float(value)
    averaged = [value / len(vectors) for value in averaged]
    norm = math.sqrt(sum(value * value for value in averaged))
    if norm > 0:
        averaged = [value / norm for value in averaged]
    return averaged


def _split_text_for_embedding(text: str) -> tuple[str, str]:
    words = text.split()
    if len(words) <= 1:
        midpoint = max(1, len(text) // 2)
        return text[:midpoint].strip(), text[midpoint:].strip()
    midpoint = len(words) // 2
    left = " ".join(words[:midpoint]).strip()
    right = " ".join(words[midpoint:]).strip()
    return left, right


def _is_input_too_long_error(exc: Exception) -> bool:
    message = str(exc)
    return "less than 512 tokens" in message or "input must have less than 512 tokens" in message


def _embed_text_resilient(
    embedding_client,
    text: str,
    *,
    logger=None,
    chunk_id: str | None = None,
    depth: int = 0,
) -> list[float]:
    try:
        return embedding_client.embed_documents([text])[0]
    except Exception as exc:
        if not _is_input_too_long_error(exc):
            raise
        left, right = _split_text_for_embedding(text)
        if not left or not right or left == text or right == text:
            raise
        if logger is not None and depth == 0:
            logger.warning(
                "Chunk %s exceeded embedding token limit; recursively splitting for embedding fallback",
                chunk_id or "<unknown>",
            )
        vectors = [
            _embed_text_resilient(embedding_client, left, logger=logger, chunk_id=chunk_id, depth=depth + 1),
            _embed_text_resilient(embedding_client, right, logger=logger, chunk_id=chunk_id, depth=depth + 1),
        ]
        return _average_vectors(vectors)


def build_faiss_index(chunks: list[ChunkRecord], settings: AppSettings, logger=None, batch_size: int = 32) -> None:
    if not chunks:
        raise ValueError("No chunks provided for index build")

    try:
        from langchain_core.documents import Document
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:
        raise ImportError("langchain-community is required to build the FAISS index") from exc

    chunk_signature = _chunks_signature(chunks)
    index_state = _load_index_state(settings.index_build_state_path)
    index_file = settings.faiss_index_dir / "index.faiss"
    expected_meta = _expected_index_meta(settings)
    metadata_matches = all(index_state.get(key) == value for key, value in expected_meta.items())
    chunk_matches = (
        index_state.get("chunk_count") == len(chunks)
        and index_state.get("chunk_signature") == chunk_signature
    )
    faiss_is_fresh = (
        index_file.exists()
        and index_file.stat().st_mtime >= settings.chunks_path.stat().st_mtime
        and index_file.stat().st_mtime >= settings.index_build_state_path.stat().st_mtime
    )

    if settings.faiss_index_dir.exists() and index_state.get("complete") is True and metadata_matches and chunk_matches and faiss_is_fresh:
        if logger is not None:
            logger.info("FAISS index already complete at %s, skipping embedding build", settings.faiss_index_dir)
        return

    if logger is not None and settings.faiss_index_dir.exists():
        reasons: list[str] = []
        if index_state.get("complete") is not True:
            reasons.append("index metadata is incomplete")
        if not metadata_matches:
            reasons.append("index metadata does not match current chunking/embedding configuration")
        if not chunk_matches:
            reasons.append("chunk signature or chunk count changed")
        if not faiss_is_fresh:
            reasons.append("index.faiss is missing or older than the current chunks/metadata")
        logger.info("Rebuilding FAISS index at %s because %s", settings.faiss_index_dir, "; ".join(reasons))

    embedding_client = EmbeddingClient(settings).client
    cache_meta_matches = _cache_meta_matches(settings)
    if cache_meta_matches:
        cached_embeddings = _load_embedding_cache(settings.embedding_cache_path)
    else:
        cached_embeddings = {}
        if settings.embedding_cache_path.exists():
            settings.embedding_cache_path.unlink()
        if logger is not None:
            logger.info(
                "Discarding embedding cache at %s because provider/model settings changed",
                settings.embedding_cache_path,
            )
    valid_chunk_ids = {chunk.chunk_id for chunk in chunks}
    cached_embeddings = {
        chunk_id: payload for chunk_id, payload in cached_embeddings.items() if chunk_id in valid_chunk_ids
    }

    pending_chunks = [chunk for chunk in chunks if chunk.chunk_id not in cached_embeddings]
    if logger is not None:
        logger.info(
            "Embedding cache status: %s completed, %s pending",
            len(cached_embeddings),
            len(pending_chunks),
        )

    for start in range(0, len(pending_chunks), batch_size):
        batch = pending_chunks[start : start + batch_size]
        vectors = [
            _embed_text_resilient(
                embedding_client,
                chunk.text,
                logger=logger,
                chunk_id=chunk.chunk_id,
            )
            for chunk in batch
        ]
        cache_payloads = []
        for chunk, vector in zip(batch, vectors):
            payload = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "embedding": vector,
                "metadata": {
                    "source_file": chunk.source_file,
                    "logical_page": chunk.logical_page,
                },
            }
            cached_embeddings[chunk.chunk_id] = payload
            cache_payloads.append(payload)
        _append_embedding_cache(settings.embedding_cache_path, cache_payloads)
        if logger is not None:
            logger.info(
                "Embedded chunks %s-%s / %s",
                start + 1,
                start + len(batch),
                len(pending_chunks),
            )

    documents = []
    text_embeddings = []
    metadatas = []
    ids = []
    for chunk in chunks:
        cached_payload = cached_embeddings.get(chunk.chunk_id)
        if cached_payload is None:
            raise ValueError(f"Missing cached embedding for chunk_id={chunk.chunk_id}")
        documents.append(
            Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "logical_page": chunk.logical_page,
                },
            )
        )
        text_embeddings.append((chunk.text, cached_payload["embedding"]))
        metadatas.append(documents[-1].metadata)
        ids.append(chunk.chunk_id)

    vector_store = FAISS.from_embeddings(text_embeddings, embedding_client, metadatas=metadatas, ids=ids)
    vector_store.save_local(str(settings.faiss_index_dir))
    write_json(settings.embedding_cache_meta_path, _expected_embedding_cache_meta(settings))
    write_json(
        settings.index_build_state_path,
        {
            "complete": True,
            **expected_meta,
            "chunk_count": len(chunks),
            "chunk_signature": chunk_signature,
            "embedding_cache_path": str(settings.embedding_cache_path),
            "faiss_index_dir": str(settings.faiss_index_dir),
        },
    )


def load_faiss_index(settings: AppSettings):
    if not settings.faiss_index_dir.exists():
        raise FileNotFoundError(f"FAISS index directory not found: {settings.faiss_index_dir}")
    if not settings.index_build_state_path.exists():
        raise FileNotFoundError(
            f"FAISS index metadata not found: {settings.index_build_state_path}. Rebuild the index first."
        )

    try:
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:
        raise ImportError("langchain-community is required to load the FAISS index") from exc

    index_meta = _load_index_state(settings.index_build_state_path)
    _validate_index_meta(settings, index_meta)

    embedding_client = EmbeddingClient(settings).client
    return FAISS.load_local(
        str(settings.faiss_index_dir),
        embedding_client,
        allow_dangerous_deserialization=True,
    )
