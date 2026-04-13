from __future__ import annotations

from collections import defaultdict

from src.config.schemas import RetrievalRecord
from src.config.settings import AppSettings
from src.retrieval.bm25_retriever import BM25Retriever
from src.ingestion.vector_store import load_faiss_index


class Retriever:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.vector_store = load_faiss_index(settings)
        self.bm25_retriever = BM25Retriever(settings)

    def _dense_retrieve(self, question: str, question_id: str) -> list[RetrievalRecord]:
        results = self.vector_store.similarity_search_with_score(question, k=self.settings.retrieval.dense_top_k)
        records: list[RetrievalRecord] = []
        for document, score in results:
            records.append(
                RetrievalRecord(
                    experiment_id=self.settings.experiment_id,
                    question_id=question_id,
                    question=question,
                    chunk_id=str(document.metadata.get("chunk_id", "")),
                    text=document.page_content,
                    score=float(score),
                    source_file=str(document.metadata.get("source_file", "")),
                    logical_page=int(document.metadata.get("logical_page", 1)),
                )
            )
        return records

    def _rrf_merge(
        self,
        dense_records: list[RetrievalRecord],
        bm25_records: list[RetrievalRecord],
    ) -> list[RetrievalRecord]:
        scores: dict[str, float] = defaultdict(float)
        payloads: dict[str, RetrievalRecord] = {}
        rrf_k = self.settings.retrieval.rrf_k

        for rank, record in enumerate(dense_records, start=1):
            scores[record.chunk_id] += 1.0 / (rrf_k + rank)
            payloads.setdefault(record.chunk_id, record)

        for rank, record in enumerate(bm25_records, start=1):
            scores[record.chunk_id] += 1.0 / (rrf_k + rank)
            payloads.setdefault(record.chunk_id, record)

        ranked_chunk_ids = sorted(scores.keys(), key=lambda chunk_id: scores[chunk_id], reverse=True)
        merged: list[RetrievalRecord] = []
        for chunk_id in ranked_chunk_ids[: self.settings.retrieval.top_k]:
            base = payloads[chunk_id]
            merged.append(base.model_copy(update={"score": float(scores[chunk_id])}))
        return merged

    def retrieve(self, question: str, question_id: str) -> list[RetrievalRecord]:
        strategy = self.settings.retrieval.strategy.strip().lower()
        if strategy == "dense":
            return self._dense_retrieve(question, question_id)[: self.settings.retrieval.top_k]
        if strategy == "bm25":
            return self.bm25_retriever.retrieve(question, question_id, k=self.settings.retrieval.top_k)
        if strategy == "hybrid_rrf":
            dense_records = self._dense_retrieve(question, question_id)
            bm25_records = self.bm25_retriever.retrieve(question, question_id)
            return self._rrf_merge(dense_records, bm25_records)
        raise ValueError(f"Unsupported retrieval strategy: {self.settings.retrieval.strategy}")
