from __future__ import annotations

from src.config.schemas import RetrievalRecord
from src.config.settings import AppSettings
from src.ingestion.vector_store import load_faiss_index


class Retriever:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.vector_store = load_faiss_index(settings)

    def retrieve(self, question: str, question_id: str) -> list[RetrievalRecord]:
        results = self.vector_store.similarity_search_with_score(question, k=self.settings.retrieval.top_k)
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
