from __future__ import annotations

from collections import defaultdict
import re

from src.config.schemas import RetrievalRecord
from src.config.settings import AppSettings
from src.retrieval.bm25_retriever import BM25Retriever
from src.ingestion.vector_store import load_faiss_index


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "the",
    "their",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "why",
    "with",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})s?\b")
TITLE_CASE_RE = re.compile(r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)?\b")
ALL_CAPS_RE = re.compile(r"\b[A-Z]{2,}(?:-[A-Z]{2,})?\b")


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

    def _extract_keywords(self, question: str) -> list[str]:
        keywords: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            normalized = value.strip()
            if not normalized:
                return
            key = normalized.lower()
            if key in seen:
                return
            seen.add(key)
            keywords.append(normalized)

        for match in YEAR_RE.findall(question):
            add(match)

        for match in re.findall(r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b", question):
            add(match)

        for match in ALL_CAPS_RE.findall(question):
            add(match)

        for match in TITLE_CASE_RE.findall(question):
            add(match)

        for token in TOKEN_RE.findall(question):
            lowered = token.lower()
            if lowered in STOPWORDS:
                continue
            if len(lowered) <= 2 and not lowered.isdigit():
                continue
            add(token)

        return keywords

    def _keyword_coverage_score(self, question: str, text: str) -> float:
        keywords = self._extract_keywords(question)
        if not keywords:
            return 0.0

        lowered_text = text.lower()
        total_weight = 0.0
        matched_weight = 0.0
        matched = 0

        def keyword_weight(keyword: str) -> float:
            if YEAR_RE.fullmatch(keyword):
                return 2.0
            if "-" in keyword:
                return 2.5
            if keyword[:1].isupper() or keyword.isupper():
                return 1.5
            return 1.0

        for keyword in keywords:
            weight = keyword_weight(keyword)
            total_weight += weight
            if keyword.lower() in lowered_text:
                matched += 1
                matched_weight += weight

        if total_weight <= 0:
            return 0.0

        coverage = matched_weight / total_weight
        exact_bonus = 0.02 * matched
        return coverage + exact_bonus

    def _rerank_with_keyword_coverage(
        self,
        question: str,
        records: list[RetrievalRecord],
    ) -> list[RetrievalRecord]:
        reranked = []
        for record in records:
            keyword_score = self._keyword_coverage_score(question, record.text)
            final_score = float(record.score) + keyword_score
            reranked.append(record.model_copy(update={"score": final_score}))

        reranked.sort(key=lambda record: record.score, reverse=True)
        return reranked[: self.settings.retrieval.top_k]

    def retrieve(self, question: str, question_id: str) -> list[RetrievalRecord]:
        strategy = self.settings.retrieval.strategy.strip().lower()
        if strategy == "dense":
            return self._dense_retrieve(question, question_id)[: self.settings.retrieval.top_k]
        if strategy == "bm25":
            return self.bm25_retriever.retrieve(question, question_id, k=self.settings.retrieval.top_k)
        if strategy == "hybrid_rrf":
            dense_records = self._dense_retrieve(question, question_id)
            bm25_records = self.bm25_retriever.retrieve(question, question_id)
            merged = self._rrf_merge(dense_records, bm25_records)
            return self._rerank_with_keyword_coverage(question, merged)
        raise ValueError(f"Unsupported retrieval strategy: {self.settings.retrieval.strategy}")
