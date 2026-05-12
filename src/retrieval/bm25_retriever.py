from __future__ import annotations

import re

from src.config.schemas import ChunkRecord, RetrievalRecord
from src.config.settings import AppSettings
from src.utils.io import read_jsonl


LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
CJK_SPAN_RE = re.compile(r"[\u4e00-\u9fff]+")


def _cjk_bigrams(text: str) -> list[str]:
    compact = "".join(char for char in text if "\u4e00" <= char <= "\u9fff")
    if len(compact) <= 1:
        return [compact] if compact else []
    return [compact[index : index + 2] for index in range(len(compact) - 1)]


def _tokenize(text: str) -> list[str]:
    tokens = [token.lower() for token in LATIN_TOKEN_RE.findall(text)]
    for span in CJK_SPAN_RE.findall(text):
        tokens.extend(_cjk_bigrams(span))
    return tokens


class BM25Retriever:
    def __init__(self, settings: AppSettings) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError("rank-bm25 is required for BM25 retrieval") from exc

        rows = read_jsonl(settings.chunks_path)
        self.settings = settings
        self.chunks = [ChunkRecord.model_validate(row) for row in rows]
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.tokenized_corpus = [_tokenize(chunk.text) for chunk in self.chunks]
        if not any(tokens for tokens in self.tokenized_corpus):
            raise ValueError("BM25 corpus is empty after tokenization; check retrieval input documents.")
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, question: str, question_id: str, k: int | None = None) -> list[RetrievalRecord]:
        top_k = k or self.settings.retrieval.bm25_top_k
        query_tokens = _tokenize(question)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]

        records: list[RetrievalRecord] = []
        for index in ranked_indices:
            chunk = self.chunks[index]
            records.append(
                RetrievalRecord(
                    experiment_id=self.settings.experiment_id,
                    question_id=question_id,
                    question=question,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(scores[index]),
                    source_file=chunk.source_file,
                    logical_page=chunk.logical_page,
                )
            )
        return records
