from __future__ import annotations

import hashlib

import numpy as np
from langchain_core.embeddings import Embeddings

from src.config.settings import AppSettings


class LocalHashEmbeddings(Embeddings):
    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = text.split()
        if not tokens:
            return vector.tolist()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], byteorder="big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


class EmbeddingClient:
    def __init__(self, settings: AppSettings) -> None:
        if settings.embedding.provider == "local_hash":
            self._client = LocalHashEmbeddings(dimension=settings.embedding.dimension)
            return

        if not settings.embedding.api_key:
            raise ValueError(
                f"Missing embedding API key. Set environment variable {settings.embedding.api_key_env}."
            )
        if settings.embedding.provider not in {"openai", "openai_compatible"}:
            raise ValueError(f"Unsupported embedding provider: {settings.embedding.provider}")

        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise ImportError("langchain-openai is required for OpenAI-compatible embeddings") from exc

        self._client = OpenAIEmbeddings(
            model=settings.embedding.model_name,
            api_key=settings.embedding.api_key,
            base_url=settings.embedding.base_url,
        )

    @property
    def client(self):
        return self._client
