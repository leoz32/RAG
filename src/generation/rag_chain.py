from __future__ import annotations

from datetime import datetime, timezone

from src.config.schemas import GenerationRecord, QuestionRecord
from src.config.settings import AppSettings
from src.generation.llm_client import LLMClient
from src.generation.prompts import build_rag_prompt
from src.retrieval.retriever import Retriever
from src.utils.enums import PipelineType


class RagChain:
    def __init__(
        self,
        settings: AppSettings,
        retriever: Retriever | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever or Retriever(settings)
        self.llm_client = llm_client or LLMClient(settings)

    def run(self, question: QuestionRecord) -> GenerationRecord:
        retrievals = self.retriever.retrieve(question.question, question.question_id)
        contexts = [record.text for record in retrievals]
        context_ids = [record.chunk_id for record in retrievals]
        prompt = build_rag_prompt(question.question, "\n\n".join(contexts))
        answer = self.llm_client.generate(prompt)
        return GenerationRecord(
            experiment_id=self.settings.experiment_id,
            question_id=question.question_id,
            question=question.question,
            ground_truth=question.ground_truth,
            question_type=question.question_type,
            pipeline_type=PipelineType.RAG,
            answer=answer,
            contexts=contexts,
            context_ids=context_ids,
            prompt=prompt,
            prompt_version=self.settings.prompt_version,
            llm_model_name=self.settings.llm.model_name,
            embedding_model_name=self.settings.embedding.model_name,
            chunk_size=self.settings.chunking.chunk_size,
            chunk_overlap=self.settings.chunking.chunk_overlap,
            retrieval_top_k=self.settings.retrieval.top_k,
            dataset_version=self.settings.dataset.dataset_version,
            timestamp=datetime.now(timezone.utc),
        )
