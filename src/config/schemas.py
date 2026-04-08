from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.utils.enums import PipelineType


class RawDocumentRecord(BaseModel):
    text: str
    metadata: dict[str, Any]


class QuestionRecord(BaseModel):
    question_id: str
    question: str
    ground_truth: str
    question_type: str


class ChunkRecord(BaseModel):
    experiment_id: str
    chunk_id: str
    text: str
    source_file: str
    source_type: str
    logical_page: int = 1
    chunk_index: int
    chunk_size: int
    chunk_overlap: int
    embedding_model_name: str
    dataset_version: str


class RetrievalRecord(BaseModel):
    experiment_id: str
    question_id: str
    question: str
    chunk_id: str
    text: str
    score: float
    source_file: str
    logical_page: int = 1
    pipeline_type: PipelineType = PipelineType.RAG


class GenerationRecord(BaseModel):
    experiment_id: str
    question_id: str
    question: str
    ground_truth: str | None = None
    question_type: str | None = None
    pipeline_type: PipelineType
    answer: str
    contexts: list[str] = Field(default_factory=list)
    context_ids: list[str] = Field(default_factory=list)
    prompt: str
    prompt_version: str
    llm_model_name: str
    embedding_model_name: str
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    dataset_version: str
    evaluation_contexts: list[str] = Field(default_factory=list)
    hallucination_label: str | None = None
    reviewer_notes: str | None = None
    timestamp: datetime


class EvaluationRecord(BaseModel):
    experiment_id: str
    question_id: str
    question: str
    ground_truth: str
    question_type: str
    pipeline_type: PipelineType
    answer: str
    contexts: list[str] = Field(default_factory=list)
    evaluation_contexts: list[str] = Field(default_factory=list)
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    hallucination_label: str | None = None
    reviewer_notes: str | None = None


class ExperimentSummary(BaseModel):
    experiment_id: str
    course_name: str
    overall_best_pipeline: str
    faithfulness_gain: float | None = None
    answer_relevance_delta: float | None = None
    adversarial_refusal_rate_baseline: float | None = None
    adversarial_refusal_rate_rag: float | None = None
    question_type_breakdown: dict[str, Any]
    notes: list[str] = Field(default_factory=list)


def schema_columns(model_cls: type[BaseModel]) -> list[str]:
    return list(model_cls.model_fields.keys())
