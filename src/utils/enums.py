from enum import Enum


class PipelineType(str, Enum):
    BASELINE = "baseline"
    RAG = "rag"


class QuestionType(str, Enum):
    FACTOID = "factoid"
    REASONING = "reasoning"
    ADVERSARIAL = "adversarial"
    OTHER = "other"
