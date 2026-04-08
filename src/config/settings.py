from __future__ import annotations

import os
import random
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv() -> bool:
        return False
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.utils.ids import make_experiment_id
from src.utils.io import ensure_dir, read_yaml, write_yaml


class PathSettings(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    eval_dir: str = "data/eval"
    vector_store_dir: str = "data/vector_store"
    results_dir: str = "results"
    runs_dir: str = "results/runs"
    reports_dir: str = "results/reports"
    figures_dir: str = "results/figures"


class LLMSettings(BaseModel):
    provider: str = "openai"
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 800
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    base_url: str | None = None


class EmbeddingSettings(BaseModel):
    provider: str = "openai"
    model_name: str
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    base_url: str | None = None
    dimension: int = 256


class ChunkingSettings(BaseModel):
    chunk_size: int = 700
    chunk_overlap: int = 120


class EvaluationSettings(BaseModel):
    llm_max_tokens: int = 4096


class RetrievalSettings(BaseModel):
    top_k: int = 4


class DatasetSettings(BaseModel):
    course_name: str
    dataset_version: str
    qa_dataset_path: str = "data/eval/qa_dataset.csv"
    raw_glob: str = "*.md"


class AppSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    project_name: str = "rag_hallucination_eval"
    project_root: Path
    experiment_id: str
    random_seed: int = 42
    prompt_version: str = "v1"
    llm: LLMSettings
    embedding: EmbeddingSettings
    chunking: ChunkingSettings
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    retrieval: RetrievalSettings
    dataset: DatasetSettings
    paths: PathSettings = Field(default_factory=PathSettings)

    @model_validator(mode="after")
    def validate_overlap(self) -> "AppSettings":
        if self.chunking.chunk_overlap >= self.chunking.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @property
    def raw_dir(self) -> Path:
        return self.project_root / self.paths.raw_dir

    @property
    def processed_dir(self) -> Path:
        return self.project_root / self.paths.processed_dir

    @property
    def eval_dir(self) -> Path:
        return self.project_root / self.paths.eval_dir

    @property
    def vector_store_dir(self) -> Path:
        return self.project_root / self.paths.vector_store_dir

    @property
    def runs_dir(self) -> Path:
        return self.project_root / self.paths.runs_dir

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.experiment_id

    @property
    def log_file(self) -> Path:
        return self.run_dir / "experiment.log"

    @property
    def config_snapshot_path(self) -> Path:
        return self.run_dir / "config_snapshot.yaml"

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.jsonl"

    @property
    def faiss_index_dir(self) -> Path:
        return self.vector_store_dir / "faiss_index"

    @property
    def qa_dataset_path(self) -> Path:
        return self.project_root / self.dataset.qa_dataset_path

    @property
    def generation_results_path(self) -> Path:
        return self.run_dir / "generation_results.csv"

    @property
    def baseline_eval_results_path(self) -> Path:
        return self.run_dir / "baseline_eval_results.csv"

    @property
    def rag_eval_results_path(self) -> Path:
        return self.run_dir / "rag_eval_results.csv"

    @property
    def question_level_results_path(self) -> Path:
        return self.run_dir / "question_level_results.csv"

    @property
    def metrics_summary_json_path(self) -> Path:
        return self.run_dir / "metrics_summary.json"

    @property
    def metrics_summary_csv_path(self) -> Path:
        return self.run_dir / "metrics_summary.csv"

    @property
    def pipeline_summary_csv_path(self) -> Path:
        return self.run_dir / "pipeline_summary.csv"

    @property
    def question_type_summary_csv_path(self) -> Path:
        return self.run_dir / "question_type_summary.csv"

    @property
    def analysis_report_path(self) -> Path:
        return self.run_dir / "analysis_report.json"

    def snapshot(self) -> dict:
        payload = self.model_dump(mode="json")
        payload["project_root"] = str(self.project_root)
        payload["resolved_paths"] = {
            "raw_dir": str(self.raw_dir),
            "processed_dir": str(self.processed_dir),
            "eval_dir": str(self.eval_dir),
            "vector_store_dir": str(self.vector_store_dir),
            "runs_dir": str(self.runs_dir),
            "run_dir": str(self.run_dir),
            "qa_dataset_path": str(self.qa_dataset_path),
        }
        return payload


def _resolve_api_keys(config: dict) -> dict:
    llm = config.get("llm", {})
    embedding = config.get("embedding", {})
    llm_env = llm.get("api_key_env", "OPENAI_API_KEY")
    emb_env = embedding.get("api_key_env", "OPENAI_API_KEY")
    llm["api_key"] = os.getenv(llm_env, llm.get("api_key"))
    embedding["api_key"] = os.getenv(emb_env, embedding.get("api_key"))
    config["llm"] = llm
    config["embedding"] = embedding
    return config


def _prepare_directories(settings: AppSettings) -> None:
    for directory in (
        settings.raw_dir,
        settings.processed_dir,
        settings.eval_dir,
        settings.vector_store_dir,
        settings.runs_dir,
        settings.run_dir,
    ):
        ensure_dir(directory)


def load_settings(config_path: str | Path, experiment_id: str | None = None) -> AppSettings:
    load_dotenv()
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw_config = read_yaml(config_path)
    raw_config = _resolve_api_keys(raw_config)
    raw_config["project_root"] = config_path.parents[1]
    raw_config["experiment_id"] = experiment_id or raw_config.get("experiment_id") or make_experiment_id()

    settings = AppSettings.model_validate(raw_config)
    random.seed(settings.random_seed)
    _prepare_directories(settings)
    write_yaml(settings.config_snapshot_path, settings.snapshot())
    return settings
