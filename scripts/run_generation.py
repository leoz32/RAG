from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.schemas import QuestionRecord
from src.config.settings import AppSettings, load_settings
from src.generation.baseline_chain import BaselineChain
from src.generation.llm_client import LLMClient
from src.generation.rag_chain import RagChain
from src.retrieval.retriever import Retriever
from src.utils.io import read_csv, write_csv
from src.utils.logger import get_logger


def _load_questions(settings: AppSettings) -> list[QuestionRecord]:
    qa_df = read_csv(settings.qa_dataset_path)
    required_columns = {"question_id", "question", "ground_truth", "question_type"}
    missing = required_columns - set(qa_df.columns)
    if missing:
        raise ValueError(f"qa_dataset.csv missing required columns: {sorted(missing)}")
    return [QuestionRecord.model_validate(row) for row in qa_df.to_dict(orient="records")]


def run_generation(settings: AppSettings) -> pd.DataFrame:
    logger = get_logger("run_generation", settings.log_file)
    questions = _load_questions(settings)
    logger.info("Loaded %s evaluation questions", len(questions))

    llm_client = LLMClient(settings)
    baseline_chain = BaselineChain(settings, llm_client=llm_client)
    rag_chain = RagChain(settings, retriever=Retriever(settings), llm_client=llm_client)

    results = []
    for question in tqdm(questions, desc="Generating answers"):
        results.append(baseline_chain.run(question).model_dump(mode="json"))
        results.append(rag_chain.run(question).model_dump(mode="json"))

    results_df = pd.DataFrame(results)
    write_csv(settings.generation_results_path, results_df)
    logger.info("Saved generation results to %s", settings.generation_results_path)
    return results_df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--experiment-id", default=None)
    args = parser.parse_args()
    settings = load_settings(args.config, experiment_id=args.experiment_id)
    run_generation(settings)


if __name__ == "__main__":
    main()
