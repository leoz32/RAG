from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import AppSettings, load_settings
from src.ingestion.cleaner import clean_text
from src.ingestion.loaders import load_markdown_documents
from src.ingestion.splitter import split_documents
from src.ingestion.vector_store import build_faiss_index
from src.utils.io import write_jsonl
from src.utils.logger import get_logger


def run_build_index(settings: AppSettings) -> None:
    logger = get_logger("build_index", settings.log_file)
    logger.info("Loading markdown documents from %s", settings.raw_dir)
    documents = load_markdown_documents(settings)
    cleaned_documents = [doc.model_copy(update={"text": clean_text(doc.text)}) for doc in documents]
    chunks = split_documents(cleaned_documents, settings)
    write_jsonl(settings.chunks_path, chunks)
    logger.info("Saved %s chunks to %s", len(chunks), settings.chunks_path)
    build_faiss_index(chunks, settings)
    logger.info("Saved FAISS index to %s", settings.faiss_index_dir)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--experiment-id", default=None)
    args = parser.parse_args()
    settings = load_settings(args.config, experiment_id=args.experiment_id)
    run_build_index(settings)


if __name__ == "__main__":
    main()
