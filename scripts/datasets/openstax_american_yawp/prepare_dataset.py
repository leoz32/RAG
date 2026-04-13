from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, write_csv, write_json


DEFAULT_REPO_ID = "ambrosfitz/openstax_american_yawp"
DEFAULT_SPLIT = "train"
DEFAULT_RAW_DIR = Path("data/raw/openstax_american_yawp")
DEFAULT_QA_PATH = Path("data/eval/openstax_american_yawp_qa_dataset.csv")
DEFAULT_METADATA_PATH = Path("data/eval/openstax_american_yawp_qa_metadata.csv")
DEFAULT_SUMMARY_PATH = Path("data/eval/openstax_american_yawp_mapping_summary.json")


FACTOID_PREFIXES = (
    "what is",
    "what was",
    "when did",
    "when was",
    "who was",
    "who were",
    "which",
    "where",
    "name ",
    "identify ",
    "define ",
    "list ",
)

REASONING_PREFIXES = (
    "how ",
    "why ",
    "to what extent",
    "compare ",
    "contrast ",
    "explain ",
    "analyze ",
    "in what ways",
    "what factors",
    "what caused",
    "what role",
    "what impact",
    "what significance",
    "what relationship",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map ambrosfitz/openstax_american_yawp into this repo's raw/eval contract."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--qa-path", default=str(DEFAULT_QA_PATH))
    parser.add_argument("--metadata-path", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def infer_question_type(question: str) -> str:
    normalized = " ".join(question.lower().strip().split())
    if normalized.startswith(REASONING_PREFIXES):
        return "reasoning"
    if normalized.startswith(FACTOID_PREFIXES):
        return "factoid"
    if "significance" in normalized or "relationship" in normalized:
        return "reasoning"
    return "factoid"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "unknown"


def _to_metadata_dict(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("source_metadata") or {}
    if isinstance(metadata, dict):
        return metadata
    return {}


def load_rows(repo_id: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required to download Hugging Face datasets.") from exc

    try:
        dataset = load_dataset(repo_id, split=split)
        return [dict(item) for item in dataset]
    except Exception as exc:
        return _load_rows_from_jsonl(repo_id=repo_id, split=split, original_error=exc)


def _load_rows_from_jsonl(repo_id: str, split: str, original_error: Exception) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Failed to use datasets.load_dataset(), and huggingface_hub is unavailable for JSONL fallback."
        ) from exc

    candidate_filenames = [
        f"{split}.jsonl",
        f"dataset/{split}.jsonl",
        f"data/{split}.jsonl",
        "qa_pairs_full_output.jsonl",
    ]

    last_error: Exception | None = None
    for filename in candidate_filenames:
        try:
            jsonl_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
            rows: list[dict[str, Any]] = []
            with Path(jsonl_path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            if rows:
                return rows
        except Exception as exc:  # pragma: no cover - depends on remote repo layout
            last_error = exc

    raise RuntimeError(
        "Unable to download dataset rows via load_dataset() or JSONL fallback. "
        f"Original load_dataset error: {original_error!r}. "
        f"Last JSONL fallback error: {last_error!r}."
    ) from last_error or original_error


def sample_rows(rows: list[dict[str, Any]], max_questions: int | None, seed: int) -> list[dict[str, Any]]:
    if max_questions is None or max_questions >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    keep = sorted(indices[:max_questions])
    return [rows[index] for index in keep]


def build_eval_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        metadata = _to_metadata_dict(row)
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue

        source = str(metadata.get("source", "unknown")).strip()
        chapter_section = str(metadata.get("chapter_section", "unknown")).strip()
        filename = str(metadata.get("filename", "unknown")).strip()
        question_hash = hashlib.sha1(f"{question}|{answer}|{filename}".encode("utf-8")).hexdigest()[:8]
        question_id = f"openstax_yawp_{index:05d}_{question_hash}"

        records.append(
            {
                "question_id": question_id,
                "question": question,
                "ground_truth": answer,
                "question_type": infer_question_type(question),
                "source_doc": source,
                "source_chapter_section": chapter_section,
                "source_filename": filename,
                "source_was_cleaned": str(metadata.get("was_cleaned", "")),
                "dataset_name": DEFAULT_REPO_ID,
            }
        )

    return pd.DataFrame(records)


def build_metadata_dataframe(eval_df: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        "question_id",
        "source_doc",
        "source_chapter_section",
        "source_filename",
        "source_was_cleaned",
        "dataset_name",
    ]
    return eval_df[metadata_columns].copy()


def build_pseudo_corpus(eval_df: pd.DataFrame, raw_dir: Path) -> list[Path]:
    ensure_dir(raw_dir)
    written_paths: list[Path] = []

    grouped = eval_df.groupby(["source_doc", "source_chapter_section", "source_filename"], dropna=False)
    for (source_doc, chapter_section, source_filename), group in grouped:
        source_slug = _slugify(str(source_doc))
        chapter_slug = _slugify(f"chapter_{chapter_section}")
        filename_stem = Path(str(source_filename)).stem or "unknown"
        output_path = raw_dir / source_slug / chapter_slug / f"{_slugify(filename_stem)}.md"
        ensure_dir(output_path.parent)

        lines = [
            f"# {source_doc} | Chapter {chapter_section}",
            "",
            "> This file is a derived pseudo-corpus created from QA answers in the Hugging Face dataset.",
            "> It is suitable for RAG pipeline testing, but it is not the original textbook paragraph corpus.",
            "",
            f"- source_doc: {source_doc}",
            f"- chapter_section: {chapter_section}",
            f"- source_filename: {source_filename}",
            f"- question_count: {len(group)}",
            "",
        ]

        for row in group.sort_values("question_id").itertuples(index=False):
            lines.extend(
                [
                    f"## {row.question_id}",
                    "",
                    f"Question: {row.question}",
                    "",
                    f"Answer: {row.ground_truth}",
                    "",
                    f"Question Type: {row.question_type}",
                    "",
                ]
            )

        output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        written_paths.append(output_path)

    return written_paths


def build_summary(eval_df: pd.DataFrame, pseudo_corpus_paths: list[Path], repo_id: str, split: str) -> dict[str, Any]:
    question_type_counts = eval_df["question_type"].value_counts(dropna=False).to_dict()
    source_counts = eval_df["source_doc"].value_counts(dropna=False).to_dict()
    return {
        "dataset_name": repo_id,
        "split": split,
        "question_count": int(len(eval_df)),
        "pseudo_corpus_file_count": len(pseudo_corpus_paths),
        "question_type_counts": question_type_counts,
        "source_doc_counts": source_counts,
        "notes": [
            "The upstream dataset is a QA dataset and does not ship the original paragraph text in each record.",
            "The generated markdown corpus is derived from answer text grouped by source metadata.",
            "This mapping is appropriate for pipeline validation and public-dataset smoke testing.",
        ],
    }


def main() -> None:
    args = parse_args()

    raw_dir = ROOT / Path(args.raw_dir)
    qa_path = ROOT / Path(args.qa_path)
    metadata_path = ROOT / Path(args.metadata_path)
    summary_path = ROOT / Path(args.summary_path)

    rows = load_rows(args.repo_id, args.split)
    rows = sample_rows(rows, args.max_questions, args.seed)
    eval_df = build_eval_dataframe(rows)
    metadata_df = build_metadata_dataframe(eval_df)
    pseudo_corpus_paths = build_pseudo_corpus(eval_df, raw_dir)
    summary = build_summary(eval_df, pseudo_corpus_paths, args.repo_id, args.split)

    write_csv(qa_path, eval_df)
    write_csv(metadata_path, metadata_df)
    write_json(summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote QA dataset to {qa_path}")
    print(f"Wrote metadata dataset to {metadata_path}")
    print(f"Wrote pseudo corpus under {raw_dir}")


if __name__ == "__main__":
    main()
