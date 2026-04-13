from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, write_json


DEFAULT_INPUT_DIR = "data/raw/openstax_us_history_original_v3"
DEFAULT_OUTPUT_DIR = "data/raw/openstax_us_history_clean_v1"


LEGAL_START_MARKERS = [
    "This book may not be used in the training of large language models",
    "Want to cite, share, or modify this book?",
]


def _strip_generator_header(text: str) -> str:
    return re.sub(r"^<!--.*?-->\s*", "", text, flags=re.DOTALL)


def _truncate_legal_tail(text: str) -> str:
    cut_positions = [text.find(marker) for marker in LEGAL_START_MARKERS if marker in text]
    cut_positions = [pos for pos in cut_positions if pos >= 0]
    if not cut_positions:
        return text
    return text[: min(cut_positions)].rstrip()


def _remove_section_block(text: str, heading: str) -> str:
    pattern = re.compile(
        rf"\n##+\s+{re.escape(heading)}\s*\n.*?(?=\n##+\s+|\Z)",
        flags=re.DOTALL,
    )
    return re.sub(pattern, "\n", text)


def _dedupe_immediate_headings(text: str) -> str:
    lines = text.splitlines()
    cleaned: list[str] = []
    prev_heading: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            normalized = re.sub(r"\s+", " ", stripped)
            if normalized == prev_heading:
                continue
            prev_heading = normalized
        elif stripped:
            prev_heading = None
        cleaned.append(line)
    return "\n".join(cleaned)


def _normalize_spacing(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def clean_markdown(text: str, drop_learning_objectives: bool = True, drop_chapter_outline: bool = True) -> str:
    cleaned = _strip_generator_header(text)
    cleaned = _truncate_legal_tail(cleaned)
    if drop_learning_objectives:
        cleaned = _remove_section_block(cleaned, "Learning Objectives")
    if drop_chapter_outline:
        cleaned = _remove_section_block(cleaned, "Chapter Outline")
    cleaned = _dedupe_immediate_headings(cleaned)
    cleaned = _normalize_spacing(cleaned)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean fetched OpenStax U.S. History Markdown files for downstream RAG indexing."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--keep-learning-objectives", action="store_true")
    parser.add_argument("--keep-chapter-outline", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_dir = ROOT / args.input_dir
    output_dir = ensure_dir(ROOT / args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    records: list[dict] = []
    for source_path in sorted(input_dir.glob("*.md")):
        target_path = output_dir / source_path.name
        if target_path.exists() and not args.force:
            records.append({"file": source_path.name, "status": "skipped_existing"})
            continue

        source_text = source_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_markdown(
            source_text,
            drop_learning_objectives=not args.keep_learning_objectives,
            drop_chapter_outline=not args.keep_chapter_outline,
        )
        target_path.write_text(cleaned, encoding="utf-8")
        records.append(
            {
                "file": source_path.name,
                "status": "cleaned",
                "input_chars": len(source_text),
                "output_chars": len(cleaned),
            }
        )
        print(f"Cleaned {source_path.name} -> {target_path}")

    manifest_path = input_dir / "manifest.json"
    if manifest_path.exists():
        copied_manifest = output_dir / "manifest.json"
        copied_manifest.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files_processed": len(records),
        "cleaned_count": sum(1 for record in records if record["status"] == "cleaned"),
        "skipped_existing_count": sum(1 for record in records if record["status"] == "skipped_existing"),
        "drop_learning_objectives": not args.keep_learning_objectives,
        "drop_chapter_outline": not args.keep_chapter_outline,
        "records": records,
    }
    write_json(output_dir / "cleaning_report.json", summary)


if __name__ == "__main__":
    main()
