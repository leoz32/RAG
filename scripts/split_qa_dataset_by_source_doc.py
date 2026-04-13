from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def split_dataset(input_path: Path, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"No header found in {input_path}")

        grouped_rows: dict[str, list[dict[str, str]]] = {}
        counts: Counter[str] = Counter()
        for row in reader:
            source_doc = (row.get("source_doc") or "unknown").strip() or "unknown"
            grouped_rows.setdefault(source_doc, []).append(row)
            counts[source_doc] += 1

    for source_doc, rows in grouped_rows.items():
        slug = source_doc.lower().replace(" ", "_")
        output_path = output_dir / f"{slug}_qa_dataset.csv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    report_path = output_dir / "split_report.json"
    report_payload = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "counts_by_source_doc": dict(counts),
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eval/openstax_american_yawp_qa_dataset.csv")
    parser.add_argument("--output-dir", default="data/eval/split_by_source_doc")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    counts = split_dataset(input_path, output_dir)
    for source_doc, count in sorted(counts.items()):
        print(f"{source_doc}: {count}")


if __name__ == "__main__":
    main()
