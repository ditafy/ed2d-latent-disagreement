#!/usr/bin/env python3
"""Prepare StrategyQA for ED2D-style benchmark experiments.

This script converts the raw StrategyQA JSON list into a stable JSONL file
with normalized YES/NO labels and formatted model inputs. It also writes a
small data report and a human-readable sample preview for validation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_PREVIEW_SIZE = 10


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert raw StrategyQA JSON into processed ED2D benchmark JSONL."
    )
    parser.add_argument(
        "--input",
        default=str(script_dir / "StrategyQA.json"),
        help="Path to the raw StrategyQA JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "processed"),
        help="Directory where processed files will be written.",
    )
    parser.add_argument(
        "--preview-size",
        type=int,
        default=DEFAULT_PREVIEW_SIZE,
        help="Number of examples to include in the sample preview.",
    )
    parser.add_argument(
        "--preview-seed",
        type=int,
        default=42,
        help="Random seed used for selecting preview examples.",
    )
    parser.add_argument(
        "--question-only",
        action="store_true",
        help="Use only the question in the model input, excluding term and description.",
    )
    return parser.parse_args()


def normalize_answer(value: Any) -> str | None:
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if value is None:
        return None

    text = str(value).strip().lower()
    yes_values = {"true", "yes", "y", "1"}
    no_values = {"false", "no", "n", "0"}
    if text in yes_values:
        return "YES"
    if text in no_values:
        return "NO"
    return None


def format_input(row: dict[str, Any], question_only: bool) -> str:
    question = str(row.get("question", "")).strip()
    term = str(row.get("term", "") or "").strip()
    description = str(row.get("description", "") or "").strip()

    parts = [f"Question: {question}"]
    if not question_only:
        if term:
            parts.append(f"Term: {term}")
        if description:
            parts.append(f"Description: {description}")
    parts.append("")
    parts.append("Decide whether the answer is YES or NO.")
    return "\n".join(parts)


def build_item(row: dict[str, Any], idx: int, question_only: bool) -> dict[str, Any] | None:
    question = str(row.get("question", "")).strip()
    if not question:
        return None

    label = normalize_answer(row.get("answer"))
    if label is None:
        return None

    qid = str(row.get("qid") or f"strategyqa_{idx:05d}")
    term = row.get("term")
    description = row.get("description")
    raw_answer = row.get("answer")

    return {
        "id": qid,
        "task_type": "strategyqa",
        "text": format_input(row, question_only),
        "label": label,
        "metadata": {
            "qid": qid,
            "question": question,
            "term": term,
            "description": description,
            "raw_answer": raw_answer,
            "input_format": "question_only" if question_only else "question_term_description",
        },
    }


def load_raw(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}.")
    return data


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_report(
    raw_rows: list[dict[str, Any]],
    items: list[dict[str, Any]],
    invalid_answer_count: int,
    empty_question_count: int,
    duplicate_id_count: int,
    question_only: bool,
) -> dict[str, Any]:
    labels = Counter(item["label"] for item in items)
    text_lengths = [len(item["text"]) for item in items]
    token_like_lengths = [len(item["text"].split()) for item in items]

    return {
        "dataset": "strategyqa",
        "input_format": "question_only" if question_only else "question_term_description",
        "total_raw": len(raw_rows),
        "total_processed": len(items),
        "yes_count": labels.get("YES", 0),
        "no_count": labels.get("NO", 0),
        "empty_question_count": empty_question_count,
        "invalid_answer_count": invalid_answer_count,
        "duplicate_id_count": duplicate_id_count,
        "avg_text_chars": round(mean(text_lengths), 2) if text_lengths else 0,
        "max_text_chars": max(text_lengths) if text_lengths else 0,
        "avg_text_words": round(mean(token_like_lengths), 2) if token_like_lengths else 0,
        "max_text_words": max(token_like_lengths) if token_like_lengths else 0,
        "label_set": sorted(labels),
        "contains_gold_answer_in_text_policy": (
            "The formatted text asks for YES/NO but does not include the gold label value."
        ),
    }


def write_preview(path: Path, items: list[dict[str, Any]], preview_size: int, seed: int) -> None:
    rng = random.Random(seed)
    sample_count = min(max(preview_size, 0), len(items))
    preview_items = rng.sample(items, sample_count) if sample_count else []

    lines = [
        "StrategyQA processed sample preview",
        f"sample_count={sample_count}",
        "",
    ]
    for idx, item in enumerate(preview_items, 1):
        lines.extend(
            [
                f"===== SAMPLE {idx} =====",
                f"id: {item['id']}",
                f"label: {item['label']}",
                "text:",
                item["text"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = load_raw(input_path)
    items: list[dict[str, Any]] = []
    empty_question_count = 0
    invalid_answer_count = 0

    for idx, row in enumerate(raw_rows):
        question = str(row.get("question", "")).strip()
        if not question:
            empty_question_count += 1
            continue
        if normalize_answer(row.get("answer")) is None:
            invalid_answer_count += 1
            continue
        item = build_item(row, idx, args.question_only)
        if item is not None:
            items.append(item)

    ids = [item["id"] for item in items]
    duplicate_id_count = len(ids) - len(set(ids))

    processed_path = output_dir / "strategyqa_processed.jsonl"
    report_path = output_dir / "strategyqa_data_report.json"
    preview_path = output_dir / "strategyqa_sample_preview.txt"

    write_jsonl(processed_path, items)
    report = build_report(
        raw_rows=raw_rows,
        items=items,
        invalid_answer_count=invalid_answer_count,
        empty_question_count=empty_question_count,
        duplicate_id_count=duplicate_id_count,
        question_only=args.question_only,
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_preview(preview_path, items, args.preview_size, args.preview_seed)

    print("StrategyQA preprocessing complete.")
    print(f"Input: {input_path}")
    print(f"Processed JSONL: {processed_path}")
    print(f"Data report: {report_path}")
    print(f"Sample preview: {preview_path}")
    print(
        "Summary: "
        f"raw={report['total_raw']} processed={report['total_processed']} "
        f"YES={report['yes_count']} NO={report['no_count']} "
        f"empty_question={report['empty_question_count']} "
        f"invalid_answer={report['invalid_answer_count']} "
        f"duplicate_id={report['duplicate_id_count']}"
    )


if __name__ == "__main__":
    main()
