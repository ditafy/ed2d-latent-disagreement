#!/usr/bin/env python3
"""Prepare PubMedQA labeled test data for ED2D-style benchmark experiments.

This script joins PubMedQA's full labeled records with the official test
ground-truth labels, then writes a leakage-safe JSONL file for debate runs.
The formatted model input includes only the question and abstract contexts;
gold decisions and long answers are kept out of the prompt text.
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
VALID_LABELS = {"yes": "YES", "no": "NO", "maybe": "MAYBE"}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert PubMedQA test ground truth into processed ED2D benchmark JSONL."
    )
    parser.add_argument(
        "--records",
        default=str(script_dir / "ori_pqal.json"),
        help="Path to PubMedQA full labeled records JSON.",
    )
    parser.add_argument(
        "--ground-truth",
        default=str(script_dir / "test_ground_truth.json"),
        help="Path to PubMedQA test ground-truth labels JSON.",
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
    return parser.parse_args()


def load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}.")
    return data


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    return VALID_LABELS.get(str(value).strip().lower())


def format_contexts(row: dict[str, Any]) -> str:
    contexts = row.get("CONTEXTS") or []
    labels = row.get("LABELS") or []
    if not isinstance(contexts, list):
        contexts = [str(contexts)]
    if not isinstance(labels, list):
        labels = []

    formatted = []
    for idx, context in enumerate(contexts, 1):
        text = str(context).strip()
        if not text:
            continue
        label = str(labels[idx - 1]).strip() if idx - 1 < len(labels) else f"CONTEXT {idx}"
        formatted.append(f"[{label}] {text}")
    return "\n\n".join(formatted)


def format_input(row: dict[str, Any]) -> str:
    question = str(row.get("QUESTION", "")).strip()
    contexts = format_contexts(row)
    return (
        f"Question: {question}\n\n"
        f"Abstract context:\n{contexts}\n\n"
        "Decide whether the answer to the biomedical question is YES, NO, or MAYBE."
    )


def build_item(pmid: str, row: dict[str, Any], gold_label: str) -> dict[str, Any] | None:
    question = str(row.get("QUESTION", "")).strip()
    contexts = row.get("CONTEXTS") or []
    if not question or not contexts:
        return None

    return {
        "id": pmid,
        "task_type": "pubmedqa",
        "text": format_input(row),
        "label": gold_label,
        "metadata": {
            "pmid": pmid,
            "question": question,
            "context_labels": row.get("LABELS") or [],
            "mesh_terms": row.get("MESHES") or [],
            "year": row.get("YEAR"),
            "source": "pqa_labeled_test",
            "context_count": len(contexts) if isinstance(contexts, list) else 1,
        },
    }


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_report(
    records: dict[str, Any],
    ground_truth: dict[str, Any],
    items: list[dict[str, Any]],
    missing_record_count: int,
    invalid_label_count: int,
    skipped_empty_count: int,
) -> dict[str, Any]:
    labels = Counter(item["label"] for item in items)
    text_lengths = [len(item["text"]) for item in items]
    word_lengths = [len(item["text"].split()) for item in items]
    context_counts = [item["metadata"]["context_count"] for item in items]

    return {
        "dataset": "pubmedqa",
        "split": "test",
        "source_records": "ori_pqal.json",
        "source_ground_truth": "test_ground_truth.json",
        "total_records_available": len(records),
        "total_ground_truth": len(ground_truth),
        "total_processed": len(items),
        "yes_count": labels.get("YES", 0),
        "no_count": labels.get("NO", 0),
        "maybe_count": labels.get("MAYBE", 0),
        "missing_record_count": missing_record_count,
        "invalid_label_count": invalid_label_count,
        "skipped_empty_question_or_context_count": skipped_empty_count,
        "avg_text_chars": round(mean(text_lengths), 2) if text_lengths else 0,
        "max_text_chars": max(text_lengths) if text_lengths else 0,
        "avg_text_words": round(mean(word_lengths), 2) if word_lengths else 0,
        "max_text_words": max(word_lengths) if word_lengths else 0,
        "avg_context_count": round(mean(context_counts), 2) if context_counts else 0,
        "label_set": sorted(labels),
        "leakage_policy": (
            "Formatted text includes QUESTION and CONTEXTS only. "
            "final_decision, LONG_ANSWER, and model prediction fields are not written into text."
        ),
    }


def write_preview(path: Path, items: list[dict[str, Any]], preview_size: int, seed: int) -> None:
    rng = random.Random(seed)
    sample_count = min(max(preview_size, 0), len(items))
    preview_items = rng.sample(items, sample_count) if sample_count else []

    lines = [
        "PubMedQA processed sample preview",
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
    records_path = Path(args.records).expanduser().resolve()
    ground_truth_path = Path(args.ground_truth).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_json_object(records_path)
    ground_truth = load_json_object(ground_truth_path)

    items: list[dict[str, Any]] = []
    missing_record_count = 0
    invalid_label_count = 0
    skipped_empty_count = 0

    for pmid, raw_label in ground_truth.items():
        label = normalize_label(raw_label)
        if label is None:
            invalid_label_count += 1
            continue

        row = records.get(str(pmid))
        if row is None:
            missing_record_count += 1
            continue
        if not isinstance(row, dict):
            skipped_empty_count += 1
            continue

        item = build_item(str(pmid), row, label)
        if item is None:
            skipped_empty_count += 1
            continue
        items.append(item)

    processed_path = output_dir / "pubmedqa_test_processed.jsonl"
    report_path = output_dir / "pubmedqa_data_report.json"
    preview_path = output_dir / "pubmedqa_sample_preview.txt"

    write_jsonl(processed_path, items)
    report = build_report(
        records=records,
        ground_truth=ground_truth,
        items=items,
        missing_record_count=missing_record_count,
        invalid_label_count=invalid_label_count,
        skipped_empty_count=skipped_empty_count,
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_preview(preview_path, items, args.preview_size, args.preview_seed)

    print("PubMedQA preprocessing complete.")
    print(f"Records: {records_path}")
    print(f"Ground truth: {ground_truth_path}")
    print(f"Processed JSONL: {processed_path}")
    print(f"Data report: {report_path}")
    print(f"Sample preview: {preview_path}")
    print(
        "Summary: "
        f"ground_truth={report['total_ground_truth']} processed={report['total_processed']} "
        f"YES={report['yes_count']} NO={report['no_count']} MAYBE={report['maybe_count']} "
        f"missing_record={report['missing_record_count']} "
        f"invalid_label={report['invalid_label_count']} "
        f"skipped_empty={report['skipped_empty_question_or_context_count']}"
    )


if __name__ == "__main__":
    main()
