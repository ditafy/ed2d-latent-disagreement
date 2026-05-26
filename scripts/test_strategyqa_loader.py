import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_loader import StrategyQALoader, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the processed StrategyQA loader output."
    )
    parser.add_argument(
        "--data-path",
        default=str(PROJECT_ROOT / "StrategyQA" / "processed" / "strategyqa_processed.jsonl"),
        help="Path to processed StrategyQA JSONL.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=687,
        help="Expected number of StrategyQA examples.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_items(items, expected_count: int) -> None:
    require(len(items) == expected_count, f"Expected {expected_count} items, got {len(items)}.")

    ids = [item.id for item in items]
    labels = [item.label for item in items]
    label_counts = Counter(labels)

    require(len(ids) == len(set(ids)), "StrategyQA ids must be unique.")
    require(set(labels) <= {"YES", "NO"}, f"Unexpected labels: {sorted(set(labels))}")
    require(label_counts["YES"] > 0, "Expected at least one YES label.")
    require(label_counts["NO"] > 0, "Expected at least one NO label.")

    missing_text = [item.id for item in items if not item.text or not item.text.strip()]
    require(not missing_text, f"Items with empty text: {missing_text[:5]}")

    missing_question_marker = [item.id for item in items if "Question:" not in item.text]
    require(
        not missing_question_marker,
        f"Items missing Question marker: {missing_question_marker[:5]}",
    )

    answer_leaks = [
        item.id
        for item in items
        if "Answer:" in item.text or "raw_answer" in item.text
    ]
    require(not answer_leaks, f"Items with answer-marker leakage: {answer_leaks[:5]}")

    bad_metadata = []
    for item in items:
        metadata = item.metadata or {}
        if metadata.get("task_type") != "strategyqa" or not metadata.get("question"):
            bad_metadata.append(item.id)
    require(not bad_metadata, f"Items with incomplete StrategyQA metadata: {bad_metadata[:5]}")

    print("StrategyQA loader validation passed.")
    print(f"Items: {len(items)}")
    print(f"Labels: YES={label_counts['YES']} NO={label_counts['NO']}")
    print(f"First id: {items[0].id}")
    print(f"First label: {items[0].label}")
    print("First text preview:")
    print(items[0].text[:300])


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)

    print("=== StrategyQALoader direct check ===")
    direct_items = StrategyQALoader().load(data_path)
    validate_items(direct_items, args.expected_count)

    print("\n=== load_dataset unified interface check ===")
    unified_items = load_dataset("strategyqa", data_path)
    validate_items(unified_items, args.expected_count)


if __name__ == "__main__":
    main()
