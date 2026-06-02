import argparse
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_loader import PubMedQALoader, load_dataset


EXPECTED_LABEL_COUNTS = {"YES": 276, "NO": 169, "MAYBE": 55}
BANNED_TEXT_MARKERS = {
    "final_decision",
    "LONG_ANSWER",
    "reasoning_required_pred",
    "reasoning_free_pred",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the processed PubMedQA loader output."
    )
    parser.add_argument(
        "--data-path",
        default=str(PROJECT_ROOT / "PubMedQA" / "processed" / "pubmedqa_test_processed.jsonl"),
        help="Path to processed PubMedQA JSONL.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=500,
        help="Expected number of PubMedQA examples.",
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

    require(len(ids) == len(set(ids)), "PubMedQA ids must be unique.")
    require(set(labels) <= set(EXPECTED_LABEL_COUNTS), f"Unexpected labels: {sorted(set(labels))}")
    require(dict(label_counts) == EXPECTED_LABEL_COUNTS, f"Unexpected label counts: {dict(label_counts)}")

    missing_text = [item.id for item in items if not item.text or not item.text.strip()]
    require(not missing_text, f"Items with empty text: {missing_text[:5]}")

    missing_question_marker = [item.id for item in items if "Question:" not in item.text]
    require(
        not missing_question_marker,
        f"Items missing Question marker: {missing_question_marker[:5]}",
    )

    missing_context_marker = [item.id for item in items if "Abstract context:" not in item.text]
    require(
        not missing_context_marker,
        f"Items missing Abstract context marker: {missing_context_marker[:5]}",
    )

    leakage_ids = [
        item.id
        for item in items
        if any(marker in item.text for marker in BANNED_TEXT_MARKERS)
    ]
    require(not leakage_ids, f"Items with possible label leakage: {leakage_ids[:5]}")

    bad_metadata = []
    for item in items:
        metadata = item.metadata or {}
        if (
            metadata.get("task_type") != "pubmedqa"
            or not metadata.get("question")
            or not metadata.get("source_path")
            or metadata.get("source") != "pqa_labeled_test"
        ):
            bad_metadata.append(item.id)
    require(not bad_metadata, f"Items with incomplete PubMedQA metadata: {bad_metadata[:5]}")

    require(items[0].id == "12377809", f"Unexpected first id: {items[0].id}")
    require(items[0].label == "YES", f"Unexpected first label: {items[0].label}")

    print("PubMedQA loader validation passed.")
    print(f"Items: {len(items)}")
    print(
        "Labels: "
        f"YES={label_counts['YES']} NO={label_counts['NO']} MAYBE={label_counts['MAYBE']}"
    )
    print(f"First id: {items[0].id}")
    print(f"First label: {items[0].label}")
    print("First text preview:")
    print(items[0].text[:300])


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)

    print("=== PubMedQALoader direct check ===")
    direct_items = PubMedQALoader().load(data_path)
    validate_items(direct_items, args.expected_count)

    print("\n=== load_dataset unified interface check ===")
    unified_items = load_dataset("pubmedqa", data_path)
    validate_items(unified_items, args.expected_count)


if __name__ == "__main__":
    main()
