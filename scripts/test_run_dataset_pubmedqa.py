import argparse
import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_dataset_tests
from dataset_loader import NewsItem
from task_specs import get_task_spec


class FakeDebate:
    def __init__(self, *, model_name, T=None, sleep=0.0, task_spec=None):
        self.model_name = model_name
        self.temperature = T
        self.sleep = sleep
        self.task_spec = task_spec

    def run(self, *, news_text, news_path):
        if "MAYBE_CASE" in news_text:
            verdict = "The abstract evidence is insufficient."
        elif "NO_CASE" in news_text:
            verdict = "Final answer: NO"
        else:
            verdict = "Verdict: YES"
        return {
            "scores": {},
            "verdict": verdict,
            "summary": "Fake PubMedQA summary.",
            "analysis_metrics": {
                "opening": {"disagreement": 0.45},
                "rebuttal": {"disagreement": 0.30},
                "free": {"disagreement": 0.20},
                "closing": {"disagreement": 0.25},
            },
        }


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def make_args(output_dir):
    return argparse.Namespace(
        dataset="pubmedqa",
        data_path=str(PROJECT_ROOT / "PubMedQA" / "processed" / "pubmedqa_test_processed.jsonl"),
        train_path=None,
        val_path=None,
        test_path=None,
        model="fake-model",
        temperature=0.0,
        sleep=0.0,
        limit=None,
        output_dir=str(output_dir),
        disable_evidence=True,
        sample_size=None,
        sample_seed=42,
        text_column="text",
        label_column="label",
        title_column="title",
        subject_column="subject",
        date_column="date",
    )


def test_pubmedqa_splits_load(args):
    splits = run_dataset_tests.load_dataset_splits(args)
    require(set(splits) == {"test"}, f"Unexpected splits: {splits.keys()}")
    require(len(splits["test"]) == 500, f"Expected 500 PubMedQA items, got {len(splits['test'])}")

    sampled = run_dataset_tests.sample_stratified_even(
        splits["test"],
        30,
        seed=42,
        task_spec=get_task_spec("pubmedqa"),
    )
    sampled_labels = {item.label for item in sampled}
    require(len(sampled) == 30, f"Expected 30 sampled items, got {len(sampled)}")
    require(
        sampled_labels == {"YES", "NO", "MAYBE"},
        f"Sample should include YES, NO, and MAYBE, got {sampled_labels}",
    )


def test_pubmedqa_run_split(args, output_dir):
    original_debate = run_dataset_tests.Debate
    try:
        run_dataset_tests.Debate = FakeDebate
        items = [
            NewsItem(
                text="Question: YES_CASE\n\nAbstract context:\n[RESULTS] yes evidence.",
                label="YES",
                id="yes_case",
                metadata={"task_type": "pubmedqa"},
            ),
            NewsItem(
                text="Question: NO_CASE\n\nAbstract context:\n[RESULTS] no evidence.",
                label="NO",
                id="no_case",
                metadata={"task_type": "pubmedqa"},
            ),
            NewsItem(
                text="Question: MAYBE_CASE\n\nAbstract context:\n[RESULTS] mixed evidence.",
                label="MAYBE",
                id="maybe_case",
                metadata={"task_type": "pubmedqa"},
            ),
        ]
        summary_path, metrics = run_dataset_tests.run_split(
            args,
            split="test",
            items=items,
            base_output=output_dir,
        )
    finally:
        run_dataset_tests.Debate = original_debate

    require(summary_path.exists(), f"Summary was not written: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    require(summary["dataset"] == "pubmedqa", "Summary dataset should be pubmedqa.")
    require(summary["task_type"] == "biomedical_qa", "Summary task_type should be biomedical_qa.")
    require(summary["answer_type"] == "yes_no_maybe", "Summary answer_type should be yes_no_maybe.")

    require(metrics["attempted"] == 3, f"Expected attempted=3, got {metrics['attempted']}")
    require(metrics["labeled"] == 3, f"Expected labeled=3, got {metrics['labeled']}")
    require(metrics["correct"] == 3, f"Expected correct=3, got {metrics['correct']}")
    require(metrics["accuracy"] == 1.0, f"Expected accuracy=1.0, got {metrics['accuracy']}")
    require(metrics["failed"] == 0, f"Expected failed=0, got {metrics['failed']}")

    for phase in ["opening", "rebuttal", "free", "closing"]:
        stats = metrics[f"{phase}_disagreement_stats"]
        require(stats["count"] == 3, f"{phase} stats should count 3 records.")
        require(stats["mean"] is not None, f"{phase} mean should be present.")

    records = summary["records"]
    require(len(records) == 3, f"Expected 3 records, got {len(records)}")
    expected = [("YES", "YES"), ("NO", "NO"), ("MAYBE", "MAYBE")]
    for record, (gold, verdict) in zip(records, expected):
        require(record["label"] == gold, f"Expected label {gold}, got {record['label']}")
        require(record["verdict"] == verdict, f"Expected verdict {verdict}, got {record['verdict']}")
        require(record["is_correct"] is True, "Record should be correct.")
        require(record["task_type"] == "biomedical_qa", "Record task_type missing.")
        require(record["answer_type"] == "yes_no_maybe", "Record answer_type missing.")
        for phase in ["opening", "rebuttal", "free", "closing"]:
            require(record[f"{phase}_disagreement"] is not None, f"{phase} disagreement missing.")

    print("PubMedQA run_dataset_tests integration validation passed.")
    print(f"Summary path: {summary_path}")
    print(f"Metrics: attempted={metrics['attempted']} correct={metrics['correct']} accuracy={metrics['accuracy']}")


def main():
    with tempfile.TemporaryDirectory(prefix="pubmedqa_runner_test_") as tmpdir:
        output_dir = Path(tmpdir)
        args = make_args(output_dir)
        test_pubmedqa_splits_load(args)
        test_pubmedqa_run_split(args, output_dir)


if __name__ == "__main__":
    main()
