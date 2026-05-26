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
        if "NO_CASE" in news_text:
            verdict = "Verdict: NO"
        else:
            verdict = "Final answer: YES"
        return {
            "scores": {},
            "verdict": verdict,
            "summary": "Fake StrategyQA summary.",
            "analysis_metrics": {
                "opening": {"disagreement": 0.40},
                "rebuttal": {"disagreement": 0.25},
                "free": {"disagreement": 0.10},
                "closing": {"disagreement": 0.15},
            },
        }


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def make_args(output_dir):
    return argparse.Namespace(
        dataset="strategyqa",
        data_path=str(PROJECT_ROOT / "StrategyQA" / "processed" / "strategyqa_processed.jsonl"),
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


def test_strategyqa_splits_load(args):
    splits = run_dataset_tests.load_dataset_splits(args)
    require(set(splits) == {"test"}, f"Unexpected splits: {splits.keys()}")
    require(len(splits["test"]) == 687, f"Expected 687 StrategyQA items, got {len(splits['test'])}")
    sampled = run_dataset_tests.sample_stratified_even(
        splits["test"],
        30,
        seed=42,
        task_spec=get_task_spec("strategyqa"),
    )
    sampled_labels = {item.label for item in sampled}
    require(len(sampled) == 30, f"Expected 30 sampled items, got {len(sampled)}")
    require(sampled_labels == {"YES", "NO"}, f"Sample should include YES and NO, got {sampled_labels}")


def test_strategyqa_run_split(args, output_dir):
    original_debate = run_dataset_tests.Debate
    try:
        run_dataset_tests.Debate = FakeDebate
        items = [
            NewsItem(
                text="Question: YES_CASE\n\nDecide whether the answer is YES or NO.",
                label="YES",
                id="yes_case",
                metadata={"task_type": "strategyqa"},
            ),
            NewsItem(
                text="Question: NO_CASE\n\nDecide whether the answer is YES or NO.",
                label="NO",
                id="no_case",
                metadata={"task_type": "strategyqa"},
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

    require(summary["dataset"] == "strategyqa", "Summary dataset should be strategyqa.")
    require(summary["task_type"] == "binary_reasoning", "Summary task_type should be binary_reasoning.")
    require(summary["answer_type"] == "yes_no", "Summary answer_type should be yes_no.")

    require(metrics["attempted"] == 2, f"Expected attempted=2, got {metrics['attempted']}")
    require(metrics["labeled"] == 2, f"Expected labeled=2, got {metrics['labeled']}")
    require(metrics["correct"] == 2, f"Expected correct=2, got {metrics['correct']}")
    require(metrics["accuracy"] == 1.0, f"Expected accuracy=1.0, got {metrics['accuracy']}")
    require(metrics["failed"] == 0, f"Expected failed=0, got {metrics['failed']}")

    for phase in ["opening", "rebuttal", "free", "closing"]:
        stats = metrics[f"{phase}_disagreement_stats"]
        require(stats["count"] == 2, f"{phase} stats should count 2 records.")
        require(stats["mean"] is not None, f"{phase} mean should be present.")

    records = summary["records"]
    require(len(records) == 2, f"Expected 2 records, got {len(records)}")
    require(records[0]["label"] == "YES", "First gold label should be YES.")
    require(records[0]["verdict"] == "YES", "First verdict should parse to YES.")
    require(records[0]["is_correct"] is True, "First record should be correct.")
    require(records[1]["label"] == "NO", "Second gold label should be NO.")
    require(records[1]["verdict"] == "NO", "Second verdict should parse to NO.")
    require(records[1]["is_correct"] is True, "Second record should be correct.")

    for record in records:
        require(record["task_type"] == "binary_reasoning", "Record task_type missing.")
        require(record["answer_type"] == "yes_no", "Record answer_type missing.")
        for phase in ["opening", "rebuttal", "free", "closing"]:
            require(record[f"{phase}_disagreement"] is not None, f"{phase} disagreement missing.")

    print("StrategyQA run_dataset_tests integration validation passed.")
    print(f"Summary path: {summary_path}")
    print(f"Metrics: attempted={metrics['attempted']} correct={metrics['correct']} accuracy={metrics['accuracy']}")


def main():
    with tempfile.TemporaryDirectory(prefix="strategyqa_runner_test_") as tmpdir:
        output_dir = Path(tmpdir)
        args = make_args(output_dir)
        test_strategyqa_splits_load(args)
        test_strategyqa_run_split(args, output_dir)


if __name__ == "__main__":
    main()
