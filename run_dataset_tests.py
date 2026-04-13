"""
Batch runner for Debate-to-Detect on Weibo21 and FakeNewsDataset.

Example:
    python run_dataset_tests.py --dataset weibo21 --test-path data/weibo21/test.pkl --model gpt-4o
    python run_dataset_tests.py --dataset fakenewsdataset --data-path data/FakeNewsDataset/test.csv --model gpt-4o-mini
    python run_dataset_tests.py --dataset fakenewsdataset --data-path data/FakeNewsDataset/test.csv --model Qwen/Qwen2.5-14B-Instruct
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import engine as debate_engine
from dataset_loader import FakeNewsDatasetLoader, NewsItem, Weibo21Loader
from engine import Debate


def parse_args() -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Run Debate-to-Detect over supported datasets.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["weibo21", "fakenewsdataset"],
        help="Which dataset loader to use.",
    )
    parser.add_argument("--data-path", help="Single dataset file (pkl for Weibo21 or csv for FakeNewsDataset).")
    parser.add_argument("--train-path", help="Weibo21 train split (.pkl).")
    parser.add_argument("--val-path", help="Weibo21 validation split (.pkl).")
    parser.add_argument("--test-path", help="Test split (.pkl for Weibo21, .csv for FakeNewsDataset).")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI chat model name.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for debate agents.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep seconds between OpenAI calls.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on how many news items to process per split (useful to control API cost).",
    )
    parser.add_argument(
        "--output-dir",
        default="batch_results",
        help="Base directory to store aggregated metrics and debate outputs.",
    )
    parser.add_argument(
        "--disable-evidence",
        action="store_true",
        help="Skip Wikipedia evidence retrieval (reduces latency and external calls).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional: sample this many items per split, stratified by label and spaced across the split.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used when filling gaps in stratified sampling.",
    )

    # FakeNewsDataset specific columns
    parser.add_argument("--text-column", default="text", help="Column name for news text.")
    parser.add_argument("--label-column", default="label", help="Column name for labels.")
    parser.add_argument("--title-column", default="title", help="Column name for titles.")
    parser.add_argument("--subject-column", default="subject", help="Column name for subjects.")
    parser.add_argument("--date-column", default="date", help="Column name for publication date.")
    return parser.parse_args()


def _evenly_pick(seq: List[tuple], k: int, seed: int) -> List[tuple]:
    """Evenly spread picks across a sequence; fill gaps randomly if needed."""
    n = len(seq)
    if k <= 0:
        return []
    if k >= n:
        return seq

    step = n / k
    base_indices = [min(int(i * step), n - 1) for i in range(k)]
    idx_set = dict.fromkeys(base_indices)  # preserve order, drop dups
    selected = [seq[i] for i in idx_set]

    if len(selected) < k:
        remaining_indices = [i for i in range(n) if i not in idx_set]
        rng = random.Random(seed)
        extra_needed = k - len(selected)
        extra = rng.sample(remaining_indices, k=min(extra_needed, len(remaining_indices)))
        selected.extend(seq[i] for i in sorted(extra))

    return selected


def sample_stratified_even(items: List[NewsItem], k: int, seed: int = 42) -> List[NewsItem]:
    """Sample k items, keeping label ratio (REAL/FAKE) and spacing across the split."""
    k = max(0, min(k, len(items)))
    if k == 0:
        return []

    indexed = [(idx, it) for idx, it in enumerate(items)]
    buckets: Dict[str, List[tuple]] = {"REAL": [], "FAKE": []}

    for idx, it in indexed:
        lbl = normalize_label(it.label)
        if lbl in buckets:
            buckets[lbl].append((idx, it))

    labeled_total = sum(len(v) for v in buckets.values())
    if labeled_total == 0:
        # No labels to stratify; just evenly pick across all items
        return [it for _, it in _evenly_pick(indexed, k, seed)]

    real_ratio = len(buckets["REAL"]) / labeled_total
    real_k = round(k * real_ratio)
    fake_k = k - real_k

    picked = []
    picked.extend(_evenly_pick(buckets["REAL"], real_k, seed))
    picked.extend(_evenly_pick(buckets["FAKE"], fake_k, seed + 1))

    # Sort back to original order
    picked.sort(key=lambda x: x[0])
    return [it for _, it in picked]


def normalize_label(label: object) -> Optional[str]:
    """Normalize dataset label to 'REAL' / 'FAKE'."""
    if label is None:
        return None

    value = str(label).strip().lower()
    if not value or value in {"nan", "none"}:
        return None

    fake_tags = {"1", "fake", "false", "rumor", "rumour", "spam", "misinformation"}
    real_tags = {"0", "real", "true", "nonrumor", "non-rumor", "legit", "legitimate"}

    if value in fake_tags:
        return "FAKE"
    if value in real_tags:
        return "REAL"
    return None


def sanitize_id(raw_id: object, fallback: int) -> str:
    """Create a filesystem-safe identifier."""
    if raw_id is None:
        return f"item_{fallback}"
    candidate = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw_id)).strip("_")
    return candidate or f"item_{fallback}"


def summarize_scalar(values: List[float]) -> Dict[str, Optional[float]]:
    """Return count/mean/std for a list of scalar values."""
    if not values:
        return {"count": 0, "mean": None, "std": None}

    count = len(values)
    mean = sum(values) / count
    variance = sum((v - mean) ** 2 for v in values) / count
    return {
        "count": count,
        "mean": round(mean, 6),
        "std": round(math.sqrt(variance), 6),
    }


def collect_phase_disagreements(result: Dict) -> Dict[str, Optional[float]]:
    """Extract disagreement scores for each analyzed debate phase."""
    analysis_metrics = result.get("analysis_metrics", {})
    phase_disagreements: Dict[str, Optional[float]] = {}
    for phase in ["opening", "rebuttal", "free", "closing"]:
        phase_metrics = analysis_metrics.get(phase, {})
        phase_disagreements[f"{phase}_disagreement"] = phase_metrics.get("disagreement")
    return phase_disagreements


def load_dataset_splits(args: argparse.Namespace) -> Dict[str, List[NewsItem]]:
    """Load dataset splits based on CLI arguments."""
    if args.dataset == "weibo21":
        loader = Weibo21Loader()
        splits: Dict[str, List[NewsItem]] = {}

        if args.train_path or args.val_path or args.test_path:
            if args.train_path:
                splits["train"] = loader.load(args.train_path)
            if args.val_path:
                splits["val"] = loader.load(args.val_path)
            if args.test_path:
                splits["test"] = loader.load(args.test_path)
        elif args.data_path:
            splits["test"] = loader.load(args.data_path)
        else:
            raise ValueError("For Weibo21, provide --data-path or at least one of --train-path/--val-path/--test-path.")

        return splits

    # FakeNewsDataset
    loader = FakeNewsDatasetLoader()
    csv_path = args.data_path or args.test_path
    if not csv_path:
        raise ValueError("For FakeNewsDataset, provide --data-path (csv) or --test-path.")

    return {
        "test": loader.load(
            csv_path,
            text_column=args.text_column,
            label_column=args.label_column,
            title_column=args.title_column,
            subject_column=args.subject_column,
            date_column=args.date_column,
        )
    }


def run_split(
    args: argparse.Namespace,
    split: str,
    items: List[NewsItem],
    base_output: Path,
) -> Tuple[Path, Dict[str, Optional[float]]]:
    """Run Debate-to-Detect on one split and write summary."""
    # Route debate outputs for this split
    debate_output_dir = base_output / args.dataset / split / "debate_outputs"
    debate_output_dir.mkdir(parents=True, exist_ok=True)
    debate_engine.SAVE_DIR = str(debate_output_dir)

    records = []
    processed = 0
    labeled = 0
    correct = 0
    failed = 0

    max_items = len(items) if args.limit is None else min(args.limit, len(items))
    progress = tqdm(total=max_items, desc=f"{args.dataset}-{split}", dynamic_ncols=True) if tqdm else None

    for idx, item in enumerate(items):
        if idx >= max_items:
            break

        processed += 1
        news_id = sanitize_id(item.id, idx)
        news_path = debate_output_dir / f"{news_id}.txt"

        try:
            debate = Debate(model_name=args.model, T=args.temperature, sleep=args.sleep)
            result = debate.run(news_text=item.text, news_path=news_path)
            verdict = result["verdict"]
            phase_disagreements = collect_phase_disagreements(result)
            gold = normalize_label(item.label)
            is_correct = None
            error_value = None
            if gold is not None:
                labeled += 1
                is_correct = verdict == gold
                error_value = 0 if is_correct else 1
                if is_correct:
                    correct += 1

            record = {
                "id": item.id if item.id is not None else idx,
                "split": split,
                "label": gold,
                "verdict": verdict,
                "is_correct": is_correct,
                "error": error_value,
            }
            record.update(phase_disagreements)
            records.append(record)
        except Exception as exc:  # pragma: no cover - runtime safety
            failed += 1
            record = {
                "id": item.id if item.id is not None else idx,
                "split": split,
                "label": normalize_label(item.label),
                "verdict": None,
                "is_correct": None,
                "error": None,
                "error_message": str(exc),
            }
            record.update(
                {
                    f"{phase}_disagreement": None
                    for phase in ["opening", "rebuttal", "free", "closing"]
                }
            )
            records.append(record)
        finally:
            if progress:
                progress.update(1)

    metrics = {
        "attempted": processed,
        "labeled": labeled,
        "correct": correct,
        "accuracy": round(correct / labeled, 4) if labeled else None,
        "failed": failed,
    }
    for phase in ["opening", "rebuttal", "free", "closing"]:
        phase_key = f"{phase}_disagreement"
        phase_values = [
            float(record[phase_key])
            for record in records
            if record.get(phase_key) is not None
        ]
        metrics[f"{phase}_disagreement_stats"] = summarize_scalar(phase_values)

    summary = {"dataset": args.dataset, "split": split, "metrics": metrics, "records": records}
    summary_path = base_output / args.dataset / f"{split}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if progress:
        progress.close()

    return summary_path, metrics


def main():
    args = parse_args()

    if args.disable_evidence:
        debate_engine.ENABLE_EVIDENCE = False

    splits = load_dataset_splits(args)
    base_output = Path(args.output_dir)

    ordered_splits = []
    for name in ["train", "val", "test"]:
        if name in splits:
            ordered_splits.append((name, splits[name]))
    for name, items in splits.items():
        if name not in {"train", "val", "test"}:
            ordered_splits.append((name, items))

    for split, items in ordered_splits:
        if args.sample_size:
            items = sample_stratified_even(items, args.sample_size, seed=args.sample_seed)
        print(f"\n=== Running {args.dataset} [{split}] with {len(items)} items ===")
        summary_path, metrics = run_split(args, split, items, base_output)
        accuracy_display = metrics["accuracy"] if metrics["accuracy"] is not None else "N/A"
        print(
            f"Finished {split}: accuracy={accuracy_display} "
            f"(correct={metrics['correct']}, labeled={metrics['labeled']}, failed={metrics['failed']})."
        )
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
