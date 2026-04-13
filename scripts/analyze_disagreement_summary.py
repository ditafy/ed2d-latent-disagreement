import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional


PHASES = ["opening", "rebuttal", "free", "closing"]


def summarize_scalar(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "std": None}

    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    return {
        "count": count,
        "mean": round(mean, 6),
        "std": round(math.sqrt(variance), 6),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group summary.json records by is_correct and summarize per-phase disagreement."
    )
    parser.add_argument(
        "summary_path",
        help="Path to a summary JSON file produced by run_dataset_tests.py.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the grouped analysis JSON.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_phase_values(records: List[Dict], phase: str) -> List[float]:
    phase_key = f"{phase}_disagreement"
    return [
        float(record[phase_key])
        for record in records
        if record.get(phase_key) is not None
    ]


def build_group_summary(records: List[Dict]) -> Dict[str, Dict[str, Optional[float]]]:
    result: Dict[str, Dict[str, Optional[float]]] = {}
    for phase in PHASES:
        result[f"{phase}_disagreement_stats"] = summarize_scalar(collect_phase_values(records, phase))
    return result


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_path).expanduser().resolve()
    summary = load_summary(summary_path)
    records = summary.get("records", [])

    correct_records = [record for record in records if record.get("is_correct") is True]
    incorrect_records = [record for record in records if record.get("is_correct") is False]
    unknown_records = [record for record in records if record.get("is_correct") is None]

    grouped = {
        "summary_path": str(summary_path),
        "dataset": summary.get("dataset"),
        "split": summary.get("split"),
        "total_records": len(records),
        "groups": {
            "correct": {
                "record_count": len(correct_records),
                "phase_stats": build_group_summary(correct_records),
            },
            "incorrect": {
                "record_count": len(incorrect_records),
                "phase_stats": build_group_summary(incorrect_records),
            },
            "unknown": {
                "record_count": len(unknown_records),
                "phase_stats": build_group_summary(unknown_records),
            },
        },
    }

    print(f"Summary file: {summary_path}")
    print(f"Dataset: {grouped['dataset']} | Split: {grouped['split']} | Records: {grouped['total_records']}")
    for group_name, group_data in grouped["groups"].items():
        print(f"\n[{group_name}] record_count={group_data['record_count']}")
        for phase in PHASES:
            stats = group_data["phase_stats"][f"{phase}_disagreement_stats"]
            print(
                f"  {phase}: count={stats['count']} mean={stats['mean']} std={stats['std']}"
            )

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_dir = Path.cwd() / "batch_results" / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{summary_path.stem}_grouped_analysis.json"
    output_path.write_text(json.dumps(grouped, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGrouped analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
