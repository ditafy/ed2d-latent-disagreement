import argparse
import json
import math
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PHASES = ["opening", "rebuttal", "free", "closing"]
DEFAULT_BINS = 20
DEFAULT_KDE_POINTS = 200
DEFAULT_KDE_BANDWIDTH_FLOOR = 1e-3
DEFAULT_JS_EPSILON = 1e-12


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
        description=(
            "Group summary.json records by is_correct, summarize per-phase disagreement, "
            "and compute discriminative statistics such as Delta, AUC, and JS divergence."
        )
    )
    parser.add_argument(
        "summary_path",
        help="Path to a summary JSON file produced by run_dataset_tests.py.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the grouped analysis JSON.",
    )
    parser.add_argument(
        "--distribution-estimator",
        choices=["histogram", "kde"],
        default="histogram",
        help="How to estimate P(D | correct) and P(D | wrong) for JS divergence.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Number of bins used when --distribution-estimator=histogram.",
    )
    parser.add_argument(
        "--kde-points",
        type=int,
        default=DEFAULT_KDE_POINTS,
        help="Number of evaluation points used when --distribution-estimator=kde.",
    )
    parser.add_argument(
        "--kde-bandwidth",
        type=float,
        default=None,
        help="Optional Gaussian KDE bandwidth. Defaults to Silverman's rule if omitted.",
    )
    parser.add_argument(
        "--js-log-base",
        choices=["2", "e"],
        default="2",
        help="Log base used for JS divergence. Base-2 yields values in [0, 1].",
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


def normalize_distribution(values: Sequence[float], epsilon: float = DEFAULT_JS_EPSILON) -> List[float]:
    adjusted = [max(float(value), 0.0) for value in values]
    total = sum(adjusted)
    if total <= 0:
        if not adjusted:
            return []
        return [1.0 / len(adjusted)] * len(adjusted)
    normalized = [value / total for value in adjusted]
    if epsilon <= 0:
        return normalized
    smoothed = [value + epsilon for value in normalized]
    smoothed_total = sum(smoothed)
    return [value / smoothed_total for value in smoothed]


def build_histogram_distribution(
    values: Sequence[float],
    support_min: float,
    support_max: float,
    bins: int,
) -> List[float]:
    if bins <= 0:
        raise ValueError("--bins must be a positive integer.")

    if not values:
        return [1.0 / bins] * bins

    width = support_max - support_min
    if width <= 0:
        histogram = [0.0] * bins
        histogram[-1] = float(len(values))
        return normalize_distribution(histogram)

    counts = [0.0] * bins
    step = width / bins
    for value in values:
        relative = (value - support_min) / step
        index = int(relative)
        if index < 0:
            index = 0
        elif index >= bins:
            index = bins - 1
        counts[index] += 1.0
    return normalize_distribution(counts)


def estimate_bandwidth(values: Sequence[float]) -> float:
    count = len(values)
    if count <= 1:
        return DEFAULT_KDE_BANDWIDTH_FLOOR

    stats = summarize_scalar(list(values))
    std = stats["std"] or 0.0
    if std <= 0:
        return DEFAULT_KDE_BANDWIDTH_FLOOR

    bandwidth = 1.06 * std * (count ** (-1.0 / 5.0))
    return max(bandwidth, DEFAULT_KDE_BANDWIDTH_FLOOR)


def gaussian_kernel(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def build_kde_distribution(
    values: Sequence[float],
    support_min: float,
    support_max: float,
    points: int,
    bandwidth: Optional[float],
) -> List[float]:
    if points <= 0:
        raise ValueError("--kde-points must be a positive integer.")

    if not values:
        return [1.0 / points] * points

    width = support_max - support_min
    if width <= 0:
        return [1.0 / points] * points

    effective_bandwidth = bandwidth if bandwidth is not None else estimate_bandwidth(values)
    effective_bandwidth = max(effective_bandwidth, DEFAULT_KDE_BANDWIDTH_FLOOR)
    step = width / max(points - 1, 1)

    density = []
    for idx in range(points):
        x = support_min + idx * step
        density_value = 0.0
        for value in values:
            density_value += gaussian_kernel((x - value) / effective_bandwidth)
        density_value /= len(values) * effective_bandwidth
        density.append(density_value)

    return normalize_distribution(density)


def kl_divergence(p: Sequence[float], q: Sequence[float], log_base: str) -> float:
    if len(p) != len(q):
        raise ValueError("KL divergence requires aligned distributions of equal length.")

    log_denom = math.log(2.0) if log_base == "2" else 1.0
    total = 0.0
    for p_value, q_value in zip(p, q):
        if p_value <= 0:
            continue
        total += p_value * (math.log(p_value / q_value) / log_denom)
    return total


def js_divergence(p: Sequence[float], q: Sequence[float], log_base: str) -> float:
    if len(p) != len(q):
        raise ValueError("JS divergence requires aligned distributions of equal length.")

    midpoint = [(p_value + q_value) * 0.5 for p_value, q_value in zip(p, q)]
    return 0.5 * kl_divergence(p, midpoint, log_base) + 0.5 * kl_divergence(q, midpoint, log_base)


def auc_probability(wrong_values: Sequence[float], correct_values: Sequence[float]) -> Optional[float]:
    if not wrong_values or not correct_values:
        return None

    sorted_correct = sorted(correct_values)
    correct_count = len(sorted_correct)
    wins = 0.0
    for wrong_value in wrong_values:
        lower = bisect_left(sorted_correct, wrong_value)
        upper = bisect_right(sorted_correct, wrong_value)
        wins += lower + 0.5 * (upper - lower)
    return wins / (len(wrong_values) * correct_count)


def round_optional(value: Optional[float], digits: int = 6) -> Optional[float]:
    return None if value is None else round(value, digits)


def build_phase_distribution(
    values: Sequence[float],
    support_min: float,
    support_max: float,
    args: argparse.Namespace,
) -> List[float]:
    if args.distribution_estimator == "histogram":
        return build_histogram_distribution(values, support_min, support_max, args.bins)
    return build_kde_distribution(values, support_min, support_max, args.kde_points, args.kde_bandwidth)


def build_phase_discriminative_stats(
    correct_values: Sequence[float],
    wrong_values: Sequence[float],
    args: argparse.Namespace,
) -> Dict[str, Optional[float] | Dict[str, Optional[float]] | str | int]:
    stats: Dict[str, Optional[float] | Dict[str, Optional[float]] | str | int] = {
        "correct_count": len(correct_values),
        "wrong_count": len(wrong_values),
        "delta": None,
        "auc_wrong_gt_correct": None,
        "js_divergence": None,
        "distribution_estimator": args.distribution_estimator,
        "distribution_config": {},
    }

    if not correct_values or not wrong_values:
        return stats

    correct_summary = summarize_scalar(list(correct_values))
    wrong_summary = summarize_scalar(list(wrong_values))
    delta = (wrong_summary["mean"] or 0.0) - (correct_summary["mean"] or 0.0)
    auc = auc_probability(wrong_values, correct_values)

    combined_values = list(correct_values) + list(wrong_values)
    support_min = min(combined_values)
    support_max = max(combined_values)

    if args.distribution_estimator == "histogram":
        distribution_config: Dict[str, Optional[float] | int] = {
            "bins": args.bins,
            "support_min": round(support_min, 6),
            "support_max": round(support_max, 6),
        }
    else:
        distribution_config = {
            "kde_points": args.kde_points,
            "kde_bandwidth": round_optional(args.kde_bandwidth),
            "support_min": round(support_min, 6),
            "support_max": round(support_max, 6),
        }

    pc = build_phase_distribution(correct_values, support_min, support_max, args)
    pw = build_phase_distribution(wrong_values, support_min, support_max, args)
    js = js_divergence(pc, pw, args.js_log_base)

    stats.update(
        {
            "delta": round_optional(delta),
            "auc_wrong_gt_correct": round_optional(auc),
            "js_divergence": round_optional(js),
            "distribution_config": distribution_config,
        }
    )
    return stats


def build_discriminative_summary(
    correct_records: List[Dict],
    incorrect_records: List[Dict],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, Optional[float] | Dict[str, Optional[float]] | str | int]]:
    result = {}
    for phase in PHASES:
        correct_values = collect_phase_values(correct_records, phase)
        wrong_values = collect_phase_values(incorrect_records, phase)
        result[phase] = build_phase_discriminative_stats(correct_values, wrong_values, args)
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
        "discriminative_stats": build_discriminative_summary(correct_records, incorrect_records, args),
    }

    print(f"Summary file: {summary_path}")
    print(f"Dataset: {grouped['dataset']} | Split: {grouped['split']} | Records: {grouped['total_records']}")
    for group_name, group_data in grouped["groups"].items():
        print(f"\n[{group_name}] record_count={group_data['record_count']}")
        for phase in PHASES:
            stats = group_data["phase_stats"][f"{phase}_disagreement_stats"]
            print(f"  {phase}: count={stats['count']} mean={stats['mean']} std={stats['std']}")

    print("\n[discriminative_stats]")
    for phase in PHASES:
        stats = grouped["discriminative_stats"][phase]
        print(
            "  "
            f"{phase}: correct_count={stats['correct_count']} wrong_count={stats['wrong_count']} "
            f"delta={stats['delta']} auc_wrong_gt_correct={stats['auc_wrong_gt_correct']} "
            f"js_divergence={stats['js_divergence']} estimator={stats['distribution_estimator']}"
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
