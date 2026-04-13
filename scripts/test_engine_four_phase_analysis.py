import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import Debate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a full Debate instance and inspect four-phase disagreement outputs."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-7B-Instruct",
        help="Hugging Face model name or local model path.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device used by the local model backend.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype used by the local model backend.",
    )
    parser.add_argument(
        "--news-text",
        default="Apple will release a new quantum computer next year.",
        help="News text used for the single-sample debate test.",
    )
    parser.add_argument(
        "--disable-evidence",
        action="store_true",
        help="Disable evidence retrieval for a lighter validation run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["LOCAL_MODEL_DEVICE"] = args.device
    os.environ["LOCAL_MODEL_DTYPE"] = args.dtype

    if args.disable_evidence:
        import engine as debate_engine

        debate_engine.ENABLE_EVIDENCE = False

    debate = Debate(model_name=args.model, T=0.0, sleep=0.0)
    output_path = PROJECT_ROOT / "tmp_four_phase_validation.txt"
    result = debate.run(news_text=args.news_text, news_path=output_path)

    print("=== Four-Phase Debate Validation ===")
    print(f"Verdict: {result['verdict']}")
    print(f"Analysis phases: {sorted(result['analysis_metrics'].keys())}")

    print("\n=== Per-Phase Metrics ===")
    for phase, metrics in result["analysis_metrics"].items():
        print(f"{phase}: similarity={metrics['similarity']:.6f}, disagreement={metrics['disagreement']:.6f}")

    print("\n=== Per-Phase Role Outputs ===")
    for phase, phase_summary in result["analysis_outputs"].items():
        print(f"{phase}:")
        phase_metrics = phase_summary.get("metrics", {})
        if phase_metrics:
            print(f"  metrics roles={phase_metrics.get('roles')} disagreement={phase_metrics.get('disagreement')}")
        for role, role_data in phase_summary.get("roles", {}).items():
            print(
                f"  {role}: pooled_vector_shape={role_data.get('pooled_vector_shape')}, "
                f"generated_token_count={role_data.get('generated_token_count')}, "
                f"sequence_length={role_data.get('sequence_length')}"
            )


if __name__ == "__main__":
    main()
