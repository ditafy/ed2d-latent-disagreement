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
        description="Minimal engine integration test for analysis-mode opening speakers."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["LOCAL_MODEL_DEVICE"] = args.device
    os.environ["LOCAL_MODEL_DTYPE"] = args.dtype

    debate = Debate(model_name=args.model, T=0.0, sleep=0.0)
    debate.news_stem = "tmp_engine_test_news"
    debate._setup_domain_context(args.news_text)

    opening_speakers = ["Affirmative_Opening", "Negative_Opening"]
    opening_template = None
    from config import PHASES  # local import keeps the script lightweight

    for phase, speakers, tpl in PHASES:
        if phase == "Opening":
            opening_template = tpl
            opening_speakers = speakers
            break

    if opening_template is None:
        raise RuntimeError("Opening phase template not found.")

    for turn, speaker in enumerate(opening_speakers, 1):
        prompt = debate._build_prompt(speaker, opening_template, args.news_text, turn, "Opening")
        return_mode = "analysis" if speaker in debate.analysis_targets else "text"
        debate._ask(speaker, prompt, return_mode=return_mode)
    debate._compute_analysis_metrics()

    print("=== Engine Integration Test ===")
    print(f"Domain: {debate.domain}")
    print(f"Profiles generated: {len(debate.profiles)}")

    print("\n=== Transcript Check ===")
    print(f"Transcript entries: {len(debate.transcript)}")
    if debate.transcript:
        print(f"First speaker: {debate.transcript[0]['speaker']}")
        print(f"First text preview: {debate.transcript[0]['text'][:200]}")

    print("\n=== Analysis Output Check ===")
    analysis_outputs = {}
    for role, data in debate.analysis_outputs.items():
        pooled_vector = data.get("pooled_vector")
        analysis_outputs[role] = {
            "pooled_vector_shape": tuple(pooled_vector.shape) if pooled_vector is not None else None,
            "generated_token_count": data.get("generated_token_count"),
            "sequence_length": data.get("sequence_length"),
        }
    print(f"Analysis roles: {list(analysis_outputs.keys())}")
    for role, data in analysis_outputs.items():
        print(f"{role}: pooled_vector_shape={data.get('pooled_vector_shape')}, "
              f"generated_token_count={data.get('generated_token_count')}, "
              f"sequence_length={data.get('sequence_length')}")

    expected_roles = {"Affirmative_Opening", "Negative_Opening"}
    print(f"Has both opening roles: {expected_roles.issubset(set(analysis_outputs.keys()))}")

    print("\n=== Similarity / Disagreement Check ===")
    metrics = debate.analysis_metrics.get("opening", {})
    print(f"Opening metrics present: {bool(metrics)}")
    if metrics:
        print(f"Roles: {metrics.get('roles')}")
        print(f"Similarity: {metrics.get('similarity'):.6f}")
        print(f"Disagreement: {metrics.get('disagreement'):.6f}")

    print("\nEngine integration test completed.")


if __name__ == "__main__":
    main()
