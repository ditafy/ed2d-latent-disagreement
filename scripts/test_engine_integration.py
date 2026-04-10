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
    parser.add_argument(
        "--analysis-phases",
        default="Opening,Rebuttal,Free,Closing",
        help="Comma-separated debate phases to enable for analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["LOCAL_MODEL_DEVICE"] = args.device
    os.environ["LOCAL_MODEL_DTYPE"] = args.dtype

    debate = Debate(model_name=args.model, T=0.0, sleep=0.0)
    debate.news_stem = "tmp_engine_test_news"
    debate.enabled_analysis_phases = {
        phase.strip() for phase in args.analysis_phases.split(",") if phase.strip()
    }
    debate._setup_domain_context(args.news_text)
    from config import PHASES  # local import keeps the script lightweight
    for phase, speakers, template in PHASES:
        seq = debate._get_speakers_sequence(phase, speakers)
        for turn, speaker in enumerate(seq, 1):
            prompt = debate._build_prompt(speaker, template, args.news_text, turn, phase)
            target_roles = set(debate.analysis_targets.get(phase, []))
            return_mode = (
                "analysis"
                if phase in debate.enabled_analysis_phases and speaker in target_roles
                else "text"
            )
            debate._ask(speaker, prompt, return_mode=return_mode)
    debate._compute_analysis_metrics()

    print("=== Engine Integration Test ===")
    print(f"Domain: {debate.domain}")
    print(f"Profiles generated: {len(debate.profiles)}")
    print(f"Enabled analysis phases: {sorted(debate.enabled_analysis_phases)}")

    print("\n=== Transcript Check ===")
    print(f"Transcript entries: {len(debate.transcript)}")
    if debate.transcript:
        print(f"First speaker: {debate.transcript[0]['speaker']}")
        print(f"First text preview: {debate.transcript[0]['text'][:200]}")

    print("\n=== Analysis Output Check ===")
    analysis_outputs = {}
    for phase, role_data in debate.analysis_outputs.items():
        analysis_outputs[phase] = {}
        for role, data in role_data.items():
            pooled_vector = data.get("pooled_vector")
            analysis_outputs[phase][role] = {
                "pooled_vector_shape": tuple(pooled_vector.shape) if pooled_vector is not None else None,
                "generated_token_count": data.get("generated_token_count"),
                "sequence_length": data.get("sequence_length"),
            }
    print(f"Analysis phases with outputs: {[phase for phase, data in analysis_outputs.items() if data]}")
    for phase, role_data in analysis_outputs.items():
        if not role_data:
            continue
        print(f"{phase}:")
        for role, data in role_data.items():
            print(f"  {role}: pooled_vector_shape={data.get('pooled_vector_shape')}, "
                  f"generated_token_count={data.get('generated_token_count')}, "
                  f"sequence_length={data.get('sequence_length')}")

    print("\n=== Similarity / Disagreement Check ===")
    print(f"Metric phases: {list(debate.analysis_metrics.keys())}")
    for phase, metrics in debate.analysis_metrics.items():
        print(f"{phase}:")
        print(f"  Roles: {metrics.get('roles')}")
        print(f"  Similarity: {metrics.get('similarity'):.6f}")
        print(f"  Disagreement: {metrics.get('disagreement'):.6f}")
        print(f"  Count: {metrics.get('count')}")
        print(f"  Mean: {metrics.get('mean'):.6f}")
        print(f"  Std: {metrics.get('std'):.6f}")

    print("\nEngine integration test completed.")


if __name__ == "__main__":
    main()
