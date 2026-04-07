import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import Agent, AgentResponse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal integration test for the new Agent interface."
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
        help="Device used by the agent backend.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype used by the agent backend.",
    )
    parser.add_argument(
        "--prompt",
        default="Please briefly assess this claim: Apple will release a quantum computer next year.",
        help="Prompt passed to the agent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["LOCAL_MODEL_DEVICE"] = args.device
    os.environ["LOCAL_MODEL_DTYPE"] = args.dtype

    agent = Agent(
        model_name=args.model,
        name="IntegrationTester",
        temperature=0.0,
        sleep_time=0.0,
    )
    agent.set_meta_prompt("You are a concise assistant.")

    print("=== Agent text mode test ===")
    text_result = agent.ask([], args.prompt, return_mode="text")
    print(f"Type: {type(text_result).__name__}")
    print(f"Text preview: {text_result[:200]}")

    print("\n=== Agent analysis mode test ===")
    analysis_result = agent.ask([], args.prompt, return_mode="analysis")
    print(f"Type: {type(analysis_result).__name__}")
    print(f"Is AgentResponse: {isinstance(analysis_result, AgentResponse)}")
    print(f"Text preview: {analysis_result.text[:200]}")
    print(f"Pooled vector present: {analysis_result.pooled_vector is not None}")
    if analysis_result.pooled_vector is not None:
        print(f"Pooled vector shape: {tuple(analysis_result.pooled_vector.shape)}")
    print(f"Generated token count: {analysis_result.generated_token_count}")
    print(f"Sequence length: {analysis_result.sequence_length}")

    print("\nAgent interface integration test passed.")


if __name__ == "__main__":
    main()
