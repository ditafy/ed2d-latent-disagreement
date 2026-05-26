import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import engine
from task_specs import get_task_spec


class FakeAgentResponse:
    def __init__(self, text, pooled_vector=None):
        self.text = text
        self.pooled_vector = pooled_vector
        self.generated_token_count = 4
        self.sequence_length = 12


class FakeRoleAgent:
    prompt_log = []

    def __init__(self, name):
        self.name = name
        self.system_prompt = ""

    def set_meta_prompt(self, prompt):
        self.system_prompt = prompt

    def ask(self, shared_memory, prompt, temperature=None, return_mode="text"):
        self.prompt_log.append((self.name, prompt, return_mode))

        if self.name == "Judge_Summary":
            if "Return ONLY one of the following labels" in prompt:
                return "Final answer: YES"
            return "Summary: the debate supports YES."

        if return_mode == "analysis":
            if self.name.startswith("Affirmative"):
                vector = torch.tensor([[1.0, 0.0, 0.0]])
            else:
                vector = torch.tensor([[0.0, 1.0, 0.0]])
            return FakeAgentResponse(
                text=f"{self.name} argues from the assigned StrategyQA stance.",
                pooled_vector=vector,
            )

        return f"{self.name} profile or text response."


class FakeDomainAgent:
    def __init__(self, model_name, name, temperature=0.0):
        self.name = name
        self.system_prompt = ""

    def set_meta_prompt(self, prompt):
        self.system_prompt = prompt

    def ask(self, shared_memory, prompt, temperature=None, return_mode="text"):
        return "commonsense reasoning"


def fake_build_agent(cfg, model_name, temperature, sleep):
    agent = FakeRoleAgent(cfg.name)
    agent.set_meta_prompt(cfg.meta_prompt)
    return agent


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    original_build_agent = engine.build_agent
    original_agent = engine.Agent
    original_auto_save = engine.AUTO_SAVE
    original_enable_evidence = engine.ENABLE_EVIDENCE

    try:
        FakeRoleAgent.prompt_log = []
        engine.build_agent = fake_build_agent
        engine.Agent = FakeDomainAgent
        engine.AUTO_SAVE = False
        engine.ENABLE_EVIDENCE = False

        spec = get_task_spec("strategyqa")
        debate = engine.Debate(model_name="fake-model", T=0.0, sleep=0.0, task_spec=spec)
        result = debate.run(
            news_text=(
                "Question: Was ship that recovered Apollo 13 named after a World War II battle?\n"
                "Term: Apollo 13\n"
                "Description: A failed crewed mission to land on the Moon\n\n"
                "Decide whether the answer is YES or NO."
            ),
            news_path=PROJECT_ROOT / "tmp_strategyqa_engine_test.txt",
        )

        require(result["verdict"] == "YES", f"Expected YES verdict, got {result['verdict']!r}")
        require(result["scores"] == {}, "StrategyQA should not use fake-news side scoring.")

        expected_phases = {"opening", "rebuttal", "free", "closing"}
        observed_phases = set(result["analysis_metrics"])
        require(observed_phases == expected_phases, f"Observed phases {observed_phases}, expected {expected_phases}")

        for phase, metrics in result["analysis_metrics"].items():
            require("similarity" in metrics, f"{phase} missing similarity")
            require("disagreement" in metrics, f"{phase} missing disagreement")
            require(metrics["roles"][0].startswith("Affirmative"), f"{phase} left role should be Affirmative")
            require(metrics["roles"][1].startswith("Negative"), f"{phase} right role should be Negative")

        prompts = "\n".join(prompt for _, prompt, _ in FakeRoleAgent.prompt_log)
        lower_prompts = prompts.lower()
        for required in ["question", "yes", "no", "answer"]:
            require(required in lower_prompts, f"StrategyQA prompts should include {required!r}")
        for forbidden in [
            "fake news",
            "real news",
            "news is true",
            "news is false",
            "original news article",
            "factual authenticity of the original news",
        ]:
            require(forbidden not in lower_prompts, f"StrategyQA prompts leaked {forbidden!r}")

        require(
            "**Your fixed stance is that the answer is YES.**" in prompts,
            "Affirmative StrategyQA stance reminder missing.",
        )
        require(
            "**Your fixed stance is that the answer is NO.**" in prompts,
            "Negative StrategyQA stance reminder missing.",
        )

        print("StrategyQA engine task-specific prompt validation passed.")
        print(f"Verdict: {result['verdict']}")
        print(f"Metric phases: {sorted(result['analysis_metrics'])}")
        for phase, metrics in sorted(result["analysis_metrics"].items()):
            print(f"{phase}: disagreement={metrics['disagreement']:.6f}")
    finally:
        engine.build_agent = original_build_agent
        engine.Agent = original_agent
        engine.AUTO_SAVE = original_auto_save
        engine.ENABLE_EVIDENCE = original_enable_evidence


if __name__ == "__main__":
    main()
