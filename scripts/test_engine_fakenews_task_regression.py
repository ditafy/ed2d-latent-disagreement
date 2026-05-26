import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import engine


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

        if self.name.startswith("Judge_") and self.name != "Judge_Summary":
            return '{"Affirmative": 4, "Negative": 3}'
        if self.name == "Judge_Summary":
            return "Summary: the debate supports REAL."

        if return_mode == "analysis":
            if self.name.startswith("Affirmative"):
                vector = torch.tensor([[1.0, 0.0, 0.0]])
            else:
                vector = torch.tensor([[0.0, 1.0, 0.0]])
            return FakeAgentResponse(
                text=f"{self.name} argues from the assigned fake-news stance.",
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
        return "technology"


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

        debate = engine.Debate(model_name="fake-model", T=0.0, sleep=0.0)
        result = debate.run(
            news_text="Apple will release a new quantum computer next year.",
            news_path=PROJECT_ROOT / "tmp_fakenews_engine_test.txt",
        )

        require(result["verdict"] == "REAL", f"Expected REAL verdict, got {result['verdict']!r}")
        require(result["scores"] == {"Affirmative": 20, "Negative": 15}, "FakeNews side scoring changed.")

        expected_phases = {"opening", "rebuttal", "free", "closing"}
        observed_phases = set(result["analysis_metrics"])
        require(observed_phases == expected_phases, f"Observed phases {observed_phases}, expected {expected_phases}")

        prompts = "\n".join(prompt for _, prompt, _ in FakeRoleAgent.prompt_log)
        lower_prompts = prompts.lower()
        for required in ["news", "news is true", "news is false", "factual authenticity"]:
            require(required in lower_prompts, f"Default fake-news prompts should include {required!r}")

        print("FakeNews engine task regression passed.")
        print(f"Verdict: {result['verdict']}")
        print(f"Scores: {result['scores']}")
        print(f"Metric phases: {sorted(result['analysis_metrics'])}")
    finally:
        engine.build_agent = original_build_agent
        engine.Agent = original_agent
        engine.AUTO_SAVE = original_auto_save
        engine.ENABLE_EVIDENCE = original_enable_evidence


if __name__ == "__main__":
    main()
