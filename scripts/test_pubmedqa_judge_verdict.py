import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import engine
from task_specs import get_task_spec, parse_pubmedqa_verdict


PUBMEDQA_INPUT = (
    "Question: Is there a connection between sublingual varices and hypertension?\n\n"
    "Abstract context:\n"
    "[BACKGROUND] Sublingual varices have earlier been related to ageing, smoking and cardiovascular disease.\n\n"
    "[METHODS] In an observational clinical study among dental patients tongue status and blood pressure were documented.\n\n"
    "[RESULTS] An association between sublingual varices and hypertension was found.\n\n"
    "Decide whether the answer to the biomedical question is YES, NO, or MAYBE."
)


class FakeRoleAgent:
    prompt_log = []
    verdict_response = "YES"

    def __init__(self, name):
        self.name = name
        self.system_prompt = ""

    def set_meta_prompt(self, prompt):
        self.system_prompt = prompt

    def ask(self, shared_memory, prompt, temperature=None, return_mode="text"):
        self.prompt_log.append((self.name, prompt, return_mode))
        if self.name == "Judge_Summary":
            if "Return ONLY one of the following labels: YES, NO, or MAYBE" in prompt:
                return self.verdict_response
            return "Summary generated from parsed verdict."
        return f"{self.name} text response."


class FakeDomainAgent:
    def __init__(self, model_name, name, temperature=0.0):
        self.name = name
        self.system_prompt = ""

    def set_meta_prompt(self, prompt):
        self.system_prompt = prompt

    def ask(self, shared_memory, prompt, temperature=None, return_mode="text"):
        return "biomedical research"


def fake_build_agent(cfg, model_name, temperature, sleep):
    agent = FakeRoleAgent(cfg.name)
    agent.set_meta_prompt(cfg.meta_prompt)
    return agent


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def test_parser_cases():
    cases = [
        ("YES", "YES"),
        ("Final answer: NO", "NO"),
        ("Verdict: MAYBE", "MAYBE"),
        ("The evidence is insufficient to determine.", "MAYBE"),
        ("YES and NO are both plausible from the abstract.", "MAYBE"),
        ("This is inconclusive.", "MAYBE"),
        ("unparseable free-form response", "MAYBE"),
    ]
    for raw, expected in cases:
        observed = parse_pubmedqa_verdict(raw)
        require(observed == expected, f"Parser case {raw!r} -> {observed!r}, expected {expected!r}")


def test_engine_judge_case(raw_verdict, expected_verdict):
    FakeRoleAgent.prompt_log = []
    FakeRoleAgent.verdict_response = raw_verdict

    spec = get_task_spec("pubmedqa")
    debate = engine.Debate(model_name="fake-model", T=0.0, sleep=0.0, task_spec=spec)
    debate.news_stem = "pubmedqa_judge_verdict_test"
    debate.transcript = [
        {"speaker": "Affirmative_Opening", "text": "The abstract supports YES."},
        {"speaker": "Negative_Opening", "text": "The abstract supports NO or uncertainty."},
    ]

    scores, verdict, summary = debate._judge_task_verdict(PUBMEDQA_INPUT)

    require(scores == {}, "PubMedQA judge should not return side scores.")
    require(verdict == expected_verdict, f"Engine verdict {verdict!r}, expected {expected_verdict!r}")
    require(summary == "Summary generated from parsed verdict.", "Summary response mismatch.")

    prompts = [prompt for name, prompt, _ in FakeRoleAgent.prompt_log if name == "Judge_Summary"]
    require(len(prompts) == 2, f"Expected verdict and summary prompts, got {len(prompts)}.")
    verdict_prompt, summary_prompt = prompts

    require(
        "Return ONLY one of the following labels: YES, NO, or MAYBE" in verdict_prompt,
        "PubMedQA judge verdict prompt missing strict label instruction.",
    )
    require(
        "Return MAYBE when the abstract evidence is insufficient, mixed, or does not clearly support YES or NO"
        in verdict_prompt,
        "PubMedQA judge verdict prompt missing MAYBE decision rule.",
    )
    require("Affirmative argues that the answer is YES" in verdict_prompt, "Affirmative judge rule missing.")
    require("Negative argues that the answer is NO" in verdict_prompt, "Negative judge rule missing.")
    require("Final verdict: " + expected_verdict in summary_prompt, "Summary prompt did not receive parsed verdict.")

    lower_prompts = "\n".join(prompts).lower()
    for forbidden in [
        "fake news",
        "real news",
        "factual authenticity",
        "original news article",
        "long_answer",
        "final_decision",
        "reasoning_required_pred",
        "reasoning_free_pred",
    ]:
        require(forbidden not in lower_prompts, f"PubMedQA judge prompt leaked {forbidden!r}")


def main():
    original_build_agent = engine.build_agent
    original_agent = engine.Agent
    original_auto_save = engine.AUTO_SAVE
    original_enable_evidence = engine.ENABLE_EVIDENCE

    try:
        engine.build_agent = fake_build_agent
        engine.Agent = FakeDomainAgent
        engine.AUTO_SAVE = False
        engine.ENABLE_EVIDENCE = False

        test_parser_cases()
        cases = [
            ("YES", "YES"),
            ("Final answer: NO", "NO"),
            ("Verdict: MAYBE", "MAYBE"),
            ("The abstract evidence is insufficient.", "MAYBE"),
            ("YES and NO are both plausible.", "MAYBE"),
            ("unexpected narrative", "MAYBE"),
        ]
        for raw, expected in cases:
            test_engine_judge_case(raw, expected)

        print("PubMedQA judge/verdict integration validation passed.")
        print(f"Cases tested: {len(cases)} engine cases + parser-only cases")
        print("Verified strict judge prompt, MAYBE fallback, parsed verdict summary handoff, and no fake-news leakage.")
    finally:
        engine.build_agent = original_build_agent
        engine.Agent = original_agent
        engine.AUTO_SAVE = original_auto_save
        engine.ENABLE_EVIDENCE = original_enable_evidence


if __name__ == "__main__":
    main()
