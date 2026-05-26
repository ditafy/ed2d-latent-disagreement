import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task_specs import get_task_spec, parse_strategyqa_verdict


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_label_normalization(spec) -> None:
    cases = [
        (True, "YES"),
        (False, "NO"),
        ("true", "YES"),
        ("false", "NO"),
        ("yes", "YES"),
        ("no", "NO"),
        ("1", "YES"),
        ("0", "NO"),
        ("unknown", None),
        (None, None),
    ]
    for raw, expected in cases:
        observed = spec.normalize_label(raw)
        require(observed == expected, f"normalize_label({raw!r}) -> {observed!r}, expected {expected!r}")


def test_verdict_parser() -> None:
    cases = [
        ("YES", "YES"),
        ("NO", "NO"),
        ("UNCERTAIN", "UNCERTAIN"),
        ("Final answer: YES", "YES"),
        ("Verdict: NO", "NO"),
        ("Answer: yes.", "YES"),
        ("Label - no", "NO"),
        ("The answer is YES.", "YES"),
        ("I am uncertain.", "UNCERTAIN"),
        ("Cannot determine from the debate.", "UNCERTAIN"),
        ("YES and NO are both possible.", "UNCERTAIN"),
        ("", "UNCERTAIN"),
        (None, "UNCERTAIN"),
    ]
    for raw, expected in cases:
        observed = parse_strategyqa_verdict(raw)
        require(observed == expected, f"parse_strategyqa_verdict({raw!r}) -> {observed!r}, expected {expected!r}")


def test_correctness(spec) -> None:
    require(spec.is_correct("YES", "YES") is True, "YES/YES should be correct.")
    require(spec.is_correct("NO", "NO") is True, "NO/NO should be correct.")
    require(spec.is_correct("YES", "NO") is False, "YES/NO should be incorrect.")
    require(spec.is_correct("UNCERTAIN", "YES") is False, "UNCERTAIN should not match YES.")
    require(spec.is_correct(None, "YES") is False, "None verdict should be incorrect.")


def test_prompt_boundaries(spec) -> None:
    joined = "\n".join(spec.phase_templates.values())
    joined += "\n" + spec.affirmative_stance
    joined += "\n" + spec.negative_stance
    joined += "\n" + spec.affirmative_stance_reminder
    joined += "\n" + spec.negative_stance_reminder
    joined += "\n" + spec.judge_prompt_template
    lower = joined.lower()

    required_terms = ["question", "yes", "no", "answer"]
    forbidden_terms = [
        "fake news",
        "real news",
        "news is true",
        "news is false",
        "original news article",
        "factual authenticity of the original news",
    ]

    for term in required_terms:
        require(term in lower, f"StrategyQA prompt text should include {term!r}.")
    for term in forbidden_terms:
        require(term not in lower, f"StrategyQA prompt text should not include {term!r}.")


def main() -> None:
    spec = get_task_spec("strategyqa")
    require(spec.name == "strategyqa", "Expected strategyqa task spec.")
    require(spec.task_type == "binary_reasoning", "Expected binary_reasoning task type.")
    require(spec.answer_type == "yes_no", "Expected yes_no answer type.")
    require(spec.verdict_labels == ("YES", "NO", "UNCERTAIN"), "Unexpected StrategyQA labels.")
    require(spec.enable_evidence is False, "StrategyQA evidence should be disabled by default.")

    test_label_normalization(spec)
    test_verdict_parser()
    test_correctness(spec)
    test_prompt_boundaries(spec)

    print("StrategyQA task spec validation passed.")
    print(f"Task: {spec.name}")
    print(f"Task type: {spec.task_type}")
    print(f"Answer type: {spec.answer_type}")
    print(f"Verdict labels: {spec.verdict_labels}")
    print("Prompt boundary check: no fake-news wording found in StrategyQA spec.")


if __name__ == "__main__":
    main()
