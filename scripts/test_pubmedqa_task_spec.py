import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task_specs import get_task_spec, parse_pubmedqa_verdict


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_label_normalization(spec) -> None:
    cases = [
        ("yes", "YES"),
        ("YES", "YES"),
        ("y", "YES"),
        ("true", "YES"),
        ("1", "YES"),
        ("no", "NO"),
        ("NO", "NO"),
        ("n", "NO"),
        ("false", "NO"),
        ("0", "NO"),
        ("maybe", "MAYBE"),
        ("MAYBE", "MAYBE"),
        ("uncertain", "MAYBE"),
        ("unknown", "MAYBE"),
        ("cannot determine", "MAYBE"),
        ("insufficient", "MAYBE"),
        ("other", None),
        (None, None),
    ]
    for raw, expected in cases:
        observed = spec.normalize_label(raw)
        require(observed == expected, f"normalize_label({raw!r}) -> {observed!r}, expected {expected!r}")


def test_verdict_parser() -> None:
    cases = [
        ("YES", "YES"),
        ("NO", "NO"),
        ("MAYBE", "MAYBE"),
        ("Final answer: YES", "YES"),
        ("Verdict: NO", "NO"),
        ("Answer: maybe.", "MAYBE"),
        ("Label - YES", "YES"),
        ("Decision: no", "NO"),
        ("The answer is YES.", "YES"),
        ("The answer is NO.", "NO"),
        ("The evidence is insufficient.", "MAYBE"),
        ("Cannot determine from the abstract.", "MAYBE"),
        ("The findings are inconclusive.", "MAYBE"),
        ("There is mixed evidence.", "MAYBE"),
        ("YES and NO are both plausible.", "MAYBE"),
        ("", "MAYBE"),
        (None, "MAYBE"),
    ]
    for raw, expected in cases:
        observed = parse_pubmedqa_verdict(raw)
        require(observed == expected, f"parse_pubmedqa_verdict({raw!r}) -> {observed!r}, expected {expected!r}")


def test_correctness(spec) -> None:
    require(spec.is_correct("YES", "YES") is True, "YES/YES should be correct.")
    require(spec.is_correct("NO", "NO") is True, "NO/NO should be correct.")
    require(spec.is_correct("MAYBE", "MAYBE") is True, "MAYBE/MAYBE should be correct.")
    require(spec.is_correct("YES", "NO") is False, "YES/NO should be incorrect.")
    require(spec.is_correct("MAYBE", "YES") is False, "MAYBE should not match YES.")
    require(spec.is_correct(None, "MAYBE") is False, "None verdict should be incorrect.")


def test_prompt_boundaries(spec) -> None:
    joined = "\n".join(spec.phase_templates.values())
    joined += "\n" + spec.affirmative_stance
    joined += "\n" + spec.negative_stance
    joined += "\n" + spec.affirmative_stance_reminder
    joined += "\n" + spec.negative_stance_reminder
    joined += "\n" + spec.judge_prompt_template
    joined += "\n" + spec.summary_prompt_template
    lower = joined.lower()

    required_terms = [
        "biomedical question",
        "abstract context",
        "yes",
        "no",
        "maybe",
        "evidence",
    ]
    forbidden_terms = [
        "fake news",
        "real news",
        "news is true",
        "news is false",
        "original news article",
        "factual authenticity of the original news",
        "long_answer",
        "final_decision",
        "reasoning_required_pred",
        "reasoning_free_pred",
    ]

    for term in required_terms:
        require(term in lower, f"PubMedQA prompt text should include {term!r}.")
    for term in forbidden_terms:
        require(term not in lower, f"PubMedQA prompt text should not include {term!r}.")


def main() -> None:
    spec = get_task_spec("pubmedqa")
    alias_spec = get_task_spec("pubmed_qa")
    require(spec is alias_spec, "Expected pubmedqa and pubmed_qa to resolve to the same task spec.")
    require(spec.name == "pubmedqa", "Expected pubmedqa task spec.")
    require(spec.task_type == "biomedical_qa", "Expected biomedical_qa task type.")
    require(spec.answer_type == "yes_no_maybe", "Expected yes_no_maybe answer type.")
    require(spec.verdict_labels == ("YES", "NO", "MAYBE"), "Unexpected PubMedQA labels.")
    require(spec.enable_evidence is False, "PubMedQA evidence should be disabled by default.")

    test_label_normalization(spec)
    test_verdict_parser()
    test_correctness(spec)
    test_prompt_boundaries(spec)

    print("PubMedQA task spec validation passed.")
    print(f"Task: {spec.name}")
    print(f"Task type: {spec.task_type}")
    print(f"Answer type: {spec.answer_type}")
    print(f"Verdict labels: {spec.verdict_labels}")
    print("Prompt boundary check: no fake-news or leakage wording found in PubMedQA spec.")


if __name__ == "__main__":
    main()
