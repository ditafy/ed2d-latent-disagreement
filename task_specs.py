"""Task-specific configuration for ED2D-style benchmark runs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, Tuple


NormalizeLabelFn = Callable[[object], str | None]
ParseVerdictFn = Callable[[str | None], str]
CorrectnessFn = Callable[[str | None, str | None], bool]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_type: str
    answer_type: str
    input_name: str
    affirmative_stance: str
    negative_stance: str
    affirmative_stance_reminder: str
    negative_stance_reminder: str
    phase_templates: Dict[str, str]
    judge_prompt_template: str
    summary_prompt_template: str
    verdict_labels: Tuple[str, ...]
    normalize_label: NormalizeLabelFn
    parse_verdict: ParseVerdictFn
    is_correct: CorrectnessFn
    enable_evidence: bool = False


def normalize_fakenews_label(label: object) -> str | None:
    if label is None:
        return None

    value = str(label).strip().lower()
    if not value or value in {"nan", "none"}:
        return None

    fake_tags = {"1", "fake", "false", "rumor", "rumour", "spam", "misinformation"}
    real_tags = {"0", "real", "true", "nonrumor", "non-rumor", "legit", "legitimate"}

    if value in fake_tags:
        return "FAKE"
    if value in real_tags:
        return "REAL"
    return None


def parse_fakenews_verdict(text: str | None) -> str:
    if text is None:
        return "UNCERTAIN"
    value = str(text).strip().upper()
    if value in {"REAL", "FAKE", "UNCERTAIN"}:
        return value
    if re.search(r"\bREAL\b", value) and not re.search(r"\bFAKE\b", value):
        return "REAL"
    if re.search(r"\bFAKE\b", value) and not re.search(r"\bREAL\b", value):
        return "FAKE"
    if re.search(r"\bUNCERTAIN\b", value):
        return "UNCERTAIN"
    return "UNCERTAIN"


def normalize_strategyqa_label(label: object) -> str | None:
    if isinstance(label, bool):
        return "YES" if label else "NO"
    if label is None:
        return None

    value = str(label).strip().lower()
    if not value or value in {"nan", "none"}:
        return None

    yes_values = {"true", "yes", "y", "1"}
    no_values = {"false", "no", "n", "0"}
    if value in yes_values:
        return "YES"
    if value in no_values:
        return "NO"
    return None


def normalize_pubmedqa_label(label: object) -> str | None:
    if label is None:
        return None

    value = str(label).strip().lower()
    if not value or value in {"nan", "none"}:
        return None

    if value in {"yes", "y", "true", "1"}:
        return "YES"
    if value in {"no", "n", "false", "0"}:
        return "NO"
    if value in {"maybe", "uncertain", "unknown", "cannot determine", "insufficient"}:
        return "MAYBE"
    return None


def parse_strategyqa_verdict(text: str | None) -> str:
    if text is None:
        return "UNCERTAIN"

    raw = str(text).strip()
    if not raw:
        return "UNCERTAIN"

    upper = raw.upper()
    exact = upper.strip(" .:;\"'")
    if exact in {"YES", "NO", "UNCERTAIN"}:
        return exact

    if re.search(r"\bUNCERTAIN\b|\bUNKNOWN\b|\bCANNOT\s+DETERMINE\b", upper):
        return "UNCERTAIN"

    explicit_patterns = [
        r"\bFINAL\s+ANSWER\s*[:=\-]\s*(YES|NO|UNCERTAIN)\b",
        r"\bVERDICT\s*[:=\-]\s*(YES|NO|UNCERTAIN)\b",
        r"\bANSWER\s*[:=\-]\s*(YES|NO|UNCERTAIN)\b",
        r"\bLABEL\s*[:=\-]\s*(YES|NO|UNCERTAIN)\b",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)

    yes_match = re.search(r"\bYES\b", upper)
    no_match = re.search(r"\bNO\b", upper)
    if yes_match and not no_match:
        return "YES"
    if no_match and not yes_match:
        return "NO"
    return "UNCERTAIN"


def parse_pubmedqa_verdict(text: str | None) -> str:
    if text is None:
        return "MAYBE"

    raw = str(text).strip()
    if not raw:
        return "MAYBE"

    upper = raw.upper()
    exact = upper.strip(" .:;\"'")
    if exact in {"YES", "NO", "MAYBE"}:
        return exact

    uncertainty_pattern = (
        r"\bMAYBE\b|\bUNCERTAIN\b|\bUNKNOWN\b|"
        r"\bCANNOT\s+DETERMINE\b|\bINSUFFICIENT\b|\bINCONCLUSIVE\b|"
        r"\bMIXED\s+EVIDENCE\b"
    )
    if re.search(uncertainty_pattern, upper):
        return "MAYBE"

    explicit_patterns = [
        r"\bFINAL\s+ANSWER\s*[:=\-]\s*(YES|NO|MAYBE)\b",
        r"\bVERDICT\s*[:=\-]\s*(YES|NO|MAYBE)\b",
        r"\bANSWER\s*[:=\-]\s*(YES|NO|MAYBE)\b",
        r"\bLABEL\s*[:=\-]\s*(YES|NO|MAYBE)\b",
        r"\bDECISION\s*[:=\-]\s*(YES|NO|MAYBE)\b",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)

    yes_match = re.search(r"\bYES\b", upper)
    no_match = re.search(r"\bNO\b", upper)
    maybe_match = re.search(r"\bMAYBE\b", upper)
    if maybe_match:
        return "MAYBE"
    if yes_match and not no_match:
        return "YES"
    if no_match and not yes_match:
        return "NO"
    return "MAYBE"


def exact_match_correct(verdict: str | None, gold: str | None) -> bool:
    return verdict is not None and gold is not None and verdict == gold


FAKENEWS_PHASE_TEMPLATES = {
    "Opening": (
        "The news is:\n\"\"\"{input_text}\"\"\"\n"
        "Give your opening statement defending your fixed stance. Concise and Comprehensive"
    ),
    "Rebuttal": "Please rebut your opponent's opening statement above. Concise and Comprehensive",
    "Free": (
        "Free-debate round {turn}. "
        "Your opponent just said:\n\"{opp}\"\nRespond accordingly. Concise and Comprehensive"
    ),
    "Free_Evidence": (
        "Free-debate round {turn} with evidence support. "
        "Your opponent just said:\n\"{opp}\"\n\n"
        "Available evidence:\n{evidence}\n\n"
        "Use the evidence above to support your arguments and respond accordingly. Concise and Comprehensive"
    ),
    "Closing": "Summarise your team's arguments and present your closing statement. Concise and Comprehensive",
}


STRATEGYQA_PHASE_TEMPLATES = {
    "Opening": (
        "The question is:\n\"\"\"{input_text}\"\"\"\n"
        "Give your opening statement defending your fixed YES/NO stance. "
        "Focus on concise factual and commonsense reasoning."
    ),
    "Rebuttal": (
        "Please rebut your opponent's opening statement above. "
        "Focus on factual or reasoning errors while defending your fixed YES/NO stance."
    ),
    "Free": (
        "Free-debate round {turn}. "
        "Your opponent just said:\n\"{opp}\"\n"
        "Respond by addressing their reasoning and defending your YES/NO stance."
    ),
    "Closing": (
        "Summarise your final reasoning and clearly state whether the answer is YES or NO."
    ),
}


PUBMEDQA_PHASE_TEMPLATES = {
    "Opening": (
        "The biomedical question and abstract context are:\n\"\"\"{input_text}\"\"\"\n"
        "Give your opening statement defending your fixed YES/NO stance. "
        "Base your argument only on the provided abstract context, and focus on evidence strength."
    ),
    "Rebuttal": (
        "Please rebut your opponent's opening statement above. "
        "Identify misread evidence, unsupported causal claims, statistical overreach, or uncertainty "
        "while defending your fixed YES/NO stance."
    ),
    "Free": (
        "Free-debate round {turn}. "
        "Your opponent just said:\n\"{opp}\"\n"
        "Respond by addressing their interpretation of the abstract evidence and defending your YES/NO stance."
    ),
    "Closing": (
        "Summarise your final biomedical reasoning from the abstract context and clearly state whether "
        "your side supports YES or NO. Mention uncertainty only when the evidence is insufficient."
    ),
}


FAKENEWS_TASK_SPEC = TaskSpec(
    name="fakenewsdataset",
    task_type="misinformation_detection",
    answer_type="real_fake",
    input_name="news",
    affirmative_stance="You believe the news is true and need to argue in its favor.",
    negative_stance="You believe the news is false and need to argue against it.",
    affirmative_stance_reminder="**Your fixed stance is that the news is true.**",
    negative_stance_reminder="**Your fixed stance is that the news is false.**",
    phase_templates=FAKENEWS_PHASE_TEMPLATES,
    judge_prompt_template=(
        "The original news article is:\n{input_text}\n"
        "Debate content:\n{debate_content}\n"
        "{evidence_context}\n"
        "Your task is to assess the factual authenticity of the original news.\n"
        "Affirmatives defend the news as factual.\n"
        "Negatives argue that the news is misleading.\n"
        "Return ONLY one label: REAL, FAKE, or UNCERTAIN."
    ),
    summary_prompt_template=(
        "The original news article is:\n{input_text}\n"
        "Debate content:\n{debate_content}\n"
        "{evidence_context}\n"
        "Final verdict: {verdict}\n\n"
        "Please provide your summary based on all the above information."
    ),
    verdict_labels=("REAL", "FAKE", "UNCERTAIN"),
    normalize_label=normalize_fakenews_label,
    parse_verdict=parse_fakenews_verdict,
    is_correct=exact_match_correct,
    enable_evidence=True,
)


STRATEGYQA_TASK_SPEC = TaskSpec(
    name="strategyqa",
    task_type="binary_reasoning",
    answer_type="yes_no",
    input_name="question",
    affirmative_stance="You believe the correct answer is YES and need to argue in favor of YES.",
    negative_stance="You believe the correct answer is NO and need to argue in favor of NO.",
    affirmative_stance_reminder="**Your fixed stance is that the answer is YES.**",
    negative_stance_reminder="**Your fixed stance is that the answer is NO.**",
    phase_templates=STRATEGYQA_PHASE_TEMPLATES,
    judge_prompt_template=(
        "The original StrategyQA question is:\n{input_text}\n\n"
        "Debate content:\n{debate_content}\n\n"
        "Your task is to decide the correct answer to the question.\n"
        "Affirmative argues that the answer is YES.\n"
        "Negative argues that the answer is NO.\n\n"
        "Return ONLY one of the following labels: YES, NO, or UNCERTAIN."
    ),
    summary_prompt_template=(
        "The original StrategyQA question is:\n{input_text}\n\n"
        "Debate content:\n{debate_content}\n\n"
        "Final verdict: {verdict}\n\n"
        "Please briefly summarize the main reasoning and the final YES/NO decision."
    ),
    verdict_labels=("YES", "NO", "UNCERTAIN"),
    normalize_label=normalize_strategyqa_label,
    parse_verdict=parse_strategyqa_verdict,
    is_correct=exact_match_correct,
    enable_evidence=False,
)


PUBMEDQA_TASK_SPEC = TaskSpec(
    name="pubmedqa",
    task_type="biomedical_qa",
    answer_type="yes_no_maybe",
    input_name="biomedical question",
    affirmative_stance=(
        "You believe the correct answer to the biomedical question is YES and need to argue in favor of YES."
    ),
    negative_stance=(
        "You believe the correct answer to the biomedical question is NO and need to argue in favor of NO."
    ),
    affirmative_stance_reminder="**Your fixed stance is that the answer is YES.**",
    negative_stance_reminder="**Your fixed stance is that the answer is NO.**",
    phase_templates=PUBMEDQA_PHASE_TEMPLATES,
    judge_prompt_template=(
        "The original PubMedQA biomedical question and abstract context are:\n{input_text}\n\n"
        "Debate content:\n{debate_content}\n\n"
        "Your task is to decide the answer to the biomedical question using only the provided abstract context "
        "and the debate. Affirmative argues that the answer is YES. Negative argues that the answer is NO. "
        "Return MAYBE when the abstract evidence is insufficient, mixed, or does not clearly support YES or NO.\n\n"
        "Return ONLY one of the following labels: YES, NO, or MAYBE."
    ),
    summary_prompt_template=(
        "The original PubMedQA biomedical question and abstract context are:\n{input_text}\n\n"
        "Debate content:\n{debate_content}\n\n"
        "Final verdict: {verdict}\n\n"
        "Please briefly summarize the key abstract evidence, the main disagreement, and the final YES/NO/MAYBE decision."
    ),
    verdict_labels=("YES", "NO", "MAYBE"),
    normalize_label=normalize_pubmedqa_label,
    parse_verdict=parse_pubmedqa_verdict,
    is_correct=exact_match_correct,
    enable_evidence=False,
)


TASK_SPECS = {
    "weibo21": FAKENEWS_TASK_SPEC,
    "fakenewsdataset": FAKENEWS_TASK_SPEC,
    "fakenews": FAKENEWS_TASK_SPEC,
    "fake_news": FAKENEWS_TASK_SPEC,
    "strategyqa": STRATEGYQA_TASK_SPEC,
    "strategy_qa": STRATEGYQA_TASK_SPEC,
    "pubmedqa": PUBMEDQA_TASK_SPEC,
    "pubmed_qa": PUBMEDQA_TASK_SPEC,
}


def get_task_spec(dataset_type: str) -> TaskSpec:
    key = dataset_type.lower()
    if key not in TASK_SPECS:
        raise ValueError(f"Unsupported task spec: {dataset_type}")
    return TASK_SPECS[key]
