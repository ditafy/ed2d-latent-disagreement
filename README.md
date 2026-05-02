# Disagreement Analysis

An extension of ED2D, a multi-agent debate framework for misinformation detection. This version keeps the debate-and-judge workflow, then adds hidden-state extraction and disagreement analysis to study when model disagreement is related to prediction error.

## Motivation

Multi-agent debate can make a model's reasoning process easier to inspect: one side argues that a claim is real, the other argues that it is fake, and judges score the debate. This project asks a second question: can the internal disagreement between the two debating sides tell us something about reliability?

The goal is not only to produce a fake-news verdict, but also to measure how the affirmative and negative agents diverge across debate phases and compare those disagreement patterns with correct and incorrect predictions.

## Key Features

- Multi-agent debate for fake-news detection with affirmative, negative, and judge roles.
- Four debate phases: opening, rebuttal, free debate, and closing.
- Optional Wikipedia evidence retrieval and stance filtering.
- Local/open-source model support through Hugging Face `transformers`.
- Hidden-state capture during generation for selected debate roles.
- Per-phase cosine similarity and disagreement scores between opposing agents.
- Batch evaluation over Weibo21 and FakeNewsDataset-style inputs.
- Post-hoc analysis that groups disagreement by correctness and computes distribution-level statistics, including KL/JS-based divergence.

## My Contributions

This repository is based on the original ED2D debate framework. My work extends it in the following ways:

- Added local causal-LM inference support using `AutoTokenizer` and `AutoModelForCausalLM`.
- Added hidden-state extraction during generation, including pooled final-layer vectors for generated tokens.
- Added an `analysis` return mode for agents so debate text and latent vectors can be collected in one pass.
- Added per-phase disagreement tracking for affirmative vs. negative speakers in opening, rebuttal, free debate, and closing.
- Implemented cosine-similarity-based disagreement metrics inside the debate engine.
- Extended dataset-level evaluation to save correctness, error labels, and phase-level disagreement values for every item.
- Added statistical analysis scripts to compare disagreement distributions between correct and incorrect predictions, including delta, AUC, and JS divergence, with KL divergence used in the JS calculation.
- Added validation scripts for local model loading, hidden-state pooling, agent interface checks, and four-phase engine integration.

The inherited ED2D components are the core role-based debate design, fixed affirmative/negative stances, judge scoring, final verdict generation, and the evidence-assisted debate structure.

## Method Overview

For each news item, the system runs a structured debate:

1. The engine detects a domain and assigns role-specific profiles.
2. Affirmative agents argue that the news is true; negative agents argue that it is false.
3. Optional Wikipedia evidence is retrieved, evaluated, and provided when useful.
4. The debate proceeds through opening, rebuttal, free debate, and closing.
5. During each phase, the system captures the final-layer hidden states from selected affirmative and negative generations.
6. Generated-token hidden states are mean-pooled into one vector per role response.
7. The engine computes cosine similarity between opposing role vectors and reports disagreement as `1 - cosine_similarity`.
8. Judges score both sides, and the final verdict is mapped to `REAL`, `FAKE`, or `UNCERTAIN`.
9. Batch summaries compare verdicts with gold labels and store phase-level disagreement values.
10. The analysis script groups records by correctness and compares disagreement distributions using summary statistics, AUC, and JS divergence.

In short: ED2D provides the debate trace; this extension adds a latent disagreement signal and tests whether that signal changes when the model is wrong.

## Project Structure

```text
.
|-- agent.py                         # Local model agent, generation, hidden-state pooling
|-- engine.py                        # Debate orchestration, judging, per-phase disagreement metrics
|-- config.py                        # Roles, phases, prompts, scoring dimensions, save settings
|-- evidence_system.py               # Keyword extraction, Wikipedia retrieval, evidence stance filtering
|-- dataset_loader.py                # Weibo21 and FakeNewsDataset loaders
|-- run_dataset_tests.py             # Batch runner with accuracy and disagreement summaries
|-- scripts/
|   |-- analyze_disagreement_summary.py
|   |-- test_local_model.py
|   |-- test_agent_interface.py
|   |-- test_engine_integration.py
|   `-- test_engine_four_phase_analysis.py
|-- scripts/test/                    # Example summary outputs
|-- fakeNewsDatasets/                # Included fake/legit text dataset
|-- Results/                         # Single-run debate outputs
`-- batch_results/                   # Batch summaries and debate outputs
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Choose the local model device and dtype:

```bash
export LOCAL_MODEL_DEVICE=auto
export LOCAL_MODEL_DTYPE=auto
```

Run a quick local model smoke test:

```bash
python scripts/test_local_model.py \
  --model Qwen/Qwen2-7B-Instruct \
  --device auto \
  --dtype auto
```

Run one four-phase debate and inspect disagreement metrics:

```bash
python scripts/test_engine_four_phase_analysis.py \
  --model Qwen/Qwen2-7B-Instruct \
  --device auto \
  --dtype auto \
  --disable-evidence
```

Run a small dataset evaluation:

```bash
python run_dataset_tests.py \
  --dataset fakenewsdataset \
  --data-path fakeNewsDatasets/fakeNewsDatasets/fakeNewsDataset \
  --model Qwen/Qwen2-7B-Instruct \
  --sample-size 30 \
  --disable-evidence
```

Analyze the relationship between disagreement and correctness:

```bash
python scripts/analyze_disagreement_summary.py \
  batch_results/fakenewsdataset/test_summary.json \
  --distribution-estimator histogram
```

Evidence retrieval uses Wikipedia and may add latency. Use `--disable-evidence` for offline or faster experiments.

## Example Output / Results

A batch summary records both task performance and latent disagreement:

```json
{
  "dataset": "fakenewsdataset",
  "split": "test",
  "metrics": {
    "attempted": 30,
    "labeled": 30,
    "correct": 19,
    "accuracy": 0.6333,
    "opening_disagreement_stats": {
      "count": 30,
      "mean": 0.098307,
      "std": 0.045395
    },
    "rebuttal_disagreement_stats": {
      "count": 30,
      "mean": 0.041016,
      "std": 0.014534
    },
    "free_disagreement_stats": {
      "count": 30,
      "mean": 0.050667,
      "std": 0.018455
    },
    "closing_disagreement_stats": {
      "count": 30,
      "mean": 0.039339,
      "std": 0.01241
    }
  }
}
```

Each record also stores the gold label, model verdict, correctness flag, error value, and per-phase disagreement:

```json
{
  "id": "biz01.legit.txt",
  "label": "REAL",
  "verdict": "REAL",
  "is_correct": true,
  "error": 0,
  "opening_disagreement": 0.04150390625,
  "rebuttal_disagreement": 0.02197265625,
  "free_disagreement": 0.02685546875,
  "closing_disagreement": 0.0263671875
}
```

The post-hoc analysis script then compares the disagreement profiles of correct and incorrect predictions. This makes it possible to inspect whether specific debate phases become more internally divided when the final verdict is wrong.

## Acknowledgement

This project builds on ED2D, the Debate-to-Detect framework for reformulating misinformation detection as a multi-agent debate with large language models. The original ED2D idea and debate structure are the foundation of this repository; the hidden-state extraction and disagreement/error analysis are my extensions.
