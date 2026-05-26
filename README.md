# ED2D Benchmark Analysis

This repository contains an ED2D-style multi-agent debate framework for benchmark analysis. The current research focus is not only final prediction accuracy, but also when debate helps, when it fails, and which process-level signals explain those outcomes.

The core benchmark design is:

```text
Datasets x Debate Pipelines x Measurements
```

## Research Scope

ED2D is used here as a structured debate pipeline:

1. Affirmative agents defend one fixed stance.
2. Negative agents defend the opposite fixed stance.
3. The debate runs through opening, rebuttal, free debate, and closing.
4. A judge produces the final verdict.
5. The runner records task accuracy and phase-level disagreement signals.

The intended benchmark datasets are:

| Dataset | Task type | Label space | Purpose |
|---|---|---|---|
| StrategyQA | commonsense / factual reasoning | YES / NO | yes-no reasoning baseline close to fake-real decisions |
| GSM8K | multi-step math reasoning | numeric answer | reasoning-heavy comparison task |
| FakeNewsDataset | misinformation detection | REAL / FAKE | main ED2D misinformation task |


## Measurements

The benchmark records final outcome metrics and debate-process metrics:

- `accuracy`
- per-item `verdict`
- per-item correctness / error
- `opening_disagreement`
- `rebuttal_disagreement`
- `free_disagreement`
- `closing_disagreement`

Hidden-state disagreement is computed from pooled final-layer generation vectors for opposing debate roles. Each phase reports disagreement as:

```text
1 - cosine_similarity(affirmative_vector, negative_vector)
```

This lets the analysis compare whether disagreement, convergence, or debate phase behavior differs between correct and incorrect predictions.

## Project Structure

```text
.
|-- agent.py                 # Local model generation and hidden-state pooling
|-- engine.py                # ED2D debate orchestration and disagreement metrics
|-- task_specs.py            # Task-specific prompts, labels, and verdict parsing
|-- dataset_loader.py        # StrategyQA and FakeNewsDataset loaders
|-- run_dataset_tests.py     # Batch benchmark runner
|-- scripts/
|   |-- analyze_disagreement_summary.py
|   |-- test_local_model.py
|   `-- test_engine_four_phase_analysis.py
|-- StrategyQA/              # StrategyQA source and processed JSONL
|-- grade-school-math-master/ # GSM8K data
|-- fakeNewsDatasets/        # FakeNewsDataset text folders
`-- batch_results/           # Benchmark summaries and debate outputs
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

For local Hugging Face models, configure device and dtype if needed:

```bash
export LOCAL_MODEL_DEVICE=auto
export LOCAL_MODEL_DTYPE=auto
```

Supported local model names are listed in `config.py`, for example:

```text
Qwen/Qwen2-7B-Instruct
Qwen/Qwen2.5-14B-Instruct
```

## Run Benchmarks

Run StrategyQA:

```bash
python run_dataset_tests.py \
  --dataset strategyqa \
  --data-path StrategyQA/processed/strategyqa_processed.jsonl \
  --model Qwen/Qwen2-7B-Instruct \
  --sample-size 30 \
  --disable-evidence
```

Run FakeNewsDataset:

```bash
python run_dataset_tests.py \
  --dataset fakenewsdataset \
  --data-path fakeNewsDatasets/fakeNewsDatasets/fakeNewsDataset \
  --model Qwen/Qwen2-7B-Instruct \
  --sample-size 30 \
  --disable-evidence
```

Use `--sample-size` for small controlled benchmark runs, or omit it to run the full split. Results are saved under:

```text
batch_results/<dataset>/test_summary.json
batch_results/<dataset>/test/debate_outputs/
```

## Analyze Disagreement

After a benchmark run, compare disagreement patterns between correct and incorrect predictions:

```bash
python scripts/analyze_disagreement_summary.py \
  batch_results/fakenewsdataset/test_summary.json \
  --distribution-estimator histogram
```

The analysis script reports phase-level disagreement statistics and distribution comparison metrics such as delta, AUC, and JS divergence.

## Notes

- Use `--disable-evidence` for faster and more reproducible runs.
- Evidence retrieval is only relevant to the misinformation setting.
- The current code measures the four-stage ED2D debate pipeline. Single-agent and majority-vote baselines should be added separately if the benchmark matrix is expanded.

## Acknowledgement

This project builds on ED2D, the Debate-to-Detect framework for reformulating misinformation detection as a multi-agent debate with large language models. The original ED2D idea and debate structure are the foundation of this repository; the hidden-state extraction and disagreement/error and benchmark analysis are my extensions.