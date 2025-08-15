# OpenAI Smoke Test Benchmark

A smoke testing tool for OpenAI-compatible endpoints.

## Features

- Concurrent user simulation with customizable query load
- Summary report with:
  - Average time to first token (TTFT)
  - Tokens per second (TPS)
  - Percentiles (p50, p90)
- Customizable prompt size (min/max word count)
- Optional single-run mode for quick testing

## Setup

```bash
make setup
```

This creates a virtual environment in .venv/, installs dependencies, and
installs the tool locally in editable mode.

## Example Usage

Below is a light run against a local Ollama instance.

```bash
 openai-smoke-test git:(main) ✗ .venv/bin/openai-smoketest --api-key ollama --api-base http://localhost:11434/v1 --model granite3.3:8b --num-users 2 --queries-per-user 1
Running queries: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:41<00:00, 20.67s/it]

--- SUMMARY REPORT ---
Total Queries: 2
Successful Queries: 2
Failed Queries: 0
+-------------------------+--------+-------+-------+
| Metric                  |   Mean |   P50 |   P90 |
+=========================+========+=======+=======+
| Time to First Token (s) |  18.54 | 18.54 | 28.07 |
+-------------------------+--------+-------+-------+
| Tokens per Second       |   6.47 |  6.47 | 10.25 |
+-------------------------+--------+-------+-------+
Successful
```

# Summarization Scoring
Below is with added summarization scores, running 5 queries simulatenously:

```bash
openai-smoke-test % openai-smoketest --model mistral-small-2503 --num-users 5
Running queries:   0%|                                        | 0/50 [00:00<?, ?it/s]Refreshing Mistral access token from service account...
Token refreshed successfully.
Running queries: 100%|███████████████████████████████| 50/50 [02:00<00:00,  2.41s/it]

--- SUMMARY REPORT ---
Total Queries: 50
Successful Queries: 50
Failed Queries: 0
+-------------------------+--------+-------+-------+
| Metric                  |   Mean |   P50 |   P90 |
+=========================+========+=======+=======+
| Time to First Token (s) |   5    |  5.02 | 10.16 |
+-------------------------+--------+-------+-------+
| Tokens/sec (Per Query)  |  27.94 | 20.13 | 62.95 |
+-------------------------+--------+-------+-------+
| Round trip (s)          |   9.38 |  8.7  | 15.34 |
+-------------------------+--------+-------+-------+

Global Throughput: 17.55 tokens/sec across 468.90 seconds
SUCCESS

--- SCORE REPORT ---
+---------------------+--------+-------+-------+
| Type                |   Mean |   P50 |   P90 |
+=====================+========+=======+=======+
| Rouge 1             |   0.58 |  0.58 |  0.66 |
+---------------------+--------+-------+-------+
| Rouge 2             |   0.31 |  0.3  |  0.42 |
+---------------------+--------+-------+-------+
| RougeLsum           |   0.41 |  0.39 |  0.54 |
+---------------------+--------+-------+-------+
| BLEU                |   0.23 |  0.23 |  0.35 |
+---------------------+--------+-------+-------+
| Unieval consistency |   0.9  |  0.93 |  0.97 |
+---------------------+--------+-------+-------+
| Unieval coherence   |   0.97 |  0.98 |  0.99 |
+---------------------+--------+-------+-------+
| Unieval fluency     |   0.94 |  0.95 |  0.96 |
+---------------------+--------+-------+-------+
| Unieval relevance   |   0.96 |  0.97 |  0.99 |
+---------------------+--------+-------+-------+
| Unieval overall     |   0.94 |  0.95 |  0.97 |
+---------------------+--------+-------+-------+
Num hiccups: 0 --- Percentage of hiccups: 0.0
Overall Score: --- mistral-small-2503: 0.67 ---
```

(You can also run `score_report.py` to recompute the reports)

## Summarization Config
This table details the configuration options for the summarization and evaluation script, managed in `config.yaml`.

| Parameter | Type | Description | Default Value |
| :--- | :--- | :--- | :--- |
| **Summarization** | | | |
| `use_dataset` | `bool` | If true, uses the Hugging Face dataset defined at `dataset_name`. Otherwise the smoke test is ran on randomly generated lorem ipsum. | `true` |
| `log_stats` | `bool` | If true, iterates through the entire dataset and logs scores to `stats.jsonl`. If false, be sure to pass `--queries-per-user` argument | `false` |
| `dataset_name` | `str` | Hugging Face dataset to load from the Mozilla organization. | `"page-summarization-eval"` |
| `system_prompt_template` | `str` | The system prompt that instructs the model on how to summarize. | `"You are an expert..."` |
| `user_prompt_template` | `str` | The user prompt containing the `{text}` placeholder for the article. | `"Summarize the following..."` |
| `temperature` | `float` | Controls the randomness of the model's output. Lower is more deterministic. | `0.1` |
| `top_p` | `float` | Controls nucleus sampling for the model's output. | `0.01` |
| `stream` | `bool` | If true, streams the response to track Time To First Token (TTFT). | `true` |
| `service_account_file` | `str` | Path to the Google Cloud service account file for Mistral authentication. | `"creds.json"` |
| **Performance Thresholds** | | | |
| `metric_threshold.ttft` | `float` | Max allowed Time To First Token in seconds (lower is better). | `1.0` |
| `metric_threshold.per_query_tps` | `int` | Min required Tokens Per Second (higher is better). | `100` |
| `metric_threshold.round_trip`| `float` | Max allowed total request time in seconds (lower is better). | `2.5` |
| **Quality Score Thresholds**| | | |
| `score_threshold.rouge.rouge1`| `float` | Minimum required ROUGE-1 score. | `0.3` |
| `score_threshold.rouge.rouge2`| `float` | Minimum required ROUGE-2 score. | `0.2` |
| `score_threshold.rouge.rougeLsum`| `float` | Minimum required ROUGE-Lsum score. | `0.25` |
| `score_threshold.bleu` | `float` | Minimum required BLEU score. | `0.1` |
| `score_threshold.unieval.consistency`| `float` | Minimum required UniEval consistency score. | `0.9` |
| `score_threshold.unieval.coherence`| `float` | Minimum required UniEval coherence score. | `0.8` |
| `score_threshold.unieval.fluency`| `float` | Minimum required UniEval fluency score. | `0.8` |
| `score_threshold.unieval.relevance`| `float` | Minimum required UniEval relevance score. | `0.75` |
| `score_threshold.unieval.overall`| `float` | Minimum required UniEval overall score. | `0.85` |
| `score_threshold.percentage_of_hiccups`| `float` | Max allowed percentage of summaries with hiccups (lower is better). | `0.05` |
| `score_threshold.overall` | `float` | Minimum required final weighted score to pass the test. | `0.65` |
| **LLM-based Evaluation** | | | |
| `llm_unieval_scoring.score_with_llm`| `bool` | If true, uses an LLM to evaluate summaries instead of unieval library (not async). | `false` |
| `llm_unieval_scoring.model_name`| `str` | The model name to use for LLM-based evaluation (e.g., GPT-4o). | `"gpt-4o"` |
| `llm_unieval_scoring.base_url` | `str` | The API endpoint for the evaluator LLM. | `"https://api.openai.com/v1/"` |
| `llm_unieval_scoring.api_key` | `str` | The name of the environment variable holding the evaluator's API key. | `LLM_UNIEVAL_SCORING_API_KEY` |
| `llm_unieval_scoring.system_prompt`| `str` | The system prompt for the evaluator model. | `"You are a meticulous..."` |
| `llm_unieval_scoring.user_prompt`| `str` | The user prompt for the evaluator model, defining criteria and format. | `"Carefully evaluate the..."` |