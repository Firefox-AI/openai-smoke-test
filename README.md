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

We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/):

1. `make`
2. `source .venv/bin/activate`

This creates a virtual environment in .venv/, installs dependencies, and installs the tool locally in editable mode.

### Lint

Ensure all checks pass:

`pre-commit run --all-files --verbose`

## Installation

`make install`

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

## Usage of Inference Test Script

### Setup API keys via .env file or env variables

`export MY_VENDOR_API_KEY="your-secret-api-key"`

```
python src/smoke/inference_test.py \
    --model "model-x" \
    --vendor "my_vendor" \
    --feature "summarization" \
    --quality-test-csv "src/smoke/900_sample_goldenfox.csv" \
    --quality-test-csv-column "text" \
    --num-users 1
```

## Usage for Quick Test script

```
python src/smoke/quick_test.py
python src/smoke/quick_test.py --config PATH-TO-custom-quick_test_multiturn_config.yaml
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

Parameter                               | Type          | Description                                                                                                                                                           | Default Value
:-------------------------------------- | :------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-----------------------------
**Summarization**                       |               |                                                                                                                                                                       |
`use_dataset`                           | `bool`        | If true, uses the Hugging Face dataset defined at `dataset_name`. Otherwise the smoke test is ran on randomly generated lorem ipsum.                                  | `false`
`log_stats`                             | `bool`        | If true, iterates through the entire dataset and logs scores to `src/smoke/stats/summary/<model_name>.jsonl`. If false, be sure to pass `--queries-per-user` argument | `false`
`dataset_name`                          | `str`         | Hugging Face dataset to load from the Mozilla organization.                                                                                                           | `"page-summarization-eval"`
`system_prompt_template`                | `str`         | The system prompt that instructs the model on how to summarize.                                                                                                       | `"You are an expert..."`
`user_prompt_template`                  | `str`         | The user prompt containing the `{text}` placeholder for the article.                                                                                                  | `"Summarize the following..."`
`temperature`                           | `float`       | Controls the randomness of the model's output. Lower is more deterministic.                                                                                           | `0.1`
`top_p`                                 | `float`       | Controls nucleus sampling for the model's output.                                                                                                                     | `0.01`
`max_completion_tokens`                 | `int or null` | Max completion tokens for summarization.                                                                                                                              | `null`
`error_on_threshold_fails`              | `bool`        | If true, throws an error if any threshold check fails                                                                                                                 | `false`
`stream`                                | `bool`        | If true, streams the response to track Time To First Token (TTFT).                                                                                                    | `true`
`service_account_file`                  | `str`         | Path to the Google Cloud service account file for Mistral authentication.                                                                                             | `"creds.json"`
**Performance Thresholds**              |               |                                                                                                                                                                       |
`metric_threshold.ttft`                 | `float`       | Max allowed Time To First Token in seconds (lower is better).                                                                                                         | `1.0`
`metric_threshold.per_query_tps`        | `int`         | Min required Tokens Per Second (higher is better).                                                                                                                    | `100`
`metric_threshold.round_trip`           | `float`       | Max allowed total request time in seconds (lower is better).                                                                                                          | `2.5`
**Quality Score Thresholds**            |               |                                                                                                                                                                       |
`score_threshold.rouge.rouge1`          | `float`       | Minimum required ROUGE-1 score.                                                                                                                                       | `0.3`
`score_threshold.rouge.rouge2`          | `float`       | Minimum required ROUGE-2 score.                                                                                                                                       | `0.2`
`score_threshold.rouge.rougeLsum`       | `float`       | Minimum required ROUGE-Lsum score.                                                                                                                                    | `0.25`
`score_threshold.bleu`                  | `float`       | Minimum required BLEU score.                                                                                                                                          | `0.1`
`score_threshold.unieval.consistency`   | `float`       | Minimum required UniEval consistency score.                                                                                                                           | `0.9`
`score_threshold.unieval.coherence`     | `float`       | Minimum required UniEval coherence score.                                                                                                                             | `0.8`
`score_threshold.unieval.fluency`       | `float`       | Minimum required UniEval fluency score.                                                                                                                               | `0.8`
`score_threshold.unieval.relevance`     | `float`       | Minimum required UniEval relevance score.                                                                                                                             | `0.75`
`score_threshold.unieval.overall`       | `float`       | Minimum required UniEval overall score.                                                                                                                               | `0.85`
`score_threshold.percentage_of_hiccups` | `float`       | Max allowed percentage of summaries with hiccups (lower is better).                                                                                                   | `0.05`
`score_threshold.overall`               | `float`       | Minimum required final weighted score to pass the test.                                                                                                               | `0.65`
**LLM-based Evaluation**                |               |                                                                                                                                                                       |
`llm_unieval_scoring.score_with_llm`    | `bool`        | If true, uses an LLM to evaluate summaries instead of unieval library (not async).                                                                                    | `false`
`llm_unieval_scoring.model_name`        | `str`         | The model name to use for LLM-based evaluation.                                                                                                                       | `"gpt-4o"`
`llm_unieval_scoring.base_url`          | `str`         | The API endpoint for the evaluator LLM.                                                                                                                               | `"https://api.openai.com/v1/"`
`llm_unieval_scoring.api_key`           | `str`         | The name of the environment variable holding the evaluator's API key.                                                                                                 | `LLM_UNIEVAL_SCORING_API_KEY`
`llm_unieval_scoring.system_prompt`     | `str`         | The system prompt for the evaluator model.                                                                                                                            | `"You are a meticulous..."`
`llm_unieval_scoring.user_prompt`       | `str`         | The user prompt for the evaluator model, defining criteria and format.                                                                                                | `"Carefully evaluate the..."`

## How to run stress test

### Example

```bash
python src/smoke/stress_test.py --test-config src/smoke/stress-test.yaml --feature chatbot --vendor ollama --model "openai/gpt-oss-120B" --mode stress
```

After the stress test is complete, you can aggregate the benchmark logs using the following command:

```bash
python3 src/smoke/aggregate_benchmark_logs.py --run-directory $benchmark_output_dir$ --test-config src/smoke/stress-test.yaml
```

### Using a different vendor

To use a different vendor, such as `gke`, you will need to update the `--vendor` and `--model` arguments. You may also need to update the `api_base` in `config.yml` or `stress-test.yaml`.

For example:

```bash
python src/smoke/stress_test.py --test-config src/smoke/stress-test.yaml --feature chatbot --vendor gke --model "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8" --mode stress
```

# [MLPA](https://github.com/Firefox-AI/MLPA) Load testing
[MLPA](https://github.com/Firefox-AI/MLPA) is a proxy gateway API that depends on 3 components:
1. [FXA](github.com/mozilla/PyFxA) - Mozilla's authorization mechanism
2. Postgres Database
3. LLM proxy service (LiteLLM or in the future AnyLLM-gateway or other similar services)

The goal of the load test is to stress the production deployment with the traffic pattern and the load that we expect to see in production (the rule of thumb is usually 3x load on the load test peak traffic) to check for potential bottlenecks, assure that scalability is configured correctly, assured that we have adequate infrastructure in place.

## Partial mocking
We would like to mock the LiteLLM calls to the real LLM providers in order to save costs on the actual reply generation from LLM - since it is not the goal of the MLPA's load test, for that we are using `/mock/` router, for example instead of `/v1/chat/completions` we are calling `mock/v1/chat/completions`, which internally mocks the LLM calls.

Another mocking might happen when we would like to mock calls to the Mozilla's auth service.

The [PyFXa](github.com/mozilla/PyFxA) works in the way that validation is happening locally in case when token is cached or JWT token is decrypted or else library is having a [fallback](https://github.com/mozilla/PyFxA/blob/d3040bfaa6d413bff63b689590287a3198fff1ef/fxa/oauth.py#L271) to calling FXA servers.  In order to avoid these calls we can use endpoint
`mock/v1/chat/completions_no_auth` - this endpoint is not calling FXA server in the case of fallbacks.

## Virtual users and authentication
In order to simulate the production traffic pattern we need to use multiple parallel users, since our users need to have valid token, we need to create `users.json` file. You can find a script in `src/stress/generate_test_fxa_users.py`

```bash
make generate-fxa-users n-users=5 env=prod
```
this needs to be done once

and then you will need to refresh tokens before running load test, since they expire pretty often

```bash
make refresh-fxa-users
```

Please delete users after your tests

```bash
make delete-fxa-users
```

## Running load test with [Locust](https://github.com/locustio/locust)
see the `src/stress/locusfiles/mlpa.py` for details

```bash
make stress-mlpa host="MLPA_HOST:8000"
```
it will open Locust's UI where you can set the number of virtual users



# Multi Turn Chat Testing

Multi turn chat tests models with context, and simulates replying back to the same bot after its response. This command supports the `--start-with-context` to begin the conversation with a random context file defined in `/src/smoke/multi_turn_chat/data/initial-context/`

A random query is chosen from `/src/smoke/multi_turn_chat/data/queries/generic_queries_2.csv`

Example usage below

```bash
multi-turn-chat-smoketest --api-key "<api_key>"  --num-users 5 --queries-per-user 5 --start-with-context --model "Qwen/Qwen3-235B-A22B-Instruct-2507-tput" --api-base https://api.together.xyz/v1/
```

Logs are stored in `/src/smoke/stats/multi_turn/<model_name>.jsonl`

## Vertex AI Deployment Log Auditing

This script queries Google Cloud Logging to provide detailed reports on Vertex AI model deployments. It tracks the entire lifecycle of endpoints--from deployment to undeployment--and provides insights into uptime, resource configuration, and estimated costs.

### Features

- **Comprehensive Event Tracking**: Monitors 'Deploy Model', 'Download model' (replica creation), and 'Undeploy Model' events.
- **Detailed Timeline Report**: Displays a chronological log of all deployment-related activities, including machine specs, accelerator details, and replica counts.
- **Uptime and Cost Analysis**: Calculates the total operational hours for each endpoint and provides an estimated cost based on a configurable pricing table.

### Usage Examples

1. **Detailed Report for a Specific Model (Default Behavior)**:

  ```bash
  python3 audit_vertexai_deployment_logs.py --search-term "the-model-name"
  ```

2. **Full Granular Report with Replica Messages**:

  ```bash
  python3 audit_vertexai_deployment_logs.py --search-term "the-model-name" --include-message
  ```

3. **High-Level Summary of All Deployments (excluding replica events)**: (Shows only deploy/undeploy events and the final uptime/cost report)

  ```bash
  python3 audit_vertexai_deployment_logs.py --no-replicas
  ```

### Command-Line Arguments

Argument            | Type      | Description
:------------------ | :-------- | :------------------------------------------------------------------------------------------------
`--search-term`     | `str`     | **Required** (unless `--no-replicas` is used). The search term to filter replica creation events.
`--no-replicas`     | `boolean` | Excludes replica-level download events from the report for a high-level summary.
`--include-message` | `boolean` | Includes the detailed message column for replica events.

### Example Output

```
--- Endpoint Uptime and Cost Report ---
Endpoint ID                                     | Uptime (hours)  | Est. Cost ($)   | Machine         | Min Rep  | Max Rep
---------------------------------------------------------------------------------------------------------------------------
2619126857315909632                             | 1.56            | 0.00            |                 | 0        | 0
3063857320518746112                             | 0.13            | 0.00            |                 | 0        | 0
3254134404775149568                             | 0.10            | 0.00            |                 | 0        | 0
3763604112621436928                             | 5.85            | 0.00            |                 | 0        | 0
4216778825125593088                             | 1.21            | 0.00            |                 | 0        | 0
7373239213958889472                             | 0.16            | 0.00            |                 | 0        | 0
818249956321132544                              | 3.75            | 330.04          | a3-highgpu-8g   | 1        | 0
mg-endpoint-6a72912c-4404-4dc5-bb7a-a8f766fa041c | 0.96            | 84.87           | a3-highgpu-8g   | 1        | 0
mg-endpoint-bcc89773-a493-46fb-a134-e4d0d7af6922 | 0.49            | 0.00            | a3-highgpu-4g   | 1        | 1
---------------------------------------------------------------------------------------------------------------------------
Totals                                          | 14.21           | 414.91
```

> **Note**: The pricing data in this script is for narrow range of use cases only. For accurate cost calculations, please update the `PRICING_DATA` dictionary with the latest official rates from the Google Cloud pricing pages.

## GKE LLM Deployment

```bash
./gke/gke_llm_start.sh -p "fx-gen-ai-sandbox" -t "$hf_secret_token" -f gke/deployments/qwen3-235b-fp8_h100.yaml -r us-central1 -z us-central1-a -o qwen3-235b-fp8-h100-pool -m a3-highgpu-4g -a "type=nvidia-h100-80gb,count=4" --max-nodes 1
```

> **Note on Instance Types:** The `gke/gke_llm_start.sh` script dynamically handles different instance provisioning types by creating a temporary deployment YAML.

> - **On-Demand (Default):** If no specific flags are provided, the script provisions standard on-demand instances.
> - **Spot Instances:** Use the `--spot` flag to provision a node pool with Spot VMs, which can provide cost savings. The deployment YAML must contain the appropriate `gke-spot` tolerations and node selectors.
> - **Reservations:** Use the `-u <RESERVATION_URL>` flag to consume a specific reservation. This is useful for guaranteeing resource availability.

> The script automatically modifies the deployment YAML to match the chosen provisioning type, ensuring the correct node selectors and affinities are applied.
