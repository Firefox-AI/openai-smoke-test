import asyncio
import os
import argparse
import csv
import sys
import time
from statistics import mean
from typing import List, Optional, Any
import datetime
import json
import re
from urllib.parse import urlparse

import numpy as np
import openai
import tiktoken
import yaml
from tabulate import tabulate
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


def build_output_filename(
    api_base: Optional[str], model_name: str, feature_name: str
) -> str:
    """Builds a sanitized, timestamped filename from the API base, model name, and feature name."""
    domain = "default_openai"
    if api_base:
        try:
            parsed_url = urlparse(api_base)
            domain = parsed_url.netloc
        except Exception:
            domain = "unknown_host"

    sanitized_domain = re.sub(r"[^\w.-]+", "_", domain)
    sanitized_model_name = re.sub(r"[^\w.-]+", "_", model_name)
    sanitized_feature_name = re.sub(r"[^\w.-]+", "_", feature_name)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{sanitized_domain}_{sanitized_model_name}_{sanitized_feature_name}_{timestamp}.jsonl"


class OutputBuilder:
    """Builds and manages the output file for test results."""

    def __init__(
        self,
        model_name: str,
        feature_name: str,
        api_base: Optional[str],
        output_dir: Optional[str] = None,
    ):
        self.output_filename = build_output_filename(
            api_base, model_name, feature_name
        )
        if output_dir:
            self.output_path = os.path.join(output_dir, self.output_filename)
            os.makedirs(output_dir, exist_ok=True)
        else:
            self.output_path = self.output_filename

    def record(self, record_data: dict):
        with open(self.output_path, "a") as f:
            f.write(json.dumps(record_data) + "\n")


def load_config(config_path="src/smoke/config.yaml"):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)


def get_vendor_config(vendor: str, config: dict) -> dict:
    vendor_config = config.get("vendors", {}).get(vendor)
    if not vendor_config:
        print(f"Error: Vendor '{vendor}' not found in the configuration.")
        sys.exit(1)

    api_key_env = vendor_config.get("api_key_env")
    api_key = os.getenv(api_key_env) if api_key_env else None

    if not api_key:
        print(
            f"Error: API key environment variable '{api_key_env}' not set for vendor '{vendor}'."
        )
        sys.exit(1)

    vendor_config["api_key"] = api_key
    return vendor_config


async def run_query(
    session_id: int,
    query_id: int,
    text: str,
    stats: List[dict],
    model_name: str,
    openai_client,
    tokenizer: Optional[Any],
    system_prompt: str,
    user_prompt_template: str,
    temperature: float,
    max_tokens: int,
    pbar=None,
    output_builder=None,
):
    user_content = user_prompt_template.format(text=text)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    try:
        start_time = time.time()
        first_token_time = None
        output_tokens = 0
        generated_text = ""

        stream = await openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        async for chunk in stream:
            if not first_token_time:
                first_token_time = time.time()
            content = chunk.choices[0].delta.content or ""
            generated_text += content
            if tokenizer:
                # isinstance check is for future-proofing; currently logic is split
                if hasattr(tokenizer, "encode"):
                    output_tokens += len(tokenizer.encode(content))

        end_time = time.time()
        total_time = end_time - start_time
        stats.append(
            {
                "session": session_id,
                "query": query_id,
                "ttft": first_token_time - start_time if first_token_time else None,
                "tps": output_tokens / total_time
                if total_time > 0 and output_tokens > 0
                else 0,
                "success": True,
                "total_time": total_time,
                "output_tokens": output_tokens,
            }
        )
        if output_builder:
            prompt_tokens = 0
            if tokenizer:
                prompt_tokens = len(tokenizer.encode(system_prompt)) + len(
                    tokenizer.encode(user_content)
                )
            record = {
                "record_id": query_id,
                "model_name": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": messages,
                "system_prompt": system_prompt,
                "user_prompt_template": user_prompt_template,
                "generated_text": generated_text,
                "prompt_tokens": prompt_tokens,
                "generation_tokens": output_tokens,
                "latency_sec": total_time,
                "ttft_sec": first_token_time - start_time
                if first_token_time
                else None,
                "tokens_per_second": output_tokens / total_time
                if total_time > 0 and output_tokens > 0
                else 0,
                "success": True,
            }
            output_builder.record(record)

    except Exception as e:
        stats.append(
            {
                "session": session_id,
                "query": query_id,
                "error": str(e),
                "success": False,
            }
        )
        if output_builder:
            record = {
                "record_id": query_id,
                "model_name": model_name,
                "messages": messages,
                "success": False,
                "error": str(e),
            }
            output_builder.record(record)

    if pbar:
        pbar.update(1)


def stats_summary(values: List[float], label: str) -> List:
    return (
        [
            label,
            f"{mean(values):.2f}",
            f"{np.percentile(values, 50):.2f}",
            f"{np.percentile(values, 90):.2f}",
        ]
        if values
        else [label, "-", "-", "-"]
    )


async def async_main(args):
    tokenizer = None
    stats = []

    config = load_config()

    api_base = None  # Define api_base to be available for OutputBuilder
    if args.vendor:
        vendor_config = get_vendor_config(args.vendor, config)
        api_base = vendor_config["api_base"]
        client_kwargs = {"api_key": vendor_config["api_key"], "base_url": api_base}

        model_configs = vendor_config.get("model_config", {})
        model_config = model_configs.get(args.model)

        if not model_config:
            sys.exit(
                f"Error: Model '{args.model}' configuration not found for vendor '{args.vendor}'.\n"
                f"Please add it to the config.yaml file. Supported models are: {list(model_configs.keys())}"
            )

        tokenizer_type = model_config.get("tokenizer_type")

        if tokenizer_type == "tiktoken":
            try:
                tokenizer = tiktoken.encoding_for_model(args.model)
            except KeyError:
                print(
                    "Warning: Model not found in tiktoken. Falling back to cl100k_base."
                )
                tokenizer = tiktoken.get_encoding("cl100k_base")
        elif tokenizer_type == "huggingface":
            if AutoTokenizer is None:
                sys.exit(
                    "`transformers` library is not installed. Please install it with `pip install transformers`."
                )

            tokenizer_path = model_config.get("tokenizer")
            if not tokenizer_path:
                sys.exit(
                    f"Error: 'tokenizer' not specified for model '{args.model}' in config.yaml"
                )
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            except Exception as e:
                sys.exit(
                    f"Failed to load tokenizer '{tokenizer_path}' from Hugging Face Hub: {e}"
                )

    else:
        api_base = args.api_base
        client_kwargs = {"api_key": args.api_key, "base_url": api_base}
        # Heuristic for non-vendor case
        if api_base is None or "openai.com" in api_base:
            try:
                tokenizer = tiktoken.encoding_for_model(args.model)
            except KeyError:
                print(
                    "Warning: Model not found for tiktoken. Token counting will be skipped."
                )

    output_builder = OutputBuilder(args.model, args.feature, api_base, args.output)
    openai_client = openai.AsyncOpenAI(**client_kwargs)

    features = config.get("features", {})
    if args.feature not in features:
        print(f"Error: Feature '{args.feature}' not found in config.yaml.")
        sys.exit(1)

    prompts_config = features[args.feature]
    system_prompt = prompts_config["system_prompt"]
    user_prompt_template = prompts_config["user_prompt_template"]

    try:
        with open(args.quality_test_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if args.quality_test_csv_column not in reader.fieldnames:
                print(
                    f"Error: Column '{args.quality_test_csv_column}' not found in {args.quality_test_csv}"
                )
                print(f"Available columns: {reader.fieldnames}")
                return 1
            prompts = [row[args.quality_test_csv_column] for row in reader]
    except FileNotFoundError:
        print(f"Error: The file {args.quality_test_csv} was not found.")
        return 1

    pbar = tqdm(total=len(prompts), desc="Running quality test")
    semaphore = asyncio.Semaphore(args.num_users)

    async def run_quality_query(session_id, query_id, text):
        async with semaphore:
            await run_query(
                session_id,
                query_id,
                text,
                stats,
                args.model,
                openai_client,
                tokenizer,
                system_prompt,
                user_prompt_template,
                args.temperature,
                args.max_tokens,
                pbar,
                output_builder,
            )

    tasks = [
        run_quality_query(i % args.num_users, i, prompt)
        for i, prompt in enumerate(prompts)
    ]
    await asyncio.gather(*tasks)
    pbar.close()

    success = all(entry.get("success", False) for entry in stats)
    ttf_times = [
        s["ttft"] for s in stats if s.get("success") and s.get("ttft") is not None
    ]
    per_query_tps = [s["tps"] for s in stats if s.get("success") and s.get("tps")]

    total_output_tokens = sum(
        s.get("output_tokens", 0) for s in stats if s.get("success")
    )
    total_duration = sum(s["total_time"] for s in stats if s.get("success"))
    global_tps = (
        total_output_tokens / total_duration
        if total_duration > 0 and total_output_tokens > 0
        else 0
    )

    total = len(stats)
    successes = sum(1 for s in stats if s.get("success"))
    failures = total - successes
    table = [
        stats_summary(ttf_times, "Time to First Token (s)"),
        stats_summary(per_query_tps, "Tokens/sec (Per Query)"),
    ]

    errors = [s for s in stats if not s.get("success") and s.get("error")]

    print("\n--- SUMMARY REPORT ---")
    print(f"Total Queries: {total}")
    print(f"Successful Queries: {successes}")
    print(f"Failed Queries: {failures}")
    print(tabulate(table, headers=["Metric", "Mean", "P50", "P90"], tablefmt="grid"))

    print(
        f"\nGlobal Throughput: {global_tps:.2f} tokens/sec across {total_duration:.2f} seconds"
    )
    if tokenizer is None:
        print(
            "Note: Token-based metrics (TPS, Global Throughput) are not available as a tokenizer could not be loaded."
        )
    print("SUCCESS" if success else "FAILURE: Some queries failed")

    if errors:
        print("\n--- FIRST ERROR ---")
        print(f"Session {errors[0]['session']} - Query {errors[0]['query']}")
        print(f"Error: {errors[0]['error']}")

    return failures


def main():
    parser = argparse.ArgumentParser(description="OpenAI Quality Test Runner")
    parser.add_argument("--model", required=True, type=str, help="Model name")
    parser.add_argument(
        "--feature",
        type=str,
        default="default",
        help="The feature to test, which determines the system and user prompts.",
    )
    parser.add_argument(
        "--vendor",
        type=str,
        default=None,
        help="The vendor name. If specified, uses the vendor's API key and base URL from config.yaml.",
    )
    parser.add_argument(
        "--num-users", type=int, default=10, help="Number of concurrent workers"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="The sampling temperature."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
        help="The maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--quality-test-csv",
        required=True,
        type=str,
        help="Path to the CSV file for quality testing.",
    )
    parser.add_argument(
        "--quality-test-csv-column",
        type=str,
        default="text",
        help="The column name in the CSV file that contains the prompts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save the output JSONL file. Defaults to current directory.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key. Cannot be used with --vendor.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL. Cannot be used with --vendor.",
    )

    args = parser.parse_args()

    if args.vendor:
        if args.api_key is not None or args.api_base is not None:
            parser.error("--api-key and --api-base cannot be used with --vendor.")
    elif args.api_key is None:
        # If no vendor, check for direct key or OPENAI_API_KEY
        env_api_key = os.getenv("OPENAI_API_KEY")
        if env_api_key is None:
            parser.error(
                "Either --vendor or --api-key (or OPENAI_API_KEY env var) is required."
            )
        args.api_key = env_api_key

    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())