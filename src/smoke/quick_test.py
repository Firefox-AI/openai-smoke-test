import asyncio
import argparse
import csv
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union
import openai
import tiktoken
import yaml
from pydantic import BaseModel
from tabulate import tabulate
from tqdm import tqdm
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
except ImportError:
    AutoTokenizer = None
    PreTrainedTokenizer = None
    PreTrainedTokenizerFast = None


class ModelConfig(BaseModel):
    """Configuration for a specific model, loaded from vendor settings.

    Attributes:
        tokenizer_type (str): The type of tokenizer, e.g., "huggingface" or "tiktoken".
        tokenizer (Optional[str]): The specific tokenizer name or path.
        n_ctx (int): The context size (window) for the model.
        truncate (bool): Whether to truncate context if it exceeds the model's window.
    """
    tokenizer_type: str = "tiktoken"
    tokenizer: Optional[str] = None
    n_ctx: int = 8192
    truncate: bool = False


def load_config(config_path: str) -> dict:
    """Loads a YAML configuration file from the specified path.

    Args:
        config_path (str): The full path to the YAML configuration file.

    Returns:
        dict: The parsed content of the YAML file as a dictionary.

    Raises:
        SystemExit: If the file is not found or if the YAML is malformed.
    """
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
    """Retrieves and validates the configuration for a specific vendor.

    This function looks up a vendor in the main config, checks for its API key
    environment variable, and fetches the key from the environment.

    Args:
        vendor (str): The name of the vendor (e.g., "together", "groq").
        config (dict): The main configuration dictionary loaded from config.yaml.

    Returns:
        dict: The configuration for the specified vendor, including the API key.

    Raises:
        ValueError: If the vendor is not found in the configuration or if the
            API key environment variable is not set.
    """
    vendor_config = config.get("vendors", {}).get(vendor)
    if not vendor_config:
        raise ValueError(f"Vendor '{vendor}' not found in the configuration.")
    api_key_env = vendor_config.get("api_key_env")
    if not api_key_env or not os.getenv(api_key_env):
        raise ValueError(
            f"API key environment variable '{api_key_env}' not set for vendor '{vendor}'."
        )
    vendor_config["api_key"] = os.getenv(api_key_env)
    return vendor_config


def setup_tokenizer(
    model_name: str, model_config: ModelConfig
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast, Any]:
    """Initializes and returns a tokenizer based on the model's configuration.

    Supports tokenizers from Hugging Face's `transformers` library and `tiktoken`.

    Args:
        model_name (str): The name of the model (e.g., "llama3-8b-8192").
        model_config (ModelConfig): The configuration object for the model, which
            specifies the tokenizer type and name.

    Returns:
        Union[PreTrainedTokenizer, PreTrainedTokenizerFast, Any]: An instantiated
        tokenizer object ready for use.

    Raises:
        ImportError: If the tokenizer_type is "huggingface" but the
            `transformers` library is not installed.
    """
    if model_config.tokenizer_type == "huggingface":
        if AutoTokenizer is None:
            raise ImportError(
                "`transformers` library is not installed. Please install it with `pip install transformers`."
            )
        tokenizer_name = model_config.tokenizer or model_name
        return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)
    else:  # tiktoken
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")


def truncate_messages(
    messages: List[Dict[str, str]],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, Any],
    model_config: ModelConfig,
    max_tokens: int,
) -> List[Dict[str, str]]:
    """Truncates the last message to ensure the conversation fits the context window.

    This function calculates the total tokens used by all but the last message
    and truncates the content of the last message if the total would exceed
    the model's context window (`n_ctx`), accounting for `max_tokens` and a
    safety buffer.

    Args:
        messages (List[Dict[str, str]]): The list of messages in the conversation.
        tokenizer: The tokenizer instance used for encoding and decoding tokens.
        model_config (ModelConfig): The model's configuration, containing `n_ctx`.
        max_tokens (int): The number of tokens to reserve for the generation.

    Returns:
        List[Dict[str, str]]: The list of messages, with the last message
        potentially truncated.
    """
    n_ctx = model_config.n_ctx
    tokens_so_far = sum(len(tokenizer.encode(msg["content"])) for msg in messages[:-1])
    last_message = messages[-1]
    # Reserve 50 tokens for a safety buffer
    available = n_ctx - (tokens_so_far + max_tokens + 50)
    encoded_last_message = tokenizer.encode(last_message["content"])

    if len(encoded_last_message) > available:
        truncated_content = tokenizer.decode(encoded_last_message[:available])
        return messages[:-1] + [
            {"role": last_message["role"], "content": truncated_content}
        ]
    return messages


async def run_quick_test(
    vendor: str,
    model_name: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    main_config: dict,
) -> Dict[str, Any]:
    """Runs a single performance test against a specified model.

    This function sends a request to the specified vendor's model, streams the
    response, and calculates performance metrics like time-to-first-token (TTFT)
    and tokens-per-second (TPS).

    Args:
        vendor (str): The name of the vendor (e.g., "groq").
        model_name (str): The specific model to test (e.g., "llama3-8b-8192").
        messages (List[Dict[str, str]]): The conversation to send to the model.
        max_tokens (int): The maximum number of tokens for the model to generate.
        temperature (float): The sampling temperature for the generation.
        main_config (dict): The main configuration dict containing all vendor info.

    Returns:
        Dict[str, Any]: A dictionary containing the test results, including
        performance metrics, the generated output, and any errors encountered.
        Example success:
        {
            "vendor": "groq", "model": "llama3-8b-8192", "success": True,
            "input": "[...]", "output": "...", "tps": 250.0, "time duration": 2.0
        }
        Example failure:
        {
            "vendor": "groq", "model": "llama3-8b-8192", "success": False,
            "error": "APIConnectionError: ...", "input": "[...]"
        }
    """
    result: Dict[str, Any] = {
        "vendor": vendor,
        "model": model_name,
        "input": json.dumps(messages),
        "success": False,
    }
    try:
        vendor_config = get_vendor_config(vendor, main_config)
        client = openai.AsyncOpenAI(
            api_key=vendor_config["api_key"], base_url=vendor_config.get("api_base")
        )
        model_config = ModelConfig(
            **vendor_config.get("model_config", {}).get(model_name, {})
        )
        tokenizer = setup_tokenizer(model_name, model_config)

        final_messages = (
            truncate_messages(messages, tokenizer, model_config, max_tokens)
            if model_config.truncate
            else messages
        )

        start_time = time.time()
        first_token_time = None
        generated_text = ""
        usage = None

        stream = await client.chat.completions.create(
            model=model_name,
            messages=final_messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if chunk.choices and chunk.choices[0].delta.content:
                if not first_token_time:
                    first_token_time = time.time()
                generated_text += chunk.choices[0].delta.content

        total_time = time.time() - start_time

        if not usage and not generated_text: # Fallback for non-streaming
            completion = await client.chat.completions.create(
                model=model_name, messages=final_messages, stream=False,
                max_tokens=max_tokens, temperature=temperature
            )
            if completion.choices:
                generated_text = completion.choices[0].message.content or ""
            if completion.usage:
                usage = completion.usage

        tps = (usage.completion_tokens / total_time) if usage and total_time > 0 else 0

        result.update({
            "success": True, "output": generated_text, "time duration": total_time,
            "tps": tps, "ttft_sec": first_token_time - start_time if first_token_time else None,
        })
    except Exception as e:
        result["error"] = str(e)
    return result


async def async_main(args: argparse.Namespace):
    """The main asynchronous entry point for the quick test script.

    This function orchestrates the entire process:
    1. Loads environment variables and configuration files.
    2. Substitutes prompt variables into message templates.
    3. Creates and runs all specified tests in parallel using asyncio.
    4. Gathers the results.
    5. Writes the detailed results to a timestamped CSV file.
    6. Prints a summary report to the console.

    Args:
        args (argparse.Namespace): The parsed command-line arguments, including
            the path to the test configuration file.
    """
    load_dotenv()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    main_config_path = os.path.join(script_dir, "config.yaml")

    main_config = load_config(main_config_path)
    quick_test_config = load_config(args.config)

    prompt_vars = quick_test_config.get("prompt_variables", {})

    final_messages = []
    for m in quick_test_config["messages"]:
        content = m["content"]
        for key, value in prompt_vars.items():
            content = content.replace(f"{{{key}}}", str(value))
        final_messages.append({"role": m["role"], "content": content})

    tasks = [
        run_quick_test(
            test["vendor"], model, final_messages, quick_test_config["max_tokens"],
            quick_test_config["temperature"], main_config,
        )
        for test in quick_test_config["tests"] for model in test["models"]
    ]

    results = [
        await future
        for future in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Running tests"
        )
    ]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"result_quick_test_{timestamp}.csv"
    csv_headers = ["vendor", "model", "input", "output", "tps", "time duration"]
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: (r["vendor"], r["model"])))
    print(f"\nResults saved to {csv_filename}")

    # Console Summary
    summary_headers = ["Vendor", "Model", "Success", "TPS", "Time (s)", "Error"]
    table = [
        [
            r["vendor"], r["model"], "✅" if r["success"] else "❌",
            f"{r.get('tps', 0):.2f}" if r["success"] else "-",
            f"{r.get('time duration', 0):.2f}" if r["success"] else "-",
            r.get("error", ""),
        ]
        for r in sorted(results, key=lambda r: (r["vendor"], r["model"]))
    ]
    print("\n--- QUICK TEST SUMMARY ---")
    print(tabulate(table, headers=summary_headers, tablefmt="grid"))


if __name__ == "__main__":
    # This block handles parsing command-line arguments and launching the async main function.
    parser = argparse.ArgumentParser(
        description="Run quick performance tests against various LLM vendors."
    )
    script_dir = os.path.dirname(os.path.realpath(__file__))
    default_config_path = os.path.join(script_dir, "quick_test_config.yaml")

    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path,
        help="Path to the test configuration YAML file. Defaults to 'quick_test_config.yaml' next to the script.",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))
