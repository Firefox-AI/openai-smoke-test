import argparse
import asyncio
import os
import yaml
import json
import csv
import time
import uuid
from datetime import datetime
from openai import AsyncOpenAI
from tqdm import tqdm

async def load_prompts(datasets_config, feature_datasets):
    """
    Asynchronously generates prompts from a list of dataset files.
    """
    total_prompts = 0
    # First, count the prompts
    for dataset_name in feature_datasets:
        dataset_info = datasets_config.get(dataset_name)
        if not dataset_info:
            raise ValueError(f"Dataset '{dataset_name}' not found in datasets configuration.")
        
        file_path = dataset_info['path']
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path, 'r') as f:
            for _ in f:
                total_prompts += 1
    
    print(f"INFO: Found a total of {total_prompts} prompts across {len(feature_datasets)} dataset(s).")

    # Now, yield the prompts
    for dataset_name in feature_datasets:
        dataset_info = datasets_config[dataset_name]
        file_path = dataset_info['path']
        dataset_type = dataset_info['type']

        if dataset_type == "custom_messages_jsonl":
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        payload = json.loads(line)
                        if "messages" in payload:
                            yield {"payload": payload, "dataset_name": dataset_name, "dataset_type": dataset_type}
                    except json.JSONDecodeError:
                        print(f"WARNING: Skipping invalid JSON line in {file_path}")
                        continue
        else:
            print(f"WARNING: Skipping unsupported dataset type '{dataset_type}' for dataset '{dataset_name}'")


async def worker(worker_id, client, prompt_generator, results_queue, semaphore, config, run_config):
    """
    A worker that sends requests to the API endpoint as fast as possible.
    """
    while True:
        try:
            prompt_data = await anext(prompt_generator)
        except StopAsyncIteration:
            break # No more prompts

        async with semaphore:
            request_id = str(uuid.uuid4())
            start_time = time.monotonic()
            first_token_time = None
            
            try:
                stream = await client.chat.completions.create(
                    model=run_config['model_name'],
                    messages=prompt_data['payload']['messages'],
                    max_tokens=run_config['max_tokens'],
                    stream=True
                )
                
                output_tokens = -1
                input_tokens = -1
                cached_tokens = -1
                last_chunk = None
                async for chunk in stream:
                    if first_token_time is None:
                        first_token_time = time.monotonic()
                    last_chunk = chunk

                end_time = time.monotonic()

                if last_chunk and hasattr(last_chunk, 'usage') and last_chunk.usage is not None:
                    input_tokens = last_chunk.usage.prompt_tokens
                    output_tokens = last_chunk.usage.completion_tokens
                
                result = {
                    "request_id": request_id,
                    "provider_name": run_config['vendor'],
                    "provider_model": run_config['model'],
                    "test_mode": run_config['mode'],
                    "traffic_level": run_config['level'],
                    "dataset_name": prompt_data['dataset_name'],
                    "dataset_type": prompt_data['dataset_type'],
                    "ttft": first_token_time - start_time if first_token_time else end_time - start_time,
                    "e2e_duration": end_time - start_time,
                    "user_tps": output_tokens / (end_time - start_time) if (end_time - start_time) > 0 and output_tokens != -1 else 0,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_tokens": cached_tokens,
                    "success": True,
                    "error_message": ""
                }

            except Exception as e:
                end_time = time.monotonic()
                result = {
                    "request_id": request_id,
                    "provider_name": run_config['vendor'],
                    "provider_model": run_config['model'],
                    "test_mode": run_config['mode'],
                    "traffic_level": run_config['level'],
                    "dataset_name": prompt_data['dataset_name'],
                    "dataset_type": prompt_data['dataset_type'],
                    "ttft": None,
                    "e2e_duration": end_time - start_time,
                    "user_tps": None,
                    "input_tokens": -1,
                    "output_tokens": -1,
                    "cached_tokens": -1,
                    "success": False,
                    "error_message": str(e)
                }
            
            await results_queue.put(result)


async def results_writer(results_queue, output_file, metrics):
    """
    Writes results from the queue to a CSV file.
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics)
        writer.writeheader()
        while True:
            result = await results_queue.get()
            if result is None: # Sentinel value to stop
                break
            writer.writerow(result)
            results_queue.task_done()


async def main():
    parser = argparse.ArgumentParser(description="Run a stress test on an OpenAI-compatible endpoint.")
    parser.add_argument("--test-config", required=True, help="Path to the stress-test.yaml file.")
    parser.add_argument("--feature", required=True, help="The feature to test (e.g., 'chatbot').")
    parser.add_argument("--vendor", required=True, help="The vendor to test (e.g., 'vertexai').")
    parser.add_argument("--model", required=True, help="The model to test (e.g., 'qwen3-1.5').")
    parser.add_argument("--mode", required=True, choices=['stress', 'qps'], help="The test mode to run.")
    args = parser.parse_args()

    # 1. Load and Validate Config
    base_config_path = 'src/smoke/config.yaml'
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    with open(args.test_config, 'r') as f:
        test_config = yaml.safe_load(f)
    
    # 2-level deep merge for configs
    for key, value in test_config.items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            for inner_key, inner_value in value.items():
                if inner_key in config[key] and isinstance(config[key][inner_key], dict) and isinstance(inner_value, dict):
                    config[key][inner_key].update(inner_value)
                else:
                    config[key][inner_key] = inner_value
        else:
            config[key] = value

    if args.vendor not in config['vendors']:
        raise ValueError(f"Vendor '{args.vendor}' not found in {base_config_path}")
    vendor_config = config['vendors'][args.vendor]

    if 'model_config' not in vendor_config:
        raise ValueError(f"Vendor '{args.vendor}' has no 'model_config' section.")
    
    if args.model not in vendor_config['model_config']:
        raise ValueError(f"Model '{args.model}' not found for vendor '{args.vendor}' in {base_config_path}")
    model_config = vendor_config['model_config'][args.model]

    if args.feature not in config['features']:
        raise ValueError(f"Feature '{args.feature}' not found in {base_config_path} or {args.test_config}")
    feature_config = config['features'][args.feature]

    # 2. Initialization
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"benchmark_{run_timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"INFO: Test results will be saved in: {run_dir}")

    api_key = os.getenv(vendor_config['api_key_env'])
    if not api_key:
        raise ValueError(f"API key environment variable '{vendor_config['api_key_env']}' not set.")

    client = AsyncOpenAI(api_key=api_key, base_url=vendor_config['api_base'])
    
    # 3. Execute Test based on mode
    if args.mode == 'stress':
        levels = config['test_config']['concurrency_levels']
        duration = config['test_config']['stress_duration_seconds']
        
        # Load prompts into a list once to avoid exhausting the generator
        print("INFO: Loading all prompts into memory...")
        prompts = [p async for p in load_prompts(config['datasets'], feature_config['datasets'])]
        print(f"INFO: Loaded {len(prompts)} prompts.")

        for level in levels:
            print(f"\n--- Running STRESS test with concurrency level: {level} for {duration}s ---")
            
            # Create a new async generator from the list for each level
            async def prompt_generator_func():
                for p in prompts:
                    yield p
            
            prompt_generator = prompt_generator_func()
            results_queue = asyncio.Queue()
            semaphore = asyncio.Semaphore(level)
            
            output_file = os.path.join(run_dir, f"raw_results_stress_{level}.csv")
            writer_task = asyncio.create_task(results_writer(results_queue, output_file, config['test_config']['per_request_metrics']))

            run_config = {
                'vendor': args.vendor, 'model': args.model, 'mode': args.mode, 'level': level,
                'model_name': args.model,
                'max_tokens': model_config.get('max_tokens')
            }

            # Start workers
            worker_tasks = [
                asyncio.create_task(worker(i, client, prompt_generator, results_queue, semaphore, config, run_config))
                for i in range(level)
            ]
            
            # Run for the specified duration
            await asyncio.sleep(duration)

            # Stop workers
            for task in worker_tasks:
                task.cancel()
            await asyncio.gather(*worker_tasks, return_exceptions=True)

            # Signal writer to stop
            await results_queue.put(None)
            await writer_task

            print(f"INFO: Finished stress test for concurrency level {level}. Results saved to {output_file}")

    elif args.mode == 'qps':
        print("ERROR: QPS mode is not yet implemented.")
        # Future implementation of QPS logic would go here

    await client.close()
    print("\n--- Stress test run complete. ---")


if __name__ == "__main__":
    asyncio.run(main())
