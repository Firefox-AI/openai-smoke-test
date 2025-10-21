import csv
import asyncio
import aiofiles
import time
import os
import argparse
from typing import List
from statistics import mean
import sys
from tabulate import tabulate
import numpy as np
import openai
import random
from tqdm import tqdm
import tiktoken

from smoke.multi_turn_chat.multi_turn_chat_client import MultiTurnChatClient
import json

dataset_lock = asyncio.Lock()
stat_file_lock = asyncio.Lock()
dataset_index = 0


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


# Used for loading /multi_turn_chat/data/queries csvs
def load_single_column_csv(file_path: str) -> List[str]:
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        return [row[0] for row in reader if row]


# Used for loading all initial_context json files
def load_all_json_files_in_dir(directory: str) -> List[List[dict]]:
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]
    if not json_files:
        raise FileNotFoundError("No .json files found in directory.")
    data = []
    for json_file in json_files:
        file_path = os.path.join(directory, json_file)
        with open(file_path, "r", encoding="utf-8") as f:
            file_data = json.load(f)
            data.append(file_data)
    return data


async def run_multi_turn_test(
    session_id: int,
    text_array: List[str],  # all questions to ask
    stats: List[dict],
    model_name: str,
    multi_turn_chat_client: MultiTurnChatClient,
    encoding,
    stop_event=None,
    pbar=None,
    debug=False,
):
    print(f"Session {session_id} starting...")
    for query_id, query in enumerate(text_array):
        if stop_event.is_set():
            return
        try:
            start_time = time.time()
            response, first_token_time = await multi_turn_chat_client.generate_next(
                model_name,
                query,
                stop_event,
            )
            end_time = time.time()
            if response == "":
                # 401 or 429
                stop_event.set()
                return
            token_count = len(encoding.encode(response))

            async with stat_file_lock:
                async with aiofiles.open(
                    f"src/smoke/stats/multi_turn/{model_name}.jsonl", mode="a"
                ) as f:
                    json_string = json.dumps(
                        {
                            "context_length": len(multi_turn_chat_client.context[:-1]),
                            "context": multi_turn_chat_client.context[:-1],
                            "response": response,
                            "ttft": first_token_time - start_time
                            if first_token_time
                            else None,
                            "tps": token_count / (end_time - start_time),
                            "total_time": end_time - start_time,
                        }
                    )
                    await f.write(json_string + "\n")

            stats.append(
                {
                    "session": session_id,
                    "query": query_id,
                    "ttft": first_token_time - start_time if first_token_time else None,
                    "tps": token_count,
                    "success": True,
                    "total_time": end_time - start_time,
                    "code": 200,
                }
            )

        except Exception as e:
            print(f"Error in session {session_id}, query {query_id}: {e}")
            stop_event.set()

        if pbar:
            pbar.update(1)


async def user_session(
    session_id: int,
    queries_per_user: int,
    text_array: List[str],
    stats: List[dict],
    model_name: str,
    multi_turn_chat_client: MultiTurnChatClient,
    encoding,
    stop_event,
    pbar,
    debug,
):
    queries = []
    for _ in range(queries_per_user):
        queries.append(random.choice(text_array))

    await run_multi_turn_test(
        session_id,
        queries,
        stats,
        model_name,
        multi_turn_chat_client,
        encoding,
        stop_event,
        pbar,
        debug,
    )


def check_thresholds(threshold_errors, threshold_config, stats):
    for key in stats.keys():
        score_value = stats[key][1]
        threshold = threshold_config.get(key, 0)

        if score_value != "-" and float(score_value) < threshold:
            threshold_errors.append(
                {
                    "name": key,
                    "value": float(score_value),
                    "threshold": threshold,
                }
            )


async def async_main(args):
    stats_file = f"src/smoke/stats/multi_turn/{args.model}.jsonl"
    # Ensure stats_file directory exists
    stats_dir = os.path.dirname(stats_file)
    if stats_dir and not os.path.exists(stats_dir):
        os.makedirs(stats_dir, exist_ok=True)

    try:
        with open(stats_file, "r") as f:
            pass
    except FileNotFoundError:
        with open(stats_file, "w") as f:
            f.write("")

    stop_event = asyncio.Event()

    queries_per_user = args.queries_per_user
    total_queries = args.num_users * queries_per_user

    openai_client = openai.AsyncOpenAI(
        api_key=args.api_key, base_url=args.api_base if args.api_base else None
    )

    try:
        encoding = tiktoken.encoding_for_model(args.model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    stats = []

    pbar = tqdm(total=total_queries, desc="Running queries")
    text_array = load_single_column_csv(
        "src/smoke/multi_turn_chat/data/queries/generic_queries_2.csv"
    )
    initial_context: List[List[dict]] = [[]]
    if args.start_with_context:
        initial_context = load_all_json_files_in_dir(
            "src/smoke/multi_turn_chat/data/initial_context"
        )

    tasks = [
        asyncio.create_task(
            user_session(
                i,
                queries_per_user,
                text_array,
                stats,
                args.model,
                MultiTurnChatClient(openai_client, random.choice(initial_context)),
                encoding,
                stop_event,
                pbar,
                args.debug,
            )
        )
        for i in range(args.num_users)
    ]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        print("Cancelled all remaining tasks due to rate limiting.")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    pbar.close()
    success = all(entry.get("success", False) for entry in stats)

    ttf_times = [
        s["ttft"] for s in stats if s.get("success") and s.get("ttft") is not None
    ]
    per_query_tps = [
        s["tps"] / s["total_time"]
        for s in stats
        if s.get("success") and s.get("tps") is not None and s.get("total_time")
    ]
    round_trip_stats = stats_summary(
        [s["total_time"] for s in stats if s.get("success") and s.get("total_time")],
        "Round trip (s)",
    )

    total_tokens = sum(s["tps"] for s in stats if s.get("success"))
    total_duration = sum(s["total_time"] for s in stats if s.get("success"))
    global_tps = total_tokens / total_duration if total_duration > 0 else 0

    total = len(stats)
    successes = sum(1 for s in stats if s.get("success"))
    failures = total - successes
    metrics_table = [
        stats_summary(ttf_times, "Time to First Token (s)"),
        stats_summary(per_query_tps, "Tokens/sec (Per Query)"),
        round_trip_stats,
    ]

    errors = [s for s in stats if not s.get("success") and s.get("error")]

    print("\n--- MULTI TURN CHAT REPORT ---")
    print(f"Total Queries: {total}")
    print(f"Successful Queries: {successes}")
    print(f"Failed Queries: {failures}")
    print(
        tabulate(
            metrics_table, headers=["Metric", "Mean", "P50", "P90"], tablefmt="grid"
        )
    )

    print(
        f"\nGlobal Throughput: {global_tps:.2f} tokens/sec across {total_duration:.2f} seconds"
    )
    print("SUCCESS" if success else "FAILURE: Some queries failed")

    if errors:
        print("\n--- FIRST ERROR ---")
        print(f"Session {errors[0]['session']} - Query {errors[0]['query']}")
        print(f"Error: {errors[0]['error']}")


def main():
    parser = argparse.ArgumentParser(description="Async OpenAI Query Benchmark")
    parser.add_argument(
        "--num-users", type=int, default=20, help="Number of concurrent users"
    )
    parser.add_argument(
        "--queries-per-user", type=int, default=5, help="Number of queries per user"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Optional custom base URL for the OpenAI API endpoint",
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name")
    parser.add_argument(
        "--start-with-context",
        action="store_true",
        help="Start each user's conversation with some initial context",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show extra debugging information",
    )

    args = parser.parse_args()

    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
