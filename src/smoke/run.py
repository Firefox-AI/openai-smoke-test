import asyncio
import aiofiles
import time
import os
import argparse
from typing import List, TypedDict
from statistics import mean
import sys
import random
import string
import json
from tabulate import tabulate
import numpy as np
import openai
from tqdm import tqdm
import tiktoken
from datasets import load_dataset

from smoke.unieval.utils import load_config
from .summary.summary_evaluator import SummaryEvaluator
from .summary.summary_generator import SummaryGenerator


class Article(TypedDict):
    url: str
    wordCount: int
    content: str
    reference_summary: str


dataset_lock = asyncio.Lock()
stat_file_lock = asyncio.Lock()
dataset_index = 0


async def iterate_thru_dataset_safe(dataset) -> Article | None:
    global dataset_index
    async with dataset_lock:
        if dataset_index < len(dataset):
            data = dataset[dataset_index]
            dataset_index += 1
            return data
        return None


def get_random_article_from_dataset(dataset) -> Article:
    return random.choice(dataset)


def generate_random_words(count=10, length=10):
    words = []
    for _ in range(count):
        word = "".join(random.choices(string.ascii_lowercase, k=length))
        words.append(word)
    return words


def generate_long_text_file(text_file: str):
    lorem = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Vestibulum vel dolor sit amet lacus ultrices ullamcorper. "
        "Suspendisse nec justo sit amet nulla dapibus sagittis. "
    )
    words = lorem.split()
    repeat_count = (5000 // len(words)) + 1
    full_text = " ".join(words * repeat_count)[: 5000 * 6]
    with open(text_file, "w") as f:
        f.write(full_text)


def generate_even_sizes(n: int, min_words: int, max_words: int) -> List[int]:
    step = (max_words - min_words) / (max(1, n - 1))
    return [int(min_words + i * step) for i in range(n)]


async def run_query(
    session_id: int,
    query_id: int,
    text: str,
    ref_summary: str,
    stats: List[dict],
    model_name: str,
    summary_generator: SummaryGenerator,
    encoding,
    summarization_config,
    summary_evaluator: SummaryEvaluator,
    stats_file: str,
    test_rate_limit=False,
    pbar=None,
    stop_event=None,
    debug=False,
):
    if stop_event.is_set():
        return
    try:
        start_time = time.time()

        summary, first_token_time = await summary_generator.generate(
            model_name, text, stop_event
        )
        end_time = time.time()
        if summary == "":
            # 401 or 429
            stop_event.set()
            return
        token_count = len(encoding.encode(summary))

        # evaluate summary
        score = {}
        if summarization_config.get("use_dataset"):
            if len(summary) > 0 and len(ref_summary) > 0:
                score = await summary_evaluator.score(
                    summary,
                    ref_summary,
                    text,
                    summarization_config.get("llm_unieval_scoring", {}).get(
                        "score_with_llm"
                    ),
                )
            if debug:
                print(
                    f"comparing \n==REF==\n{ref_summary} \nto \n==SUM==\n{summary}\n======= \nyields"
                )
                print(score)

        if summarization_config.get("log_stats"):
            # write score to file with
            async with stat_file_lock:
                async with aiofiles.open(stats_file, mode="a") as f:
                    json_string = json.dumps(
                        {
                            "original_text": text,
                            "reference_summary": ref_summary,
                            "predicted_summary": summary,
                            "ttft": first_token_time - start_time
                            if first_token_time
                            else None,
                            "tps": token_count / (end_time - start_time),
                            "total_time": end_time - start_time,
                            "summary_score": score,
                        }
                    )

                    await f.write(json_string + "\n")

        stats.append(
            {
                "session": session_id,
                "query": query_id,
                "ttft": first_token_time - start_time if first_token_time else None,
                "tps": token_count,
                "summary_score": score,
                "success": True,
                "total_time": end_time - start_time,
                "code": 200,
            }
        )

    except Exception as e:
        try:
            code = int(e.code)
        except Exception:
            code = -1

        stats.append(
            {
                "session": session_id,
                "query": query_id,
                "error": str(e),
                "success": False,
                "code": code,
            }
        )
        if code == 429 and test_rate_limit:
            print("Found a 429 error. Rate limiting test passed.")
            stop_event.set()
            raise asyncio.CancelledError()

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


async def user_session(
    session_id: int,
    sizes: List[int],
    stats: List[dict],
    queries_per_user: int,
    base_text: str,
    ref_summary: str,
    model_name: str,
    dataset,
    same_text: bool,
    same_text_size: int,
    summary_generator: SummaryGenerator,
    encoding,
    summarization_config,
    summary_evaluator: SummaryEvaluator,
    stats_file: str,
    test_rate_limit,
    pbar,
    stop_event,
    debug,
):
    for query_id in range(queries_per_user):
        if stop_event.is_set():
            break

        if dataset and not same_text:
            if summarization_config.get("log_stats"):
                article = await iterate_thru_dataset_safe(dataset)
            else:
                article = get_random_article_from_dataset(dataset)
            if not article:
                break
            text = article["content"]
            ref_summary = article["reference_summary"]
        else:
            if same_text:
                text = base_text[:same_text_size]
            else:
                word_count = random.choice(sizes) - 100
                text = base_text.split()[:word_count]
                text.extend(generate_random_words(10, 10))
                random.shuffle(text)
                text = " ".join(text)

        await run_query(
            session_id,
            query_id,
            text,
            ref_summary,
            stats,
            model_name,
            summary_generator,
            encoding,
            summarization_config,
            summary_evaluator,
            stats_file,
            test_rate_limit,
            pbar,
            stop_event,
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
    stats_file = f"src/smoke/stats/summary/{args.model}.jsonl"
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

    config = load_config()
    summarization_config = config.get("features", {}).get("summarization", {})
    use_dataset, dataset_name = (
        summarization_config.get("use_dataset", False),
        summarization_config.get("dataset_name", ""),
    )
    log_stats = summarization_config.get("log_stats", False)

    base_text, ref_summary = "", ""
    dataset = None
    queries_per_user = args.queries_per_user
    total_queries = args.num_users * queries_per_user
    if use_dataset:
        dataset = load_dataset(f"Mozilla/{dataset_name}")["train"]
        if log_stats:
            # load from stats_file and skip entries with the same original_text
            with open(stats_file, "r") as f:
                stats = [json.loads(line)["original_text"] for line in f]
            dataset = [
                article for article in dataset if article["content"] not in stats
            ]
            total_queries = len(dataset)
            queries_per_user = total_queries // args.num_users

        if args.single_run:
            article = get_random_article_from_dataset(dataset_name)
            base_text = article["content"]
            ref_summary = article["reference_summary"]
    else:
        if log_stats:
            raise Exception("use_dataset must be True when log_stats is True")
        if not os.path.exists(args.text_file):
            generate_long_text_file(args.text_file)

        with open(args.text_file, "r") as f:
            base_text = f.read()

    openai_client = openai.AsyncOpenAI(
        api_key=args.api_key, base_url=args.api_base if args.api_base else None
    )
    summary_generator = SummaryGenerator(openai_client, summarization_config)
    summary_evaluator = (
        SummaryEvaluator(summarization_config.get("llm_unieval_scoring", {}))
        if use_dataset
        else None
    )

    try:
        encoding = tiktoken.encoding_for_model(args.model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    stats = []
    if args.single_run:
        print("Running a single test query with 100 words...")
        sample_text = " ".join(base_text.split()[:100])
        await run_query(
            0,
            0,
            sample_text,
            ref_summary,
            stats,
            args.model,
            summary_generator,
            encoding,
            summarization_config,
            summary_evaluator,
        )
    else:
        sizes = generate_even_sizes(total_queries, args.min_words, args.max_words)

        pbar = tqdm(total=total_queries, desc="Running queries")

        tasks = [
            asyncio.create_task(
                user_session(
                    i,
                    sizes,
                    stats,
                    queries_per_user,
                    base_text,
                    ref_summary,
                    args.model,
                    dataset,
                    args.same_text,
                    args.same_text_size,
                    summary_generator,
                    encoding,
                    summarization_config,
                    summary_evaluator,
                    stats_file,
                    args.test_rate_limit,
                    pbar,
                    stop_event,
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

    if not args.test_rate_limit:
        success = all(entry.get("success", False) for entry in stats)
    else:
        # we want to find at least one 429
        success = any(entry["code"] == 429 for entry in stats)
        if success:
            print("Found a 429 error. Rate limiting test passed.")
            return 0
        else:
            print("No 429 error. Rate limiting test failed.")
            return 1

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

    all_rouge_stats = {
        "rouge1": stats_summary(
            [
                s["summary_score"]["rouge"]["rouge1"]
                for s in stats
                if s.get("summary_score", {}).get("rouge", {}).get("rouge1") is not None
            ],
            "Rouge 1",
        ),
        "rouge2": stats_summary(
            [
                s["summary_score"]["rouge"]["rouge2"]
                for s in stats
                if s.get("summary_score", {}).get("rouge", {}).get("rouge2") is not None
            ],
            "Rouge 2",
        ),
        "rougeLsum": stats_summary(
            [
                s["summary_score"]["rouge"]["rougeLsum"]
                for s in stats
                if s.get("summary_score", {}).get("rouge", {}).get("rougeLsum")
                is not None
            ],
            "RougeLsum",
        ),
    }
    bleu_stats = stats_summary(
        [
            s["summary_score"]["bleu"]["bleu"]
            for s in stats
            if s.get("summary_score", {}).get("bleu", {}).get("bleu") is not None
        ],
        "BLEU",
    )
    all_unieval_stats = {
        "unieval_consistency": stats_summary(
            [
                s["summary_score"]["unieval"]["consistency"]
                for s in stats
                if s.get("summary_score", {}).get("unieval", {}).get("consistency")
                is not None
            ],
            "Unieval consistency",
        ),
        "unieval_coherence": stats_summary(
            [
                s["summary_score"]["unieval"]["coherence"]
                for s in stats
                if s.get("summary_score", {}).get("unieval", {}).get("coherence")
                is not None
            ],
            "Unieval coherence",
        ),
        "unieval_fluency": stats_summary(
            [
                s["summary_score"]["unieval"]["fluency"]
                for s in stats
                if s.get("summary_score", {}).get("unieval", {}).get("fluency")
                is not None
            ],
            "Unieval fluency",
        ),
        "unieval_relevance": stats_summary(
            [
                s["summary_score"]["unieval"]["relevance"]
                for s in stats
                if s.get("summary_score", {}).get("unieval", {}).get("relevance")
                is not None
            ],
            "Unieval relevance",
        ),
        "unieval_overall": stats_summary(
            [
                s["summary_score"]["unieval"]["overall"]
                for s in stats
                if s.get("summary_score", {}).get("unieval", {}).get("overall")
                is not None
            ],
            "Unieval overall",
        ),
    }

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
    score_table = [
        *all_rouge_stats.values(),
        bleu_stats,
        *all_unieval_stats.values(),
    ]

    errors = [s for s in stats if not s.get("success") and s.get("error")]

    print("\n--- SUMMARY REPORT ---")
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

    threshold_errors = []
    if metrics_table[0][1] != "-" and float(
        metrics_table[0][1]
    ) > summarization_config.get("metric_threshold", {}).get("ttft"):
        threshold_errors.append(
            {
                "name": "TTFT",
                "value": float(metrics_table[0][1]),
                "threshold": summarization_config.get("metric_threshold", {}).get(
                    "ttft"
                ),
            }
        )
    if metrics_table[1][1] != "-" and float(
        metrics_table[1][1]
    ) < summarization_config.get("metric_threshold", {}).get("per_query_tps"):
        threshold_errors.append(
            {
                "name": "PER QUERY TPS",
                "value": float(metrics_table[1][1]),
                "threshold": summarization_config.get("metric_threshold", {}).get(
                    "per_query_tps"
                ),
            }
        )
    if metrics_table[2][1] != "-" and float(
        metrics_table[2][1]
    ) > summarization_config.get("metric_threshold", {}).get("round_trip"):
        threshold_errors.append(
            {
                "name": "ROUND TRIP",
                "value": float(metrics_table[2][1]),
                "threshold": summarization_config.get("metric_threshold", {}).get(
                    "round_trip"
                ),
            }
        )

    if use_dataset:
        print("\n--- SCORE REPORT ---")
        print(
            tabulate(
                score_table, headers=["Type", "Mean", "P50", "P90"], tablefmt="grid"
            )
        )
        num_hiccups = sum(
            [
                s["summary_score"]["hiccup"]
                for s in stats
                if s.get("summary_score", {}).get("hiccup")
            ]
        )
        percentage_of_hiccups = num_hiccups / max(1, total)
        print(
            f"Num hiccups: {num_hiccups} --- Percentage of hiccups: {percentage_of_hiccups}"
        )
        # Overall score calculation
        weights = {
            "rouge1": 0.1,
            "rouge2": 0.1,
            "rougeLsum": 0.2,
            "bleu": 0.1,
            "unieval_consistency": 0.05,
            "unieval_coherence": 0.05,
            "unieval_fluency": 0.05,
            "unieval_relevance": 0.05,
            "unieval_overall": 0.2,
            "percentage_of_hiccups": 0.1,
        }
        rouge_score = sum(
            float(all_rouge_stats[key][1]) * weights[key]
            for key in all_rouge_stats
            if all_rouge_stats[key][1] != "-"
        )
        unieval_score = sum(
            float(all_unieval_stats[key][1]) * weights[key]
            for key in all_unieval_stats
            if all_unieval_stats[key][1] != "-"
        )
        overall_score = (
            rouge_score
            + (float(bleu_stats[1]) * weights["bleu"] if bleu_stats[1] != "-" else 0)
            + unieval_score
            + ((1 - percentage_of_hiccups) * weights["percentage_of_hiccups"])
        )
        print(f"Overall Score: --- {args.model}: {overall_score:.2f} ---")
        score_threshold_config = summarization_config.get("score_threshold", {})

        check_thresholds(
            threshold_errors, score_threshold_config.get("rouge", {}), all_rouge_stats
        )

        if bleu_stats[1] != "-" and float(bleu_stats[1]) < score_threshold_config.get(
            "bleu", 0
        ):
            threshold_errors.append(
                {
                    "name": "bleu",
                    "value": float(bleu_stats[1]),
                    "threshold": score_threshold_config.get("bleu"),
                }
            )

        check_thresholds(
            threshold_errors,
            score_threshold_config.get("unieval", {}),
            all_unieval_stats,
        )

        if percentage_of_hiccups > score_threshold_config.get(
            "percentage_of_hiccups", 1
        ):
            threshold_errors.append(
                {
                    "name": "percentage of hiccups",
                    "value": float(percentage_of_hiccups),
                    "threshold": score_threshold_config.get("percentage of hiccups"),
                }
            )

        if overall_score < score_threshold_config.get("overall", 0):
            threshold_errors.append(
                {
                    "name": "overall",
                    "value": float(overall_score),
                    "threshold": score_threshold_config.get("overall"),
                }
            )

    if len(threshold_errors) > 0:
        print("\n--- THRESHOLD ERROR ---")
        error_details = "\n".join(
            [
                f"  - {error['name']}: value {error['value']:.4f} is beyond threshold {error['threshold']:.4f}"
                for error in threshold_errors
            ]
        )
        if summarization_config.get("error_on_threshold_fails", False):
            raise Exception(f"Threshold Error:\n{error_details}")
        else:
            print(error_details)

    return failures


def main():
    parser = argparse.ArgumentParser(description="Async OpenAI Query Benchmark")
    parser.add_argument(
        "--num-users", type=int, default=20, help="Number of concurrent users"
    )
    parser.add_argument(
        "--queries-per-user", type=int, default=10, help="Number of queries per user"
    )
    parser.add_argument(
        "--same-text",
        action="store_true",
        help="Use the same text for all queries",
        default=False,
    )
    parser.add_argument(
        "--same-text-size",
        type=int,
        default=500,
        help="Same text size",
    )
    parser.add_argument(
        "--test-rate-limit",
        action="store_true",
        default=False,
        help="Test the rate limiter",
    )
    parser.add_argument(
        "--min-words", type=int, default=500, help="Minimum number of words per query"
    )
    parser.add_argument(
        "--max-words", type=int, default=5000, help="Maximum number of words per query"
    )
    parser.add_argument(
        "--text-file", type=str, default="long_text.txt", help="Path to base text file"
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
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run a single test query with 100 words",
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show extra debugging information",
    )

    args = parser.parse_args()

    # tests fastly rate limiter
    if args.test_rate_limit:
        args.same_text = True
        args.same_text_size = 10
        args.num_users = 250
        args.queries_per_user = 1

    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
