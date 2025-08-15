import json
from statistics import mean, median, quantiles
from tabulate import tabulate
from unieval.utils import load_config

STATS_FILE = "src/smoke/mistral-small-2503.jsonl"

def stats_summary(values, name):
    if not values:
        return [name, "-", "-", "-"]
    values_sorted = sorted(values)
    p50 = median(values_sorted)
    p90 = quantiles(values_sorted, n=10)[8] if len(values_sorted) >= 2 else values_sorted[0]
    return [name, f"{mean(values_sorted):.4f}", f"{p50:.4f}", f"{p90:.4f}"]

def check_thresholds(threshold_errors, threshold_config, stats_dict):
    for key, threshold in threshold_config.items():
        if key in stats_dict and stats_dict[key][1] != "-" and float(stats_dict[key][1]) < threshold:
            threshold_errors.append({
                "name": key,
                "value": float(stats_dict[key][1]),
                "threshold": threshold
            })

def main(stats_path: str, summarization_config: dict):
    if summarization_config is None:
        summarization_config = {"metric_threshold": {}, "score_threshold": {}}

    stats = []
    with open(stats_path, "r") as f:
        for line in f:
            stats.append(json.loads(line))

    ttf_times = [s["ttft"] for s in stats if s.get("ttft") is not None]
    per_query_tps = [
        s["tps"] / s["total_time"]
        for s in stats
        if s.get("tps") is not None and s.get("total_time")
    ]
    round_trip_stats = stats_summary(
        [s["total_time"] for s in stats if s.get("total_time")],
        "Round trip (s)"
    )

    all_rouge_stats = {
        "rouge1": stats_summary(
            [s["summary_score"]["rouge"]["rouge1"] for s in stats if s.get("summary_score", {}).get("rouge", {}).get("rouge1") is not None],
            "Rouge 1"
        ),
        "rouge2": stats_summary(
            [s["summary_score"]["rouge"]["rouge2"] for s in stats if s.get("summary_score", {}).get("rouge", {}).get("rouge2") is not None],
            "Rouge 2"
        ),
        "rougeLsum": stats_summary(
            [s["summary_score"]["rouge"]["rougeLsum"] for s in stats if s.get("summary_score", {}).get("rouge", {}).get("rougeLsum") is not None],
            "RougeLsum"
        ),
    }
    bleu_stats = stats_summary(
        [s["summary_score"]["bleu"]["bleu"] for s in stats if s.get("summary_score", {}).get("bleu", {}).get("bleu") is not None],
        "BLEU"
    )
    all_unieval_stats = {
        # "unieval_consistency": stats_summary(
        #     [s["summary_score"]["unieval"]["consistency"] for s in stats if s.get("summary_score", {}).get("unieval", {}).get("consistency") is not None],
        #     "Unieval consistency"
        # ),
        # "unieval_coherence": stats_summary(
        #     [s["summary_score"]["unieval"]["coherence"] for s in stats if s.get("summary_score", {}).get("unieval", {}).get("coherence") is not None],
        #     "Unieval coherence"
        # ),
        # "unieval_fluency": stats_summary(
        #     [s["summary_score"]["unieval"]["fluency"] for s in stats if s.get("summary_score", {}).get("unieval", {}).get("fluency") is not None],
        #     "Unieval fluency"
        # ),
        # "unieval_relevance": stats_summary(
        #     [s["summary_score"]["unieval"]["relevance"] for s in stats if s.get("summary_score", {}).get("unieval", {}).get("relevance") is not None],
        #     "Unieval relevance"
        # ),
        # "unieval_overall": stats_summary(
        #     [s["summary_score"]["unieval"]["overall"] for s in stats if s.get("summary_score", {}).get("unieval", {}).get("overall") is not None],
        #     "Unieval overall"
        # ),
    }

    total_tokens = sum(s["tps"] for s in stats if s.get("tps") is not None)
    total_duration = sum(s["total_time"] for s in stats if s.get("total_time") is not None)
    global_tps = total_tokens / total_duration if total_duration > 0 else 0

    total = len(stats)
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

    print("\n--- SUMMARY REPORT ---")
    print(f"Total Queries: {total}")
    print(tabulate(metrics_table, headers=["Metric", "Mean", "P50", "P90"], tablefmt="grid"))

    print(
        f"\nGlobal Throughput: {global_tps:.2f} tokens/sec across {total_duration:.2f} seconds"
    )

    threshold_errors = []
    metric_threshold = summarization_config.get("metric_threshold", {})
    if metrics_table[0][1] != "-" and float(metrics_table[0][1]) > metric_threshold.get("ttft", float("inf")):
        threshold_errors.append({
            "name": "TTFT",
            "value": float(metrics_table[0][1]),
            "threshold": metric_threshold.get("ttft")
        })
    if metrics_table[1][1] != "-" and float(metrics_table[1][1]) < metric_threshold.get("per_query_tps", float("-inf")):
        threshold_errors.append({
            "name": "PER QUERY TPS",
            "value": float(metrics_table[1][1]),
            "threshold": metric_threshold.get("per_query_tps")
        })
    if metrics_table[2][1] != "-" and float(metrics_table[2][1]) > metric_threshold.get("round_trip", float("inf")):
        threshold_errors.append({
            "name": "ROUND TRIP",
            "value": float(metrics_table[2][1]),
            "threshold": metric_threshold.get("round_trip")
        })

    print("\n--- SCORE REPORT ---")
    print(tabulate(score_table, headers=["Type", "Mean", "P50", "P90"], tablefmt="grid"))
    hiccups_found = [s for s in stats if s.get("summary_score", {}).get("hiccup")]
    num_hiccups = sum([s["summary_score"]["hiccup"] for s in stats if s.get("summary_score", {}).get("hiccup")])
    percentage_of_hiccups = num_hiccups / max(1, total)
    print(f"Num hiccups: {num_hiccups} --- Percentage of hiccups: {percentage_of_hiccups:.4f}")
    weights = {
        "rouge1": .1,
        "rouge2": .1,
        "rougeLsum": 0.2,
        "bleu": 0.1,
        "unieval_consistency": 0.05,
        "unieval_coherence": 0.05,
        "unieval_fluency": 0.05,
        "unieval_relevance": 0.05,
        "unieval_overall": 0.2,
        "percentage_of_hiccups": 0.1,
    }
    rouge_score = sum(float(all_rouge_stats[key][1]) * weights[key] for key in all_rouge_stats if all_rouge_stats[key][1] != "-")
    unieval_score = sum(float(all_unieval_stats[key][1]) * weights[key] for key in all_unieval_stats if all_unieval_stats[key][1] != "-")
    overall_score = (
        rouge_score +
        (float(bleu_stats[1]) * weights["bleu"] if bleu_stats[1] != "-" else 0) +
        unieval_score +
        ((1-percentage_of_hiccups) * weights["percentage_of_hiccups"])
    )
    print(f"Overall Score: --- {overall_score:.2f} ---")
    score_threshold_config = summarization_config.get("score_threshold", {})

    check_thresholds(threshold_errors, score_threshold_config.get("rouge", {}), all_rouge_stats)

    if bleu_stats[1] != "-" and float(bleu_stats[1]) < score_threshold_config.get("bleu", 0):
        threshold_errors.append({
            "name": "bleu",
            "value": float(bleu_stats[1]),
            "threshold": score_threshold_config.get("bleu")
        })

    check_thresholds(threshold_errors, score_threshold_config.get("unieval", {}), all_unieval_stats)

    if percentage_of_hiccups > score_threshold_config.get("percentage_of_hiccups", 1):
        threshold_errors.append({
            "name": "percentage of hiccups",
            "value": float(percentage_of_hiccups),
            "threshold": score_threshold_config.get("percentage of hiccups")
        })

    if overall_score < score_threshold_config.get("overall", 0):
        threshold_errors.append({
            "name": "overall",
            "value": float(overall_score),
            "threshold": score_threshold_config.get("overall")
        })

    if len(hiccups_found) > 0:
        print("\n--- HICCUPS FOUND IN PREDICTED SUMMARIES ---")
        for i, hiccup in enumerate(hiccups_found):
            print(f"{i+1}.\tOriginal Text: {hiccup["original_text"]}\n\n\tReference Summary: {hiccup["reference_summary"]}\n\n\tPredicted Summary: {hiccup["predicted_summary"]}\n")

    if len(threshold_errors) > 0:
        print("\n--- THRESHOLD ERROR ---")
        error_details = "\n".join([
            f"  - {error['name']}: value {error['value']:.4f} is beyond threshold {error['threshold']:.4f}" for error in threshold_errors
        ])
        raise Exception(f"Threshold Error:\n{error_details}")

if __name__ == "__main__":
    config = load_config("src/smoke/config.yaml")
    main(STATS_FILE, config.get("summarization_config", {}))