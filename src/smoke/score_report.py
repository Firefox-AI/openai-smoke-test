import asyncio
import csv
import json
import random
from statistics import mean, median, quantiles
from tabulate import tabulate
from tqdm import tqdm
from unieval.utils import load_config
import os
import numpy as np
import openai
import collections
from collections import Counter

STATS_DIR = "src/smoke/stats/"


def normalize(data, lower_is_better=False):
        min_val, max_val = min(data), max(data)
        if max_val == min_val:
            return [1.0] * len(data)
        normalized = [(x - min_val) / (max_val - min_val) for x in data]
        return [1.0 - x for x in normalized] if lower_is_better else normalized

def calculate_performance_score(value, good_threshold, bad_threshold, lower_is_better=True):
    """
    Calculates a 0-1 score for a single metric value based on thresholds.
    Clamps the score between 0.0 and 1.0.
    """
    if lower_is_better:
        if value <= good_threshold:
            return 1.0
        if value >= bad_threshold:
            return 0.0
        # Linearly scale the score between the thresholds
        score = (bad_threshold - value) / (bad_threshold - good_threshold)
    else: # Higher is better
        if value >= good_threshold:
            return 1.0
        if value <= bad_threshold:
            return 0.0
        score = (value - bad_threshold) / (good_threshold - bad_threshold)
    
    return max(0.0, min(1.0, score)) # Ensure score is always [0, 1]

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

def get_report(stats_path: str, summarization_config: dict):
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

    ttft_stats = stats_summary(ttf_times, "Time to First Token (s)")
    per_query_tps_stats = stats_summary(per_query_tps, "Tokens/sec (Per Query)")
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
        "unieval_consistency": stats_summary(
            [s["summary_score"]["unieval"]["consistency"] for s in stats if s.get("summary_score", {}).get("unieval", {}).get("consistency") is not None],
            "Unieval consistency"
        ),
        "unieval_coherence": stats_summary(
            [s["summary_score"]["unieval"]["coherence"] for s in stats if s.get("summary_score", {}).get("unieval", {}).get("coherence") is not None],
            "Unieval coherence"
        ),
        "unieval_fluency": stats_summary(
            [s["summary_score"]["unieval"]["fluency"] for s in stats if s.get("summary_score", {}).get("unieval", {}).get("fluency") is not None],
            "Unieval fluency"
        ),
        "unieval_relevance": stats_summary(
            [s["summary_score"]["unieval"]["relevance"] for s in stats if s.get("summary_score", {}).get("unieval", {}).get("relevance") is not None],
            "Unieval relevance"
        ),
        "unieval_overall": stats_summary(
            [s["summary_score"]["unieval"]["overall"] for s in stats if s.get("summary_score", {}).get("unieval", {}).get("overall") is not None],
            "Unieval overall"
        ),
    }

    total_tokens = sum(s["tps"] for s in stats if s.get("tps") is not None)
    total_duration = sum(s["total_time"] for s in stats if s.get("total_time") is not None)
    global_tps = total_tokens / total_duration if total_duration > 0 else 0

    total = len(stats)
    metrics_table = [
        ttft_stats,
        per_query_tps_stats,
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

    return {"metrics":
        [
            ttft_stats,
            per_query_tps_stats,
            round_trip_stats,
        ],
        "score_table": [
            *all_rouge_stats.values(),
            bleu_stats,
            *all_unieval_stats.values(),
        ],
        "num_hiccups": num_hiccups, 
        "percentage_of_hiccups": percentage_of_hiccups,
        "overall_score": overall_score
}

def score_reports():
    config = load_config("src/smoke/config.yaml")
    stats_files = []
    for root, _, files in os.walk(STATS_DIR):
        for file in files:
            if file.endswith(".jsonl"):
                stats_files.append(os.path.join(root, file))
    
    if not stats_files:
        raise FileNotFoundError(f"No .jsonl files found in {STATS_DIR} or its subdirectories")
    summarization_config = config.get("summarization_config", {})
    
    print(f"Found {len(stats_files)} files to process...")
    all_reports = [get_report(f, summarization_config) for f in stats_files]

    print(f"\n{'='*40}\n🏆 COMPARATIVE SUMMARY\n{'='*40}")

    TTFT_THRESHOLDS = {'good': 0.1, 'bad': 1.5}        # Lower is better (seconds)
    ROUND_TRIP_THRESHOLDS = {'good': 0.5, 'bad': 3.0}  # Lower is better (seconds)
    TPS_THRESHOLDS = {'good': 200, 'bad': 20}          # Higher is better (tokens/sec)

    for r in all_reports:
        r["summary_score"] = r["overall_score"]
        # --- Get raw performance values ---
        ttft = float(r["metrics"][0][1])
        tps = float(r["metrics"][1][1])
        round_trip = float(r["metrics"][2][1])

        # --- Convert performance metrics to a 0-1 score ---
        ttft_score = calculate_performance_score(
            ttft, TTFT_THRESHOLDS['good'], TTFT_THRESHOLDS['bad'], lower_is_better=True
        )
        tps_score = calculate_performance_score(
            tps, TPS_THRESHOLDS['good'], TPS_THRESHOLDS['bad'], lower_is_better=False
        )
        round_trip_score = calculate_performance_score(
            round_trip, ROUND_TRIP_THRESHOLDS['good'], ROUND_TRIP_THRESHOLDS['bad'], lower_is_better=True
        )

        # --- Define the weights for the final score calculation ---
        weights = {
            'summary': 0.50, # The original score from ROUGE, BLEU, etc.
            'tps': 0.15,     # Tokens per Second
            'ttft': 0.15,    # Time to First Token
            'round_trip': 0.20
        }

        # --- Calculate the new, blended overall score ---
        r["overall_score"] = (
            r["summary_score"] * weights['summary'] +
            tps_score * weights['tps'] +
            ttft_score * weights['ttft'] +
            round_trip_score * weights['round_trip']
        )
    
    # --- 1. Sort reports by overall score for a clear ranking ---
    all_reports.sort(key=lambda r: r['overall_score'], reverse=True)

    # --- 2. Prepare summary table ---
    summary_table = [
        [
            stats_files[i].replace("src/smoke/stats/", "").replace(".jsonl", ""),
            r["overall_score"],
            f"{float(r['metrics'][1][1]):.2f}", #tokens/sec
            f"{float(r['metrics'][0][1]):.2f}", #ttft
            f"{float(r['metrics'][2][1]):.2f}", #round trip

            f"{float(r['score_table'][1][1]):.2f}", #rouge1
            f"{float(r['score_table'][2][1]):.2f}", #rouge2
            f"{float(r['score_table'][3][1]):.2f}", #rougelsum
            f"{float(r['score_table'][4][1]):.2f}", #bleu
            f"{float(r['score_table'][8][1]):.2f}", #unieval
            r['num_hiccups'],
            f"{r['summary_score']:.4f}",
        ] for i, r in enumerate(all_reports)
    ]
    print(tabulate(summary_table, headers=["Model", "Overall Score", "Tokens/Sec", "TTFT (Mean)", "Round Trip", "ROUGE 1", "ROUGE 2", "ROUGE L", "BLEU", "Unieval", "Hiccups", "Overall Summary Score"], tablefmt="grid"))

    csv_path = "comparative_summary.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["File Path", "Overall Score", "Tokens/Sec", "TTFT (Mean)", "Round Trip", "ROUGE 1", "ROUGE 2", "ROUGE L", "BLEU", "Unieval", "Hiccups", "Overall Summary Score"])
        writer.writerows(summary_table)

async def compare_with_ai():
    # Path to the reference file
    reference_path = os.path.join(STATS_DIR, "mistral-small-2503.jsonl")
    if not os.path.exists(reference_path):
        print(f"Reference file {reference_path} not found.")
        return

    # Load reference stats
    with open(reference_path, "r") as f:
        reference_stats = [json.loads(line) for line in f]

    # Find all other stats files
    other_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(STATS_DIR)
        for file in files
        if file.endswith(".jsonl") and file != "mistral-small-2503.jsonl"
    ]
    openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Compare each reference entry with matching entries in other files (by original_text)
    all_other_stats = []
    for other_path in other_files:
        with open(other_path, "r") as f:
            other_stats = [json.loads(line) for line in f]
            all_other_stats.append((other_path, other_stats))

    LIMIT = 100
    preferences = []

    async def get_gpt_preference(prompt, openai_client):
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    # Shuffle reference_stats for random sampling
    indices = list(range(len(reference_stats)))
    random.shuffle(indices)
    batch_prompts = []
    batch_info = []
    count = 0
    for idx in tqdm(indices[:min(LIMIT, len(reference_stats))]):
        ref_entry = reference_stats[idx]
        for other_path, other_stats in all_other_stats:
            matching = next((e for e in other_stats if e.get("original_text") == ref_entry.get("original_text")), None)
            if not matching:
                continue
            shuffle = random.randint(0, 1) == 1
            m1 = "mistral-small-2503" if shuffle else other_path.replace(STATS_DIR, "").replace(".jsonl", "")
            m2 = other_path.replace(STATS_DIR, "").replace(".jsonl", "") if shuffle else "mistral-small-2503"
            s1 = ref_entry.get("predicted_summary", "") if shuffle else matching.get("predicted_summary", "")
            s2 = matching.get("predicted_summary", "") if shuffle else ref_entry.get("predicted_summary", "")
            prompt = (
                f"Compare the following two summaries and state which you prefer ('{m1}' or '{m2}' or 'no preference'). "
                f"Respond in JSON: {{\"preference\": \"{m1}\" or \"{m2}\"}}\n\n"
                f"Original Text: {ref_entry.get('original_text', '')}\n"
                f"({m1}): {s1}\n"
                f"({m2}): {s2}\n"
            )
            batch_prompts.append(prompt)
            batch_info.append((m1, m2, ref_entry, matching))
            count += 1
            if count % 4 == 0:
                # Call get_gpt_preference in a batch using asyncio.gather
                contents = await asyncio.gather(
                    *[get_gpt_preference(p, openai_client) for p in batch_prompts]
                )
                for i, content in enumerate(contents):
                    try:
                        # Remove code block markers if present
                        if content.startswith("```"):
                            content = content.replace("```json", "").replace("```", "").strip()
                        result = json.loads(content)
                        m1, m2, _, _ = batch_info[i]
                        preferred = result.get("preference")
                        preferences.append({"model_1": m1, "model_2": m2, "preferred": preferred})
                    except Exception as e:
                        print(e)
                        m1, m2, _, _ = batch_info[i]
                        preferences.append({"model_1": m1, "model_2": m2, "preferred": "error"})
                batch_prompts = []
                batch_info = []
    # Process any remaining prompts
    if batch_prompts:
        contents = await asyncio.gather(
            *[get_gpt_preference(p, openai_client) for p in batch_prompts]
        )
        for i, content in enumerate(contents):
            try:
                result = json.loads(content)
                m1, m2, _, _ = batch_info[i]
                preferred = result.get("preference")
                preferences.append({"model_1": m1, "model_2": m2, "preferred": preferred})
            except Exception as e:
                print(e)
                m1, m2, _, _ = batch_info[i]
                preferences.append({"model_1": m1, "model_2": m2, "preferred": "error"})

    print("AI Preferences:", preferences)
    # Count preferences for each model
    preference_counter = Counter([p["preferred"] for p in preferences if "preferred" in p])
    total_preferences = sum(preference_counter.values())

    print("\n--- Model Preference Stats ---")
    for model, count in preference_counter.items():
        pct = (count / total_preferences) * 100 if total_preferences else 0
        print(f"{model}: {count} times ({pct:.2f}%)")
    with open("ai_preferences.json", "w") as f:
        json.dump(preferences, f, indent=2)
    print("Preferences written to ai_preferences.json")

def score_ai_comparisons():
    # {"reference_score": 0.9, "predicted_score": 0.85, "preference": "reference", "reason": "...", "original_text": "...", "reference_model": "mistral-small-2503", "predicted_model": "meta-llama/Llama-4-Scout-17B-16E-Instruct"}
    comparison_path = "mistral-ai-comparison.jsonl"
    reference_scores = []
    predicted_scores = []
    preference_counts = collections.Counter()
    reference_model = None
    predicted_model = None
    total = 0

    with open(comparison_path, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
            except Exception:
                continue
            if reference_model is None:
                reference_model = record.get("reference_model", "reference")
            if predicted_model is None:
                predicted_model = record.get("predicted_model", "predicted")
            ref_score = record.get("reference_score")
            pred_score = record.get("predicted_score")
            pref = record.get("preference")
            if isinstance(ref_score, (int, float)):
                reference_scores.append(ref_score)
            if isinstance(pred_score, (int, float)):
                predicted_scores.append(pred_score)
            if pref:
                preference_counts[record.get("reference_model") if pref == "reference" else record.get("predicted_model")] += 1
            total += 1

    # Show average predicted scores for each predicted model
    model_scores = collections.defaultdict(list)
    model_scores["mistral-2503-small"].append(sum(reference_scores)/len(reference_scores))
    with open(comparison_path, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
            except Exception:
                continue
            pred_model = record.get("predicted_model")
            pred_score = record.get("predicted_score")
            if pred_model and isinstance(pred_score, (int, float)):
                model_scores[pred_model].append(pred_score)
    print("\nAverage scores (scoring by GPT-4o):")
    table = []
    for model, scores in model_scores.items():
        avg_score = sum(scores) / len(scores) if scores else 0
        preferred_count = preference_counts.get(model, 0)
        preferred_pct = (preferred_count / total * 100) if total else 0
        # For mistral-2503-small, also show its preference percentage
        if model == "mistral-2503-small":
            mistral_pref_pct = (dict(preference_counts).get("mistral-small-2503", 0) / total * 100) if total else 0
            table.append([model, f"{avg_score:.4f}", f"{mistral_pref_pct:.2f}%"])
        else:
            table.append([model, f"{avg_score:.4f}", f"{preferred_pct:.2f}%"])
    print(tabulate(table, headers=["Model", "Average Score", "Preferred %"], tablefmt="grid"))

    mistral_pref_pct = (dict(preference_counts).get("mistral-small-2503", 0) / total * 100) if total else 0
    print(f"\nMistral preferred: {mistral_pref_pct:.2f}% of the time")
    print(f"Total comparisons: {total}")

def score_ai_comparisons2():
    # Load AI preferences
    with open("ai_preferences.json", "r") as f:
        preferences = json.load(f)

    # Collect stats
    total = len(preferences)
    model_wins = {}
    model_counts = {}

    for record in preferences:
        m1 = record.get("model_1")
        m2 = record.get("model_2")
        preferred = record.get("preferred")
        # Count total appearances
        model_counts[m1] = model_counts.get(m1, 0) + 1
        model_counts[m2] = model_counts.get(m2, 0) + 1
        # Count wins (including 'no preference')
        model_wins[preferred] = model_wins.get(preferred, 0) + 1

    print("\n--- AI Preference Stats ---")
    table = []
    for model in sorted(model_counts.keys()):
        wins = model_wins.get(model, 0)
        appearances = model_counts[model]
        win_pct = (wins / appearances * 100) if appearances else 0
        # Calculate losses and losses (no preference)
        losses = sum(1 for r in preferences if (r.get("model_1") == model or r.get("model_2") == model) and r.get("preferred") not in [model, "no preference"])
        no_pref_losses = sum(1 for r in preferences if (r.get("model_1") == model or r.get("model_2") == model) and r.get("preferred") == "no preference")
        table.append([model, wins, losses, no_pref_losses, appearances, f"{win_pct:.2f}%"])
    # Show 'no preference' count and percentage
    no_pref_count = model_wins.get("no preference", 0)
    no_pref_pct = (no_pref_count / total * 100) if total else 0
    table.append(["no preference", no_pref_count, "-", "-", "-", f"{no_pref_pct:.2f}%"])
    # Sort table by Win % (descending), skip header row
    table_sorted = sorted(table, key=lambda x: float(x[5].replace('%', '')), reverse=True)
    print(tabulate(table_sorted, headers=["Model", "Wins", "Losses", "Losses (no pref)", "Appearances", "Win %"], tablefmt="grid"))

    print(f"\nTotal comparisons: {total}")

if __name__ == "__main__":
    # score_reports()
    # asyncio.run(compare_with_ai())
    # score_ai_comparisons()
    score_ai_comparisons2()