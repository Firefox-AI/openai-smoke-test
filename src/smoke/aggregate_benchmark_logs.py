import argparse
import os
import pandas as pd
import numpy as np
import yaml


def calculate_statistics(series):
    """Calculates mean, stdev, percentiles, and a simple distribution for a pandas Series."""
    if series.empty or series.isnull().all():
        return {
            "mean": None,
            "stdev": None,
            "p05": None,
            "p50": None,
            "p80": None,
            "p95": None,
            "p99": None,
            "p999": None,
            "distribution": None,
        }

    # Simplified histogram representation
    hist, _ = np.histogram(series, bins=5)
    distribution_str = "|".join(map(str, hist))

    return {
        "mean": series.mean(),
        "stdev": series.std(),
        "p05": series.quantile(0.05),
        "p50": series.quantile(0.50),
        "p80": series.quantile(0.80),
        "p95": series.quantile(0.95),
        "p99": series.quantile(0.99),
        "p999": series.quantile(0.999),
        "distribution": distribution_str,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark logs into a final report."
    )
    parser.add_argument(
        "--run-directory",
        required=True,
        help="The timestamped directory containing the raw benchmark CSVs.",
    )
    parser.add_argument(
        "--test-config",
        default="stress-test.yaml",
        help="Path to the stress-test.yaml file to get the metrics list.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.run_directory):
        raise FileNotFoundError(f"Run directory not found: {args.run_directory}")

    with open(args.test_config, "r") as f:
        config = yaml.safe_load(f)

    final_report_columns = config["test_config"].get("aggregation_metrics", [])
    if not final_report_columns:
        raise ValueError(
            "'aggregation_metrics' not found in test_config. Cannot generate report."
        )

    # Find all raw CSV files in the directory
    csv_files = [
        f
        for f in os.listdir(args.run_directory)
        if f.startswith("raw_results_") and f.endswith(".csv")
    ]
    if not csv_files:
        print(
            f"No raw result CSVs found in {args.run_directory}. Nothing to aggregate."
        )
        return

    # Load all data into a single DataFrame
    df_list = [pd.read_csv(os.path.join(args.run_directory, f)) for f in csv_files]
    full_df = pd.concat(df_list, ignore_index=True)

    # Group by traffic level to aggregate results
    grouped = full_df.groupby("traffic_level")

    aggregated_data = []

    for level, group in grouped:
        successful_requests = group[group["success"]]

        row = {
            "provider_name": group["provider_name"].iloc[0],
            "provider_model": group["provider_model"].iloc[0],
            "traffic_mode": group["test_mode"].iloc[0],
            "traffic_level": level,
            "dataset_name": group["dataset_name"].iloc[
                0
            ],  # Assumes same dataset for the run
        }

        # Input/Output Token Stats
        row["input_total_tokens"] = successful_requests["input_tokens"].sum()
        row["output_total_tokens"] = successful_requests["output_tokens"].sum()
        row["input_avg_len"] = successful_requests["input_tokens"].mean()
        row["input_stdev_len"] = successful_requests["input_tokens"].std()
        row["input_min_len"] = successful_requests["input_tokens"].min()
        row["input_max_len"] = successful_requests["input_tokens"].max()
        row["output_avg_len"] = successful_requests["output_tokens"].mean()
        row["output_stdev_len"] = successful_requests["output_tokens"].std()
        row["output_min_len"] = successful_requests["output_tokens"].min()
        row["output_max_len"] = successful_requests["output_tokens"].max()

        # Cached Token Stats
        row["cached_total_tokens"] = successful_requests["cached_tokens"].sum()
        row["cached_avg_len"] = successful_requests["cached_tokens"].mean()
        row["cached_stdev_len"] = successful_requests["cached_tokens"].std()
        row["cached_min_len"] = successful_requests["cached_tokens"].min()
        row["cached_max_len"] = successful_requests["cached_tokens"].max()

        # TTFT, User TPS, E2E Duration Stats
        for metric in ["ttft", "user_tps", "e2e_duration"]:
            stats = calculate_statistics(successful_requests[metric])
            # Rename e2e_duration to e2e for the report
            report_metric_name = "e2e" if metric == "e2e_duration" else metric
            for stat_name, value in stats.items():
                row[f"{report_metric_name}_{stat_name}"] = value

        # Summary Stats
        total_requests = len(group)
        failed_requests = total_requests - len(successful_requests)
        total_duration = successful_requests["e2e_duration"].sum()

        row["summary_total_num_requests"] = total_requests
        row["summary_num_failed_requests"] = failed_requests
        row["acceptance_rate"] = (
            (len(successful_requests) / total_requests) if total_requests > 0 else 0
        )
        row["summary_total_elapsed_time_s"] = total_duration

        # Note: This is a simplified QPS. A more accurate measure would use the test's wall-clock time.
        row["summary_actual_qps"] = (
            len(successful_requests) / config["test_config"]["stress_duration_seconds"]
        )
        row["summary_job_level_tps"] = (
            row["output_total_tokens"] / total_duration if total_duration > 0 else 0
        )

        aggregated_data.append(row)

    # Create the final DataFrame and save to CSV
    report_df = pd.DataFrame(aggregated_data)

    # Ensure all required columns are present, filling missing ones with None
    for col in final_report_columns:
        if col not in report_df.columns:
            report_df[col] = None

    # Order columns as specified in the config
    report_df = report_df[final_report_columns]

    output_path = os.path.join(args.run_directory, "aggregated_report.csv")
    report_df.to_csv(output_path, index=False)

    print("\n--- Aggregation complete. ---")
    print(f"Final report saved to: {output_path}")


if __name__ == "__main__":
    main()
