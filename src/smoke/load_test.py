import argparse
import asyncio
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import sys
from .run import smoke_test
import numpy as np

# import vertexai
# from vertexai import model_garden

# vertexai.init(project="fx-gen-ai-sandbox", location="us-west1")

# model = model_garden.OpenModel("qwen/qwen3@qwen3-235b-a22b-instruct-2507-fp8")
# endpoint = model.deploy(
#   accept_eula=True,
#   machine_type="a3-highgpu-4g",
#   accelerator_type="NVIDIA_H100_80GB",
#   accelerator_count=4,
#   serving_container_image_uri="us-docker.pkg.dev/deeplearning-platform-release/vertex-model-garden/sglang-serve.cu124.0-4.ubuntu2204.py310:20250428-1803-rc0",
#   endpoint_display_name="qwen_qwen3-235b-a22b-instruct-2507-fp8-mg-one-click-deploy",
#   model_display_name="qwen_qwen3-235b-a22b-instruct-2507-fp8-1757088988583",
#   use_dedicated_endpoint=True,
# )

"""
1. Create instance of VM with GPU on GCP
2. Start the VM and wait for it to be ready
3. Load test against the deployed model
4. Collect and analyze the results
5. Find maximum throughput and latency
"""

async def async_main():
	MODEL_NAME = "qwen3:235b-a22b-instruct-2507 8g us"
	N_START = 1
	N_END = 1000
	N_STEP = 50
	GENERATE = True

	# Ensure stats_dir directory exists
	stats_dir = os.path.dirname(f"src/smoke/stats/load_test/{MODEL_NAME}/")
	if stats_dir and not os.path.exists(stats_dir):
		os.makedirs(stats_dir, exist_ok=True)
	stats_file = os.path.join(stats_dir, "stats.csv")
	

	# do testing here via ip
	stats = []
	if not GENERATE:
		if os.path.exists(stats_file):
			df = pd.read_csv(stats_file)
			for _, row in df.iterrows():
				stats.append({
					"num_users": int(row["num_users"]),
					"metrics": [
						["Time to First Token (s)", row["ttft_avg"]],
						["Tokens/sec", row["tokens_per_sec_avg"]],
						["Round Trip", row["round_trip_avg"]],
					],
					"failures": row["failures"]
				})
	else:
		for i in range(N_START, N_END+1, N_STEP): #simulate from ex: (1 to 50) simulataneous requests
			args = argparse.Namespace(
				num_users=i,
				queries_per_user=1, # test over X seconds or requests
				one_rpups=False,
				model=MODEL_NAME,
				api_base="",
				api_key="ollama",
				min_words=500,
				max_words=5000,
				single_run=False,
				same_text=True,
				same_text_size=500,
				test_rate_limit=False,
				text_file="long_text.txt",
				use_vertex=True,
				debug=True,
			)
			data = await smoke_test(args)
			stats.append({"num_users": i, "metrics": data["metrics_table"], "failures": data["failures"]})

		flat_stats = []
		for stat in stats:
			flat_stat = {
				'num_users': stat['num_users'],
				'ttft_avg': stat['metrics'][0][1],
				'tokens_per_sec_avg': stat['metrics'][1][1],
				'round_trip_avg': stat['metrics'][2][1],
				'failures': stat['failures'],
			}
			flat_stats.append(flat_stat)

		df = pd.DataFrame(flat_stats)
		df.to_csv(stats_file, index=False)
		print(f"Stats logged to {stats_file}")

		# Be sure to clean up VM at the end
		# vm.delete()
	
	if not stats:
		print("No stats to plot.")
		return
	
	num_users = [stat["num_users"] for stat in stats]
	ttft_avgs = [float(stat["metrics"][0][1]) if stat["metrics"][0][1] != '-' else np.nan for stat in stats]
	tokens_per_sec_avgs = [float(stat["metrics"][1][1]) if stat["metrics"][1][1] != '-' else np.nan for stat in stats]
	round_trips = [float(stat["metrics"][2][1]) if stat["metrics"][2][1] != '-' else np.nan for stat in stats]
	failures = [stat["failures"] for stat in stats]

	# Plot Round Trip Time and TTFT Avg on one graph
	plt.figure(figsize=(10, 6))
	plt.plot(num_users, round_trips, marker='o', linestyle='-', label='Round Trip Time (s)')
	plt.plot(num_users, failures, marker='x', linestyle='-', label='Failures', color='red')
	plt.plot(num_users, ttft_avgs, marker='s', linestyle='--', label='TTFT Avg (s)')
	plt.xlabel("Number of Req/s")
	plt.ylabel("Time (s)")
	plt.legend()
	plt.title(f"{MODEL_NAME} Load Test: Round Trip & TTFT")
	plt.grid(True)
	plt.xticks(num_users)

	ymin = 0
	ymax = np.nanmax(np.array(round_trips + ttft_avgs))
	yticks = np.linspace(ymin, ymax, num=10)
	plt.yticks(yticks)
	plot_filename = os.path.join(stats_dir, "round_trip_ttft.png")
	plt.savefig(plot_filename)
	print(f"Plot saved to {plot_filename}")

	# Plot Tokens/sec Avg on a separate graph
	plt.figure(figsize=(10, 6))
	plt.plot(num_users, tokens_per_sec_avgs, marker='^', linestyle=':', label='Tokens/sec Avg')
	plt.plot(num_users, failures, marker='x', linestyle='-', label='Failures', color='red')
	plt.xlabel("Number of Req/s")
	plt.ylabel("Tokens/sec Avg")
	plt.legend()
	plt.title(f"{MODEL_NAME} Load Test: Tokens/sec Avg")
	plt.grid(True)
	plt.xticks(num_users)

	tokens_min = 0
	tokens_max = np.nanmax(np.array(tokens_per_sec_avgs))
	yticks_tokens = np.linspace(tokens_min, tokens_max, num=10)
	plt.yticks(yticks_tokens)
	plot_filename_tokens = os.path.join(stats_dir, "tokens_per_sec.png")
	plt.savefig(plot_filename_tokens)
	print(f"Tokens/sec plot saved to {plot_filename_tokens}")

def main():
	return asyncio.run(async_main())

if __name__ == "__main__":
	sys.exit(main())