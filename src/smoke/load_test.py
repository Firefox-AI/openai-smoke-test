import argparse
import asyncio
import os
import matplotlib.pyplot as plt
import pandas as pd
import sys
from .run import smoke_test
import numpy as np

import vertexai
from vertexai import model_garden

"""
1. Deploy VertexAI deployment
2. Load test against the deployed model
3. Collect and analyze the results
4. Find maximum throughput and latency
5. Undeploy model
"""

MODEL_NAME_BASE = "qwen_qwen3-235b-a22b-instruct-2507-fp8"
MODEL_NAME = "qwen/qwen3@qwen3-235b-a22b-instruct-2507-fp8"
MODEL_ID = "1757088988583"
NUM_GPUS = 8
REGION = "us-west1"

#ensure GOOGLE_APPLICATION_CREDENTIALS is set env var
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creds.json"

vertexai.init(project="fx-gen-ai-sandbox", location=REGION)

def deploy_model():
	model = model_garden.OpenModel(MODEL_NAME)
	endpoint = model.deploy(
		accept_eula=True,
		machine_type=f"a3-highgpu-{NUM_GPUS}g",
		accelerator_type="NVIDIA_H100_80GB",
		accelerator_count=NUM_GPUS,
		endpoint_display_name=f"{MODEL_NAME_BASE}-load-test-deployment",
		model_display_name=f"{MODEL_NAME_BASE}-{MODEL_ID}",
		use_dedicated_endpoint=True,
	)
	return endpoint

def delete_deployment(endpoint):
	endpoint.undeploy_all()
	endpoint.delete()

async def async_main():
	MODEL_NAME = f"{MODEL_NAME_BASE} concurrent # 1"
	N_START = 1
	N_END = 50
	N_STEP = 1
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
						["Total Tokens", row["total_tokens_avg"]]
					],
					"failures": row["failures"]
				})
	else:
		endpoint = deploy_model()
		for i in range(N_START, N_END+1, N_STEP): #simulate from ex: (1 to 50) simulataneous requests
			args = argparse.Namespace(
				num_users=i,
				queries_per_user=10, # test over X seconds if one_rpups, else requests
				one_rpups=True,
				model=MODEL_NAME,
				api_base="",
				api_key="ollama",
				min_words=1000,
				max_words=99999999,
				single_run=False,
				same_text=True,
				same_text_size=5000,
				test_rate_limit=False,
				text_file="long_text.txt",
				vertex_region=REGION,
				vertex_uri=endpoint.name,
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
				'total_tokens_avg': stat['metrics'][3][1],
				'failures': stat['failures'],
			}
			flat_stats.append(flat_stat)

		df = pd.DataFrame(flat_stats)
		df.to_csv(stats_file, index=False)
		print(f"Stats logged to {stats_file}")

		# Be sure to clean up VM at the end
		delete_deployment(endpoint)

	if not stats:
		print("No stats to plot.")
		return
	
	num_users = [stat["num_users"] for stat in stats]
	ttft_avgs = [float(stat["metrics"][0][1]) if stat["metrics"][0][1] != '-' else np.nan for stat in stats]
	round_trips = [float(stat["metrics"][2][1]) if stat["metrics"][2][1] != '-' else np.nan for stat in stats]
	total_tokens = [float(stat["metrics"][3][1]) if stat["metrics"][3][1] != '-' else np.nan for stat in stats]
	tokens_per_sec_avgs = [float(stat["metrics"][1][1]) if stat["metrics"][1][1] != '-' else np.nan for stat in stats]
	failures = [stat["failures"] for stat in stats]

	x_axis = num_users
	# OR
	# x_axis = total_tokens

	# Plot Round Trip Time and TTFT Avg on one graph
	plt.figure(figsize=(10, 6))
	plt.plot(x_axis, round_trips, marker='o', linestyle='', label='Round Trip Time (s)')
	plt.plot(x_axis, failures, marker='x', linestyle='', label='Failures', color='red')
	plt.plot(x_axis, ttft_avgs, marker='s', linestyle='', label='TTFT Avg (s)')
	plt.xlabel("Number of Concurrent requests")
	plt.ylabel("Time (s)")
	plt.legend()
	plt.title(f"{MODEL_NAME} Load Test: Round Trip & TTFT")
	plt.grid(True)
	# xmin = np.nanmin(np.array(num_users))
	# xmax = np.nanmax(np.array(num_users))
	# xticks = np.linspace(xmin, xmax, num=10)
	# plt.xticks(xticks)

	# ymin = 0
	# ymax = np.nanmax(np.array(round_trips + ttft_avgs))
	# yticks = np.linspace(ymin, ymax, num=10)
	# plt.yticks(yticks)
	plot_filename = os.path.join(stats_dir, "round_trip_ttft.png")
	plt.savefig(plot_filename)
	print(f"Plot saved to {plot_filename}")

	# Plot Tokens/sec Avg on a separate graph
	plt.figure(figsize=(10, 6))
	plt.plot(x_axis, tokens_per_sec_avgs, marker='^', linestyle='', label='Tokens/sec Avg')
	plt.plot(x_axis, failures, marker='x', linestyle='', label='Failures', color='red')
	plt.xlabel("Number of Concurrent requests")
	plt.ylabel("Tokens/sec Avg")
	plt.legend()
	plt.title(f"{MODEL_NAME} Load Test: Tokens/sec Avg")
	plt.grid(True)
	# plt.xticks(xticks)

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