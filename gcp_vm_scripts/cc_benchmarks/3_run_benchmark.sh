#!/bin/bash

# ==============================================================================
# Script 3: Run Benchmark Client (CC Benchmarks)
#
# Purpose:
#   Runs the genai-perf benchmark client against the running TRT-LLM server.
#   It uses the optimal request-rate for CC Benchmarks (H100).
#
# Usage:
#   ./3_run_benchmark.sh --hardware <HW> --model <MODEL>
#
# Example:
#   ./3_run_benchmark.sh --hardware H100 --model Qwen
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
CONTAINER_NAME="trtllm_server"

# Associative arrays for configurations
declare -A MODEL_PATHS
MODEL_PATHS["Qwen_H100"]="Qwen/Qwen3-30B-A3B"
MODEL_PATHS["Mistral_H100"]="mistralai/Mistral-7B-v0.1"

declare -A REQUEST_RATES
REQUEST_RATES["Qwen_H100"]=5
REQUEST_RATES["Mistral_H100"]=10

# --- Helper Functions ---
usage() {
    echo "Usage: $0 --hardware H100 --model <Qwen|Mistral>"
    exit 1
}

# --- Main Execution ---
main() {
    HARDWARE=""
    MODEL=""

    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --hardware) HARDWARE="$2"; shift ;;
            --model) MODEL="$2"; shift ;;
            *) usage ;;
        esac
        shift
    done

    if [ -z "$HARDWARE" ] || [ -z "$MODEL" ]; then
        echo "Error: --hardware and --model are required arguments."
        usage
    fi

    CONFIG_KEY="${MODEL}_${HARDWARE}"
    MODEL_PATH=${MODEL_PATHS[$CONFIG_KEY]}
    REQUEST_RATE=${REQUEST_RATES[$CONFIG_KEY]}

    # Determine endpoint type (Mistral base model needs 'completions' endpoint)
    ENDPOINT_TYPE="chat"
    if [[ "$MODEL" == "Mistral" ]]; then
        ENDPOINT_TYPE="completions"
    fi

    if [ -z "$MODEL_PATH" ]; then
        echo "Error: Invalid hardware/model combination. Only H100 with Qwen or Mistral is supported."
        exit 1
    fi

    echo "--- [Step 1/2] Running benchmark with configuration ---"
    echo "Hardware: $HARDWARE"
    echo "Model: $MODEL"
    echo "Model Path: $MODEL_PATH"
    echo "Request Rate: $REQUEST_RATE"
    echo "-------------------------------------------------------"

    if ! [ "$(sudo docker ps -q -f name=$CONTAINER_NAME)" ]; then
        echo "Error: The server container '$CONTAINER_NAME' is not running."
        echo "Please start it first by running ./2_start_server.sh"
        exit 1
    fi

    echo "--- [Step 2/2] Executing genai-perf inside the container ---"
    
    # First, ensure genai-perf is installed inside the container
    sudo docker exec $CONTAINER_NAME pip install genai-perf

    # Execute the benchmark command
    sudo docker exec $CONTAINER_NAME genai-perf profile \
        -m "$MODEL_PATH" \
        --tokenizer "$MODEL_PATH" \
        --endpoint-type $ENDPOINT_TYPE \
        --random-seed 123 \
        --prefix-prompt-length 2500 \
        --synthetic-input-tokens-mean 7500 \
        --synthetic-input-tokens-stddev 0 \
        --output-tokens-mean 1000 \
        --output-tokens-stddev 0 \
        --request-count 1000 \
        --request-rate $REQUEST_RATE \
        --url localhost:8000 \
        --streaming \
        --extra-inputs ignore_eos:true

    echo "--- Benchmark finished ---"
}

main "$@"
