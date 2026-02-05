#!/bin/bash

# ==============================================================================
# Script 2: Start TRT-LLM Server (CC Benchmarks)
#
# Purpose:
#   Starts the TRT-LLM server in a detached Docker container with the
#   optimal configuration for CC Benchmarks (H100).
#
# Usage:
#   ./2_start_server.sh --hardware <HW> --model <MODEL>
#
# Example:
#   ./2_start_server.sh --hardware H100 --model Qwen
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
DOCKER_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc1"
CONTAINER_NAME="trtllm_server"

# Associative arrays for configurations
declare -A MODEL_PATHS
MODEL_PATHS["Qwen_H100"]="Qwen/Qwen3-30B-A3B"
MODEL_PATHS["Mistral_H100"]="mistralai/Mistral-7B-v0.1"

declare -A MAX_BATCH_SIZES
MAX_BATCH_SIZES["Qwen_H100"]=512
MAX_BATCH_SIZES["Mistral_H100"]=2056

declare -A TP_SIZES
TP_SIZES["Qwen_H100"]=1
TP_SIZES["Mistral_H100"]=1

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
    MAX_BATCH_SIZE=${MAX_BATCH_SIZES[$CONFIG_KEY]}
    TP_SIZE=${TP_SIZES[$CONFIG_KEY]}

    if [ -z "$MODEL_PATH" ]; then
        echo "Error: Invalid hardware/model combination. Only H100 with Qwen or Mistral is supported."
        exit 1
    fi

    echo "--- [Step 1/2] Starting server with configuration ---"
    echo "Hardware: $HARDWARE"
    echo "Model: $MODEL"
    echo "Model Path: $MODEL_PATH"
    echo "Max Batch Size: $MAX_BATCH_SIZE"
    echo "TP Size: $TP_SIZE"
    echo "----------------------------------------------------"

    if [ "$(sudo docker ps -q -f name=$CONTAINER_NAME)" ]; then
        echo "Error: A container with the name '$CONTAINER_NAME' is already running."
        echo "Please stop it first by running ./4_stop_server.sh"
        exit 1
    fi
    
    if [ "$(sudo docker ps -aq -f status=exited -f name=$CONTAINER_NAME)" ]; then
        echo "Removing existing stopped container..."
        sudo docker rm $CONTAINER_NAME
    fi

    if [ "$(sudo docker ps -aq -f name=$CONTAINER_NAME)" ]; then
        echo "Removing existing container..."
        sudo docker rm -f $CONTAINER_NAME
    fi

    echo "Creating local directory for artifacts..."
    mkdir -p ~/llm_benchmarks/artifacts
    echo "Creating local directory for scripts..."
    mkdir -p ~/scripts
    echo "Creating local directory for genai-bench output..."
    mkdir -p ~/genai-bench-output

    echo "Starting Docker container '$CONTAINER_NAME' in detached mode..."
    if [ -z "$HF_TOKEN" ]; then
        echo "HF_TOKEN is not set. Please enter your Hugging Face token:"
        read -s HF_TOKEN
        echo "HF_TOKEN set: $HF_TOKEN"
    fi
    
    sudo docker run --ipc host --gpus all -p 8000:8000 -v ~/llm_benchmarks/artifacts:/app/tensorrt_llm/artifacts -v ~/scripts:/app/scripts -v ~/genai-bench-output:/genai-bench -e HF_TOKEN=$HF_TOKEN -d --name $CONTAINER_NAME $DOCKER_IMAGE sleep infinity

    echo "--- [Step 2/2] Launching trtllm-serve inside the container ---"
    
    EXEC_CMD="trtllm-serve \"$MODEL_PATH\" \
        --host 0.0.0.0 \
        --max_batch_size $MAX_BATCH_SIZE \
        --max_num_tokens 16384 \
        --max_seq_len 16384 \
        --tp_size $TP_SIZE"

    sudo docker exec -d $CONTAINER_NAME bash -c "$EXEC_CMD > /var/log/trtllm_server.log 2>&1 &"

    echo "Server is starting in the background. It may take a few minutes to become ready."
    echo "You can check the logs with: sudo docker exec -it $CONTAINER_NAME tail -f /var/log/trtllm_server.log"
    echo "To get an interactive shell inside the container, run: sudo docker exec -it $CONTAINER_NAME bash"
}

main "$@"
