#!/bin/bash

# ==============================================================================
# Script 4: Stop TRT-LLM Server
#
# Purpose:
#   Stops and removes the running TRT-LLM server container.
#
# Usage:
#   ./4_stop_server.sh
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
CONTAINER_NAME="trtllm_server"

# --- Main Execution ---
echo "--- Stopping and removing the server container ---"

if [ "$(sudo docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping container '$CONTAINER_NAME'..."
    sudo docker stop $CONTAINER_NAME
    echo "Container stopped."
else
    echo "Container '$CONTAINER_NAME' is not running."
fi

if [ "$(sudo docker ps -aq -f status=exited -f name=$CONTAINER_NAME)" ]; then
    echo "Removing container '$CONTAINER_NAME'..."
    sudo docker rm $CONTAINER_NAME
    echo "Container removed."
elif ! [ "$(sudo docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "No stopped container named '$CONTAINER_NAME' to remove."
fi

echo "--- Cleanup complete ---"
