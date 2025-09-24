#!/bin/bash

# ==============================================================================
# Script to monitor the running Qwen3 inference server on GKE.
# It shows GPU utilization and tails the container logs.
#
# Prerequisites:
#   - Google Cloud SDK ('gcloud') installed and authenticated.
#   - Kubernetes command-line tool ('kubectl') installed.
#   - The GKE cluster must be running.
#
# Usage:
#   ./gke_llm_monitor.sh [options]
# ==============================================================================

# --- Prerequisite Checks ---
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud command not found. Please install the Google Cloud CLI."
    exit 1
fi

if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl command not found. Please install kubectl."
    exit 1
fi

# --- Configuration ---
CLUSTER_NAME="qwen-inference-cluster"
REGION="us-central1"
PROJECT_ID=""

# --- Helper Functions ---
usage() {
  echo "Usage: $0 [-p <PROJECT_ID>] [-c <CLUSTER_NAME>] [-r <REGION>]"
  echo "  -p: Google Cloud Project ID (optional, uses current gcloud config if not specified)"
  echo "  -c: GKE Cluster Name (default: qwen-inference-cluster)"
  echo "  -r: GKE Cluster Region (default: us-central1)"
  exit 1
}

show_pod_status() {
  echo "### Pod Status ###"
  kubectl get pods -l app=qwen3-server -o wide
  echo ""
}

show_deployment_status() {
  echo "### Deployment Status ###"
  kubectl get deployment vllm-qwen3-deployment
  echo ""
}

show_service_status() {
  echo "### Service Status ###"
  kubectl get service qwen3-service
  echo ""
}

monitor_gpu() {
  local pod_name=$1
  echo "### GPU Utilization for pod: ${pod_name} ###"
  echo "Press [CTRL+C] to stop GPU monitoring and view logs."
  
  trap 'echo ""; echo "Stopping GPU monitoring..."; return' INT
  
  while true; do
    echo "--- $(date) ---"
    if kubectl exec "${pod_name}" -- nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null; then
      echo ""
    else
      echo "Warning: Could not retrieve GPU information. The pod may still be starting up."
    fi
    sleep 10
  done
  
  trap - INT
}

# --- Argument Parsing ---
while getopts ":p:c:r:h" opt; do
  case ${opt} in
    p ) PROJECT_ID=$OPTARG;;
    c ) CLUSTER_NAME=$OPTARG;;
    r ) REGION=$OPTARG;;
    h ) usage;;
    \? ) usage;;
  esac
done

# --- Main Execution ---
echo "### Step 1: Configuring gcloud CLI ###"
if [ -n "${PROJECT_ID}" ]; then
  gcloud config set project "${PROJECT_ID}"
fi
gcloud config set compute/region "${REGION}"

echo "### Step 2: Getting Cluster Credentials ###"
if ! gcloud container clusters get-credentials "${CLUSTER_NAME}" --region="${REGION}" --quiet 2>/dev/null; then
  echo "Error: Could not connect to cluster ${CLUSTER_NAME} in region ${REGION}."
  echo "Please ensure the cluster exists and you have the correct permissions."
  exit 1
fi

echo "### Step 3: Checking Cluster Resources ###"
show_deployment_status
show_service_status
show_pod_status

echo "### Step 4: Finding the inference pod ###"
POD_NAME=$(kubectl get pods -l app=qwen3-server -o jsonpath="{.items[0].metadata.name}" 2>/dev/null)

if [ -z "${POD_NAME}" ]; then
  echo "Error: Could not find a running pod for the qwen3-server."
  echo "Please check if the deployment is running:"
  echo "  kubectl get pods -l app=qwen3-server"
  exit 1
fi

echo "Found pod: ${POD_NAME}"
echo ""

# Check pod status
POD_STATUS=$(kubectl get pod "${POD_NAME}" -o jsonpath="{.status.phase}")
echo "Pod status: ${POD_STATUS}"

if [ "${POD_STATUS}" != "Running" ]; then
  echo "Warning: Pod is not in Running state. Current status: ${POD_STATUS}"
  echo "Pod events:"
  kubectl describe pod "${POD_NAME}" | tail -20
  echo ""
fi

echo "### Step 5: Monitoring Options ###"
echo "Choose monitoring option:"
echo "1) Show GPU utilization (refreshes every 10 seconds)"
echo "2) Stream container logs"
echo "3) Show both (GPU monitoring first, then logs)"
echo "4) Show pod resource usage"
echo "5) Exit"
echo ""

while true; do
  read -p "Enter your choice (1-5): " choice
  case $choice in
    1)
      monitor_gpu "${POD_NAME}"
      break
      ;;
    2)
      echo "### Streaming logs from pod: ${POD_NAME} ###"
      echo "Press [CTRL+C] to exit."
      kubectl logs -f "${POD_NAME}"
      break
      ;;
    3)
      monitor_gpu "${POD_NAME}"
      echo ""
      echo "### Now streaming logs from pod: ${POD_NAME} ###"
      echo "Press [CTRL+C] to exit."
      kubectl logs -f "${POD_NAME}"
      break
      ;;
    4)
      echo "### Resource Usage for pod: ${POD_NAME} ###"
      kubectl top pod "${POD_NAME}" 2>/dev/null || echo "Metrics not available. Ensure metrics-server is running."
      echo ""
      kubectl describe pod "${POD_NAME}" | grep -A 10 "Requests:\|Limits:"
      echo ""
      ;;
    5)
      echo "Exiting monitor."
      exit 0
      ;;
    *)
      echo "Invalid choice. Please enter 1-5."
      ;;
  esac
done
