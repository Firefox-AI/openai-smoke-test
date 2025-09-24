#!/bin/bash

# ==============================================================================
# Script to tear down the GKE cluster and all related resources.
#
# Prerequisites:
#   - Google Cloud SDK ('gcloud') installed and authenticated.
#   - Kubernetes command-line tool ('kubectl') installed.
#
# Usage:
#   ./gke_llm_stop.sh -p <YOUR_PROJECT_ID> [options]
# ==============================================================================

set -e

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
PROJECT_ID=""
REGION="us-central1"
CLUSTER_NAME="qwen-inference-cluster"
YAML_FILE="qwen3-235b-deploy.yaml"
SECRET_NAME="hf-secret"

# --- Helper Functions ---
usage() {
  echo "Usage: $0 -p <PROJECT_ID> [-r <REGION>] [-c <CLUSTER_NAME>]"
  echo "  -p: Google Cloud Project ID (required)"
  echo "  -r: GKE Cluster Region (default: us-central1)"
  echo "  -c: GKE Cluster Name (default: qwen-inference-cluster)"
  exit 1
}

# --- Argument Parsing ---
while getopts ":p:r:c:" opt; do
  case ${opt} in
    p ) PROJECT_ID=$OPTARG;;
    r ) REGION=$OPTARG;;
    c ) CLUSTER_NAME=$OPTARG;;
    \? ) usage;;
  esac
done

# Check for required arguments
if [ -z "${PROJECT_ID}" ]; then
  usage
fi

# --- Main Execution ---
echo "### Step 1: Configuring gcloud CLI ###"
gcloud config set project "${PROJECT_ID}"
gcloud config set compute/region "${REGION}"

echo "### Step 2: Getting Cluster Credentials ###"
if gcloud container clusters get-credentials "${CLUSTER_NAME}" --region="${REGION}" 2>/dev/null; then
  echo "Successfully connected to cluster ${CLUSTER_NAME}"
  
  echo "### Step 3: Deleting Kubernetes Resources from ${YAML_FILE} ###"
  if [ -f "${YAML_FILE}" ]; then
    kubectl delete -f "${YAML_FILE}" --ignore-not-found=true
    echo "Deleted resources defined in ${YAML_FILE}."
  else
    echo "Warning: ${YAML_FILE} not found. Attempting to delete resources by label..."
    kubectl delete deployment,service,podmonitoring -l app=qwen3-server --ignore-not-found=true
  fi

  echo "### Step 4: Deleting Kubernetes Secret ###"
  kubectl delete secret "${SECRET_NAME}" --ignore-not-found=true
  echo "Deleted secret ${SECRET_NAME}."
  
else
  echo "Warning: Could not connect to cluster ${CLUSTER_NAME}. It may already be deleted or not exist."
fi

echo "### Step 5: Deleting GKE Cluster: ${CLUSTER_NAME} ###"
if gcloud container clusters describe "${CLUSTER_NAME}" --region="${REGION}" &>/dev/null; then
  gcloud container clusters delete "${CLUSTER_NAME}" --region="${REGION}" --quiet
  echo "Successfully deleted GKE cluster ${CLUSTER_NAME}."
else
  echo "Cluster ${CLUSTER_NAME} does not exist or has already been deleted."
fi

echo "### Step 6: Cleaning up local files ###"
if [ -f "${YAML_FILE}" ]; then
  rm -f "${YAML_FILE}"
  echo "Removed local file ${YAML_FILE}."
fi

echo ""
echo "### Teardown Complete! ###"
echo "The following resources have been terminated:"
echo "  - GKE Cluster: ${CLUSTER_NAME}"
echo "  - Kubernetes Deployment: vllm-qwen3-deployment"
echo "  - Kubernetes Service: qwen3-service"
echo "  - Kubernetes Secret: ${SECRET_NAME}"
echo "  - PodMonitoring: vllm-qwen3-monitoring"
echo "  - Local YAML file: ${YAML_FILE}"
