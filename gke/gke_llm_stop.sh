#!/bin/bash

# ==============================================================================
# Script to tear down the GKE cluster and all related resources.
#
# Prerequisites:
#   - Google Cloud SDK ('gcloud') installed and authenticated.
#
# Usage:
#   ./gke_llm_stop.sh -p <YOUR_PROJECT_ID> [options]
# ==============================================================================

set -e

# --- Configuration ---
PROJECT_ID=""
REGION="us-west1" # Corrected to match the start script
CLUSTER_NAME="qwen-inference-cluster"
SECRET_NAME="hf-secret"

# --- Helper Functions ---
usage() {
  echo "Usage: $0 -p <PROJECT_ID> [-r <REGION>] [-c <CLUSTER_NAME>] [--deployment-only]"
  echo "  -p: Google Cloud Project ID (required)"
  echo "  -r: GKE Cluster Region (default: us-central1)"
  echo "  -c: GKE Cluster Name (default: qwen-inference-cluster)"
  echo "  --deployment-only: If set, only deletes the Kubernetes deployment, service, and secret, leaving the cluster intact."
  exit 1
}

# --- Argument Parsing ---
DEPLOYMENT_ONLY=false

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -p)
      PROJECT_ID="$2"
      shift # past argument
      shift # past value
      ;;
    -r)
      REGION="$2"
      shift # past argument
      shift # past value
      ;;
    -c)
      CLUSTER_NAME="$2"
      shift # past argument
      shift # past value
      ;;
    --deployment-only)
      DEPLOYMENT_ONLY=true
      shift # past argument
      ;;
    *)
      # Unknown option
      usage
      ;;
  esac
done

# Check for required arguments
if [ -z "${PROJECT_ID}" ]; then
  echo "Error: Project ID (-p) is required."
  usage
fi

# --- Main Execution ---
echo "### Step 1: Configuring gcloud CLI ###"
gcloud config set project "${PROJECT_ID}"
gcloud config set compute/region "${REGION}"

echo "### Step 2: Getting Cluster Credentials ###"
if gcloud container clusters get-credentials "${CLUSTER_NAME}" --region="${REGION}" --project="${PROJECT_ID}" 2>/dev/null; then
  echo "Successfully connected to cluster ${CLUSTER_NAME}"

  echo "### Step 3: Deleting Kubernetes AI Model Resources ###"
  # Use a label selector for reliability instead of a specific file
  kubectl delete deployment,service -l ai.gke.io/model --ignore-not-found=true
  echo "Deleted AI model resources."

  echo "### Step 4: Deleting Kubernetes Secret ###"
  kubectl delete secret "${SECRET_NAME}" --ignore-not-found=true
  echo "Deleted secret ${SECRET_NAME}."

else
  echo "Warning: Could not connect to cluster ${CLUSTER_NAME}. It may already be deleted."
fi

if [ "${DEPLOYMENT_ONLY}" = true ]; then
  echo "### Deployment-only teardown complete! ###"
else
  echo "### Step 5: Deleting GKE Cluster: ${CLUSTER_NAME} ###"
  if gcloud container clusters describe "${CLUSTER_NAME}" --region="${REGION}" --project="${PROJECT_ID}" &>/dev/null; then
    read -p "Are you sure you want to delete the entire GKE cluster '${CLUSTER_NAME}'? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      gcloud container clusters delete "${CLUSTER_NAME}" --region="${REGION}" --project="${PROJECT_ID}" --quiet
      echo "✅ Successfully deleted GKE cluster ${CLUSTER_NAME}."
    else
      echo "❌ Cluster deletion cancelled by user."
      exit 1
    fi
  else
    echo "Cluster ${CLUSTER_NAME} does not exist or has already been deleted."
  fi
fi

echo ""
echo "### Teardown Complete! ###"
