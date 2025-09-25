#!/bin/bash

# ==============================================================================
# Script to provision a GKE cluster, create a GPU node pool, and deploy a model.
#
# Prerequisites:
#   - Google Cloud SDK ('gcloud') installed and authenticated.
#   - Kubernetes command-line tool ('kubectl') installed.
#   - A valid Google Cloud Project with billing enabled.
#   - A Hugging Face account with a 'read' access token.
#   - A specific reservation for GPU resources in Google Cloud.
#
# Usage:
#   ./gke_llm_start.sh -p <PROJECT_ID> -t <HF_TOKEN> -f <YAML_FILE> [options]
# ==============================================================================

set -e

# --- Configuration ---
PROJECT_ID=""
HUGGING_FACE_TOKEN=""
YAML_FILE=""
REGION="us-central1"
ZONE="us-central1-a"
CLUSTER_NAME="qwen-inference-cluster"
NODE_POOL_NAME=""
MACHINE_TYPE=""
ACCELERATOR=""
MAX_NODES=4
NETWORK="sandbox-vpc-default"
SUBNETWORK="sandbox-vpc-default"
RESERVATION_URL=""
SECRET_NAME="hf-secret"

# --- Helper Functions ---
usage() {
  echo "Usage: $0 -p <PROJECT_ID> -t <HUGGING_FACE_TOKEN> -f <YAML_FILE> [options]"
  echo "  -p: Google Cloud Project ID (required)"
  echo "  -t: Hugging Face Read Token (required)"
  echo "  -f: Path to deployment YAML file (required)"
  echo "  -r: GKE Cluster Region (default: us-central1)"
  echo "  -z: GKE Cluster Zone (default: us-central1-a)"
  echo "  -c: GKE Cluster Name (default: qwen-inference-cluster)"
  echo "  -o: Node Pool Name (required)"
  echo "  -m: Machine Type (required)"
  echo "  -a: Accelerator, e.g. 'type=nvidia-l4,count=1' (required)"
  echo "  -n: Network (default: sandbox-vpc-default)"
  echo "  -s: Subnetwork (default: sandbox-vpc-default)"
  echo "  -u: Reservation URL (optional)"
  echo "  --max-nodes: Maximum number of nodes for the node pool (default: 4)"
  echo "  --spot: If set, creates a spot node pool."
  echo "  --recreate-nodepool: If set, deletes and recreates the node pool if it already exists."
  exit 1
}

# --- Argument Parsing ---
RECREATE_NODEPOOL=false
SPOT_INSTANCE=false

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -p)
      PROJECT_ID="$2"
      shift # past argument
      shift # past value
      ;;
    -t)
      HUGGING_FACE_TOKEN="$2"
      shift # past argument
      shift # past value
      ;;
    -f)
      YAML_FILE="$2"
      shift # past argument
      shift # past value
      ;;
    -r)
      REGION="$2"
      shift # past argument
      shift # past value
      ;;
    -z)
      ZONE="$2"
      shift # past argument
      shift # past value
      ;;
    -c)
      CLUSTER_NAME="$2"
      shift # past argument
      shift # past value
      ;;
    -o)
      NODE_POOL_NAME="$2"
      shift # past argument
      shift # past value
      ;;
    -m)
      MACHINE_TYPE="$2"
      shift # past argument
      shift # past value
      ;;
    -a)
      ACCELERATOR="$2"
      shift # past argument
      shift # past value
      ;;
    -n)
      NETWORK="$2"
      shift # past argument
      shift # past value
      ;;
    -s)
      SUBNETWORK="$2"
      shift # past argument
      shift # past value
      ;;
    -u)
      RESERVATION_URL="$2"
      shift # past argument
      shift # past value
      ;;
    --max-nodes)
      MAX_NODES="$2"
      shift # past argument
      shift # past value
      ;;
    --recreate-nodepool)
      RECREATE_NODEPOOL=true
      shift # past argument
      ;;
    --spot)
      SPOT_INSTANCE=true
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
if [ -z "${HUGGING_FACE_TOKEN}" ]; then
  echo "Error: Hugging Face Token (-t) is required. Make sure the shell variable you are using is not empty."
  usage
fi
if [ -z "${YAML_FILE}" ]; then
  echo "Error: YAML file (-f) is required."
  usage
fi

if [ -z "${NODE_POOL_NAME}" ]; then
    echo "Error: Node Pool Name (-o) is required."
    usage
fi
if [ -z "${MACHINE_TYPE}" ]; then
    echo "Error: Machine Type (-m) is required."
    usage
fi
if [ -z "${ACCELERATOR}" ]; then
    echo "Error: Accelerator (-a) is required."
    usage
fi

if [ "${SPOT_INSTANCE}" = true ] && [ -n "${RESERVATION_URL}" ] && [ "${RESERVATION_URL}" != "" ]; then
  echo "Error: The --spot option and a reservation URL (-u) cannot be used at the same time."
  exit 1
fi

# Check if YAML file exists
if [ ! -f "${YAML_FILE}" ]; then
  echo "Error: YAML file '${YAML_FILE}' not found."
  exit 1
fi

echo "--> Using Region: ${REGION}, Zone: ${ZONE}"

# --- Main Execution ---
echo "### Step 1: Configuring gcloud CLI ###"
gcloud config set project "${PROJECT_ID}"
gcloud config set compute/region "${REGION}"

echo "### Step 2: Enabling required APIs ###"
gcloud services enable container.googleapis.com compute.googleapis.com

echo "### Step 3: Creating GKE Standard Cluster: ${CLUSTER_NAME} ###"
# Check if the cluster already exists
if ! gcloud container clusters describe "${CLUSTER_NAME}" --region "${REGION}" --project "${PROJECT_ID}" &> /dev/null; then
  echo "Cluster '${CLUSTER_NAME}' not found. Creating it now..."
  gcloud container clusters create "${CLUSTER_NAME}" \
      --project="${PROJECT_ID}" \
      --region="${REGION}" \
      --release-channel=rapid \
      --network="${NETWORK}" \
      --subnetwork="${SUBNETWORK}" \
      --enable-shielded-nodes \
      --shielded-secure-boot \
      --shielded-integrity-monitoring \
      --enable-ip-alias \
      --workload-pool="${PROJECT_ID}.svc.id.goog"
else
  echo "Cluster '${CLUSTER_NAME}' already exists. Skipping creation."
fi

echo "### Step 3b: Checking and Creating Reservation ###"
if [ -n "${RESERVATION_URL}" ] && [ "${RESERVATION_URL}" != "" ]; then
  # Note: gcloud describe returns a non-zero exit code if the resource is not found
  if ! gcloud compute reservations describe "${RESERVATION_URL}" --zone="${ZONE}" &> /dev/null; then
    echo "Reservation '${RESERVATION_URL}' not found. Creating it now in zone '${ZONE}'..."
    gcloud compute reservations create "${RESERVATION_URL}" \
      --project="${PROJECT_ID}" \
      --zone="${ZONE}" \
      --machine-type="${MACHINE_TYPE}" \
      --accelerator="${ACCELERATOR}" \
      --vm-count="${MAX_NODES}"
  else
    echo "Reservation '${RESERVATION_URL}' already exists. Skipping creation."
  fi
else
    echo "No reservation specified. Skipping reservation check."
fi

echo "### Step 3a: Creating/Updating GPU Node Pool: ${NODE_POOL_NAME} ###"
NODE_POOL_EXISTS=$(gcloud container node-pools describe "${NODE_POOL_NAME}" --cluster "${CLUSTER_NAME}" --region "${REGION}" --project "${PROJECT_ID}" &> /dev/null; echo $?)

if [ "${RECREATE_NODEPOOL}" = true ] && [ "${NODE_POOL_EXISTS}" -eq 0 ]; then
  echo "Node pool '${NODE_POOL_NAME}' exists and --recreate-nodepool is set. Deleting it now..."
  gcloud container node-pools delete "${NODE_POOL_NAME}" --cluster "${CLUSTER_NAME}" --region "${REGION}" --project "${PROJECT_ID}" --quiet
  echo "Node pool '${NODE_POOL_NAME}' deleted."
  NODE_POOL_EXISTS=1 # Mark as non-existent for recreation
fi

if [ "${NODE_POOL_EXISTS}" -ne 0 ]; then
  echo "Node pool '${NODE_POOL_NAME}' not found or being recreated. Creating it now..."
  GCLOUD_CMD=(gcloud container node-pools create "${NODE_POOL_NAME}"
    --cluster "${CLUSTER_NAME}"
    --project "${PROJECT_ID}"
    --region "${REGION}"
    --machine-type "${MACHINE_TYPE}"
    --accelerator "${ACCELERATOR}"
    --enable-autoscaling
    --min-nodes=0
    --max-nodes="${MAX_NODES}"
    --shielded-secure-boot
    --shielded-integrity-monitoring
    --enable-gvnic
  )

  if [ -n "${RESERVATION_URL}" ] && [ "${RESERVATION_URL}" != "" ]; then
    GCLOUD_CMD+=(--reservation-affinity=specific --reservation="${RESERVATION_URL}")
    echo "Creating node pool with reservation: ${RESERVATION_URL}"
  elif [ "${SPOT_INSTANCE}" = true ]; then
    GCLOUD_CMD+=(--spot)
    echo "Creating node pool using spot instances."
  else
    echo "Creating on-demand node pool."
  fi

  "${GCLOUD_CMD[@]}"
else
  echo "Node pool '${NODE_POOL_NAME}' already exists. Skipping creation."
fi

echo "### Step 4: Getting Cluster Credentials ###"
gcloud container clusters get-credentials "${CLUSTER_NAME}" --region="${REGION}" --project="${PROJECT_ID}"


echo "### Step 6: Creating Kubernetes Secret for Hugging Face Token ###"
kubectl create secret generic "${SECRET_NAME}" \
    --from-literal=hf_token="${HUGGING_FACE_TOKEN}" \
    --dry-run=client -o yaml | kubectl apply -f -

echo "### Step 7: Preparing and Applying Deployment YAML (${YAML_FILE}) ###"
# This part of your original script is complex and has been preserved.
# It correctly handles the reservation URL replacement.

# Create temporary deployment file with timestamp
TEMP_YAML="/tmp/deployment_$(date +%s).yaml"
cp "${YAML_FILE}" "${TEMP_YAML}"
# Remove strict GPU driver label that nodes typically don't have by default
sed -i.bak '/cloud.google.com\/gke-gpu-driver-version:/d' "${TEMP_YAML}"

if [ -n "${RESERVATION_URL}" ] && [ "${RESERVATION_URL}" != "" ]; then
  echo "Using reservation: ${RESERVATION_URL}"
  sed -i.bak "s|RESERVATION_URL|${RESERVATION_URL}|g" "${TEMP_YAML}"
  # Remove spot toleration and node selector when using a reservation
  sed -i.bak '/cloud.google.com\/gke-spot:/d' "${TEMP_YAML}"
  sed -i.bak '/- key: cloud.google.com\/gke-spot/,/effect: NoSchedule/d' "${TEMP_YAML}"
elif [ "${SPOT_INSTANCE}" = true ]; then
  if ! grep -q "gke-spot" "${YAML_FILE}"; then
    echo "Error: --spot is specified, but the YAML file '${YAML_FILE}' does not contain a 'gke-spot' nodeSelector or toleration."
    exit 1
  fi
  echo "Using spot instance - removing reservation nodeSelector lines"
  sed -i.bak '/cloud.google.com\/reservation-name:/d' "${TEMP_YAML}"
  sed -i.bak '/cloud.google.com\/reservation-affinity:/d' "${TEMP_YAML}"
else
  echo "Using on-demand instance - removing reservation and spot nodeSelector lines"
  sed -i.bak '/cloud.google.com\/reservation-name:/d' "${TEMP_YAML}"
  sed -i.bak '/cloud.google.com\/reservation-affinity:/d' "${TEMP_YAML}"
  sed -i.bak '/cloud.google.com\/gke-spot:/d' "${TEMP_YAML}"
  sed -i.bak '/- key: cloud.google.com\/gke-spot/,/effect: NoSchedule/d' "${TEMP_YAML}"
fi

echo "Applying final deployment YAML..."
kubectl apply -f "${TEMP_YAML}"

echo "### Step 8: Waiting for Deployment to be available ###"
DEPLOYMENT_NAME=$(grep "name:" "${YAML_FILE}" | head -1 | awk '{print $2}')
echo "Waiting for deployment '${DEPLOYMENT_NAME}' to become available. This can take up to 30 minutes..."
kubectl wait \
    --for=condition=Available \
    --timeout=1800s deployment/"${DEPLOYMENT_NAME}"

echo "✅ Deployment is now available!"

# The final port-forwarding and curl instructions
SERVICE_NAME=$(grep -A 10 "kind: Service" "${YAML_FILE}" | grep "name:" | awk '{print $2}')
echo "In a new terminal, run the following command to forward the port:"
echo "kubectl port-forward service/${SERVICE_NAME} 8000:8000"
echo ""
echo "### Deployment Complete! ###"
