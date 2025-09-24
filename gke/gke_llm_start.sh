#!/bin/bash

# ==============================================================================
# Script to provision a GKE cluster and deploy the Qwen3 model for inference.
#
# Prerequisites:
#   - Google Cloud SDK ('gcloud') installed and authenticated.
#   - Kubernetes command-line tool ('kubectl') installed.
#   - A valid Google Cloud Project with billing enabled.
#   - A Hugging Face account with a 'read' access token.
#   - (Optional) A specific reservation for GPU resources in Google Cloud.
#
# Usage:
#   ./gke_llm_start.sh -p <YOUR_PROJECT_ID> -t <YOUR_HUGGING_FACE_TOKEN> [options]
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
# Default values, can be overridden by command-line flags
PROJECT_ID="fx-gen-ai-sandbox"
REGION="us-west1"
CLUSTER_NAME="qwen-inference-cluster"
NETWORK="default"
SUBNETWORK="default"
RESERVATION_URL="" # Optional: specify a reservation URL if needed
HUGGING_FACE_TOKEN=""
MODEL_ID="Qwen/Qwen3-235B-A22B-Instruct-2507"
DEPLOYMENT_NAME="vllm-qwen3-deployment"
SERVICE_NAME="qwen3-service"
SECRET_NAME="hf-secret"
YAML_FILE="qwen3-235b-deploy.yaml"

# --- Helper Functions ---
usage() {
  echo "Usage: $0 -p <PROJECT_ID> -t <HUGGING_FACE_TOKEN> [-r <REGION>] [-c <CLUSTER_NAME>] [-n <NETWORK>] [-s <SUBNETWORK>] [-u <RESERVATION_URL>]"
  echo "  -p: Google Cloud Project ID (required)"
  echo "  -t: Hugging Face Read Token (required)"
  echo "  -r: GKE Cluster Region (default: us-central1)"
  echo "  -c: GKE Cluster Name (default: qwen-inference-cluster)"
  echo "  -n: Network (default: default)"
  echo "  -s: Subnetwork (default: default)"
  echo "  -u: Reservation URL (optional)"
  exit 1
}

# --- Argument Parsing ---
while getopts ":p:t:r:c:n:s:u:" opt; do
  case ${opt} in
    p ) PROJECT_ID=$OPTARG;;
    t ) HUGGING_FACE_TOKEN=$OPTARG;;
    r ) REGION=$OPTARG;;
    c ) CLUSTER_NAME=$OPTARG;;
    n ) NETWORK=$OPTARG;;
    s ) SUBNETWORK=$OPTARG;;
    u ) RESERVATION_URL=$OPTARG;;
    \? ) usage;;
  esac
done

# Check for required arguments
if [ -z "${PROJECT_ID}" ] || [ -z "${HUGGING_FACE_TOKEN}" ]; then
  usage
fi

# --- Main Execution ---
echo "### Step 1: Configuring gcloud CLI ###"
gcloud config set project "${PROJECT_ID}"
gcloud config set compute/region "${REGION}"

echo "### Step 2: Enabling required APIs ###"
gcloud services enable container.googleapis.com

echo "### Step 3: Creating GKE Cluster: ${CLUSTER_NAME} in ${REGION} ###"
gcloud container clusters create-auto "${CLUSTER_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --release-channel=rapid \
    --network="${NETWORK}" \
    --subnetwork="${SUBNETWORK}"

echo "### Step 4: Getting Cluster Credentials ###"
gcloud container clusters get-credentials "${CLUSTER_NAME}" --region="${REGION}"

echo "### Step 5: Creating Kubernetes Secret for Hugging Face Token ###"
kubectl create secret generic "${SECRET_NAME}" \
    --from-literal=hf_token="${HUGGING_FACE_TOKEN}" \
    --dry-run=client -o yaml | kubectl apply -f -

echo "### Step 6: Creating Deployment YAML (${YAML_FILE}) ###"
cat > "${YAML_FILE}" <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${DEPLOYMENT_NAME}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: qwen3-server
  template:
    metadata:
      labels:
        app: qwen3-server
        ai.gke.io/model: ${MODEL_ID}
        ai.gke.io/inference-server: vllm
    spec:
      containers:
      - name: qwen-inference-server
        image: us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20250801_0916_RC01
        resources:
          requests:
            cpu: "10"
            memory: "1000Gi"
            ephemeral-storage: "500Gi"
            nvidia.com/gpu: "8"
          limits:
            cpu: "10"
            memory: "1000Gi"
            ephemeral-storage: "500Gi"
            nvidia.com/gpu: "8"
        command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
        args:
        - --model=\$(MODEL_ID)
        - --tensor-parallel-size=8
        - --host=0.0.0.0
        - --port=8000
        - --max-model-len=8192
        - --max-num-seqs=4
        - --dtype=bfloat16
        env:
        - name: MODEL_ID
          value: "${MODEL_ID}"
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: ${SECRET_NAME}
              key: hf_token
        volumeMounts:
        - mountPath: /dev/shm
          name: dshm
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 1320
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 1320
          periodSeconds: 5
      volumes:
      - name: dshm
        emptyDir:
          medium: Memory
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-b200
        cloud.google.com/reservation-name: "${RESERVATION_URL}"
        cloud.google.com/reservation-affinity: "specific"
        cloud.google.com/gke-gpu-driver-version: latest
---
apiVersion: v1
kind: Service
metadata:
  name: ${SERVICE_NAME}
spec:
  selector:
    app: qwen3-server
  type: ClusterIP
  ports:
    - protocol: TCP
      port: 8000
      targetPort: 8000
---
apiVersion: monitoring.googleapis.com/v1
kind: PodMonitoring
metadata:
  name: vllm-qwen3-monitoring
spec:
  selector:
    matchLabels:
      app: qwen3-server
  endpoints:
  - port: 8000
    path: /metrics
    interval: 30s
EOF

echo "### Step 7: Applying Deployment to GKE Cluster ###"
kubectl apply -f "${YAML_FILE}"

echo "### Step 8: Waiting for Deployment to be available (this may take up to 25 minutes) ###"
kubectl wait \
    --for=condition=Available \
    --timeout=1500s deployment/"${DEPLOYMENT_NAME}"

echo "### Step 9: Setting up port-forwarding ###"
echo "In a new terminal, run the following command to forward the port:"
echo "kubectl port-forward service/${SERVICE_NAME} 8000:8000"
echo ""
echo "### Deployment Complete! ###"
echo "You can now send requests to http://127.0.0.1:8000/v1/chat/completions"
echo ""
echo "Example curl command:"
echo "curl http://127.0.0.1:8000/v1/chat/completions \\"
echo "  -X POST \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{"
echo "    \"model\": \"${MODEL_ID}\","
echo "    \"messages\": ["
echo "      {"
echo "        \"role\": \"user\","
echo "        \"content\": \"Describe a GPU in one short sentence?\""
echo "      }"
echo "    ]"
echo "  }'"
