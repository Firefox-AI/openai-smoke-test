#!/bin/bash
# A robust startup script for installing Ollama, downloading a custom FP8 model,
# converting it, and configuring the service.

# 1. Configuration
LOG_FILE="/var/log/startup-script.log"
MODEL_REPO="https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
MODEL_NAME="qwen3-235b-fp8"
MODEL_DIR="/root/models/Qwen3-235B-A22B-Instruct-2507-FP8"
GGUF_OUTPUT_FILE="/root/models/${MODEL_NAME}.gguf"
LLAMA_CPP_DIR="/root/llama.cpp"

# 2. Logging Setup
exec > >(tee -a ${LOG_FILE}) 2>&1
echo "--- Startup script started at $(date) ---"

# 3. Idempotency Check (Don't run if the final model is already created)
# We check for the custom model tag as the last step of a successful run.
if HOME=/root ollama list | grep -q "${MODEL_NAME}"; then
    echo "Custom model '${MODEL_NAME}' is already installed. Ensuring service is running."
    systemctl enable ollama
    systemctl restart ollama
    echo "--- Startup script finished ---"
    exit 0
fi

# 4. Ollama Installation (if not present)
if [ ! -f "/usr/local/bin/ollama" ]; then
    echo "Installing Ollama..."
    sudo /opt/deeplearning/install-driver.sh
    curl -fsSL https://ollama.com/install.sh | sh
    if [ ! -f "/usr/local/bin/ollama" ]; then
        echo "ERROR: Ollama binary not found after installation."
        exit 1
    fi
    echo "Ollama installed successfully."
else
    echo "Ollama is already installed."
fi

# 5. Install Dependencies for Model Conversion
echo "Installing dependencies (git, git-lfs, python-venv)..."
apt-get update
apt-get install -y git git-lfs python3.10-venv

# 6. Set up llama.cpp for model conversion
echo "Cloning llama.cpp repository..."
git clone https://github.com/ggerganov/llama.cpp.git "${LLAMA_CPP_DIR}"
cd "${LLAMA_CPP_DIR}"
echo "Setting up Python virtual environment for llama.cpp..."
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
deactivate
cd /

# 7. Download the FP8 Model from Hugging Face
echo "Downloading FP8 model from ${MODEL_REPO}..."
echo "This will take a very long time..."
mkdir -p /root/models
git lfs install
git clone "${MODEL_REPO}" "${MODEL_DIR}"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download model from Hugging Face."
    exit 1
fi

# 8. Convert the Model to GGUF Format
echo "Converting model to GGUF format. This is a long, CPU-intensive process..."
cd "${LLAMA_CPP_DIR}"
source .venv/bin/activate
# Using --outtype f16 as a stable intermediate. Direct FP8 GGUF conversion can be complex.
# This creates a high-quality GGUF that Ollama can then use on the FP8-capable hardware.
python3 convert.py "${MODEL_DIR}" --outtype f16 --outfile "${GGUF_OUTPUT_FILE}"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to convert model to GGUF."
    exit 1
fi
deactivate
cd /
echo "Model conversion successful."

# 9. Configure and Start Ollama Service
echo "Configuring and starting Ollama service..."
mkdir -p /etc/systemd/system/ollama.service.d
cat <<EOF > /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_NUM_PARALLEL=1000"
EOF
systemctl daemon-reload
systemctl enable ollama
systemctl restart ollama
echo "Waiting 20 seconds for the service to start..."
sleep 20

# 10. Create the Custom Model in Ollama
echo "Creating custom model '${MODEL_NAME}' in Ollama..."
cat <<EOF > /root/${MODEL_NAME}.modelfile
FROM ${GGUF_OUTPUT_FILE}
EOF
HOME=/root ollama create "${MODEL_NAME}" -f "/root/${MODEL_NAME}.modelfile"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create custom model in Ollama."
    exit 1
fi

# 11. Cleanup (Optional: uncomment to save space)
# echo "Cleaning up original model files to save disk space..."
# rm -rf "${MODEL_DIR}"

echo "--- Startup script finished successfully at $(date) ---"
echo "Ollama is now running with the custom FP8 model: '${MODEL_NAME}'"