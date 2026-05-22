#!/bin/bash

# ==============================================================================
# Script: Package and Download Artifacts
#
# Purpose:
#   Packages benchmark artifacts and logs into a zip file and provides
#   the command to download it to your local machine. This script handles
#   root-owned files by using sudo and changing ownership.
#
# Usage:
#   Run this script on the remote VM after SSH'ing in:
#   ./5_package_and_download_artifacts.sh
#
#   Then copy and run the provided gcloud compute scp command on your laptop.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

echo "===================================================================="
echo "Package and Download Artifacts Script"
echo "===================================================================="
echo ""

# --- Step 1: Install zip if not already installed ---
echo "[Step 1/4] Checking if zip is installed..."
if ! command -v zip &> /dev/null; then
    echo "zip is not installed. Installing..."
    sudo apt-get update
    sudo apt-get install -y zip
    echo "zip installed successfully."
else
    echo "zip is already installed."
fi
echo ""

# --- Step 2: Create timestamped zip archive ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ZIP_FILENAME="llm_benchmarks_${TIMESTAMP}.zip"
ZIP_PATH="/tmp/${ZIP_FILENAME}"

echo "[Step 2/4] Creating zip archive: ${ZIP_FILENAME}"
echo "This may take a moment..."

mkdir -p ~/llm_benchmarks/log
sudo docker cp trtllm_server:/var/log/trtllm_server.log ~/llm_benchmarks/log/trtllm_server.log

# Use sudo to zip the root-owned files
sudo zip -r "${ZIP_PATH}" \
    ~/llm_benchmarks/artifacts \
    ~/llm_benchmarks/log/trtllm_server.log \
    2>/dev/null || true

if [ ! -f "${ZIP_PATH}" ]; then
    echo "Error: Failed to create zip archive."
    exit 1
fi

echo "Archive created successfully at: ${ZIP_PATH}"
echo ""

# --- Step 3: Change ownership to current user ---
echo "[Step 3/4] Changing ownership of zip file to current user..."
sudo chown $(whoami):$(whoami) "${ZIP_PATH}"
echo "Ownership changed successfully."
echo ""

# --- Step 4: Provide download instructions ---
echo "[Step 4/4] Archive is ready for download!"
echo ""
echo "===================================================================="
echo "DOWNLOAD INSTRUCTIONS"
echo "===================================================================="
echo ""
echo "The archive is ready at: ${ZIP_PATH}"
echo ""
echo "On your LOCAL LAPTOP, run the download script:"
echo ""
echo "  ./gcp_vm_scripts/6_download_artifacts.sh"
echo ""
echo "Or manually download with:"
echo ""
echo "  gcloud compute scp [INSTANCE_NAME]:${ZIP_PATH} ~/Downloads/"
echo ""
echo "===================================================================="
echo ""
