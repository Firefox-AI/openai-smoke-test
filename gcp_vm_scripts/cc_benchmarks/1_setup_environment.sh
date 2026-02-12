#!/bin/bash

# ==============================================================================
# Script 1: Setup Environment
#
# Purpose:
#   This script performs a one-time setup for a new machine by installing
#   Docker and the NVIDIA Container Toolkit. It is idempotent, meaning it
#   can be safely re-run without causing issues.
#
# Usage:
#   ./1_setup_environment.sh
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# Parse command-line arguments
CONFIDENTIAL=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --confidential) CONFIDENTIAL=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "--- [Step 1/1] Setting up environment ---"

# --- Install Docker ---
if command -v docker &> /dev/null; then
    echo "Docker is already installed. Skipping installation."
else
    echo "Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    echo "Docker installation complete."
fi

# Enable Confidential Computing before verifying CUDA Toolkit.
if [ "$CONFIDENTIAL" = true ]; then
    echo "Enabling Confidential Computing..."
    # Enable Linux Kernel Crypto API 
    echo "install nvidia /sbin/modprobe ecdsa_generic; /sbin/modprobe ecdh; /sbin/modprobe --ignore-install nvidia" | sudo tee /etc/modprobe.d/nvidia-lkca.conf
    sudo update-initramfs -u

    # Enable Confidential Compute GPUs Ready state
    sudo nvidia-smi conf-compute -srs 1

    # Set startup unit to enable Confidential Compute GPUs Ready state on each boot
    sudo tee /etc/systemd/system/cc-gpu-ready.service > /dev/null << 'EOF'
[Unit]
Description=Set Confidential Compute GPU to Ready mode
After=multi-user.target
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStartPre=/bin/sleep 2
ExecStart=/usr/bin/nvidia-smi conf-compute -srs 1
ExecStartPost=/usr/bin/nvidia-smi conf-compute -grs
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable cc-gpu-ready.service

    nvidia-smi conf-compute -f    # should say CC status: ON
    nvidia-smi conf-compute -grs  # should say ready
fi

# --- Install NVIDIA Container Toolkit ---
if dpkg -l | grep -q nvidia-container-toolkit; then
    echo "NVIDIA Container Toolkit is already installed. Skipping installation."
else
    echo "Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "NVIDIA Container Toolkit installation complete."
fi

echo "--- Environment setup is complete. ---"

sleep 3
echo "Enabling persistence mode..."

# Enable persistence mode to establish a secure Security Protocol and Data Model (SPDM) connection
sudo mkdir -p /etc/systemd/system/nvidia-persistenced.service.d
cat <<EOF | sudo tee /etc/systemd/system/nvidia-persistenced.service.d/override.conf            
[Service]
# Clear the original ExecStart then provide our desired command:
ExecStart=
ExecStart=/usr/bin/nvidia-persistenced --user nvidia-persistenced --uvm-persistence-mode --verbose
EOF

sudo systemctl daemon-reload
sudo systemctl enable nvidia-persistenced.service
sudo reboot
