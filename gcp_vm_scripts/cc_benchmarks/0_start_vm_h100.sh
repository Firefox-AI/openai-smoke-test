#!/bin/bash

# ==============================================================================
# Script: 0_start_vm_h100.sh
# Description: Provisions an H100 GPU VM (a3-highgpu-1g) for CC benchmarks.
#              Supports multi-zone fallback and Confidential Compute (TDX).
#
# Usage:
#   ./0_start_vm_h100.sh [--confidential] [--secure-boot] [--zone <zone>]
#
# Flags:
#   --confidential : Enable Confidential Compute (TDX) for the VM.
#   --secure-boot   : Enable Shielded Secure Boot for the VM.
#   --zone         : Target a specific zone (overrides fallback list).
# ==============================================================================

# Configuration
PROJECT_ID="fx-gen-ai-sandbox"
VM_NAME="h100-test-vm"
MACHINE_TYPE="a3-highgpu-1g"
GPU_COUNT=1
DISK_SIZE=250
SERVICE_ACCOUNT="18209811701-compute@developer.gserviceaccount.com"
IMAGE="projects/ubuntu-os-accelerator-images/global/images/ubuntu-accelerator-2404-amd64-with-nvidia-580-v20251021"

# List of zones to try in sequential order (fallback mechanism)
ZONES=("us-central1-a" "us-central1-c" "europe-west4-b")

# Default Flags
CONFIDENTIAL_FLAG=""
SECURE_BOOT_FLAG="--no-shielded-secure-boot"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --confidential)
      # Enable Confidential Computing with Intel TDX
      CONFIDENTIAL_FLAG="--confidential-compute-type=TDX"
      echo "Enabling Confidential Computing (TDX)..."
      ;;
    --secure-boot)
      # Enable Shielded Secure Boot
      SECURE_BOOT_FLAG="--shielded-secure-boot"
      echo "Enabling Shielded Secure Boot..."
      ;;
    --zone)
      # Override zones list with a specific user-provided zone
      if [[ -n "$2" && "$2" != --* ]]; then
        ZONES=("$2")
        echo "Targeting specific zone: $2"
        shift
      else
        echo "Error: --zone requires a value."
        exit 1
      fi
      ;;
    *)
      echo "Unknown argument: $1"
      ;;
  esac
  shift
done

# Iterate through zones until a VM is successfully created
for ZONE in "${ZONES[@]}"; do
    REGION="${ZONE%-*}"
    echo "--------------------------------------------------------"
    echo "Attempting to start VM $VM_NAME in zone $ZONE..."
    echo "--------------------------------------------------------"

    # 1. Ensure Snapshot Schedule exists in the region
    # Resource policies are regional, so we ensure it exists for the target region.
    echo "Checking/Creating snapshot schedule 'default-schedule-1' in region $REGION..."
    gcloud compute resource-policies create snapshot-schedule default-schedule-1 \
        --project=$PROJECT_ID \
        --region=$REGION \
        --max-retention-days=14 \
        --on-source-disk-delete=keep-auto-snapshots \
        --daily-schedule \
        --start-time=00:00 \
        2>/dev/null || echo "Snapshot schedule already exists or could not be created."

    # 2. Configure Disk based on Zone
    if [[ "$ZONE" == europe-west4* ]]; then
        DISK_TYPE="hyperdisk-balanced"
        DISK_EXTRAS=",provisioned-iops=6000,provisioned-throughput=890"
    else
        DISK_TYPE="pd-balanced"
        DISK_EXTRAS=""
    fi

    # 3. Construct and Execute the gcloud create command
    # Confidential Compute (TDX) and Secure Boot flags are applied here if set via arguments.
    if gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --network-interface="network-tier=PREMIUM,nic-type=GVNIC,stack-type=IPV4_ONLY,subnet=sandbox-vpc-default" \
        --metadata="enable-osconfig=TRUE" \
        --no-restart-on-failure \
        --maintenance-policy="TERMINATE" \
        --provisioning-model="SPOT" \
        --instance-termination-action="STOP" \
        --discard-local-ssds-at-termination-timestamp=true \
        --service-account="$SERVICE_ACCOUNT" \
        --scopes="https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append" \
        --accelerator="count=$GPU_COUNT,type=nvidia-h100-80gb" \
        --create-disk="auto-delete=yes,boot=yes,device-name=$VM_NAME,disk-resource-policy=projects/$PROJECT_ID/regions/$REGION/resourcePolicies/default-schedule-1,image=$IMAGE,mode=rw,size=$DISK_SIZE,type=$DISK_TYPE$DISK_EXTRAS" \
        $SECURE_BOOT_FLAG \
        $CONFIDENTIAL_FLAG \
        --shielded-vtpm \
        --shielded-integrity-monitoring \
        --labels="goog-ops-agent-policy=v2-x86-template-1-4-0,goog-ec-src=vm_add-gcloud" \
        --reservation-affinity="none"; then

        echo "Successfully created VM in $ZONE."

        # 4. Post-creation: Configure Ops Agent
        echo "Configuring Ops Agent..."
        printf 'agentsRule:\n  packageState: installed\n  version: latest\ninstanceFilter:\n  inclusionLabels:\n  - labels:\n      goog-ops-agent-policy: v2-x86-template-1-4-0\n' > config.yaml

        POLICY_NAME="goog-ops-agent-v2-x86-template-1-4-0-${ZONE}"
        echo "Applying Ops Agent policy: $POLICY_NAME"
        gcloud compute instances ops-agents policies create "$POLICY_NAME" \
            --project="$PROJECT_ID" \
            --zone="$ZONE" \
            --file=config.yaml || \
        gcloud compute instances ops-agents policies update "$POLICY_NAME" \
            --project="$PROJECT_ID" \
            --zone="$ZONE" \
            --file=config.yaml || \
        echo "Warning: Failed to create or update Ops Agent policy, but VM is up."

        echo "All set!"
        exit 0
    else
        echo "Failed to create VM in $ZONE. Trying next zone..."
    fi
done

echo "Error: Could not start VM in any of the specified zones."
exit 1
