#!/bin/bash

# ==============================================================================
# Script: Download Artifacts from VM
#
# Purpose:
#   Downloads the packaged artifacts zip file from the remote VM to your
#   local Downloads folder. Run this script on your LOCAL LAPTOP after
#   running 5_package_and_download_artifacts.sh on the VM.
#
# Usage:
#   ./6_download_artifacts.sh [INSTANCE_NAME] [--zone ZONE] [--project PROJECT]
#
# Examples:
#   ./6_download_artifacts.sh my-vm-instance
#   ./6_download_artifacts.sh h200-euro --zone europe-west1-b --project fx-gen-ai-sandbox
#   ./6_download_artifacts.sh --zone us-central1-a --project my-project
#
# If you don't provide an instance name, the script will try to auto-detect it.
# Zone and project are optional; if not provided, gcloud will use defaults.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

echo "===================================================================="
echo "Download Artifacts from VM"
echo "===================================================================="
echo ""

# Parse command-line arguments
INSTANCE_NAME=""
ZONE=""
PROJECT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --zone)
            ZONE="$2"
            shift 2
            ;;
        --project)
            PROJECT="$2"
            shift 2
            ;;
        *)
            if [ -z "$INSTANCE_NAME" ]; then
                INSTANCE_NAME="$1"
            fi
            shift
            ;;
    esac
done

# If no instance name provided, try to auto-detect
if [ -z "$INSTANCE_NAME" ]; then
    echo "No instance name provided. Attempting to auto-detect..."
    echo ""
    
    # List all running instances
    echo "Available GCP instances:"
    gcloud compute instances list --format="table(name,zone,status)"
    echo ""
    
    # Try to find instances with common naming patterns
    INSTANCES=$(gcloud compute instances list --filter="status=RUNNING" --format="value(name)")
    INSTANCE_COUNT=$(echo "$INSTANCES" | wc -l | tr -d ' ')
    
    if [ "$INSTANCE_COUNT" -eq 1 ]; then
        INSTANCE_NAME=$(echo "$INSTANCES" | head -n 1)
        echo "Auto-detected instance: $INSTANCE_NAME"
        echo ""
    else
        echo "Error: Multiple or no running instances found."
        echo "Please specify the instance name:"
        echo ""
        echo "Usage: $0 [INSTANCE_NAME]"
        echo ""
        exit 1
    fi
fi

echo "Target instance: $INSTANCE_NAME"

# Auto-detect zone and project if not provided
if [ -z "$ZONE" ] || [ -z "$PROJECT" ]; then
    echo "Auto-detecting zone and project for instance..."
    
    # Get instance details
    INSTANCE_INFO=$(gcloud compute instances list --filter="name=$INSTANCE_NAME" --format="value(zone,selfLink)" 2>/dev/null | head -n 1)
    
    if [ -n "$INSTANCE_INFO" ]; then
        # Extract zone if not provided
        if [ -z "$ZONE" ]; then
            DETECTED_ZONE=$(echo "$INSTANCE_INFO" | awk '{print $1}')
            if [ -n "$DETECTED_ZONE" ]; then
                # Extract just the zone name from the full path (e.g., "us-central1-a" from "projects/.../zones/us-central1-a")
                ZONE=$(basename "$DETECTED_ZONE")
                echo "Auto-detected zone: $ZONE"
            fi
        fi
        
        # Extract project if not provided
        if [ -z "$PROJECT" ]; then
            SELF_LINK=$(echo "$INSTANCE_INFO" | awk '{print $2}')
            if [[ "$SELF_LINK" =~ projects/([^/]+)/ ]]; then
                PROJECT="${BASH_REMATCH[1]}"
                echo "Auto-detected project: $PROJECT"
            fi
        fi
    else
        echo "Warning: Could not auto-detect zone/project. Using gcloud defaults."
    fi
fi

if [ -n "$ZONE" ]; then
    echo "Using zone: $ZONE"
fi
if [ -n "$PROJECT" ]; then
    echo "Using project: $PROJECT"
fi
echo ""

# Build gcloud command with optional zone and project
GCLOUD_SSH_CMD="gcloud compute ssh"
GCLOUD_SCP_CMD="gcloud compute scp"

if [ -n "$ZONE" ]; then
    GCLOUD_SSH_CMD="$GCLOUD_SSH_CMD --zone $ZONE"
    GCLOUD_SCP_CMD="$GCLOUD_SCP_CMD --zone $ZONE"
fi

if [ -n "$PROJECT" ]; then
    GCLOUD_SSH_CMD="$GCLOUD_SSH_CMD --project $PROJECT"
    GCLOUD_SCP_CMD="$GCLOUD_SCP_CMD --project $PROJECT"
fi

# Find the most recent zip file in /tmp/ on the VM
echo "Looking for the most recent artifacts zip file on the VM..."
REMOTE_ZIP=$($GCLOUD_SSH_CMD "$INSTANCE_NAME" --command="ls -t /tmp/llm_benchmarks_*.zip 2>/dev/null | head -n 1" 2>/dev/null || echo "")

if [ -z "$REMOTE_ZIP" ]; then
    echo "Error: No artifacts zip file found on the VM."
    echo ""
    echo "Please run the following on the VM first:"
    echo "  ./5_package_and_download_artifacts.sh"
    echo ""
    exit 1
fi

ZIP_FILENAME=$(basename "$REMOTE_ZIP")
echo "Found: $ZIP_FILENAME"
echo ""

# Download the file
echo "Downloading to ~/Downloads/$ZIP_FILENAME..."
$GCLOUD_SCP_CMD "${INSTANCE_NAME}:${REMOTE_ZIP}" ~/Downloads/

if [ $? -eq 0 ]; then
    echo ""
    echo "===================================================================="
    echo "SUCCESS!"
    echo "===================================================================="
    echo ""
    echo "File downloaded to: ~/Downloads/$ZIP_FILENAME"
    echo ""
    echo "You can now extract it with:"
    echo "  unzip ~/Downloads/$ZIP_FILENAME -d ~/Downloads/"
    echo ""
    echo "To clean up the temporary file on the VM, run:"
    if [ -n "$ZONE" ] && [ -n "$PROJECT" ]; then
        echo "  gcloud compute ssh $INSTANCE_NAME --zone $ZONE --project $PROJECT --command=\"rm $REMOTE_ZIP\""
    elif [ -n "$ZONE" ]; then
        echo "  gcloud compute ssh $INSTANCE_NAME --zone $ZONE --command=\"rm $REMOTE_ZIP\""
    elif [ -n "$PROJECT" ]; then
        echo "  gcloud compute ssh $INSTANCE_NAME --project $PROJECT --command=\"rm $REMOTE_ZIP\""
    else
        echo "  gcloud compute ssh $INSTANCE_NAME --command=\"rm $REMOTE_ZIP\""
    fi
    echo ""
else
    echo ""
    echo "Error: Download failed."
    exit 1
fi
