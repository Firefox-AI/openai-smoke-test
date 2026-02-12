#!/bin/bash

# ==============================================================================
# Script: Upload CC Benchmark Scripts to VM
#
# Purpose:
#   Uploads the necessary CC benchmark scripts to the remote VM's
#   home directory.
#
# Usage:
#   ./upload_cc_benchmarks.sh [INSTANCE_NAME] [--zone ZONE] [--project PROJECT]
#
# Examples:
#   ./upload_cc_benchmarks.sh h100-test-vm
#   ./upload_cc_benchmarks.sh h100-test-vm --zone us-central1-c --project fx-gen-ai-sandbox
#
# If you don't provide an instance name, the script will try to auto-detect it.
# Zone and project are optional and will be auto-detected if not provided.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

echo "===================================================================="
echo "Upload CC Benchmark Scripts to VM"
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
GCLOUD_SCP_CMD="gcloud compute scp"

if [ -n "$ZONE" ]; then
    GCLOUD_SCP_CMD="$GCLOUD_SCP_CMD --zone $ZONE"
fi

if [ -n "$PROJECT" ]; then
    GCLOUD_SCP_CMD="$GCLOUD_SCP_CMD --project $PROJECT"
fi

# List of files to upload from the cc_benchmarks directory
SCRIPT_DIR=$(dirname "$0")
CC_DIR="${SCRIPT_DIR}"

FILES_TO_UPLOAD=(
    "${CC_DIR}/1_setup_environment.sh"
    "${CC_DIR}/2_start_server.sh"
    "${CC_DIR}/3_run_benchmark.sh"
    "${CC_DIR}/4_stop_server.sh"
    "${CC_DIR}/5_package_and_download_artifacts.sh"
)

# Upload the files
echo "Uploading scripts to ~ on $INSTANCE_NAME..."
$GCLOUD_SCP_CMD "${FILES_TO_UPLOAD[@]}" "${INSTANCE_NAME}:~/"

if [ $? -eq 0 ]; then
    echo ""
    echo "===================================================================="
    echo "SUCCESS!"
    echo "===================================================================="
    echo ""
    echo "The following files were uploaded to the home directory on $INSTANCE_NAME:"
    for file in "${FILES_TO_UPLOAD[@]}"; do
        echo "  - $(basename "$file")"
    done
    echo ""
else
    echo ""
    echo "Error: Upload failed."
    exit 1
fi
