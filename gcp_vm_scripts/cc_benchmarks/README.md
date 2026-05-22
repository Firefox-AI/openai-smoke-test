# CC Benchmarks Suite (H100)

This directory contains scripts to run benchmarks for Qwen (30B) and Mistral (7B) models on NVIDIA H100 hardware.

## Workflow

### 1. Preparation (from Local Machine)
Ensure all local scripts are executable:
```bash
chmod +x gcp_vm_scripts/cc_benchmarks/*.sh
```

### 2. Provision VM (from Local Machine)
Use the unified start script to create an H100 VM. By default, the script targets `a3-highgpu-1g` and automatically tries multiple zones (`us-central1-a`, `us-central1-c`, `europe-west4-b`) if the initial attempt fails.

```bash
# Standard VM (Non-confidential)
./gcp_vm_scripts/cc_benchmarks/0_start_vm_h100.sh

# VM with Confidential Compute (TDX) enabled
./gcp_vm_scripts/cc_benchmarks/0_start_vm_h100.sh --confidential

# VM with Confidential Compute (TDX) AND Secure Boot enabled
./gcp_vm_scripts/cc_benchmarks/0_start_vm_h100.sh --confidential --secure-boot

# Target a specific zone
./gcp_vm_scripts/cc_benchmarks/0_start_vm_h100.sh --zone us-central1-a
```

### 3. Upload Scripts to VM (from Local Machine)
Once the VM is running, upload the benchmark scripts to the VM's home directory.
```bash
./gcp_vm_scripts/cc_benchmarks/upload_scripts.sh h100-test-vm --zone "$gcp_zone" --project "$gcp_project_id"
```
*Replace `h100-test-vm` with your actual instance name if different.*

### 4. Connect to VM
SSH into the VM.
```bash
gcloud compute ssh --zone "$gcp_zone" "h100-test-vm" --project "$gcp_project_id"
```

### 5. Setup Environment and Permissions
You can run the setup script directly on the VM (Option A) or trigger it from your local machine (Option B).

**Option A: Run on VM**
SSH into the VM (as shown in Step 4) and run:
```bash
chmod +x *.sh
./1_setup_environment.sh
```

**Option B: Run from Local Machine**
Run the following command from your local machine to execute the script remotely. This allows you to see the output locally, ensuring you don't lose the logs when the VM reboots.

First, export your GCP configuration variables:
```bash
export gcp_zone=us-central1-a
export gcp_project_id=fx-gen-ai-sandbox
```

Then run the setup script:
```bash
gcloud compute ssh --zone "$gcp_zone" "h100-test-vm" --project "$gcp_project_id" --command "chmod +x 1_setup_environment.sh && ./1_setup_environment.sh"
```
*Note: The script reboots the VM upon completion, which will automatically close the SSH connection. This is expected.*

### 6. Start Server (on VM)
Start the TRT-LLM server with the desired model.
Supported models: `Qwen` (Qwen3-30B-A3B), `Mistral` (Mistral-7B-v0.1).
Supported hardware: `H100` (optimized).

```bash
./2_start_server.sh --hardware H100 --model Qwen
```
or
```bash
./2_start_server.sh --hardware H100 --model Mistral
```

The server will start in the background. You can check logs with:
```bash
sudo docker exec -it trtllm_server tail -f /var/log/trtllm_server.log
```

### 7. Run Benchmark (on VM)
Run the benchmark client.
```bash
./3_run_benchmark.sh --hardware H100 --model Qwen
```
Make sure to match the model you started the server with.

### 8. Package Artifacts (on VM)
After the benchmark completes, package the results on the VM.
```bash
./5_package_and_download_artifacts.sh
```
This will create a zip file in `/tmp/` and provide instructions.

### 9. Download Artifacts (from Local Machine)
On your **local machine**, download the artifacts.
```bash
./gcp_vm_scripts/cc_benchmarks/6_download_artifacts.sh h100-test-vm --zone "$gcp_zone" --project "$gcp_project_id"
```

### 10. Cleanup (on VM/Local Machine)
Stop the server on the VM.
```bash
./4_stop_server.sh
```
Don't forget to delete the VM when done to avoid costs (from local machine).
```bash
gcloud compute instances delete h100-test-vm --zone "$gcp_zone" --project "$gcp_project_id"
```
