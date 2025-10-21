"""
Vertex AI Deployment Information and Cost Estimator

This script queries Google Cloud Logging to provide detailed reports on Vertex AI model deployments.
It tracks the entire lifecycle of endpoints—from deployment to undeployment—and provides insights into
uptime, resource configuration, and estimated costs.

Features:
- **Comprehensive Event Tracking**: Monitors 'Deploy Model', 'Download model' (replica creation),
  and 'Undeploy Model' events.
- **Detailed Timeline Report**: Displays a chronological log of all deployment-related activities,
  including machine specs, accelerator details, and replica counts.
- **Uptime and Cost Analysis**: Calculates the total operational hours for each endpoint and provides
  an estimated cost based on a configurable pricing table.
- **Flexible Reporting**: Command-line flags allow you to customize the output, from a high-level
  summary to a granular, replica-level view.

Usage Examples:

1.  **Detailed Report for a Specific Model (Default Behavior)**:
    ```bash
    python3 audit_vertexai_deployment_logs.py --search-term "the-model-name"
    ```

2.  **Full Granular Report with Replica Messages**:
    ```bash
    python3 audit_vertexai_deployment_logs.py --search-term "the-model-name" --include-message
    ```

3.  **High-Level Summary of All Deployments (excluding replica events)**:
    (Shows only deploy/undeploy events and the final uptime/cost report)
    ```bash
    python3 audit_vertexai_deployment_logs.py --no-replicas
    ```

Note: The pricing data in this script is for demonstration purposes only. For accurate cost
      calculations, please update the `PRICING_DATA` dictionary with the latest official rates
      from the Google Cloud pricing pages.
"""

import datetime
import argparse
import sys
from google.cloud import logging

# NOTE: The pricing data below is for demonstration purposes only and may not be accurate.
# Please refer to the official Google Cloud pricing pages for up-to-date information.
PRICING_DATA = {
    "us-west1": {
        "machines": {
            "n1-standard-8": 0.379,
            "a2-highgpu-1g": 2.045,
            "g2-standard-8": 0.56,
            "a3-highgpu-8g": 88.0,
        },
        "accelerators": {
            "NVIDIA_TESLA_T4": 0.35,
            "NVIDIA_A100_40GB": 3.22,
            "NVIDIA_L4": 0.67,
            "NVIDIA_NVIDIA_H100_80GB": 12,
        },
    }
    # Other regions can be added here
}


def get_creation_events(client, project_id, search_term, location, start_time):
    """Fetches replica creation events."""
    print("--- Searching for Replica Creation Events ---")
    timestamp_filter = f'timestamp >= "{start_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")}"'
    log_filter = (
        f'logName="projects/{project_id}/logs/aiplatform.googleapis.com%2Fprediction_container" '
        f'"{search_term}" AND "Downloading model" '
        f'resource.labels.location="{location}" AND {timestamp_filter}'
    )
    print(f"Using filter: {log_filter}")

    events = []
    seen_replicas = set()
    try:
        entries = client.list_entries(filter_=log_filter, order_by=logging.ASCENDING)
        for entry in entries:
            replica_id = entry.labels.get("replica_id")
            if replica_id and replica_id not in seen_replicas:
                seen_replicas.add(replica_id)
                message = entry.payload.get("message", "N/A")
                events.append(
                    {
                        "timestamp": entry.timestamp,
                        "event_type": "Download model",
                        "replica_id": replica_id,
                        "endpoint_id": entry.resource.labels.get("endpoint_id", "N/A"),
                        "message": message,
                    }
                )
    except Exception as e:
        print(f"An error occurred during creation event search: {e}")

    print(f">>> Found {len(events)} unique creation event(s).")
    return events


def get_unload_events(client, project_id, start_time):
    """Fetches model unload events."""
    print("\n--- Searching for Replica Unload Events ---")
    timestamp_filter = f'timestamp >= "{start_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")}"'
    log_filter = f'"EndpointService.UndeployModel" AND {timestamp_filter}'
    print(f"Using filter: {log_filter}")

    events = []
    try:
        entries = client.list_entries(filter_=log_filter, order_by=logging.ASCENDING)
        for entry in entries:
            resource_name = entry.payload.get("resourceName", "N/A")
            endpoint_id = (
                resource_name.split("/")[-1] if resource_name != "N/A" else "N/A"
            )
            events.append(
                {
                    "timestamp": entry.timestamp,
                    "event_type": "Undeploy model",
                    "replica_id": "N/A",
                    "endpoint_id": endpoint_id,
                }
            )
    except Exception as e:
        print(f"An error occurred during unload event search: {e}")

    print(f">>> Found {len(events)} unload event(s).")
    return events


def get_deploy_events(client, project_id, start_time):
    """Fetches model deploy events."""
    print("\n--- Searching for Replica Deploy Events ---")
    timestamp_filter = f'timestamp >= "{start_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")}"'
    log_filter = f'"EndpointService.DeployModel" AND {timestamp_filter}'
    print(f"Using filter: {log_filter}")

    events = []
    try:
        entries = client.list_entries(filter_=log_filter, order_by=logging.ASCENDING)
        for entry in entries:
            request_payload = entry.payload.get("request", {})
            endpoint_path = request_payload.get("endpoint", "N/A")
            endpoint_id = (
                endpoint_path.split("/")[-1] if endpoint_path != "N/A" else "N/A"
            )

            deployed_model = request_payload.get("deployedModel", {})
            dedicated_resources = deployed_model.get("dedicatedResources", {})
            machine_spec = dedicated_resources.get("machineSpec", {})

            min_rep = dedicated_resources.get("minReplicaCount", 0)
            max_rep = dedicated_resources.get("maxReplicaCount", 0)
            acc_count = machine_spec.get("acceleratorCount", 0)

            event_details = {
                "timestamp": entry.timestamp,
                "event_type": "Deploy model",
                "replica_id": "N/A",
                "endpoint_id": endpoint_id,
                "machineType": machine_spec.get("machineType", ""),
                "acceleratorType": machine_spec.get("acceleratorType", ""),
                "acceleratorCount": int(acc_count) if acc_count else 0,
                "minReplicaCount": int(min_rep) if min_rep else 0,
                "maxReplicaCount": int(max_rep) if max_rep else 0,
            }
            events.append(event_details)
    except Exception as e:
        print(f"An error occurred during deploy event search: {e}")

    print(f">>> Found {len(events)} deploy event(s).")
    return events


def report_replica_events(
    project_id,
    search_term,
    location,
    days_ago=7,
    include_replicas=False,
    include_message=False,
):
    """Reports replica creation and unload events in a unified timeline."""
    client = logging.Client(project=project_id)
    start_time = datetime.datetime.utcnow() - datetime.timedelta(days=days_ago)

    creation_events = []
    if include_replicas:
        creation_events = get_creation_events(
            client, project_id, search_term, location, start_time
        )

    unload_events = get_unload_events(client, project_id, start_time)
    deploy_events = get_deploy_events(client, project_id, start_time)

    all_events = creation_events + unload_events + deploy_events

    if not all_events:
        print("\nNo replica events found for the specified criteria.")
        return

    all_events.sort(key=lambda x: x["timestamp"])

    # Dynamically determine column widths
    max_replica_id_len = len("Replica ID")
    max_endpoint_id_len = len("Endpoint ID")

    if all_events:
        if include_replicas:
            replica_ids = [str(e.get("replica_id", "")) for e in all_events]
            if replica_ids:
                max_replica_id_len = max(len(rid) for rid in replica_ids)
                max_replica_id_len = max(max_replica_id_len, len("Replica ID"))

        endpoint_ids = [str(e.get("endpoint_id", "")) for e in all_events]
        if endpoint_ids:
            max_endpoint_id_len = max(len(eid) for eid in endpoint_ids)
            max_endpoint_id_len = max(max_endpoint_id_len, len("Endpoint ID"))

    print("\n--- Combined Replica Event Report (UTC) ---")
    header_parts = [f"{'Timestamp':<20}", f"{'Event':<18}"]
    if include_replicas:
        header_parts.append(f"{'Replica ID':<{max_replica_id_len}}")

    header_parts.extend(
        [
            f"{'Endpoint ID':<{max_endpoint_id_len}}",
            f"{'Machine':<15}",
            f"{'Accelerator':<25}",
            f"{'Acc. Count':<10}",
            f"{'Min Rep':<8}",
            f"{'Max Rep':<8}",
        ]
    )

    if include_message:
        header_parts.append("Message")

    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))

    for event in all_events:
        timestamp_str = event["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

        row_data = {
            "timestamp": timestamp_str,
            "event_type": event.get("event_type", ""),
            "replica_id": event.get("replica_id", ""),
            "endpoint_id": event.get("endpoint_id", ""),
            "machineType": event.get("machineType", ""),
            "acceleratorType": event.get("acceleratorType", ""),
            "acceleratorCount": event.get("acceleratorCount", ""),
            "minReplicaCount": event.get("minReplicaCount", ""),
            "maxReplicaCount": event.get("maxReplicaCount", ""),
            "message": event.get("message", ""),
        }

        row_parts = [f"{row_data['timestamp']:<20}", f"{row_data['event_type']:<18}"]
        if include_replicas:
            row_parts.append(f"{str(row_data['replica_id']):<{max_replica_id_len}}")

        row_parts.extend(
            [
                f"{str(row_data['endpoint_id']):<{max_endpoint_id_len}}",
                f"{str(row_data['machineType']):<15}",
                f"{str(row_data['acceleratorType']):<25}",
                f"{str(row_data['acceleratorCount']):<10}",
                f"{str(row_data['minReplicaCount']):<8}",
                f"{str(row_data['maxReplicaCount']):<8}",
            ]
        )

        if include_message:
            row_parts.append(row_data["message"])

        print(" | ".join(row_parts))

    calculate_and_report_cost(all_events, location)


def calculate_and_report_cost(all_events, location):
    """Calculates and reports the uptime and estimated cost for each endpoint by pairing events."""

    endpoint_events = {}
    for event in all_events:
        endpoint_id = event.get("endpoint_id")
        if not endpoint_id or endpoint_id == "N/A":
            continue
        if endpoint_id not in endpoint_events:
            endpoint_events[endpoint_id] = []
        endpoint_events[endpoint_id].append(event)

    # Determine the column width for Endpoint ID dynamically
    max_endpoint_len = len("Endpoint ID")
    if endpoint_events:
        max_key_len = max(len(k) for k in endpoint_events.keys())
        max_endpoint_len = max(max_key_len, max_endpoint_len)

    print("\n--- Endpoint Uptime and Cost Report ---")
    header = f"{'Endpoint ID':<{max_endpoint_len}} | {'Uptime (hours)':<15} | {'Est. Cost ($)':<15} | {'Machine':<15} | {'Min Rep':<8} | {'Max Rep':<8}"
    print(header)
    print("-" * len(header))

    total_uptime_hours = 0
    total_estimated_cost = 0

    regional_pricing = PRICING_DATA.get(location, {})
    machine_prices = regional_pricing.get("machines", {})

    for endpoint_id, events in sorted(endpoint_events.items()):
        endpoint_total_duration_hours = 0
        start_event = None

        machine_info = {"machineType": "", "minReplicaCount": 0, "maxReplicaCount": 0}
        for event in events:
            if event.get("event_type") == "Deploy model":
                machine_info.update(
                    {
                        "machineType": event.get("machineType", ""),
                        "minReplicaCount": event.get("minReplicaCount", 0),
                        "maxReplicaCount": event.get("maxReplicaCount", 0),
                    }
                )

        for event in events:
            event_type = event.get("event_type")

            if event_type == "Deploy model":
                start_event = event
            elif event_type == "Download model":
                if start_event is None:
                    start_event = event
            elif event_type == "Undeploy model":
                if start_event is not None:
                    duration = event["timestamp"] - start_event["timestamp"]
                    endpoint_total_duration_hours += duration.total_seconds() / 3600
                    start_event = None

        is_running = start_event is not None

        if endpoint_total_duration_hours > 0:
            total_uptime_hours += endpoint_total_duration_hours

            machine_cost_per_hour = machine_prices.get(machine_info["machineType"], 0)
            num_replicas = (
                machine_info["maxReplicaCount"] or machine_info["minReplicaCount"] or 1
            )

            estimated_cost = (
                endpoint_total_duration_hours * machine_cost_per_hour * num_replicas
            )
            total_estimated_cost += estimated_cost

            print(
                f"{endpoint_id:<{max_endpoint_len}} | {endpoint_total_duration_hours:<15.2f} | {estimated_cost:<15.2f} | {machine_info['machineType']:<15} | {machine_info['minReplicaCount']:<8} | {machine_info['maxReplicaCount']:<8}"
            )

        elif is_running:
            print(
                f"{endpoint_id:<{max_endpoint_len}} | {'Still running':<15} | {'N/A':<15} | {machine_info['machineType']:<15} | {machine_info['minReplicaCount']:<8} | {machine_info['maxReplicaCount']:<8}"
            )

    print("-" * len(header))
    print(
        f"{'Totals':<{max_endpoint_len}} | {total_uptime_hours:<15.2f} | {total_estimated_cost:<15.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Report on replica creation and unload events."
    )
    parser.add_argument(
        "--search-term",
        type=str,
        help="The search term to filter replica creation events. Required by default.",
    )
    parser.add_argument(
        "--no-replicas",
        action="store_true",
        help="Exclude replica-level download events from the report.",
    )
    parser.add_argument(
        "--include-message",
        action="store_true",
        help="Include the message column for replica events. Has no effect if --no-replicas is used.",
    )
    args = parser.parse_args()

    include_replicas = not args.no_replicas

    if include_replicas and not args.search_term:
        print(
            "Error: --search-term is required unless --no-replicas is specified.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.include_message and not include_replicas:
        print(
            "Warning: --include-message has no effect when --no-replicas is specified.",
            file=sys.stderr,
        )

    PROJECT_ID = "fx-gen-ai-sandbox"
    LOCATION = "us-west1"
    DAYS_TO_SEARCH = 7

    report_replica_events(
        PROJECT_ID,
        args.search_term,
        LOCATION,
        DAYS_TO_SEARCH,
        include_replicas=include_replicas,
        include_message=args.include_message,
    )
