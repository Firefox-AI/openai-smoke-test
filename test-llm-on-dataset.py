import argparse
import json
import os
import time
from openai import OpenAI

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Send requests to an OpenAI-compatible endpoint.")
    parser.add_argument("--api-key", required=True, help="OpenAI API key.")
    parser.add_argument("--base-url", required=True, help="OpenAI base URL (proxy endpoint).")
    parser.add_argument("--input-file", required=True, help="Path to the input .jsonl file.")
    parser.add_argument("--model", default="gpt-4", help="The model to use for chat completions.")
    parser.add_argument("--max-tokens", type=int, default=1000, help="The maximum number of tokens to generate.")
    parser.add_argument("--long-response", action="store_true", help="Request extremely detailed and verbose responses from the model.")
    
    args = parser.parse_args()

    # Generate the output filename
    base_name = os.path.splitext(args.input_file)[0]
    suffix = "_long_response" if args.long_response else ""
    output_file = f"{base_name}{suffix}_output_data.jsonl"

    # Initialize the OpenAI client with arguments
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    process_jsonl_file(client, args.input_file, output_file, args.model, args.max_tokens, args.long_response)

def process_jsonl_file(client, input_file, output_file, model, max_tokens, long_response):
    """
    Reads an input .jsonl file, sends each line's content to the OpenAI chat completion endpoint,
    and writes the response to an output .jsonl file.
    """
    durations = []
    request_count = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_tokens = 0
    total_tokens_list = []
    prompt_tokens_list = []
    completion_tokens_list = []
    
    try:
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in f_in:
                try:
                    # Parse the JSON object from the line
                    payload = json.loads(line.strip())
                    messages = payload.get("messages")

                    if not messages:
                        print("Skipping line: 'messages' key not found.")
                        continue

                    # If long_response flag is set, modify the last user message
                    if long_response:
                        # Find the last user message and append the verbose request
                        for i in range(len(messages) - 1, -1, -1):
                            if messages[i].get("role") == "user":
                                original_content = messages[i].get("content", "")
                                messages[i]["content"] = original_content + "\n\nPlease provide an extremely detailed, verbose, and comprehensive response. Ensure the answer is as long and thorough as possible, exploring all facets of the topic in depth."
                                break

                    # Record start time for e2e duration
                    start_time = time.time()

                    # Send the request to the chat completions endpoint
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=model,
                        max_tokens=max_tokens,
                    )

                    # Record end time and calculate duration
                    end_time = time.time()
                    duration = end_time - start_time
                    durations.append(duration)
                    request_count += 1

                    # Extract the last user message as the question
                    question = ""
                    for msg in reversed(messages):
                        if msg.get("role") == "user":
                            question = msg.get("content", "")
                            break
                    
                    # Extract the assistant's response as the answer
                    answer = chat_completion.choices[0].message.content

                    # Extract token usage information
                    usage = chat_completion.usage
                    prompt_tokens = usage.prompt_tokens if usage else 0
                    completion_tokens = usage.completion_tokens if usage else 0
                    total_tokens = usage.total_tokens if usage else 0
                    
                    # Check for cached tokens (some APIs provide this)
                    cached_tokens = getattr(usage, 'cached_tokens', 0) if usage else 0

                    # Accumulate statistics
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    total_cached_tokens += cached_tokens
                    total_tokens_list.append(total_tokens)
                    prompt_tokens_list.append(prompt_tokens)
                    completion_tokens_list.append(completion_tokens)

                    # Create simplified output format with token usage
                    output_data = {
                        "question": question,
                        "answer": answer,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "cached_tokens": cached_tokens,
                        "total_tokens": total_tokens,
                        "e2e_duration": round(duration, 2)
                    }

                    # Write the simplified response to the output file
                    f_out.write(json.dumps(output_data) + '\n')
                    print(f"Request {request_count}: E2E duration: {duration:.2f}s | Tokens - Input: {prompt_tokens}, Cached: {cached_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")

                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")
                except Exception as e:
                    print(f"An error occurred: {e}")
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    
    # Generate final report
    if durations and total_tokens_list:
        # Calculate percentiles
        def calculate_percentile(data, percentile):
            sorted_data = sorted(data)
            index = int(len(sorted_data) * percentile / 100)
            if index >= len(sorted_data):
                index = len(sorted_data) - 1
            return sorted_data[index]
        
        # Duration statistics
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        p50_duration = calculate_percentile(durations, 50)
        p90_duration = calculate_percentile(durations, 90)
        
        # Token statistics
        avg_total_tokens = sum(total_tokens_list) / len(total_tokens_list)
        p50_total_tokens = calculate_percentile(total_tokens_list, 50)
        p90_total_tokens = calculate_percentile(total_tokens_list, 90)
        
        avg_prompt_tokens = sum(prompt_tokens_list) / len(prompt_tokens_list)
        avg_completion_tokens = sum(completion_tokens_list) / len(completion_tokens_list)
        
        # Create comprehensive report
        report = {
            "summary": {
                "total_requests_processed": request_count,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_cached_tokens": total_cached_tokens,
                "total_tokens_consumed": total_prompt_tokens + total_completion_tokens
            },
            "duration_stats": {
                "avg_e2e_duration_seconds": round(avg_duration, 2),
                "min_e2e_duration_seconds": round(min_duration, 2),
                "max_e2e_duration_seconds": round(max_duration, 2),
                "p50_e2e_duration_seconds": round(p50_duration, 2),
                "p90_e2e_duration_seconds": round(p90_duration, 2)
            },
            "token_stats": {
                "avg_total_tokens_per_request": round(avg_total_tokens, 2),
                "avg_prompt_tokens_per_request": round(avg_prompt_tokens, 2),
                "avg_completion_tokens_per_request": round(avg_completion_tokens, 2),
                "p50_total_tokens_per_request": p50_total_tokens,
                "p90_total_tokens_per_request": p90_total_tokens
            },
            "model_info": {
                "model": model,
                "max_tokens_limit": max_tokens
            }
        }
        
        # Save report to JSON file
        base_name = os.path.splitext(input_file)[0]
        suffix = "_long_response" if long_response else ""
        report_file = f"{base_name}{suffix}_final_report.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary to console
        print(f"\n=== Performance Summary ===")
        print(f"Total requests processed: {request_count}")
        print(f"Average E2E duration: {avg_duration:.2f}s")
        print(f"P50 E2E duration: {p50_duration:.2f}s")
        print(f"P90 E2E duration: {p90_duration:.2f}s")
        print(f"Total tokens consumed: {total_prompt_tokens + total_completion_tokens:,}")
        print(f"Total cached tokens: {total_cached_tokens:,}")
        print(f"Average tokens per request: {avg_total_tokens:.1f}")
        print(f"P50 tokens per request: {p50_total_tokens}")
        print(f"P90 tokens per request: {p90_total_tokens}")
        print(f"\nDetailed report saved to: {report_file}")
    else:
        print("No requests were processed successfully.")

if __name__ == "__main__":
    main()
