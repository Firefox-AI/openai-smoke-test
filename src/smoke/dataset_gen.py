import argparse
import csv
import json
import os
import random
import time
import sys
from typing import List, Dict, Any

import openai
import tenacity
from transformers import AutoTokenizer
import math
import numpy as np

# Suppress the tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Hardcoded base system prompt for the FINAL dataset
CLEAN_SYSTEM_PROMPT = "You are a personal browser assistant, designed to assist the user in navigating the web.\n\nYou can use the following tools when needed:\n- @get_page_contents(url): returns the text content of a web page given the url.\n- @search_history(search_term): returns the most relevant history items related to search term with each containing url, title, visited time and a description of the page if available.\n- @get_preferences(query=\"\"): retrieve the user's saved preferences (location, dietary, hobbies, interests, etc.) which could help in personalizing the response. If a query is provided, it will be used to filter for relevant preferences. \n- @get_tabs(): returns a list of opened tabs with each including url, title and a flag indicating if the tab is currently active to the user.\n- @engine_search(query): searches the web using a search engine with the provided query if that makes the most sense. It will direct the user to browser's search result page and end the conversation.\n\nTool calling rules:\n1. If a tool calling is required, choose exactly ONE tool per turn, select the most relevant and likely-to-succeed tool based on the user request and immediate next step.\n2. Ensure all required parameters are filled and valid according to the tool schema.\n3. Do not make up URLs in tool call arguments.\n4. If no tool calling is required, respond in natural language.\n5. Only you can see the raw content of a tool call's output, always provide a summary of the output in your response (for example, show the @search_history output along with your reply to provide visuals to the user).\n6. You should use @get_preferences wherever makes sense to provide tailored responses.\n7. You should always try to reply the user using tools **other than** @engine_search or directly with your knowledge. Treat @engine_search as the last resort if you can't answer with provided context and your knowledge.\n\nAlways follow these rules strictly.\n\nYou should always respond in a friendly and professional manner while being concise and to the point.\n\nThe user is currently on this tab:\nNone"

# System prompt for the GENERATION model
GENERATION_SYSTEM_PROMPT = """
You are an expert data generator. Your task is to create a realistic, multi-turn conversation between a "user" and an "assistant".
The conversation MUST be based *only* on the information provided in the following context document.
Do not use any external knowledge.
The conversation should be natural and coherent.
"""

def load_tokenizer(tokenizer_name: str):
    """Loads a Hugging Face tokenizer."""
    try:
        print(f"Loading tokenizer '{tokenizer_name}'...")
        return AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

def filter_long_texts(
    dataset_path: str,
    text_column: str,
    tokenizer: Any,
    min_tokens: int
) -> List[str]:
    """Loads a CSV and filters for texts longer than a minimum token count."""
    long_texts = []
    print(f"Reading dataset from {dataset_path}...")
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get(text_column)
                if not text:
                    continue
                
                tokens = tokenizer.encode(text)
                if len(tokens) >= min_tokens:
                    long_texts.append(text)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading or processing CSV file: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Found {len(long_texts)} texts with at least {min_tokens} tokens.")
    return long_texts

@tenacity.retry(
    wait=tenacity.wait_fixed(3),
    stop=tenacity.stop_after_attempt(2),
    retry=tenacity.retry_if_exception_type((openai.RateLimitError, openai.APIStatusError)),
    before_sleep=lambda retry_state: print(f"Retrying API call due to error: {retry_state.outcome.exception()} (Attempt {retry_state.attempt_number})")
)
def generate_themes_for_text(client: openai.OpenAI, model: str, num_themes: int, text: str) -> List[str]:
    """Generates a list of conversational themes relevant to a specific text."""
    print(f"Generating {num_themes} unique themes for the current text...")
    
    system_prompt = f"""
You are an expert at analyzing text and identifying key themes for discussion.
Based *only* on the context document provided, generate {num_themes} distinct themes for a conversation.
The themes should be diverse and directly related to the content of the document.
Return the themes as a single JSON object with a key "themes" which is a list of strings.
"""
    
    user_prompt = f"CONTEXT DOCUMENT:\n---\n{text}\n---"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=1.0,
    )
    
    response_content = response.choices[0].message.content
    try:
        data = extract_json_from_response(response_content)
        if "themes" in data and isinstance(data["themes"], list):
            return data["themes"]
        else:
            raise ValueError("Generated JSON is missing 'themes' key or it's not a list.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing themes JSON: {e}\nRaw response: {response_content}", file=sys.stderr)
        return []

def extract_json_from_response(response_content: str) -> Dict[str, Any]:
    """Extracts a JSON object from a model's response, cleaning up markdown fences."""
    json_start_index = response_content.find('{')
    if json_start_index == -1:
        raise ValueError("No JSON object found in the model's response.")

    json_end_index = response_content.rfind('}')
    if json_end_index == -1:
        raise ValueError("No valid JSON object end found in the model's response.")

    json_string = response_content[json_start_index : json_end_index + 1]
    
    return json.loads(json_string)

@tenacity.retry(
    wait=tenacity.wait_fixed(3),
    stop=tenacity.stop_after_attempt(2),
    retry=tenacity.retry_if_exception_type((openai.RateLimitError, openai.APIStatusError)),
    before_sleep=lambda retry_state: print(f"Retrying API call due to error: {retry_state.outcome.exception()} (Attempt {retry_state.attempt_number})")
)
def generate_conversation(
    client: openai.OpenAI,
    model: str,
    system_prompt: str,
    num_turns: int,
    theme: str
) -> List[Dict[str, str]]:
    """Calls the LLM to generate a conversation and returns it as a list of messages."""
    
    generation_prompt = f"""
Generate a conversation with exactly {num_turns} turns, focusing on the theme of: "{theme}".
A turn consists of one user message and one assistant message.
Your response MUST be a single, valid JSON object. Do not include any introductory text, explanations, or code fences.
The JSON object must have a key "conversation" which contains a list of message objects.
Each message object must have "role" and "content" keys.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": generation_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=2000,
    )
    
    response_content = response.choices[0].message.content

    try:
        data = extract_json_from_response(response_content)
        if "conversation" in data and isinstance(data["conversation"], list):
            return data["conversation"]
        else:
            raise ValueError("Generated JSON is missing the 'conversation' key or it's not a list.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing generated JSON: {e}\nRaw response: {response_content}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description="Generate a conversational dataset from long texts.")
    parser.add_argument("--golden-dataset-path", required=True, help="Path to the source CSV file.")
    parser.add_argument("--dataset-text-column", default="text", help="Column name in the CSV containing the text.")
    parser.add_argument("--output-file", required=True, help="Path to save the output .jsonl file.")
    parser.add_argument("--num-samples", type=int, default=None, help="Optional: The target number of samples for the final dataset.")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-235B-A22B-Instruct-2507", help="Hugging Face tokenizer name.")
    parser.add_argument("--min-tokens", type=int, default=3000, help="Minimum token count for a text to be used.")
    parser.add_argument("--model", required=True, help="Name of the model to use for generation (e.g., on TogetherAI).")
    parser.add_argument("--api-key", required=True, help="TogetherAI API key.")
    parser.add_argument("--api-base", default="https://api.together.xyz/v1/", help="API base URL.")
    
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    long_texts = filter_long_texts(args.golden_dataset_path, args.dataset_text_column, tokenizer, args.min_tokens)

    if not long_texts:
        print("No long texts found to process. Exiting.")
        return

    client = openai.OpenAI(api_key=args.api_key, base_url=args.api_base)

    input_token_counts = []
    all_themes_used = {}

    target_samples = args.num_samples if args.num_samples is not None else len(long_texts)
    num_long_texts = len(long_texts)
    
    generated_count = 0
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        while generated_count < target_samples:
            for i, text in enumerate(long_texts):
                if generated_count >= target_samples:
                    break

                if args.num_samples is not None:
                    samples_per_text_pass = math.ceil((target_samples - generated_count) / (num_long_texts - i))
                    
                    themes_for_this_text = generate_themes_for_text(client, args.model, samples_per_text_pass, text)
                    if not themes_for_this_text:
                        print(f"Warning: Could not generate themes for text {i+1}. Skipping.", file=sys.stderr)
                        continue
                    
                    print(f"\n--- Generated Themes for Text {i+1} ---")
                    for t in themes_for_this_text:
                        print(f"  - {t}")
                    print("-------------------------------------\n")
                    all_themes_used[f"text_{i+1}"] = themes_for_this_text
                    
                    for theme in themes_for_this_text:
                        if generated_count >= target_samples:
                            break
                        
                        print(f"\n--- Processing sample {generated_count + 1}/{target_samples} (Text {i+1}, Theme: '{theme}') ---")
                        
                        generation_system_prompt = f"{GENERATION_SYSTEM_PROMPT}\n\nCONTEXT DOCUMENT:\n---\n{text}\n---"
                        clean_system_prompt = f"{CLEAN_SYSTEM_PROMPT}\n\nCONTEXT DOCUMENT:\n---\n{text}\n---"
                        num_turns = random.randint(2, 4)
                        print(f"Generating a conversation with {num_turns} turns...")

                        try:
                            generated_conversation = generate_conversation(client, args.model, generation_system_prompt, num_turns, theme)

                            if not generated_conversation or len(generated_conversation) != num_turns * 2:
                                print(f"Skipping entry due to invalid conversation generated.", file=sys.stderr)
                                continue

                            final_payload = { "messages": [ {"role": "system", "content": clean_system_prompt} ] + generated_conversation[:-1] }
                            
                            total_tokens = sum(len(tokenizer.encode(msg["content"])) for msg in final_payload["messages"])
                            input_token_counts.append(total_tokens)

                            f_out.write(json.dumps(final_payload) + "\n")
                            print(f"Successfully generated and wrote payload for sample {generated_count + 1} ({total_tokens} tokens).")
                            generated_count += 1
                        except Exception as e:
                            print(f"Failed to process sample {generated_count + 1} after retries: {e}", file=sys.stderr)
                        
                        time.sleep(1)
                else:
                    print(f"\n--- Processing sample {generated_count + 1}/{target_samples} (Text {i+1}, Theme: 'general inquiry') ---")
                    
                    generation_system_prompt = f"{GENERATION_SYSTEM_PROMPT}\n\nCONTEXT DOCUMENT:\n---\n{text}\n---"
                    clean_system_prompt = f"{CLEAN_SYSTEM_PROMPT}\n\nCONTEXT DOCUMENT:\n---\n{text}\n---"
                    num_turns = random.randint(2, 4)
                    print(f"Generating a conversation with {num_turns} turns...")

                    try:
                        generated_conversation = generate_conversation(client, args.model, generation_system_prompt, num_turns, "general inquiry")

                        if not generated_conversation or len(generated_conversation) != num_turns * 2:
                            print(f"Skipping entry due to invalid conversation generated.", file=sys.stderr)
                            continue

                        final_payload = { "messages": [ {"role": "system", "content": clean_system_prompt} ] + generated_conversation[:-1] }
                        
                        total_tokens = sum(len(tokenizer.encode(msg["content"])) for msg in final_payload["messages"])
                        input_token_counts.append(total_tokens)

                        f_out.write(json.dumps(final_payload) + "\n")
                        print(f"Successfully generated and wrote payload for sample {generated_count + 1} ({total_tokens} tokens).")
                        generated_count += 1
                    except Exception as e:
                        print(f"Failed to process sample {generated_count + 1} after retries: {e}", file=sys.stderr)

                    time.sleep(1)

            if args.num_samples is None:
                break

    print(f"\nDataset generation complete. {generated_count} samples saved to {args.output_file}")

    if input_token_counts:
        stats = {
            "total_samples": len(input_token_counts),
            "themes_used_per_text": all_themes_used if args.num_samples is not None else "N/A (Simple Mode)",
            "token_statistics": {
                "min": int(np.min(input_token_counts)),
                "avg": int(np.mean(input_token_counts)),
                "max": int(np.max(input_token_counts)),
                "p50": int(np.percentile(input_token_counts, 50)),
                "p90": int(np.percentile(input_token_counts, 90)),
            }
        }
        
        stats_filename = os.path.splitext(args.output_file)[0] + "_data.json"
        with open(stats_filename, 'w', encoding='utf-8') as f_stats:
            json.dump(stats, f_stats, indent=4)
        print(f"Token statistics saved to {stats_filename}")


if __name__ == "__main__":
    main()
