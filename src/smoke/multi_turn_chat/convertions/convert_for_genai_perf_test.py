import json
import random
import uuid

"""
Convert `messages: [{"role": "user"/"assistant"/"tool"/"system", "content": "..."}]`
into `input: "..."`, `delay: int` format for GenAI performance test.
Skips null content messages.

Open question: Do we include the system prompt in a separate input element? Or prefix it to the first user message?
"""


def convert_json_format(data, output_path):
    converted_data = []
    for conversation in data:
        id = uuid.uuid4().hex
        first_prompt = conversation["messages"][0]
        if first_prompt["role"] == "system" and first_prompt["content"] is not None:
            new_item = {
                "session_id": id,
                "text": first_prompt["content"],
            }
            converted_data.append(new_item)
        for i in range(1, len(conversation["messages"])):
            message = conversation["messages"][i]
            if message["role"] != "user" or message["content"] is None:
                continue  # Skip null content messages
            is_last_message_in_conversation = i == len(conversation["messages"]) - 1
            new_item = {
                "session_id": id,
                # delay: simulate time between user sending messages (ms)
                "delay": random.randint(2000, 20000),  # 2 to 20 seconds
                "text": message["content"],
            }
            if is_last_message_in_conversation:
                del new_item["delay"]
            converted_data.append(new_item)

    with open(output_path, "w") as outfile:
        for item in converted_data:
            json_line = json.dumps(item)
            outfile.write(json_line + "\n")


# Example usage
if __name__ == "__main__":
    datasets = [
        "../multi_turn_chat/data/generated_goldenfox_dataset_500_min_2000tokens.jsonl_truncated_at_11000.jsonl",
        "../multi_turn_chat/data/generated_goldenfox_dataset_500_min_2000tokens_long_response.jsonl_truncated_at_11000.jsonl",
        "../multi_turn_chat/data/generated_goldenfox_dataset_500_min_5000tokens.jsonl_truncated_at_11000.jsonl",
        "../multi_turn_chat/data/generated_goldenfox_dataset_500_min_5000tokens_long_response.jsonl_truncated_at_11000.jsonl",
    ]
    for dataset_path in datasets:
        with open(dataset_path, "r") as f:
            dataset = [json.loads(line) for line in f.readlines()]
            convert_json_format(
                dataset,
                dataset_path.replace(".jsonl", "") + "_genai_perf_converted.jsonl",
            )
