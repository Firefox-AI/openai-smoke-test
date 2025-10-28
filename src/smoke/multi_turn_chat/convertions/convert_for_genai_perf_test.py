import json
import random
import uuid
import os
import tiktoken

"""
Convert `messages: [{"role": "user"/"assistant"/"tool"/"system", "content": "..."}]`
into `input: "..."`, `delay: int` format for GenAI performance test.
Skips null content messages.

Open questions:
1. Do we include the system prompt in a separate input element? Or prefix it to the first user message?
2. Do we want to include delay?
"""

# add delay
ADD_DELAY = False
ADD_TEXT = True


def get_next_assistant_message(conversation, start_index):
    for message in conversation["messages"][start_index:]:
        if message["role"] == "assistant" and message["content"] is not None:
            return message["content"]
    return None


def convert_for_genai_perf_test_with_text(data, output_path):
    encoding = tiktoken.get_encoding("cl100k_base")
    converted_data = []
    for conversation in data:
        session_id = uuid.uuid4().hex
        first_prompt = conversation["messages"][0]
        if first_prompt["role"] == "system" and first_prompt["content"] is not None:
            new_item = {
                "session_id": session_id,
                "text": first_prompt["content"],
            }
            if not ADD_TEXT:
                del new_item["text"]
                new_item["input_length"] = len(encoding.encode(first_prompt["content"]))
                new_item["output_length"] = random.randint(50, 200)
            converted_data.append(new_item)

        for i in range(1, len(conversation["messages"])):
            message = conversation["messages"][i]
            next_assistant_message = get_next_assistant_message(conversation, i)
            if message["role"] != "user" or message["content"] is None:
                continue  # Skip null content messages
            is_last_message_in_conversation = i == len(conversation["messages"]) - 1
            new_item = {
                "session_id": session_id,
                # delay: simulate time between user sending messages (ms)
                "delay": random.randint(2000, 20000),  # 2 to 20 seconds
                "text": message["content"],
            }
            if not ADD_TEXT:
                del new_item["text"]
                new_item["input_length"] = len(encoding.encode(message["content"]))
                new_item["output_length"] = (
                    len(encoding.encode(next_assistant_message))
                    if next_assistant_message
                    else random.randint(50, 200)
                )

            if not ADD_DELAY or is_last_message_in_conversation:
                del new_item["delay"]
            converted_data.append(new_item)

    with open(output_path, "w") as outfile:
        for item in converted_data:
            json_line = json.dumps(item)
            outfile.write(json_line + "\n")


if __name__ == "__main__":
    data_dir = "../../multi_turn_chat/data"
    datasets = [
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if fname.endswith(".jsonl") or fname.endswith(".jsonl")
    ]
    print(datasets)
    for dataset_path in datasets:
        with open(dataset_path, "r") as f:
            dataset = [json.loads(line) for line in f.readlines()]
            if ADD_TEXT:
                output_dir = "./genai-perf-dataset-with-text"
            else:
                output_dir = "./genai-perf-dataset-with-lengths"
            os.makedirs(output_dir, exist_ok=True)
            original_file_name = os.path.basename(dataset_path)
            output_path = os.path.join(
                output_dir, original_file_name + "_genai_perf_converted.jsonl"
            )
            convert_for_genai_perf_test_with_text(
                dataset,
                output_path,
            )
