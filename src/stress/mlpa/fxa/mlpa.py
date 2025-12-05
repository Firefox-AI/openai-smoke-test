from pathlib import Path
import random
import json
import os
from itertools import cycle
from locust import HttpUser, task, between
from datasets import load_dataset


WAIT_TIME_MIN = 0.01
WAIT_TIME_MAX = 0.02
DEFAULT_MODEL = "qwen3-235b-a22b-instruct-2507-maas"

dataset = load_dataset("Mozilla/chat-eval", split="train")
TEST_CONVERSATIONS = []
for conv in dataset["conversation"]:
    if isinstance(conv, list) and len(conv) > 1:
        combined = {}
        for msg in conv:
            if msg.get("content") is not None and msg.get("role") is not None:
                role = msg["role"]
                content = msg["content"]
                if role not in combined:
                    combined[role] = content
                else:
                    combined[role] += f"\n\n{content}"
        TEST_CONVERSATIONS.append(
            [{"role": role, "content": content} for role, content in combined.items()]
        )
USERS_FILE = Path(__file__).parent.resolve() / "users.json"

if not os.path.exists(USERS_FILE):
    raise FileNotFoundError(f"User file not found: {USERS_FILE}")

with open(USERS_FILE) as f:
    USERS = json.load(f)

if not isinstance(USERS, list) or not all("token" in u for u in USERS):
    raise ValueError("users.json must be a list of objects containing a 'token' field")

USER_CYCLE = cycle(USERS)


class MLPAUser(HttpUser):
    wait_time = between(WAIT_TIME_MIN, WAIT_TIME_MAX)

    def on_start(self):
        user_data = next(USER_CYCLE)
        self.fxa_token = user_data.get("token")

    def _make_chat_request(
        self, messages, stream: bool = False, mock_response: bool = False
    ):
        payload = {
            "messages": messages,
            "model": DEFAULT_MODEL,
            "stream": stream,
        }
        if mock_response:
            payload["mock_response"] = str(messages)

        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.fxa_token}",
            "service-type": "ai",
        }

        endpoint = "/v1/chat/completions" if mock_response else "/mock/chat/completions"
        return self.client.post(
            endpoint, json=payload, headers=headers, catch_response=True
        )

    @task(4)
    def chat_completion(self):
        messages = random.choice(TEST_CONVERSATIONS)
        with self._make_chat_request(messages, stream=False) as response:
            self._handle_response(response, "Chat Completion")

    @task(3)
    def chat_completion_streaming(self):
        messages = random.choice(TEST_CONVERSATIONS)
        with self._make_chat_request(messages, stream=True) as response:
            self._handle_response(response, "Streaming")

    @task(2)
    def health_check(self):
        with self.client.get("/health/liveness", catch_response=True) as response:
            self._handle_response(response, "Health Check")

    @task(1)
    def health_check(self):
        with self.client.get("/health/readiness", catch_response=True) as response:
            self._handle_response(response, "Health Check")
