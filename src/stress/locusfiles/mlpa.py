import random
import json
import os
from itertools import cycle
from locust import HttpUser, task, between
from datasets import load_dataset


WAIT_TIME_MIN = 0.01
WAIT_TIME_MAX = 0.02
DEFAULT_MODEL = "openai/gpt-4o"

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
USER_FILE = os.getenv("USER_FILE", "users.json")

if not os.path.exists(USER_FILE):
    raise FileNotFoundError(f"User file not found: {USER_FILE}")

with open(USER_FILE) as f:
    USERS = json.load(f)

if not isinstance(USERS, list) or not all("token" in u for u in USERS):
    raise ValueError("users.json must be a list of objects containing a 'token' field")

USER_CYCLE = cycle(USERS)


class MLPAUser(HttpUser):
    wait_time = between(WAIT_TIME_MIN, WAIT_TIME_MAX)

    def on_start(self):
        user_data = next(USER_CYCLE)
        self.fxa_token = user_data.get("token")

    def _make_chat_request(self, messages, stream: bool = False):
        payload = {
            "messages": messages,
            "model": DEFAULT_MODEL,
            "stream": stream,
        }

        headers = {
            "Content-Type": "application/json",
            "x-fxa-authorization": f"Bearer {self.fxa_token}",
        }

        endpoint = "/mock/chat/completions"
        return self.client.post(
            endpoint, json=payload, headers=headers, catch_response=True
        )

    def _handle_response(self, response, request_type: str):
        if response.status_code == 200:
            response.success()
        elif response.status_code == 401:
            response.failure("Authentication failed - check FxA token")
        elif response.status_code == 403:
            response.failure("User blocked")
        elif response.status_code == 400:
            response.failure("Bad request")
        else:
            response.failure(
                f"{request_type} failed with status: {response.status_code}"
            )

    @task(3)
    def chat_completion(self):
        messages = random.choice(TEST_CONVERSATIONS)
        with self._make_chat_request(messages, stream=False) as response:
            self._handle_response(response, "Chat Completion")

    @task(2)
    def chat_completion_streaming(self):
        messages = random.choice(TEST_CONVERSATIONS)
        with self._make_chat_request(messages, stream=True) as response:
            self._handle_response(response, "Streaming")

    @task(1)
    def health_check(self):
        with self.client.get("/health/liveness", catch_response=True) as response:
            self._handle_response(response, "Health Check")
