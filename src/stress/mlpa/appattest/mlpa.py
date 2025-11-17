import base64
from pathlib import Path
import random
import json
import os
from itertools import cycle
import sys
from locust import HttpUser, task, between
from datasets import load_dataset


try:
    from .utils import register_device, request_completion
except ImportError:
    # Support running the script directly (python generate_test_appattest_users.py)
    sys.path.append(str(Path(__file__).resolve().parent))
    from utils import register_device, request_completion

WAIT_TIME_MIN = 0.01
WAIT_TIME_MAX = 0.02
DEFAULT_MODEL = "mistral-small-2503"


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

if not isinstance(USERS, list) or not all("key_id_b64" in u for u in USERS):
    raise ValueError(
        "users.json must be a list of objects containing a 'key_id_b64' field"
    )

USER_CYCLE = cycle(USERS)


class MLPAUser(HttpUser):
    wait_time = between(WAIT_TIME_MIN, WAIT_TIME_MAX)

    def on_start(self):
        user_data = next(USER_CYCLE)
        self.data = {
            "key_id_bytes": base64.urlsafe_b64decode(user_data["key_id_b64"]),
            **user_data,
        }
        self.counter = 0

    def _register_device(self):
        return register_device(self.client, self.data)

    def _make_chat_request(self, messages, stream: bool = False):
        self.counter += 1
        return request_completion(
            self.client, self.data, messages, stream, self.counter
        )

    def _handle_response(self, response, request_type: str):
        if response.status_code == 200 or response.status_code == 201:
            response.success()
        elif response.status_code == 401:
            response.failure(
                f"Authentication failed - attest/assert verification failed {response.json()}"
            )
        elif response.status_code == 403:
            response.failure("User blocked")
        elif response.status_code == 400:
            response.failure("Bad request")
        else:
            response.failure(
                f"{request_type} failed with status: {response.status_code}"
            )

    # @task(4)
    # def chat_completion_streaming(self):
    #     messages = random.choice(TEST_CONVERSATIONS)
    #     with self._make_chat_request(messages, stream=True) as response:
    #         self._handle_response(response, "Streaming")

    @task(3)
    def chat_completion(self):
        messages = random.choice(TEST_CONVERSATIONS)
        with self._make_chat_request(messages, stream=False) as response:
            self._handle_response(response, "Chat Completion")

    @task(2)
    def register_device(self):
        with register_device(self.client, self.data) as response:
            self._handle_response(response, "Device Registration")

    # @task(1)
    # def health_check(self):
    #     with self.client.get("/health/liveness", catch_response=True) as response:
    #         self._handle_response(response, "Health Check")
