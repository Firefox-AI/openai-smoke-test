import random
import json
import os
from itertools import cycle
from locust import HttpUser, task, between


WAIT_TIME_MIN = 0.01
WAIT_TIME_MAX = 0.02
DEFAULT_MODEL = "openai/gpt-4o"
DEFAULT_TEMPERATURE_MIN = 0.3
DEFAULT_TEMPERATURE_MAX = 0.9
DEFAULT_MAX_TOKENS_MIN = 50
DEFAULT_MAX_TOKENS_MAX = 200

TEST_MESSAGES = [
    "Hello! Can you help me with a simple question?",
    "Tell me a short story about a robot.",
    "Explain quantum computing in simple terms",
    "Write a Python function to sort a list",
    "What are the benefits of renewable energy?",
    "Tell me about the history of the internet",
    "How does machine learning work?",
    "What is the capital of France?",
    "Explain photosynthesis in simple terms",
    "Write a haiku about programming",
]

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

    def _make_chat_request(
        self,
        message: str,
        stream: bool = False,
        temperature: float = None,
        max_tokens: int = None,
    ):
        payload = {
            "messages": [{"role": "user", "content": message}],
            "model": DEFAULT_MODEL,
            "temperature": temperature
            or random.uniform(DEFAULT_TEMPERATURE_MIN, DEFAULT_TEMPERATURE_MAX),
            "max_completion_tokens": max_tokens
            or random.randint(DEFAULT_MAX_TOKENS_MIN, DEFAULT_MAX_TOKENS_MAX),
            "stream": stream,
        }

        headers = {
            "Content-Type": "application/json",
            "x-fxa-authorization": f"Bearer {self.fxa_token}",
        }

        endpoint = "/mock/v1/chat/completions"
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
        message = random.choice(TEST_MESSAGES)
        with self._make_chat_request(message, stream=False) as response:
            self._handle_response(response, "Chat Completion")

    @task(2)
    def chat_completion_streaming(self):
        message = random.choice(TEST_MESSAGES)
        with self._make_chat_request(message, stream=True) as response:
            self._handle_response(response, "Streaming")

    @task(1)
    def health_check(self):
        with self.client.get("/health/liveness", catch_response=True) as response:
            self._handle_response(response, "Health Check")
