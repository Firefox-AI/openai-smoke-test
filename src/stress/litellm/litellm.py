import random
import os
from locust import HttpUser, task, between
from datasets import load_dataset
import dotenv

dotenv.load_dotenv()

N_USERS = 10_000_000
WAIT_TIME_MIN = 0.01
WAIT_TIME_MAX = 0.02
DEFAULT_MODEL = "mistral-small-2503"
LITELLM_V_KEY = os.getenv("LITELLM_VIRTUAL_KEY")

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

USER_CYCLE = iter(range(1, N_USERS + 1))


class LiteLLMUser(HttpUser):
    wait_time = between(WAIT_TIME_MIN, WAIT_TIME_MAX)

    def on_start(self):
        self.end_user_id = f"stress_test_user_{random.randint(1, 1_000_000)}"

    def _make_chat_request(self, messages, stream: bool = False):
        payload = {
            "messages": messages,
            "model": DEFAULT_MODEL,
            "stream": stream,
            "mock_response": "Ok sure",
            "user": self.end_user_id,
        }

        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {LITELLM_V_KEY}",
        }

        endpoint = "/v1/chat/completions"
        return self.client.post(
            endpoint,
            json=payload,
            headers=headers,
            catch_response=True,
            name="/v1/chat/completions",
        )

    def _handle_response(self, response, request_type: str):
        if response.status_code == 200:
            response.success()
        elif response.status_code == 401:
            response.failure("Authentication failed")
        elif response.status_code == 403:
            response.failure("User blocked")
        elif response.status_code == 400:
            response.failure("Bad request")
        else:
            response.failure(
                f"{request_type} failed with status: {response.status_code}"
            )

    @task(1)
    def chat_completion(self):
        messages = random.choice(TEST_CONVERSATIONS)
        with self._make_chat_request(messages, stream=False) as response:
            self._handle_response(response, "Chat Completion")
