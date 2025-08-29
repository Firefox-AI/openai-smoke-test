import time
import google.auth
from google.oauth2 import service_account
import google.auth.transport.requests
import asyncio
import aiohttp
import json
import certifi
import ssl
from typing import List

REQUIRED_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

class MistralClient:
    def __init__(self, summarization_config):
        self.config = summarization_config
        credentials_path = self.config.get("service_account_file", "creds.json")
        self.creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=REQUIRED_SCOPES)

        self._mistral_token = None
        self._mistral_token_refresh_time = 0
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    @property
    def mistral_token(self):
        """A property that returns a valid access token, refreshing it if necessary."""
        if not self.creds.valid:
            print("Refreshing Mistral access token from service account...")
            self._refresh_mistral_token()

        return self._mistral_token

    def _refresh_mistral_token(self):
        """Uses the loaded service account credentials to get a new token."""
        try:
            # The credentials object handles its own refreshing
            auth_req = google.auth.transport.requests.Request()
            self.creds.refresh(auth_req)

            self._mistral_token = self.creds.token
            self._mistral_token_refresh_time = time.time()
            print("Token refreshed successfully.")

        except Exception as e:
            print(f"Error refreshing token: {e}")
            raise

    async def completion(self, model_name: str, messages: List[dict], temperature: float, top_p: float, stop_event: asyncio.Event):
        REGION = "us-central1"
        PROJECT_ID = "fx-gen-ai-sandbox"

        first_token_time = None
        stream = self.config.get("stream")
        base_url = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/publishers/mistralai/models/{model_name}"
        url = f"{base_url}:streamRawPredict" if stream else f"{base_url}:rawPredict"
        payload = {
            "model": model_name,
            "top_p": top_p,
            "temperature": temperature,
            "messages": messages
        }

        headers = {
            "Authorization": f"Bearer {self.mistral_token}",
            "Accept": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url=url, json=payload, headers=headers, ssl=self.ssl_context) as response:
                if response.status != 200:
                    print(f"Request failed with status code: {response.status}")
                    return "", first_token_time

                if stream:
                    summary = ""
                    async for line in response.content:
                        if stop_event.is_set():
                            return summary, first_token_time
                        if not first_token_time:
                            first_token_time = time.time()
                        try:
                            line_str = line.decode('utf-8')

                            # Streaming APIs often send Server-Sent Events (SSE) that start with "data: "
                            # We need to remove this prefix before parsing JSON.
                            if line_str.startswith('data: '):
                                line_str = line_str[6:]

                            chunk = json.loads(line_str)
                            content = chunk["choices"][0]["delta"]["content"]
                            summary += content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                    return summary, first_token_time
                else:
                    try:
                        response_dict = await response.json()
                        return response_dict["choices"][0]["message"]["content"], first_token_time
                    except aiohttp.ContentTypeError as e:
                        print(f"Error decoding JSON: {e}")
                        print(f"Raw response: {await response.text()}")
                        return "", first_token_time
