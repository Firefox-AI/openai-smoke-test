import time
import asyncio
from typing import List
from ..mistral_client import MistralClient
import openai


class MultiTurnChatClient:
    def __init__(
        self, openai_client: openai.AsyncClient, initial_context: List[str] = []
    ):
        """
        Initializes the generator with API clients and configuration.
        This client is made for multi-turn chat interactions.

        :param openai_client: An initialized async OpenAI client.
        """
        self.openai_client = openai_client
        self.mistral_client = MistralClient(
            {"stream": True, "service_account_file": "creds.json"}
        )
        self.context = initial_context

    def add_exchange_to_context(self, query: str, response: str):
        self.context.append({"role": "user", "content": query})
        if response and len(response.strip()) > 0:
            self.context.append({"role": "assistant", "content": response})

    async def generate_next(
        self, model_name: str, query: str, stop_event: asyncio.Event
    ) -> tuple[str, float]:
        """
        :param model_name: The name of the model to use (e.g., "mistral-7b", "gpt-4").
        :param stop_event: An initialized StopEvent object.
        :return: The LLM response.
        """
        first_token_time = None
        if "mistral" in model_name.lower():
            try:
                response, first_token_time = await self.mistral_client.completion(
                    model_name,
                    [*self.context, {"role": "user", "content": query}],
                    0.1,  # temperature
                    0.01,  # top_p
                    stop_event,
                )
            except Exception as e:
                print(f"mistral_api error: {e}")
                response = ""
        else:
            stream = await self.openai_client.chat.completions.create(
                model=model_name,
                messages=[*self.context, {"role": "user", "content": query}],
                temperature=0.1,
                top_p=0.01,
                stream=True,
                max_completion_tokens=None,
            )

            response = ""
            async for chunk in stream:
                if stop_event.is_set():
                    self.add_exchange_to_context(query, response)
                    return response, first_token_time
                if not first_token_time:
                    first_token_time = time.time()
                try:
                    content = chunk.choices[0].delta.content
                    if content:
                        response += content
                except (AttributeError, KeyError, IndexError):
                    continue
        self.add_exchange_to_context(query, response)
        return response, first_token_time
