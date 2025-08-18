import time
import asyncio
from .mistral_client import MistralClient

class SummaryGenerator:
    def __init__(self, openai_client, summarization_config):
        """
        Initializes the generator with API clients and configuration.

        :param openai_client: An initialized async OpenAI client.
        :param summarization_config: A dict with prompts and settings.
        """
        self.openai_client = openai_client
        self.mistral_client = MistralClient(summarization_config)
        self.config = summarization_config

    async def generate(self, model_name: str, text: str, stop_event: asyncio.Event) -> tuple[str, float]:
        """
        Generates a summary using the specified model.

        :param model_name: The name of the model to use (e.g., "mistral-7b", "gpt-4").
        :param text: The text to summarize.
        :param stop_event: An initialized StopEvent object.
        :return: The generated summary as a string.
        """
        system_prompt = self.config.get("system_prompt_template")
        user_prompt = self.config.get("user_prompt_template", "{text}").replace("{text}", text)

        first_token_time = None
        if "mistral" in model_name.lower():
            try:
                summary, first_token_time = await self.mistral_client.completion(
                    model_name,
                    system_prompt,
                    user_prompt,
                    self.config.get("temperature", 0.1),
                    self.config.get("top_p", 0.01),
                    stop_event,
                )
            except Exception as e:
                print(f"mistral_api error: {e}")
                summary = ""
        else:
            if self.config.get("stream"):
                stream = await self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": self.config.get("system_prompt_template")},
                        {"role": "user",
                         "content": self.config.get("user_prompt_template", "{text}").replace("{text}", text)},
                    ],
                    temperature=self.config.get("temperature", 0.1),
                    top_p=self.config.get("top_p", 0.01),
                    stream=True,
                    max_completion_tokens=self.config.get("max_completion_tokens"),
                )

                summary = ""
                async for chunk in stream:
                    if stop_event.is_set():
                        return summary, first_token_time
                    if not first_token_time:
                        first_token_time = time.time()
                    content = chunk.choices[0].delta.content or ""
                    if len(content) > 0:
                        summary += content
            else:
                # --- Call OpenAI-compatible API ---
                response = await self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    stream=False,
                    max_completion_tokens=self.config.get("max_completion_tokens"),
                )
                summary = response.choices[0].message.content
        return summary, first_token_time
