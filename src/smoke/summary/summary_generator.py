import time
import asyncio
import openai

class SummaryGenerator:
    def __init__(self, openai_client: openai.AsyncClient, vertex_client, summarization_config: dict):
        """
        Initializes the generator with API clients and configuration.

        :param openai_client: An initialized async OpenAI client.
        :param summarization_config: A dict with prompts and settings.
        """
        self.openai_client = openai_client
        self.vertex_client = vertex_client
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
        meta_info = {}
        
        if self.vertex_client.vertex_region and self.vertex_client.vertex_uri:
            summary, first_token_time, meta_info = await self.vertex_client.vertex_completion(
                {
                    "instances": [
                    {
                        "text": f"""<|im_start|>system
                        {system_prompt}
                        <|im_end|>
                        <|im_start|>user
                        {user_prompt}
                        <|im_end|>
                        <|im_start|>assistant
                        """
                    }
                ],
                "parameters": {
                    "sampling_params": {
                        "temperature": self.config.get("temperature", 0.1),
                        "top_p": self.config.get("top_p", 0.01),
                    }
                }}
            )
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
                    timeout=10,
                )

                summary = ""
                async for chunk in stream:
                    if stop_event.is_set():
                        return summary, first_token_time, meta_info
                    if not first_token_time:
                        first_token_time = time.time()
                    try:
                        content = chunk.choices[0].delta.content
                        summary += content if content else ""
                    except (AttributeError, KeyError, IndexError):
                        if chunk.usage:
                            meta_info = {
                                "completion_tokens": chunk.usage.completion_tokens,
                                "input_tokens": chunk.usage.prompt_tokens
                            }
                        continue
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
                    timeout=10,
                )
                summary = response.choices[0].message.content
                meta_info = {"completion_tokens": response.usage.completion_tokens, "input_tokens": response.usage.prompt_tokens}
        return summary, first_token_time, meta_info
