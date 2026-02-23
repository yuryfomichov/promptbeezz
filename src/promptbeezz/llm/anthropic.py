"""Anthropic implementation of the LLM client interface."""

import asyncio

from anthropic import AsyncAnthropic
from pydantic import ValidationError

from promptbeezz.llm.base import BaseLLMClient, TModel


class AnthropicLLMClient(BaseLLMClient):
    """LLM client backed by Anthropic Messages API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        default_max_tokens: int = 2048,
        structured_retries: int = 3,
        structured_retry_delay_seconds: float = 1.0,
    ):
        if base_url is None:
            self.client = AsyncAnthropic(api_key=api_key)
        else:
            self.client = AsyncAnthropic(api_key=api_key, base_url=base_url)
        self.model = model
        self.default_max_tokens = default_max_tokens
        self.structured_retries = max(0, structured_retries)
        self.structured_retry_delay_seconds = max(0.0, structured_retry_delay_seconds)

    async def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        response = await self._messages_create(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return "".join(getattr(block, "text", "") for block in response.content)

    async def generate_structured(
        self,
        prompt: str,
        *,
        schema: type[TModel],
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> TModel:
        tool_name = "return_structured_output"
        last_error: Exception | None = None
        attempts = self.structured_retries + 1

        for attempt in range(1, attempts + 1):
            try:
                response = await self._messages_create(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=[
                        {
                            "name": tool_name,
                            "description": "Return output matching schema exactly.",
                            "input_schema": schema.model_json_schema(),
                        }
                    ],
                    tool_choice={"type": "tool", "name": tool_name},
                )

                for block in response.content:
                    if (
                        getattr(block, "type", None) == "tool_use"
                        and getattr(block, "name", None) == tool_name
                    ):
                        return schema.model_validate(getattr(block, "input", {}))

                last_error = ValueError(
                    "Anthropic structured output missing required tool_use payload"
                )
            except ValidationError as exc:
                last_error = exc

            if attempt < attempts and self.structured_retry_delay_seconds > 0:
                await asyncio.sleep(self.structured_retry_delay_seconds)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Structured generation failed without an explicit error")

    async def _messages_create(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, object]] | None = None,
        tool_choice: dict[str, str] | None = None,
    ):
        messages = [{"role": "user", "content": self._user_content(prompt)}]
        temp = temperature if temperature is not None else 0.7
        max_out = max_tokens or self.default_max_tokens

        if tools is None:
            if system_prompt is None:
                return await self.client.messages.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_out,
                )
            return await self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=messages,
                temperature=temp,
                max_tokens=max_out,
            )

        choice = tool_choice or {"type": "auto"}
        if system_prompt is None:
            return await self.client.messages.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_out,
                tools=tools,
                tool_choice=choice,
            )
        return await self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=messages,
            temperature=temp,
            max_tokens=max_out,
            tools=tools,
            tool_choice=choice,
        )

    def _user_content(self, prompt: str) -> str | list[dict[str, str]]:
        return prompt
