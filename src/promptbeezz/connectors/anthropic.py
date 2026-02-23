"""Anthropic connector for testing target models."""

from anthropic import AsyncAnthropic

from promptbeezz.connectors.base import BaseConnector


class AnthropicConnector(BaseConnector):
    """Connector for testing prompts against Anthropic-compatible models."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        default_max_tokens: int = 2048,
    ):
        if base_url is None:
            self.client = AsyncAnthropic(api_key=api_key)
        else:
            self.client = AsyncAnthropic(api_key=api_key, base_url=base_url)
        self.model = model
        self.default_max_tokens = default_max_tokens

    async def test_prompt(self, system_prompt: str, message: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": message}],
            max_tokens=self.default_max_tokens,
        )
        return "".join(getattr(block, "text", "") for block in response.content)
