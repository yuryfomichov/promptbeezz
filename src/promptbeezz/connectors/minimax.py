"""Minimax connector for testing target models."""

from promptbeezz.connectors.anthropic import AnthropicConnector


class MinimaxConnector(AnthropicConnector):
    """Connector for testing prompts against Minimax models."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.minimax.io/anthropic",
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            default_max_tokens=8192,
        )

    async def test_prompt(self, system_prompt: str, message: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": [{"type": "text", "text": message}]}],
            max_tokens=self.default_max_tokens,
        )
        return "".join(getattr(block, "text", "") for block in response.content)
