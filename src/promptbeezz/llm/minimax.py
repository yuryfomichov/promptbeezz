"""Minimax implementation built on Anthropic-compatible client."""

from promptbeezz.llm.anthropic import AnthropicLLMClient


class MinimaxLLMClient(AnthropicLLMClient):
    """Minimax client that reuses Anthropic structured-output transport."""

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

    def _user_content(self, prompt: str) -> list[dict[str, str]]:
        return [{"type": "text", "text": prompt}]
