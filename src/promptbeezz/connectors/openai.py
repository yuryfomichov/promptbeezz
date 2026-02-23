"""OpenAI connector for testing target models."""

import logging

from openai import AsyncOpenAI

from promptbeezz.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class OpenAIConnector(BaseConnector):
    """Connector for testing prompts against OpenAI models."""

    def __init__(self, *, api_key: str, model: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        logger.info("OpenAIConnector initialized with model %s", model)

    async def test_prompt(self, system_prompt: str, message: str) -> str:
        response = await self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=message,
        )
        return response.output_text or ""
