"""Dummy connector for testing - provides fast, deterministic responses."""

import hashlib

from promptbeezz.connectors.base import BaseConnector


class DummyConnector(BaseConnector):
    """A dummy connector that returns simple, fixed responses for testing."""

    async def test_prompt(self, system_prompt: str, message: str) -> str:
        """Return deterministic response derived from prompt and message."""
        refinement_depth = system_prompt.count("refined")
        digest = hashlib.md5(f"{system_prompt}|{message}".encode()).hexdigest()[:12]
        return f"test response depth={refinement_depth} {digest}"
