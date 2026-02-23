"""Provider-agnostic LLM client interface."""

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

TModel = TypeVar("TModel", bound=BaseModel)


class BaseLLMClient(ABC):
    """Abstract interface for text and structured generation."""

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate free-form text."""

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        *,
        schema: type[TModel],
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> TModel:
        """Generate output validated against a Pydantic schema."""
