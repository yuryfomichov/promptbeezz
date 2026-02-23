"""OpenAI implementation of the LLM client interface."""

from openai import AsyncOpenAI, BadRequestError

from promptbeezz.llm.base import BaseLLMClient, TModel


class OpenAILLMClient(BaseLLMClient):
    """LLM client backed by OpenAI Responses API."""

    def __init__(self, *, api_key: str, model: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        request_args = {
            "model": self.model,
            "instructions": system_prompt,
            "input": prompt,
        }
        if temperature is not None:
            request_args["temperature"] = temperature
        if max_tokens is not None:
            request_args["max_output_tokens"] = max_tokens

        try:
            response = await self.client.responses.create(**request_args)
        except BadRequestError as exc:
            if temperature is not None and _is_unsupported_temperature_error(exc):
                request_args.pop("temperature", None)
                response = await self.client.responses.create(**request_args)
            else:
                raise
        return response.output_text or ""

    async def generate_structured(
        self,
        prompt: str,
        *,
        schema: type[TModel],
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> TModel:
        request_args = {
            "model": self.model,
            "instructions": system_prompt,
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "structured_output",
                    "schema": schema.model_json_schema(),
                    "strict": True,
                }
            },
        }
        if temperature is not None:
            request_args["temperature"] = temperature
        if max_tokens is not None:
            request_args["max_output_tokens"] = max_tokens

        try:
            response = await self.client.responses.create(**request_args)
        except BadRequestError as exc:
            if temperature is not None and _is_unsupported_temperature_error(exc):
                request_args.pop("temperature", None)
                response = await self.client.responses.create(**request_args)
            else:
                raise

        raw = response.output_text or ""
        if not raw:
            raise ValueError("OpenAI structured output returned empty body")
        return schema.model_validate_json(raw)


def _is_unsupported_temperature_error(exc: BadRequestError) -> bool:
    message = ""
    if isinstance(exc.body, dict):
        error = exc.body.get("error")
        if isinstance(error, dict):
            message = str(error.get("message", ""))
    return "Unsupported parameter" in message and "temperature" in message
