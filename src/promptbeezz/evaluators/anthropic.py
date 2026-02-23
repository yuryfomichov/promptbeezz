"""Anthropic-backed evaluator."""

from promptbeezz.evaluators.base import BaseEvaluator, evaluate_with_llm_judge
from promptbeezz.llm.anthropic import AnthropicLLMClient
from promptbeezz.schemas import EvaluationScore, TaskSpec, TestCase


class AnthropicEvaluator(BaseEvaluator):
    """Evaluator that uses an Anthropic judge model."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        judge_prompt_template: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ):
        self._client = AnthropicLLMClient(api_key=api_key, model=model)
        self._judge_prompt_template = judge_prompt_template
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def evaluate(
        self,
        response: str,
        test_case: TestCase,
        task_spec: TaskSpec,
        scoring_weights: dict[str, float],
    ) -> EvaluationScore:
        return await evaluate_with_llm_judge(
            llm_client=self._client,
            response=response,
            test_case=test_case,
            task_spec=task_spec,
            scoring_weights=scoring_weights,
            judge_prompt_template=self._judge_prompt_template,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
