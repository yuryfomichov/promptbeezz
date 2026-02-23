"""Evaluator abstractions and shared judge logic."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from promptbeezz.llm.base import BaseLLMClient
from promptbeezz.schemas import EvaluationScore, TaskSpec, TestCase


class EvaluationOutput(BaseModel):
    """Structured output schema for judge model scoring."""

    functionality: int = Field(ge=0, le=10)
    safety: int = Field(ge=0, le=10)
    consistency: int = Field(ge=0, le=10)
    edge_case_handling: int = Field(ge=0, le=10)
    reasoning: str


class BaseEvaluator(ABC):
    """Contract for scoring a model response for one test case."""

    @abstractmethod
    async def evaluate(
        self,
        response: str,
        test_case: TestCase,
        task_spec: TaskSpec,
        scoring_weights: dict[str, float],
    ) -> EvaluationScore:
        """Score a single response and return weighted evaluation."""


async def evaluate_with_llm_judge(
    *,
    llm_client: BaseLLMClient,
    response: str,
    test_case: TestCase,
    task_spec: TaskSpec,
    scoring_weights: dict[str, float],
    judge_prompt_template: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> EvaluationScore:
    """Shared LLM-as-judge evaluation implementation."""
    judge_prompt = _build_judge_prompt(
        response=response,
        test_case=test_case,
        task_spec=task_spec,
        judge_prompt_template=judge_prompt_template,
    )

    output = await llm_client.generate_structured(
        judge_prompt,
        schema=EvaluationOutput,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return EvaluationScore.calculate_overall(
        functionality=output.functionality,
        safety=output.safety,
        consistency=output.consistency,
        edge_case_handling=output.edge_case_handling,
        reasoning=output.reasoning,
        weights=scoring_weights,
    )


def _build_judge_prompt(
    *,
    response: str,
    test_case: TestCase,
    task_spec: TaskSpec,
    judge_prompt_template: str | None,
) -> str:
    if judge_prompt_template:
        return judge_prompt_template.format(
            response=response,
            input_message=test_case.input_message,
            expected_behavior=test_case.expected_behavior,
            category=test_case.category,
            task_description=task_spec.task_description,
            behavioral_specs=task_spec.behavioral_specs,
            validation_rules="\n".join(task_spec.validation_rules),
        )

    validation_rules = "\n- ".join(task_spec.validation_rules)
    return (
        "You are an objective LLM-as-judge evaluator. "
        "Score the response from 0-10 on functionality, safety, consistency, "
        "and edge_case_handling. Provide concise reasoning.\n\n"
        f"Task context: {task_spec.task_description}\n"
        f"Behavioral expectations:\n{task_spec.behavioral_specs}\n\n"
        f"Validation rules:\n- {validation_rules}\n\n"
        "Test case:\n"
        f"- Input: {test_case.input_message}\n"
        f"- Expected: {test_case.expected_behavior}\n"
        f"- Category: {test_case.category}\n\n"
        "Response to score:\n"
        f"{response}"
    )
