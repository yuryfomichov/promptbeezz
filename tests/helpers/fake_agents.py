"""Fake runtime components for promptbeezz tests."""

import hashlib
import re

from promptbeezz.agents.prompt_generator_agent import GeneratedPrompt, GeneratedPromptsOutput
from promptbeezz.agents.refiner_agent import RefinedPromptOutput
from promptbeezz.agents.test_designer_agent import TestCasesOutput
from promptbeezz.llm.base import BaseLLMClient
from promptbeezz.schemas import EvaluationScore, TaskSpec, TestCase


class FakeLLMClient(BaseLLMClient):
    """Fake LLM client returning deterministic structured outputs."""

    async def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return f"fake-text:{len(prompt)}:{len(system_prompt or '')}:{temperature}:{max_tokens}"

    async def generate_structured(
        self,
        prompt: str,
        *,
        schema,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        del prompt, temperature, max_tokens
        instructions = system_prompt or ""
        schema_name = schema.__name__

        if schema_name == "GeneratedPromptsOutput":
            return self._build_generated_prompts(instructions)
        if schema_name == "TestCasesOutput":
            return self._build_test_cases(instructions)
        if schema_name == "RefinedPromptOutput":
            return self._build_refined_prompt(instructions)

        raise ValueError(f"Unsupported schema for FakeLLMClient: {schema_name}")

    def _build_generated_prompts(self, instructions: str) -> GeneratedPromptsOutput:
        match = re.search(r"Generate exactly (\d+) diverse", instructions, re.IGNORECASE)
        n = int(match.group(1)) if match else 15
        prompts = [
            GeneratedPrompt(id=f"p{i}", strategy=f"s{i}", prompt_text=f"prompt{i}")
            for i in range(n)
        ]
        return GeneratedPromptsOutput(prompts=prompts)

    def _build_test_cases(self, instructions: str) -> TestCasesOutput:
        matches = re.findall(r"\*\*(\w+)\*\*: EXACTLY (\d+) tests", instructions)
        distribution = {category: int(count) for category, count in matches}
        if not distribution:
            distribution = {"core": 2, "edge": 1}

        test_cases = []
        idx = 0
        for category, count in distribution.items():
            for _ in range(count):
                test_cases.append(
                    TestCase(
                        id=f"t{idx}",
                        input_message="test",
                        expected_behavior="expected",
                        category=category,
                    )
                )
                idx += 1
        return TestCasesOutput(test_cases=test_cases)

    def _build_refined_prompt(self, instructions: str) -> RefinedPromptOutput:
        current_prompt = "prompt"
        if "CURRENT PROMPT" in instructions and "```" in instructions:
            parts = instructions.split("```")
            if len(parts) >= 2:
                current_prompt = parts[1].strip()
        return RefinedPromptOutput(
            improved_prompt=current_prompt + "\nrefined",
            changes_made="refined",
        )


class FakeEvaluator:
    """Fake evaluator returning deterministic but varied scores."""

    async def evaluate(
        self,
        response: str,
        test_case: TestCase,
        task_spec: TaskSpec,
        scoring_weights: dict[str, float],
    ) -> EvaluationScore:
        depth_match = re.search(r"depth=(\d+)", response)
        refinement_depth = int(depth_match.group(1)) if depth_match else 0

        seed = hashlib.md5(
            f"{response}|{test_case.id}|{test_case.category}|{task_spec.task_description}".encode()
        ).hexdigest()
        n = int(seed[:8], 16)

        bonus = min(refinement_depth * 2, 4)
        functionality = min(10, 5 + (n % 4) + bonus)
        safety = min(10, 5 + ((n >> 3) % 4) + bonus)
        consistency = min(10, 5 + ((n >> 6) % 4) + bonus)
        edge_case_handling = min(10, 5 + ((n >> 9) % 4) + bonus)

        return EvaluationScore.calculate_overall(
            functionality=functionality,
            safety=safety,
            consistency=consistency,
            edge_case_handling=edge_case_handling,
            reasoning="fake-eval",
            weights=scoring_weights,
        )
