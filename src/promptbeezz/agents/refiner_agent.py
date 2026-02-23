"""Prompt refinement service built on BaseLLMClient."""

from pydantic import BaseModel, Field

from promptbeezz.config import LLMConfig
from promptbeezz.llm.base import BaseLLMClient
from promptbeezz.schemas import TaskSpec


class RefinedPromptOutput(BaseModel):
    """Output structure for refined prompts."""

    improved_prompt: str = Field(description="The improved system prompt text")
    changes_made: str = Field(description="Brief description of improvements made")


class RefinerAgent:
    """Refiner agent using a provider-agnostic LLM client."""

    def __init__(
        self,
        *,
        llm_client: BaseLLMClient,
        llm_config: LLMConfig,
        task_spec: TaskSpec,
        current_prompt: str,
        weaknesses: str,
        failed_tests: list[str],
        iteration: int,
    ):
        self.name = "PromptRefiner"
        self.llm_client = llm_client
        self.llm_config = llm_config
        self.instructions = _build_instructions(
            task_spec=task_spec,
            current_prompt=current_prompt,
            weaknesses=weaknesses,
            failed_tests=failed_tests,
            iteration=iteration,
        )

    async def run(self) -> RefinedPromptOutput:
        return await self.llm_client.generate_structured(
            "Refine this system prompt based on weaknesses.",
            system_prompt=self.instructions,
            schema=RefinedPromptOutput,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
        )


def create_refiner_agent(
    llm_client: BaseLLMClient,
    llm_config: LLMConfig,
    task_spec: TaskSpec,
    current_prompt: str,
    weaknesses: str,
    failed_tests: list[str],
    iteration: int,
) -> RefinerAgent:
    """Factory for refiner service."""
    return RefinerAgent(
        llm_client=llm_client,
        llm_config=llm_config,
        task_spec=task_spec,
        current_prompt=current_prompt,
        weaknesses=weaknesses,
        failed_tests=failed_tests,
        iteration=iteration,
    )


def _build_instructions(
    *,
    task_spec: TaskSpec,
    current_prompt: str,
    weaknesses: str,
    failed_tests: list[str],
    iteration: int,
) -> str:
    failed_tests_str = "\n".join(f"- {test}" for test in failed_tests) if failed_tests else "None"

    return f"""You are a prompt optimization specialist who surgically improves system prompts.

**TASK**: {task_spec.task_description}

**REQUIRED BEHAVIOR**:
{task_spec.behavioral_specs}

**VALIDATION RULES**:
{chr(10).join(f"- {rule}" for rule in task_spec.validation_rules)}

**CURRENT PROMPT** (Iteration {iteration}):
```
{current_prompt}
```

**IDENTIFIED WEAKNESSES**:
{weaknesses}

**FAILED TEST CASES**:
{failed_tests_str}

**YOUR JOB**:
Create an improved version of the system prompt that:

1. **Preserves Strengths**: Keep what works well (high-scoring aspects)
2. **Addresses Failures**: Fix specific failure modes from the test results
3. **Adds Clarifications**: Where confusion or ambiguity occurred
4. **Strengthens Boundaries**: Where rules were violated or misunderstood
5. **Maintains Conciseness**: Don't make it unnecessarily verbose

**IMPROVEMENT STRATEGIES**:
- Add explicit examples for areas where the AI failed
- Strengthen language around violated rules
- Add constraints or formatting instructions if needed
- Clarify tone/style requirements if inconsistent
- Reorder instructions to emphasize critical parts

Focus on targeted, surgical improvements. This is iteration {iteration}, so build on previous refinements.
""".strip()
