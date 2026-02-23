"""Test case designer service built on BaseLLMClient."""

from pydantic import BaseModel, Field

from promptbeezz.config import LLMConfig, TestDistribution
from promptbeezz.llm.base import BaseLLMClient
from promptbeezz.schemas import TaskSpec, TestCase


class TestCasesOutput(BaseModel):
    """Output structure for test cases."""

    test_cases: list[TestCase] = Field(description="List of test cases")


class TestDesignerAgent:
    """Test designer agent using a provider-agnostic LLM client."""

    def __init__(
        self,
        *,
        llm_client: BaseLLMClient,
        llm_config: LLMConfig,
        task_spec: TaskSpec,
        test_distribution: TestDistribution,
        stage: str,
    ):
        self.name = "TestDesigner"
        self.llm_client = llm_client
        self.llm_config = llm_config
        self.instructions = _build_instructions(task_spec, test_distribution, stage)

    async def run(self) -> TestCasesOutput:
        return await self.llm_client.generate_structured(
            "Create evaluation test cases.",
            system_prompt=self.instructions,
            schema=TestCasesOutput,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
        )


def create_test_designer_agent(
    llm_client: BaseLLMClient,
    llm_config: LLMConfig,
    task_spec: TaskSpec,
    test_distribution: TestDistribution,
    stage: str = "quick",
) -> TestDesignerAgent:
    """Factory for test designer service."""
    return TestDesignerAgent(
        llm_client=llm_client,
        llm_config=llm_config,
        task_spec=task_spec,
        test_distribution=test_distribution,
        stage=stage,
    )


def _build_instructions(
    task_spec: TaskSpec,
    test_distribution: TestDistribution,
    stage: str,
) -> str:
    category_descriptions = {
        field_name: field_info.description
        for field_name, field_info in TestDistribution.model_fields.items()
    }

    dist_items = [
        f"- **{cat}**: EXACTLY {count} tests ({category_descriptions[cat]})"
        for cat, count in test_distribution.to_dict().items()
        if count > 0
    ]
    dist_str = "\n".join(dist_items)

    stage_emphasis = (
        "Focus on high-signal tests that quickly reveal prompt quality."
        if stage == "quick"
        else "Be creative and thorough within each category. Think like a QA engineer trying to find weaknesses."
    )

    focus = f"""Create EXACTLY {test_distribution.total} evaluation tests following this EXACT distribution:
{dist_str}

CRITICAL: You MUST create the exact number specified for each category. Do not deviate from these counts.

{stage_emphasis}"""

    return f"""You are a meticulous QA engineer who designs comprehensive test cases.

**TASK BEING TESTED**: {task_spec.task_description}

**EXPECTED BEHAVIOR**:
{task_spec.behavioral_specs}

**VALIDATION RULES**:
{chr(10).join(f"- {rule}" for rule in task_spec.validation_rules)}

{focus}

**Categories**: core, edge, boundary, adversarial, consistency, format

**USER MESSAGE STYLE REQUIREMENTS**:
- Phrase each input_message as a natural request or instruction consistent with the task scenario. Do not prefix requests with meta labels (e.g., "Behavioral question:", "Format test:", "Coding task:") unless the domain normally requires those markers.
- Reference supporting resources only when that interaction is part of the task expectations, and describe them using the terminology the scenario would naturally use (e.g., files for coding agents, brief summaries for interview coaching).
- When you mention prior context, do so conversationally rather than listing resource names unless explicit identifiers are normal in that domain.

Make tests specific, actionable, and diverse. Each test should reveal something important about prompt quality.
""".strip()
