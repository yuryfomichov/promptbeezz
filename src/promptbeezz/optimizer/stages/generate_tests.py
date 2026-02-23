"""Generate tests stage: Create test cases for evaluation."""

from promptbeezz.agents.test_designer_agent import (
    TestCasesOutput,
    create_test_designer_agent,
)
from promptbeezz.optimizer.base_stage import BaseStage
from promptbeezz.optimizer.context import RunContext
from promptbeezz.schemas import TestCase
from promptbeezz.storage import TestCaseConverter


class GenerateTestsStage(BaseStage):
    """Generate test cases for a specific testing stage."""

    def __init__(self, test_stage: str, *args, **kwargs):
        """
        Initialize test generation stage.

        Args:
            test_stage: "quick" or "rigorous"
            *args, **kwargs: Passed to BaseStage
        """
        super().__init__(*args, **kwargs)
        self.test_stage = test_stage

    @property
    def name(self) -> str:
        """Return the stage name."""
        return f"Generate {self.test_stage.capitalize()} Tests"

    async def _run_async(self, context: RunContext) -> RunContext:
        """
        Generate test cases (async mode).

        Args:
            context: Run context with task_spec

        Returns:
            Updated context (test cases saved to database)
        """
        num_tests = (
            self.config.num_quick_tests
            if self.test_stage == "quick"
            else self.config.num_rigorous_tests
        )
        distribution = (
            self.config.quick_test_distribution
            if self.test_stage == "quick"
            else self.config.rigorous_test_distribution
        )

        self._print_progress(f"Designing {num_tests} {self.test_stage} tests...")

        if self.config.test_designer_client is None:
            raise ValueError("OptimizerConfig.test_designer_client is required")
        test_designer_client = self.config.test_designer_client

        test_designer = create_test_designer_agent(
            test_designer_client,
            self.config.test_designer_llm,
            context.task_spec,
            distribution,
            stage=self.test_stage,
        )
        tests = self._parse_test_cases(await test_designer.run())

        # Save all test cases to database
        db_tests = [
            TestCaseConverter.to_db(test, context.run_id, self.test_stage) for test in tests
        ]
        context.test_repo.save_many(db_tests)

        self._print_progress(f"Generated {len(tests)} {self.test_stage} test cases")

        return context

    async def _run_sync(self, context: RunContext) -> RunContext:
        """
        Generate test cases (sync mode - same as async for this stage).

        Args:
            context: Run context with task_spec

        Returns:
            Updated context with quick_tests or rigorous_tests populated
        """
        # This stage doesn't benefit from parallel execution since it's a single agent call
        return await self._run_async(context)

    def _parse_test_cases(self, agent_output: TestCasesOutput) -> list[TestCase]:
        """Parse agent output into TestCase objects."""
        return agent_output.test_cases
