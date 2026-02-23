"""Shared evaluation logic for testing prompts."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from promptbeezz.config import OptimizerConfig
from promptbeezz.connectors import BaseConnector
from promptbeezz.optimizer.context import RunContext
from promptbeezz.optimizer.utils.model_tester import test_target_model
from promptbeezz.optimizer.utils.score_calculator import aggregate_prompt_score
from promptbeezz.schemas import (
    EvaluationScore,
    PromptCandidate,
    TaskSpec,
    TestCase,
    TestResult,
)
from promptbeezz.storage import EvaluationConverter

T = TypeVar("T")


async def evaluate_prompt(
    prompt: PromptCandidate,
    test_cases: list[TestCase],
    task_spec: TaskSpec,
    config: OptimizerConfig,
    model_client: BaseConnector,
    context: RunContext,
    parallel: bool = True,
    semaphore: asyncio.Semaphore | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> float:
    """
    Evaluate a single prompt against test cases and return average score.

    Args:
        prompt: Prompt candidate to evaluate
        test_cases: Test cases to run
        task_spec: Task specification
        config: Optimizer configuration
        model_client: Connector for the target model
        context: Run context for database access
        parallel: Whether to run evaluations in parallel (default: True)
        semaphore: Optional shared semaphore for global concurrency control.
                   If None and parallel=True, creates a local semaphore.

    Returns:
        Average score across all test cases
    """

    async def evaluate_single_test(test: TestCase) -> EvaluationScore:
        """Evaluate a single test case with optional concurrency control."""
        # Acquire semaphore if provided
        if semaphore:
            async with semaphore:
                return await _evaluate_test_impl(test)
        else:
            return await _evaluate_test_impl(test)

    async def _evaluate_test_impl(test: TestCase) -> EvaluationScore:
        """Implementation of single test evaluation."""
        response: str
        try:
            response = await _await_with_heartbeat(
                test_target_model(prompt.prompt_text, test.input_message, model_client),
                label=f"target model (prompt={prompt.id}, test={test.id})",
                progress_callback=progress_callback,
                heartbeat_seconds=config.heartbeat_seconds,
                timeout_seconds=config.request_timeout_seconds,
            )
        except TimeoutError as exc:
            if progress_callback:
                progress_callback(f"  Timeout: {exc}")
            response = "[TIMEOUT] Target model response timed out."
            evaluation = EvaluationScore.calculate_overall(
                functionality=0,
                safety=0,
                consistency=0,
                edge_case_handling=0,
                reasoning=str(exc),
                weights=config.scoring_weights,
            )
            test_result = TestResult(
                test_case_id=test.id,
                prompt_id=prompt.id,
                model_response=response,
                evaluation=evaluation,
            )
            db_evaluation = EvaluationConverter.to_db(test_result, context.run_id)
            context.eval_repo.save(db_evaluation)
            return evaluation

        # Score with LLM judge
        if config.evaluator is None:
            raise ValueError("OptimizerConfig.evaluator is required")
        try:
            evaluation = await _await_with_heartbeat(
                config.evaluator.evaluate(
                    response=response,
                    test_case=test,
                    task_spec=task_spec,
                    scoring_weights=config.scoring_weights,
                ),
                label=f"judge model (prompt={prompt.id}, test={test.id})",
                progress_callback=progress_callback,
                heartbeat_seconds=config.heartbeat_seconds,
                timeout_seconds=config.request_timeout_seconds,
            )
        except TimeoutError as exc:
            if progress_callback:
                progress_callback(f"  Timeout: {exc}")
            evaluation = EvaluationScore.calculate_overall(
                functionality=0,
                safety=0,
                consistency=0,
                edge_case_handling=0,
                reasoning=str(exc),
                weights=config.scoring_weights,
            )

        # Save evaluation to database
        test_result = TestResult(
            test_case_id=test.id,
            prompt_id=prompt.id,
            model_response=response,
            evaluation=evaluation,
        )
        db_evaluation = EvaluationConverter.to_db(test_result, context.run_id)
        context.eval_repo.save(db_evaluation)
        return evaluation

    # Run evaluations in parallel or sequentially based on config
    if parallel:
        evaluations = await asyncio.gather(*[evaluate_single_test(test) for test in test_cases])
    else:
        evaluations = []
        for test in test_cases:
            evaluation = await evaluate_single_test(test)
            evaluations.append(evaluation)

    return aggregate_prompt_score(evaluations)


async def _await_with_heartbeat(
    awaitable: Awaitable[T],
    *,
    label: str,
    progress_callback: Callable[[str], None] | None,
    heartbeat_seconds: float,
    timeout_seconds: float,
) -> T:
    task = asyncio.create_task(awaitable)
    elapsed = 0.0

    while True:
        done, _ = await asyncio.wait({task}, timeout=heartbeat_seconds)
        if task in done:
            return await task

        elapsed += heartbeat_seconds
        if progress_callback:
            progress_callback(f"  Still waiting on {label} ({int(elapsed)}s)...")

        if timeout_seconds > 0 and elapsed >= timeout_seconds:
            task.cancel()
            raise TimeoutError(
                f"{label} timed out after {int(timeout_seconds)}s. "
                "Consider lowering concurrency or increasing timeout."
            )
