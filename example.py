import asyncio
import os

from dotenv import load_dotenv

from promptbeezz import MinimaxConnector, MinimaxEvaluator, OptimizationRunner, OptimizerConfig
from promptbeezz.config import LLMConfig, TestDistribution
from promptbeezz.llm import MinimaxLLMClient
from promptbeezz.schemas import TaskSpec


async def main() -> None:
    load_dotenv()
    api_key = os.getenv("MINIMAX_API_KEY", "").strip()
    if not api_key:
        raise ValueError("MINIMAX_API_KEY is empty. Set it in .env or export it in shell.")
    model = os.getenv("MINIMAX_MODEL", "MiniMax-M2.5")

    llm = MinimaxLLMClient(api_key=api_key, model=model)
    config = OptimizerConfig(
        task_spec=TaskSpec(
            task_description="Helpful support assistant for SaaS product questions",
            behavioral_specs="Be concise, accurate, and polite. Ask clarifying questions if needed.",
            validation_rules=["Do not invent facts", "Refuse unsafe requests politely"],
            current_prompt="You are a helpful support assistant.",
        ),
        num_initial_prompts=2,
        quick_test_distribution=TestDistribution(
            core=1, edge=1, boundary=0, adversarial=0, consistency=0, format=0
        ),
        top_k_advance=1,
        rigorous_test_distribution=TestDistribution(
            core=2, edge=1, boundary=1, adversarial=0, consistency=0, format=0
        ),
        top_m_refine=1,
        max_iterations_per_track=1,
        generator_llm=LLMConfig(model=model, max_tokens=4000),
        test_designer_llm=LLMConfig(model=model, max_tokens=3000),
        evaluator_llm=LLMConfig(model=model, max_tokens=2000),
        refiner_llm=LLMConfig(model=model, max_tokens=4000),
        generator_client=llm,
        test_designer_client=llm,
        refiner_client=llm,
        evaluator=MinimaxEvaluator(api_key=api_key, model=model),
        connector=MinimaxConnector(api_key=api_key, model=model),
        parallel_execution=True,
        max_concurrent_evaluations=6,
        heartbeat_seconds=20,
        request_timeout_seconds=240,
        output_dir="example_output",
    )

    result, run_dir = await OptimizationRunner(connector=None, config=config).run()
    print(f"\nChampion score: {result.best_prompt.rigorous_score:.2f}")
    print(f"Output dir: {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
