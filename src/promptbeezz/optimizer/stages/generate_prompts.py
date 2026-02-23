"""Generate prompts stage: Create diverse initial prompt variations."""

import uuid

from promptbeezz.agents.prompt_generator_agent import create_generator_agent
from promptbeezz.optimizer.base_stage import BaseStage
from promptbeezz.optimizer.context import RunContext
from promptbeezz.schemas import PromptCandidate
from promptbeezz.storage import PromptConverter


class GeneratePromptsStage(BaseStage):
    """Stage 1: Generate diverse initial prompt variations."""

    @property
    def name(self) -> str:
        """Return the stage name."""
        return "Generate Prompts"

    async def _run_async(self, context: RunContext) -> RunContext:
        """
        Generate initial prompt variations (async mode).

        Args:
            context: Run context with task_spec

        Returns:
            Updated context (prompts saved to database)
        """
        prompts = []

        # Include original system prompt if provided
        if context.task_spec.current_prompt:
            self._print_progress("Including original system prompt for testing...")
            original_prompt = PromptCandidate(
                id=f"original_system_prompt_{uuid.uuid4().hex[:8]}",
                prompt_text=context.task_spec.current_prompt,
                stage="initial",
                strategy="original_system_prompt",
                is_original_system_prompt=True,
            )
            prompts.append(original_prompt)
            self._print_progress("Added original system prompt to testing pipeline")

        # Generate additional variations
        num_to_generate = (
            self.config.num_initial_prompts - 1
            if context.task_spec.current_prompt
            else self.config.num_initial_prompts
        )

        self._print_progress(f"Generating {num_to_generate} diverse prompt variations...")

        if self.config.generator_client is None:
            raise ValueError("OptimizerConfig.generator_client is required")
        generator_client = self.config.generator_client

        generator = create_generator_agent(
            generator_client,
            self.config.generator_llm,
            context.task_spec,
            num_to_generate,
        )
        generated_prompts = self._parse_generated_prompts(await generator.run())
        prompts.extend(generated_prompts)

        self._print_progress(
            f"Generated {len(prompts)} total prompts "
            f"({1 if context.task_spec.current_prompt else 0} original + "
            f"{len(generated_prompts)} variations)"
        )

        # Save all prompts to database
        for prompt in prompts:
            db_prompt = PromptConverter.to_db(prompt, context.run_id)
            context.prompt_repo.save(db_prompt)

        return context

    async def _run_sync(self, context: RunContext) -> RunContext:
        """
        Generate initial prompt variations (sync mode - same as async for this stage).

        Args:
            context: Run context with task_spec

        Returns:
            Updated context with initial_prompts populated
        """
        # This stage doesn't benefit from parallel execution since it's a single agent call
        return await self._run_async(context)

    def _parse_generated_prompts(self, agent_output) -> list[PromptCandidate]:
        """Parse agent output into PromptCandidate objects."""
        prompts = []
        for item in agent_output.prompts:
            prompts.append(
                PromptCandidate(
                    id=item.id,
                    prompt_text=item.prompt_text,
                    stage="initial",
                    strategy=item.strategy,
                )
            )
        return prompts
