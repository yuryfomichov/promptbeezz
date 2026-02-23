"""Configuration models for promptbeezz runtime."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from promptbeezz.connectors.base import BaseConnector
from promptbeezz.evaluators.base import BaseEvaluator
from promptbeezz.llm.base import BaseLLMClient
from promptbeezz.schemas import TaskSpec


class LLMConfig(BaseModel):
    """Configuration for a single model call profile."""

    model: str = Field(description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None)


class TestDistribution(BaseModel):
    """Distribution of test cases across categories."""

    core: int = Field(default=20, ge=0, description="Primary capability tests")
    edge: int = Field(default=10, ge=0, description="Rare and unusual inputs")
    boundary: int = Field(default=10, ge=0, description="Boundary and refusal tests")
    adversarial: int = Field(default=5, ge=0, description="Prompt attack tests")
    consistency: int = Field(default=3, ge=0, description="Tone and consistency tests")
    format: int = Field(default=2, ge=0, description="Formatting and structure tests")

    @property
    def total(self) -> int:
        return (
            self.core
            + self.edge
            + self.boundary
            + self.adversarial
            + self.consistency
            + self.format
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "core": self.core,
            "edge": self.edge,
            "boundary": self.boundary,
            "adversarial": self.adversarial,
            "consistency": self.consistency,
            "format": self.format,
        }


class OptimizerConfig(BaseModel):
    """Configuration for the full optimization pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_initial_prompts: int = Field(default=15)
    quick_test_distribution: TestDistribution = Field(
        default_factory=lambda: TestDistribution(
            core=2,
            edge=2,
            boundary=1,
            adversarial=1,
            consistency=1,
            format=0,
        )
    )
    top_k_advance: int = Field(default=5)

    rigorous_test_distribution: TestDistribution = Field(default_factory=TestDistribution)
    top_m_refine: int = Field(default=3)

    max_iterations_per_track: int = Field(default=10)
    convergence_threshold: float = Field(default=0.02)
    early_stopping_patience: int = Field(default=2)

    scoring_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "functionality": 0.4,
            "safety": 0.3,
            "consistency": 0.2,
            "edge_case_handling": 0.1,
        }
    )

    generator_llm: LLMConfig = Field(default=LLMConfig(model="gpt-4o", temperature=0.8))
    test_designer_llm: LLMConfig = Field(default=LLMConfig(model="gpt-4o", temperature=0.7))
    evaluator_llm: LLMConfig = Field(default=LLMConfig(model="gpt-4o", temperature=0.3))
    refiner_llm: LLMConfig = Field(default=LLMConfig(model="gpt-4o", temperature=0.7))

    generator_client: BaseLLMClient | None = Field(default=None)
    test_designer_client: BaseLLMClient | None = Field(default=None)
    refiner_client: BaseLLMClient | None = Field(default=None)
    connector: BaseConnector | None = Field(default=None)
    evaluator: BaseEvaluator | None = Field(default=None)

    output_dir: str | Path = Field(default="promptbeezz/data")
    parallel_execution: bool = Field(default=False)
    max_concurrent_evaluations: int = Field(default=5, ge=1)
    heartbeat_seconds: float = Field(default=30.0, gt=0.0)
    request_timeout_seconds: float = Field(default=300.0, gt=0.0)
    verbose: bool = Field(default=True)

    task_spec: TaskSpec = Field(...)

    @property
    def num_quick_tests(self) -> int:
        return self.quick_test_distribution.total

    @property
    def num_rigorous_tests(self) -> int:
        return self.rigorous_test_distribution.total

    @property
    def database_path(self) -> Path:
        return Path(self.output_dir) / "storage" / "optimizer.db"

    @property
    def results_path(self) -> Path:
        return Path(self.output_dir) / "results"

    def validate_runtime_dependencies(self) -> None:
        """Ensure runtime clients are wired before execution."""
        missing: list[str] = []
        if self.generator_client is None:
            missing.append("generator_client")
        if self.test_designer_client is None:
            missing.append("test_designer_client")
        if self.refiner_client is None:
            missing.append("refiner_client")
        if self.evaluator is None:
            missing.append("evaluator")
        if missing:
            raise ValueError("OptimizerConfig requires runtime dependencies: " + ", ".join(missing))
