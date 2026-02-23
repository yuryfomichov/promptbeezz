# Promptbeezz Agent Notes

## Scope
- This repository contains the standalone prompt optimization framework.
- Keep framework concerns here; app-specific behavior belongs to consuming repos (for example, `interviewee`).

## Architecture
- `src/promptbeezz/config.py`: pipeline and runtime configuration models.
- `src/promptbeezz/optimizer/`: stage orchestration and execution flow.
- `src/promptbeezz/llm/`: provider clients used by generator/test/refiner/evaluator internals.
- `src/promptbeezz/evaluators/`: pluggable LLM-as-judge implementations.
- `src/promptbeezz/connectors/`: target-model connectors under test.
- `src/promptbeezz/storage/`: SQLAlchemy persistence and repositories.

## Important conventions
- Use native structured output paths for providers.
- Keep API surfaces provider-agnostic at orchestration level.
- Prefer explicit config fields over environment variable lookups in library code.
- Preserve deterministic tests by mocking clients/evaluators in `tests/helpers`.

## Quality gates
- Run before finishing changes:
  - `ruff check src tests`
  - `pytest -q`

## Compatibility notes
- Minimax is treated as Anthropic-compatible transport with a different base URL and message content shape.
- Avoid introducing app-specific prompts, secrets, or deployment logic in this repo.
