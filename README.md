# promptbeezz

Prompt optimization framework.

## What it provides

- Multi-stage optimization pipeline for system prompts.
- Pluggable target connectors (OpenAI, Anthropic, Minimax, custom).
- Pluggable evaluator providers (OpenAI, Anthropic, Minimax).
- Provider-agnostic LLM abstraction for generation, testing, and refinement.

## Runtime configuration

`OptimizerConfig` controls execution behavior and includes request observability settings

## Development

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
ruff check src tests
pytest -v
```

## Quick run (Minimax)

```bash
export MINIMAX_API_KEY=your_key
python example.py
```
