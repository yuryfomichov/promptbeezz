# promptbeezz

Prompt optimization framework extracted from `interviewee`.

## What it provides

- Multi-stage optimization pipeline for system prompts.
- Pluggable target connectors (OpenAI, Anthropic, Minimax, custom).
- Pluggable evaluator providers (OpenAI, Anthropic, Minimax).
- Provider-agnostic LLM abstraction for generation, testing, and refinement.
- Structured outputs through provider-native schemas (JSON schema / tool schema).

## Runtime configuration

`OptimizerConfig` controls execution behavior and includes request observability settings

These are regular config fields (not env-driven).

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
