# Contributing

This is a portfolio project, but feedback and suggestions are welcome.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
uvicorn app.main:app --reload
```

## Running tests

```bash
# Unit tests — no API keys needed
pytest tests/unit/ -v

# All tests
pytest
```

## Code style

- Follow existing patterns — Pydantic models for all I/O, structured logging everywhere
- New endpoints need a unit or integration test
- Keep the responsible AI constraints in the system prompt intact