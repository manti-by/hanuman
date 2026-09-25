---
title: Groq to OpenRouter migration
date: 2026-09-25
type: implementation
status: resolved
session_id: ses_f27a6cc6dffew0N0Z8PwLU48FM
services: []
branch: -
tickets: []
tags: [llm, openrouter, langchain, config]
related: []
---

# Groq to OpenRouter migration

## TL;DR

Replaced all Groq integration with OpenRouter via `langchain-openai` `ChatOpenAI` pointed at `https://openrouter.ai/api/v1`. Renamed `groq_api_key` / `GROQ_API_KEY` to `openrouter_api_key` / `OPENROUTER_API_KEY`, switched default chat model to `meta-llama/llama-3.1-8b-instruct`, updated docs, compose, and tests. All 16 tests pass.

---

## Overview

`hanuman/search.py` used `langchain-groq` `ChatGroq` with a Groq-only model id. Goal: use OpenRouter's OpenAI-compatible endpoint instead, with no remaining `groq` references in code, config, or docs.

## Step 1 — LLM client swap

**File:** `hanuman/search.py:4,30`

Before:

```python
from langchain_groq import ChatGroq
llm = ChatGroq(
    model=settings.chat.model,
    groq_api_key=settings.groq_api_key,
    ...
)
```

After:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=settings.chat.model,
    api_key=settings.openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=settings.chat.temperature,
    max_tokens=settings.chat.max_tokens,
)
```

Guard message changed to `OPENROUTER_API_KEY is not set` (`hanuman/search.py:26`).

## Step 2 — Settings and default model

**File:** `hanuman/settings.py:24,35`

- `groq_api_key: SecretStr | None` → `openrouter_api_key: SecretStr | None`
- `ChatSettings.model`: `llama-3.1-8b-instant` (Groq id) → `meta-llama/llama-3.1-8b-instruct` (OpenRouter id)

## Step 3 — Dependencies, env, docs

- `pyproject.toml:13`: `langchain-groq>=1.1.3` → `langchain-openai>=1.0.0` (resolved to `1.6.6` + `openai==3.19.2`; `groq==0.37.1` / `langchain-groq==1.1.3` uninstalled; `uv.lock` regenerated with no `groq` refs)
- `.env.example:1`: `OPENROUTER_API_KEY=your_openrouter_api_key_here`
- `docker-compose.yml:29`: `- OPENROUTER_API_KEY=${OPENROUTER_API_KEY}`
- `README.md`: feature line, tech stack, setup, env-var table updated to OpenRouter
- `tests/test_main.py`, `tests/test_settings.py`: `ChatGroq` patches → `ChatOpenAI`; `groq_api_key` mocks → `openrouter_api_key`; model assertion and `HANUMAN_OPENROUTER_API_KEY` env test updated

## Test Results

- `uv run pytest tests/ -v`: 16 passed
- `uv run ty check`: all checks passed
- `uv run pre-commit run --all-files`: passed (ruff, ruff-format, bandit, pyupgrade)
- `grep -rni groq --exclude-dir=.venv --exclude-dir=.git`: no matches (excluding history)

---

## Follow-ups

- None

## References

- External: https://openrouter.ai/docs
