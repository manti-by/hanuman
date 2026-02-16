# AGENTS.md

## Project Overview

Hanuman is a "classic" RAG application.

## Project Structure

- `main.py`: Entry point with `index` and `search` commands
- `hanuman/settings.py`: Core configuration and environment variables
- `hanuman/__init__.py`: Package initialization
- `docker-compose.yml`: Docker services (PostgreSQL + pgvector, app)
- `Dockerfile`: Python application container
- `tests/`: Test suite
- `.env.example`: Environment variables template

## Git Workflow

This project adheres strictly to the Git Flow branching model. AI agents must follow these guidelines:

### Main Branch:

- The `master` branch always contains production-ready, stable code.
- Never commit directly to `master`.
- Do not use `git push --force` on the `master` branch.
- Do not merge branches into `master` without explicit approval.

### Feature Branches:

- Create feature branches using the naming convention `<agent-name>/feature/<issue-id>-<descriptive-name>` (e.g., `opencode/feature/mnt-10-add-user-authentication`).
- Use the [Conventional Commits](https://www.conventionalcommits.org) specification for commit messages (e.g., `feat:`, `fix:`, `docs:`).
- Ensure all local tests pass before committing.
- Use `git push --force-with-lease` if needed on your feature branch, but never on `master`.

### Pull Requests (PRs):

- Open a Pull Request for every completed feature branch.
- PRs must be reviewed and pass all CI checks before merging.
- The PR title should follow the Conventional Commits specification.

## Linear Workflow

- When starting implementation of any issue from `TODO`, move it to `In Progress` column.
- When feature is completed and PR is created, move it to `In Review` column.
- After approval, merge the feature branch into `master` and move the issue to `Done` column.
- If the feature branch is not merged into `master`, move it back to `In Progress` column.
- If the feature branch is closed without merging, move it to `Closed` column.

## Development Commands

### Package Management

```bash
# Install dependencies (including dev extras)
uv sync --all-extras --dev

# Upgrade dependencies and pre-commit hooks
uv run uv-bump
uv sync --all-extras --dev
uv run pre-commit autoupdate
```

### Makefile Targets

```bash
make check          # Run type checking and pre-commit checks
make pip            # Install dependencies
make update         # Upgrade dependencies and pre-commit hooks
make test           # Run tests
make ci             # Shorthand: pip check test
```

### Running Modules

From the project root, after creating a virtualenv and installing dependencies:

```bash
uv run main.py
```

## Language & Environment

- Python >=3.13.6, <3.14.0 (see `pyproject.toml`)
- Follow PEP 8 style guidelines, with Ruff enforcing style and linting (120 char line length)
- Use type hints for public functions and complex code paths
- Prefer f-strings over `.format()` or `%`
- Use list/dict/set comprehensions instead of `map`/`filter` where it improves readability
- Prefer `pathlib.Path` over `os.path` for filesystem paths
- Follow PEP 257 for docstrings where docstrings are used
- Prefer EAFP (try/except) over LBYL (if checks) in Python code

## Code Style & Tooling

Configured in `pyproject.toml`:

- **Ruff** for linting and import management (`[tool.ruff]`, `[tool.ruff.lint]`)
- **Bandit** for basic security checks (`[tool.bandit]`)
- **pre-commit** is used to run the tools before commits
- **ty** for type checking

Run manually:

```bash
uv run pre-commit run --all-files
uv run ruff check .
uv run ty check
uv run bandit -c pyproject.toml .
```

## Testing Guidelines

- Use `pytest` for tests
- Tests live in `tests/` directory
- Run with `make test` or `uv run pytest tests/`

## Security Guidelines

- Never commit secrets, passwords, or API tokens
- Configure sensitive values via environment variables
- Run `bandit` periodically or in CI
- Validate any external input before using it in system calls or network operations

## AI Behavior

Response style -- concise and minimal:

- Provide minimal, working code without unnecessary explanation
- Omit comments unless essential for understanding
- Skip boilerplate and obvious patterns unless requested
- Use type inference and shorthand syntax where possible
- Focus on the core solution, skip tangential suggestions
- Assume familiarity with language idioms and patterns
- Let code speak for itself through clear naming and structure
- Avoid over-explaining standard patterns and conventions
- Provide just enough context to understand the solution
- Trust the developer to handle obvious cases independently
