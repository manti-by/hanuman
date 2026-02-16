check:
	git add .
	uv run ty check
	uv run pre-commit run

pip:
	uv sync --all-extras --dev

update:
	uv run uv-bump
	uv sync --all-extras --dev
	uv run pre-commit autoupdate

test:
	uv run pytest tests/

ci: pip check test

index:
	uv run main.py index data/example.txt

search:
	uv run main.py search
