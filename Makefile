check:
	git add .
	uv run ty check
	uv run pre-commit run

install:
	uv sync --all-extras --dev

update:
	uv run uv-bump
	uv sync --all-extras --dev
	uv run pre-commit autoupdate

test:
	uv run pytest tests/

ci: install check test

prepare:
	uv run main.py prepare

index:
	uv run main.py index

search:
	uv run main.py search

truncate:
	psql -d hanuman -c "TRUNCATE TABLE langchain_pg_embedding;"

clean:
	rm -rf data/chunks/*
	rm -rf data/markdown/*

cleanup: truncate clean
