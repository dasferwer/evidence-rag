.PHONY: sync format lint type test check up down smoke seed

sync:
	uv sync --extra dev

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff format --check .
	uv run ruff check .

type:
	uv run mypy src

test:
	uv run pytest

check: lint type test
	docker compose config --quiet

up:
	docker compose up --build -d

down:
	docker compose down

smoke:
	python scripts/smoke.py

seed:
	docker compose exec api python -m evidencerag.seed
