# EvidenceRAG

Платформа ответов по внутренней базе знаний с проверяемыми цитатами,
асинхронной индексацией и трассировкой каждого запроса. Это не обёртка над
одним вызовом LLM: проект показывает полный RAG-конвейер, согласованную
доставку заданий, гибридный поиск, защиту от инструкций внутри документов и
контур оценки качества.

> English overview: a citation-first RAG backend built with FastAPI,
> PostgreSQL/pgvector and RabbitMQ. It includes transactional ingestion,
> vector retrieval with lexical reranking, provider abstraction, prompt
> versioning, feedback and Prometheus metrics.

## Статус проекта

Проект создан как самостоятельная портфолио-работа для демонстрации изученных
LLM/RAG-подходов. Он не заявляется как коммерческий опыт разработки моделей.
Модель не обучается с нуля: приложение интегрирует готовый OpenAI-совместимый
API либо использует воспроизводимый локальный режим.

## Что реализовано

- базы знаний, документы и идемпотентная загрузка по `source_key`;
- transactional outbox между PostgreSQL и RabbitMQ;
- отдельные dispatcher и ingestion worker, publisher confirms и DLQ;
- чанкинг с перекрытием и пакетное построение embeddings;
- поиск кандидатов через pgvector и лексический reranking;
- ответы только по найденным данным с обязательными цитатами;
- защита промпта: инструкции из документов считаются недоверенными данными;
- локальный deterministic provider без ключей и OpenAI-compatible provider;
- query tracing: версия промпта, evidence, latency и используемый provider;
- пользовательская оценка ответа и Prometheus-метрики;
- Alembic, Docker health checks, тесты и end-to-end smoke-сценарий.

## Стек

Python 3.13, FastAPI, SQLAlchemy 2 async, PostgreSQL 17, pgvector, Alembic,
RabbitMQ, aio-pika, Pydantic 2, HTTPX, Prometheus client, pytest, Ruff, mypy,
Docker Compose.

## Быстрый запуск

```bash
cp .env.example .env
docker compose up --build -d
docker compose ps
python scripts/smoke.py
```

Опциональные демонстрационные данные:

```bash
docker compose exec api python -m evidencerag.seed
```

После запуска:

- Swagger UI: <http://localhost:8081/docs>
- health check: <http://localhost:8081/health>
- Prometheus metrics: <http://localhost:8081/metrics>
- RabbitMQ UI: <http://localhost:15688> (`guest` / `guest`)

По умолчанию используется `LLM_PROVIDER=local`, поэтому проект полностью
работает без внешних ключей. Для реального OpenAI-совместимого endpoint нужно
заполнить `OPENAI_API_KEY` и установить `LLM_PROVIDER=openai`.

Остановка:

```bash
docker compose down
```

## Quality gate

```bash
docker compose --profile test run --rm test
docker compose config --quiet
```

Локально через uv:

```bash
uv sync --extra dev
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

## Основные маршруты

| Метод | Маршрут | Назначение |
|---|---|---|
| `POST` | `/api/v1/knowledge-bases` | создать базу знаний |
| `POST` | `/api/v1/knowledge-bases/{id}/documents` | принять документ и outbox event |
| `GET` | `/api/v1/documents/{id}` | проверить статус индексации |
| `POST` | `/api/v1/knowledge-bases/{id}/query` | получить ответ и citations |
| `POST` | `/api/v1/traces/{id}/feedback` | сохранить оценку ответа |
| `GET` | `/api/v1/knowledge-bases/{id}/stats` | получить агрегаты базы |

Архитектура, модель согласованности и отказные сценарии описаны в
[docs/architecture.md](docs/architecture.md). English documentation is
available in [README.en.md](README.en.md).

## Repository map

```text
src/evidencerag/       API, retrieval, providers, dispatcher, worker and seed
migrations/            PostgreSQL and pgvector schema
tests/                 deterministic unit tests
scripts/smoke.py       real asynchronous end-to-end scenario
docs/architecture.md   diagrams, consistency and failure analysis
docker-compose.yml     database, RabbitMQ, API, dispatcher and worker
```
