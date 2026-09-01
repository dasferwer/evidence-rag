# EvidenceRAG

EvidenceRAG is a citation-first knowledge assistant built as an executable
portfolio project. It demonstrates an end-to-end RAG pipeline rather than a
single LLM call: durable asynchronous ingestion, vector retrieval, lexical
reranking, grounded answers, evaluation feedback and observability.

This repository demonstrates learned LLM/RAG engineering practices. It does
not claim commercial model-development experience and does not train a model
from scratch. The application integrates an OpenAI-compatible provider and has
a deterministic offline provider for tests and local demos.

## Highlights

- FastAPI and async SQLAlchemy with PostgreSQL/pgvector;
- transactional outbox, RabbitMQ publisher confirms, idempotent worker and DLQ;
- overlap-aware chunking and batched embeddings;
- cosine retrieval with lexical reranking;
- prompt-injection boundary and mandatory source citations;
- prompt/provider/latency/evidence traces and answer feedback;
- Prometheus metrics, Alembic migrations, Docker health checks and CI.

## Run

```bash
cp .env.example .env
docker compose up --build -d
python scripts/smoke.py
```

Open <http://localhost:8081/docs>. The default `local` provider requires no
secret. Set `LLM_PROVIDER=openai` and `OPENAI_API_KEY` to use an
OpenAI-compatible API.

See [docs/architecture.md](docs/architecture.md) for consistency guarantees,
retrieval design and failure scenarios.

