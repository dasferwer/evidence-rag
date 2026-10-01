import asyncio
from datetime import timedelta
from uuid import UUID, uuid4

import aio_pika
import httpx
import pytest
import pytest_asyncio
from sqlalchemy import select, text, update

from evidencerag.broker import DLQ_NAME, ROUTING_KEY, connect, declare_topology
from evidencerag.config import get_settings
from evidencerag.db import engine, session_factory
from evidencerag.main import app
from evidencerag.models import (
    Chunk,
    Document,
    DocumentStatus,
    KnowledgeBase,
    OutboxEvent,
    QueryTrace,
    utcnow,
)
from evidencerag.retrieval import retrieve
from evidencerag.worker import handle_message, ingest_document

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest_asyncio.fixture(scope="module", loop_scope="session", autouse=True)
async def clean_test_database():
    if engine.url.database != "evidencerag_test":
        pytest.skip("Интеграционные тесты выполняются только в evidencerag_test")
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "TRUNCATE feedback, query_traces, outbox_events, chunks, "
                "documents, knowledge_bases CASCADE"
            )
        )
    yield


@pytest.fixture
def client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


def headers(owner="test"):
    return {
        "Authorization": "Bearer "
        + ("test-client-key-long" if owner == "test" else "other-client-key-long")
    }


async def make_document(client, owner="test", slug=None, source=None, content=None):
    slug = slug or "kb-" + uuid4().hex[:8]
    kb_response = await client.post(
        "/api/v1/knowledge-bases",
        headers=headers(owner),
        json={"name": "База знаний", "slug": slug},
    )
    assert kb_response.status_code == 201, kb_response.text
    kb = kb_response.json()
    response = await client.post(
        f"/api/v1/knowledge-bases/{kb['id']}/documents",
        headers=headers(owner),
        json={
            "source_key": source or uuid4().hex,
            "title": "Порядок действий",
            "content": content
            or "При аварии нужно открыть общий канал и назначить ответственного. " * 2,
        },
    )
    assert response.status_code == 202, response.text
    return kb, response.json()


async def test_owner_isolation_and_idempotency(client):
    async with client:
        kb, document = await make_document(client)
        other = headers("other")
        assert (await client.get("/api/v1/knowledge-bases", headers=other)).json() == []
        assert (
            await client.get(f"/api/v1/documents/{document['id']}", headers=other)
        ).status_code == 404
        assert (
            await client.post(
                f"/api/v1/knowledge-bases/{kb['id']}/query",
                headers=other,
                json={"question": "Что делать при аварии?"},
            )
        ).status_code == 404
        assert (
            await client.get(f"/api/v1/knowledge-bases/{kb['id']}/stats", headers=other)
        ).status_code == 404
        assert (await client.get(f"/api/v1/documents/{document['id']}")).status_code == 401
        same_slug = await client.post(
            "/api/v1/knowledge-bases",
            headers=other,
            json={"name": "Чужая база", "slug": kb["slug"]},
        )
        assert same_slug.status_code == 201
        path = f"/api/v1/knowledge-bases/{kb['id']}/documents"
        data = {
            "source_key": "same-key",
            "title": "Инструкция",
            "content": "Текст инструкции о восстановлении. " * 3,
        }
        responses = await asyncio.gather(
            *[client.post(path, headers=headers(), json=data) for _ in range(6)]
        )
        assert all(response.status_code == 202 for response in responses)
        ids = {response.json()["id"] for response in responses}
        assert len(ids) == 1
        with_changes = {**data, "title": "Другая инструкция"}
        assert (await client.post(path, headers=headers(), json=with_changes)).status_code == 409
        async with session_factory() as session:
            count = await session.scalar(
                select(OutboxEvent.id).where(OutboxEvent.aggregate_id == UUID(next(iter(ids))))
            )
            assert count is not None


async def test_revisions_fence_slow_worker(client, monkeypatch):
    async with client:
        kb, document = await make_document(client)
        started, release = asyncio.Event(), asyncio.Event()
        from evidencerag.worker import provider

        original = provider.embed

        async def slow(texts):
            started.set()
            await release.wait()
            return await original(texts)

        monkeypatch.setattr(provider, "embed", slow)
        task = asyncio.create_task(ingest_document(UUID(document["id"])))
        await asyncio.wait_for(started.wait(), 5)
        path = f"/api/v1/documents/{document['id']}"
        updated = await client.put(
            path,
            headers={**headers(), "If-Match": '"1"'},
            json={
                "title": "Новая редакция",
                "content": "Новая процедура требует согласования у ответственного. " * 2,
            },
        )
        assert updated.status_code == 200 and updated.json()["generation"] == 2
        release.set()
        assert await asyncio.wait_for(task, 5) is False
        monkeypatch.setattr(provider, "embed", original)
        assert (
            await client.put(
                path,
                headers={**headers(), "If-Match": '"1"'},
                json={"title": "Старая редакция", "content": "Устаревший текст. " * 4},
            )
        ).status_code == 412
        assert await ingest_document(UUID(document["id"]))
        async with session_factory() as session:
            rows = list(
                (
                    await session.scalars(
                        select(Chunk).where(Chunk.document_id == UUID(document["id"]))
                    )
                ).all()
            )
            assert rows and all(
                row.generation == 2 and "Новая процедура" in row.text for row in rows
            )
        answer = await client.post(
            f"/api/v1/knowledge-bases/{kb['id']}/query",
            headers=headers(),
            json={"question": "Что требует согласования?"},
        )
        assert answer.status_code == 200 and answer.json()["citations"]
        trace = answer.json()["trace_id"]
        assert (
            await client.post(
                f"/api/v1/traces/{trace}/feedback", headers=headers("other"), json={"rating": 1}
            )
        ).status_code == 404
        assert (
            await client.post(
                f"/api/v1/traces/{trace}/feedback", headers=headers(), json={"rating": 1}
            )
        ).status_code == 201


async def test_failure_budget_and_retry_by_revision(client, monkeypatch):
    async with client:
        _, document = await make_document(client)
        from evidencerag.worker import provider

        async def fail(texts):
            raise RuntimeError("Намеренный сбой провайдера")

        original = provider.embed
        monkeypatch.setattr(provider, "embed", fail)
        for attempt in range(get_settings().max_ingestion_attempts):
            assert await ingest_document(UUID(document["id"])) is False
            async with session_factory.begin() as session:
                stored = await session.get(Document, UUID(document["id"]))
                assert stored.attempts == attempt + 1
                stored.next_attempt_at = utcnow() - timedelta(seconds=1)
        async with session_factory() as session:
            assert (
                await session.get(Document, UUID(document["id"]))
            ).status == DocumentStatus.failed
        monkeypatch.setattr(provider, "embed", original)
        path = f"/api/v1/documents/{document['id']}"
        revised = await client.put(
            path,
            headers={**headers(), "If-Match": '"1"'},
            json={
                "title": "Исправленный материал",
                "content": "Исправленный материал описывает порядок восстановления. " * 2,
            },
        )
        assert revised.status_code == 200
        assert await ingest_document(UUID(document["id"]))


async def test_lexical_candidate_is_in_union(client):
    async with client:
        kb, target = await make_document(
            client, content="СЛОВОПАРОЛЬ описывает поиск нужного материала. " * 2
        )
        other_response = await client.post(
            f"/api/v1/knowledge-bases/{kb['id']}/documents",
            headers=headers(),
            json={
                "source_key": uuid4().hex,
                "title": "Другая тема",
                "content": "Совершенно посторонний текст о погоде. " * 2,
            },
        )
        assert other_response.status_code == 202
        other = other_response.json()
        assert await ingest_document(UUID(target["id"]))
        assert await ingest_document(UUID(other["id"]))
        async with session_factory() as session:
            other_chunk = await session.scalar(
                select(Chunk).where(Chunk.document_id == UUID(other["id"]))
            )
            results = await retrieve(
                session,
                knowledge_base_id=UUID(kb["id"]),
                query="СЛОВОПАРОЛЬ",
                query_vector=other_chunk.embedding,
                candidates=1,
                top_k=2,
            )
            assert any(item.document_id == UUID(target["id"]) for item in results)


async def test_invalid_notification_goes_to_dlq():
    connection = await connect()
    try:
        channel = await connection.channel(publisher_confirms=True)
        exchange, queue = await declare_topology(channel)
        dlq = await channel.declare_queue(DLQ_NAME, durable=True)
        await queue.purge()
        await dlq.purge()
        await exchange.publish(
            aio_pika.Message(body=b"not-json", delivery_mode=aio_pika.DeliveryMode.PERSISTENT),
            routing_key=ROUTING_KEY,
            mandatory=True,
        )
        message = await queue.get(timeout=3)
        assert message is not None
        await handle_message(message)
        async with dlq.iterator() as incoming:
            rejected = await asyncio.wait_for(anext(incoming), timeout=3)
            assert rejected.body == b"not-json"
            await rejected.ack()
    finally:
        await connection.close()


@pytest.mark.parametrize("phase", ["embed", "answer"])
async def test_external_query_calls_release_database_connection(client, monkeypatch, phase):
    from evidencerag.main import provider

    async with client:
        kb, document = await make_document(client)
        assert await ingest_document(UUID(document["id"]))
        started, release = asyncio.Event(), asyncio.Event()
        original = getattr(provider, phase)

        async def slow(*args):
            started.set()
            await release.wait()
            return await original(*args)

        monkeypatch.setattr(provider, phase, slow)
        task = asyncio.create_task(
            client.post(
                f"/api/v1/knowledge-bases/{kb['id']}/query",
                headers=headers(),
                json={"question": "Что делать при аварии?"},
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 5)
            assert engine.pool.checkedout() == 0
        finally:
            release.set()
            response = await asyncio.wait_for(task, 5)
        assert response.status_code == 200, response.text


@pytest.mark.parametrize("phase", ["embed", "answer"])
async def test_query_rechecks_owner_after_external_call(client, monkeypatch, phase):
    from evidencerag.main import provider

    async with client:
        kb, document = await make_document(client)
        assert await ingest_document(UUID(document["id"]))
        started, release = asyncio.Event(), asyncio.Event()
        original = getattr(provider, phase)

        async def slow(*args):
            started.set()
            await release.wait()
            return await original(*args)

        monkeypatch.setattr(provider, phase, slow)
        task = asyncio.create_task(
            client.post(
                f"/api/v1/knowledge-bases/{kb['id']}/query",
                headers=headers(),
                json={"question": "Что делать при аварии?"},
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 5)
            async with session_factory.begin() as session:
                await session.execute(
                    update(KnowledgeBase)
                    .where(KnowledgeBase.id == UUID(kb["id"]))
                    .values(owner_id="other")
                )
        finally:
            release.set()
            response = await asyncio.wait_for(task, 5)
        assert response.status_code == 404, response.text
        async with session_factory() as session:
            assert (
                await session.scalar(
                    select(QueryTrace.id).where(QueryTrace.knowledge_base_id == UUID(kb["id"]))
                )
                is None
            )


async def test_query_rechecks_document_generation_after_answer(client, monkeypatch):
    from evidencerag.main import provider

    async with client:
        kb, document = await make_document(client)
        assert await ingest_document(UUID(document["id"]))
        started, release = asyncio.Event(), asyncio.Event()
        original = provider.answer

        async def slow(*args):
            started.set()
            await release.wait()
            return await original(*args)

        monkeypatch.setattr(provider, "answer", slow)
        task = asyncio.create_task(
            client.post(
                f"/api/v1/knowledge-bases/{kb['id']}/query",
                headers=headers(),
                json={"question": "Что делать при аварии?"},
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 5)
            updated = await client.put(
                f"/api/v1/documents/{document['id']}",
                headers={**headers(), "If-Match": '"1"'},
                json={"title": "Новая редакция", "content": "Новый порядок действий. " * 3},
            )
            assert updated.status_code == 200
            assert await ingest_document(UUID(document["id"]))
        finally:
            release.set()
            response = await asyncio.wait_for(task, 5)
        assert response.status_code == 409, response.text
        async with session_factory() as session:
            assert (
                await session.scalar(
                    select(QueryTrace.id).where(QueryTrace.knowledge_base_id == UUID(kb["id"]))
                )
                is None
            )


async def test_invalid_embedding_batch_never_publishes_partial_chunks(client, monkeypatch):
    from evidencerag import worker
    from evidencerag.ai import AIProvider
    from evidencerag.config import Settings

    invalid_provider = AIProvider(
        Settings(llm_provider="openai", openai_api_key="test-placeholder")
    )

    async def invalid_response(path, payload):
        assert len(payload["input"]) > 1
        data = [
            {"index": index, "embedding": [0.5] * 384} for index in range(len(payload["input"]))
        ]
        data[-1]["index"] = 0
        return {"data": data}

    monkeypatch.setattr(invalid_provider, "_post", invalid_response)
    monkeypatch.setattr(worker, "provider", invalid_provider)
    async with client:
        _, document = await make_document(client, content="Инструкция по восстановлению. " * 500)
        document_id = UUID(document["id"])
        assert await ingest_document(document_id) is False
        async with session_factory() as session:
            assert (
                await session.scalar(select(Chunk.id).where(Chunk.document_id == document_id))
                is None
            )
            stored = await session.get(Document, document_id)
            assert stored.status == DocumentStatus.queued
            assert stored.error == "ValueError"
