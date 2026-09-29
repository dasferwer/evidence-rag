import hashlib
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from evidencerag.ai import PROMPT_VERSION, AIProvider
from evidencerag.auth import OwnerId
from evidencerag.config import get_settings
from evidencerag.db import engine, get_session
from evidencerag.models import (
    Document,
    DocumentStatus,
    Feedback,
    KnowledgeBase,
    OutboxEvent,
    QueryTrace,
    utcnow,
)
from evidencerag.retrieval import retrieve
from evidencerag.schemas import (
    Citation,
    DocumentCreate,
    DocumentRead,
    DocumentRevision,
    FeedbackCreate,
    FeedbackRead,
    KnowledgeBaseCreate,
    KnowledgeBaseRead,
    QueryRequest,
    QueryResponse,
)

settings = get_settings()
provider = AIProvider(settings)
SessionDep = Annotated[AsyncSession, Depends(get_session)]
QUERY_COUNTER = Counter("rag_queries_total", "RAG queries", ["provider"])
QUERY_LATENCY = Histogram("rag_query_seconds", "RAG query latency")
DOCUMENT_COUNTER = Counter("rag_documents_total", "Documents accepted")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    await engine.dispose()


app = FastAPI(
    title="EvidenceRAG API",
    version="0.1.0",
    description="Ответы по базе знаний с цитатами и проверяемой историей запросов.",
    lifespan=lifespan,
)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", tags=["system"])
async def ready(session: SessionDep) -> dict[str, str]:
    await session.execute(text("SELECT 1"))
    return {"status": "ready"}


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def owned_kb(
    session: AsyncSession, knowledge_base_id: uuid.UUID, owner_id: str
) -> KnowledgeBase:
    knowledge_base = await session.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.id == knowledge_base_id, KnowledgeBase.owner_id == owner_id
        )
    )
    if knowledge_base is None:
        raise HTTPException(status_code=404, detail="База знаний не найдена")
    return knowledge_base


async def owned_document(session: AsyncSession, document_id: uuid.UUID, owner_id: str) -> Document:
    document = await session.scalar(
        select(Document)
        .join(KnowledgeBase)
        .where(Document.id == document_id, KnowledgeBase.owner_id == owner_id)
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Документ не найден")
    return document


def fingerprint(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


@app.post(
    "/api/v1/knowledge-bases",
    response_model=KnowledgeBaseRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_knowledge_base(
    payload: KnowledgeBaseCreate, session: SessionDep, owner_id: OwnerId
) -> KnowledgeBase:
    knowledge_base = KnowledgeBase(**payload.model_dump(), owner_id=owner_id)
    session.add(knowledge_base)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail="Адрес базы знаний уже занят") from exc
    await session.refresh(knowledge_base)
    return knowledge_base


@app.get("/api/v1/knowledge-bases", response_model=list[KnowledgeBaseRead])
async def list_knowledge_bases(session: SessionDep, owner_id: OwnerId) -> list[KnowledgeBase]:
    return list(
        (
            await session.scalars(
                select(KnowledgeBase)
                .where(KnowledgeBase.owner_id == owner_id)
                .order_by(KnowledgeBase.name)
            )
        ).all()
    )


@app.post(
    "/api/v1/knowledge-bases/{knowledge_base_id}/documents",
    response_model=DocumentRead,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_document(
    knowledge_base_id: uuid.UUID,
    payload: DocumentCreate,
    session: SessionDep,
    owner_id: OwnerId,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", max_length=160),
) -> Document:
    await owned_kb(session, knowledge_base_id, owner_id)
    source_key = idempotency_key or payload.source_key
    if len(source_key) < 1:
        raise HTTPException(status_code=400, detail="Пустой ключ источника")
    await session.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
        {"key": f"document:{knowledge_base_id}:{source_key}"},
    )
    checksum = fingerprint(payload.content)
    existing = await session.scalar(
        select(Document).where(
            Document.knowledge_base_id == knowledge_base_id,
            Document.source_key == source_key,
        )
    )
    if existing is not None:
        if existing.checksum != checksum or existing.title != payload.title:
            raise HTTPException(
                status_code=409, detail="Ключ источника уже использован с другими данными"
            )
        return existing

    document = Document(
        knowledge_base_id=knowledge_base_id,
        source_key=source_key,
        title=payload.title,
        content=payload.content,
        checksum=checksum,
    )
    session.add(document)
    await session.flush()
    session.add(
        OutboxEvent(
            aggregate_id=document.id,
            event_type="document.ingest.requested",
            payload={"document_id": str(document.id), "generation": document.generation},
        )
    )
    await session.commit()
    await session.refresh(document)
    DOCUMENT_COUNTER.inc()
    return document


@app.get("/api/v1/documents/{document_id}", response_model=DocumentRead)
async def get_document(document_id: uuid.UUID, session: SessionDep, owner_id: OwnerId) -> Document:
    return await owned_document(session, document_id, owner_id)


@app.put("/api/v1/documents/{document_id}", response_model=DocumentRead)
async def revise_document(
    document_id: uuid.UUID,
    payload: DocumentRevision,
    session: SessionDep,
    owner_id: OwnerId,
    if_match: str | None = Header(default=None, alias="If-Match"),
) -> Document:
    if if_match is None:
        raise HTTPException(status_code=428, detail="Передайте If-Match с поколением документа")
    if (
        len(if_match) > 20
        or not if_match.startswith('"')
        or not if_match.endswith('"')
        or not if_match[1:-1].isascii()
        or not if_match[1:-1].isdigit()
        or int(if_match[1:-1]) < 1
    ):
        raise HTTPException(status_code=400, detail='Ожидается If-Match: "1"')
    document = await owned_document(session, document_id, owner_id)
    await session.refresh(document, with_for_update=True)
    if document.generation != int(if_match[1:-1]):
        raise HTTPException(status_code=412, detail="Документ уже изменён")
    document.title, document.content = payload.title, payload.content
    document.checksum = fingerprint(payload.content)
    document.generation += 1
    document.status = DocumentStatus.queued
    document.indexed_generation = None
    document.lease_token = document.lease_until = None
    document.attempts = 0
    document.next_attempt_at = utcnow()
    document.error = None
    session.add(
        OutboxEvent(
            aggregate_id=document.id,
            event_type="document.ingest.requested",
            payload={"document_id": str(document.id), "generation": document.generation},
        )
    )
    await session.commit()
    await session.refresh(document)
    return document


@app.post("/api/v1/knowledge-bases/{knowledge_base_id}/query", response_model=QueryResponse)
async def query_knowledge_base(
    knowledge_base_id: uuid.UUID,
    payload: QueryRequest,
    session: SessionDep,
    owner_id: OwnerId,
) -> QueryResponse:
    await owned_kb(session, knowledge_base_id, owner_id)

    started = time.perf_counter()
    query_vector = (await provider.embed([payload.question]))[0]
    retrieved = await retrieve(
        session,
        knowledge_base_id=knowledge_base_id,
        query=payload.question,
        query_vector=query_vector,
        candidates=settings.retrieval_candidates,
        top_k=min(payload.top_k, settings.top_k_max),
    )
    await session.commit()
    answer = await provider.answer(payload.question, [item.text for item in retrieved])
    versions = {item.document_id: item.generation for item in retrieved}
    if versions:
        current = {
            document.id: document
            for document in (
                await session.scalars(
                    select(Document)
                    .where(Document.id.in_(versions))
                    .order_by(Document.id)
                    .with_for_update()
                )
            ).all()
        }
        if any(
            document_id not in current
            or current[document_id].generation != generation
            or current[document_id].status != DocumentStatus.ready
            or current[document_id].indexed_generation != generation
            for document_id, generation in versions.items()
        ):
            raise HTTPException(status_code=409, detail="Источники изменились во время ответа")
    latency_ms = int((time.perf_counter() - started) * 1_000)
    citations = [
        Citation(
            rank=index,
            generation=item.generation,
            document_id=item.document_id,
            title=item.title,
            source_key=item.source_key,
            chunk_ordinal=item.ordinal,
            score=item.score,
            excerpt=item.text[:500],
        )
        for index, item in enumerate(retrieved, start=1)
    ]
    trace = QueryTrace(
        knowledge_base_id=knowledge_base_id,
        query=payload.question,
        answer=answer,
        provider=provider.name,
        prompt_version=PROMPT_VERSION,
        latency_ms=latency_ms,
        retrieved=[item.model_dump(mode="json") for item in citations],
    )
    session.add(trace)
    await session.commit()
    await session.refresh(trace)
    QUERY_COUNTER.labels(provider=provider.name).inc()
    QUERY_LATENCY.observe(latency_ms / 1_000)
    return QueryResponse(
        trace_id=trace.id,
        answer=answer,
        citations=citations,
        provider=provider.name,
        latency_ms=latency_ms,
    )


@app.post(
    "/api/v1/traces/{trace_id}/feedback",
    response_model=FeedbackRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_feedback(
    trace_id: uuid.UUID,
    payload: FeedbackCreate,
    session: SessionDep,
    owner_id: OwnerId,
) -> Feedback:
    if payload.rating == 0:
        raise HTTPException(status_code=422, detail="Оценка должна быть -1 или 1")
    trace = await session.scalar(
        select(QueryTrace)
        .join(KnowledgeBase)
        .where(QueryTrace.id == trace_id, KnowledgeBase.owner_id == owner_id)
    )
    if trace is None:
        raise HTTPException(status_code=404, detail="Трасса не найдена")
    feedback = Feedback(trace_id=trace_id, **payload.model_dump())
    session.add(feedback)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail="Оценка уже сохранена") from exc
    await session.refresh(feedback)
    return feedback


@app.get("/api/v1/knowledge-bases/{knowledge_base_id}/stats")
async def knowledge_base_stats(
    knowledge_base_id: uuid.UUID, session: SessionDep, owner_id: OwnerId
) -> dict[str, int]:
    await owned_kb(session, knowledge_base_id, owner_id)
    document_count = await session.scalar(
        select(func.count())
        .select_from(Document)
        .where(Document.knowledge_base_id == knowledge_base_id)
    )
    query_count = await session.scalar(
        select(func.count())
        .select_from(QueryTrace)
        .where(QueryTrace.knowledge_base_id == knowledge_base_id)
    )
    return {"documents": document_count or 0, "queries": query_count or 0}
