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
from evidencerag.config import get_settings
from evidencerag.db import engine, get_session
from evidencerag.models import Document, Feedback, KnowledgeBase, OutboxEvent, QueryTrace
from evidencerag.retrieval import retrieve
from evidencerag.schemas import (
    Citation,
    DocumentCreate,
    DocumentRead,
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
    description="Citation-first RAG with asynchronous ingestion and traceable answers.",
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


@app.post(
    "/api/v1/knowledge-bases",
    response_model=KnowledgeBaseRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_knowledge_base(payload: KnowledgeBaseCreate, session: SessionDep) -> KnowledgeBase:
    knowledge_base = KnowledgeBase(**payload.model_dump())
    session.add(knowledge_base)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail="knowledge base slug already exists") from exc
    await session.refresh(knowledge_base)
    return knowledge_base


@app.get("/api/v1/knowledge-bases", response_model=list[KnowledgeBaseRead])
async def list_knowledge_bases(session: SessionDep) -> list[KnowledgeBase]:
    return list((await session.scalars(select(KnowledgeBase).order_by(KnowledgeBase.name))).all())


@app.post(
    "/api/v1/knowledge-bases/{knowledge_base_id}/documents",
    response_model=DocumentRead,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_document(
    knowledge_base_id: uuid.UUID,
    payload: DocumentCreate,
    session: SessionDep,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> Document:
    if await session.get(KnowledgeBase, knowledge_base_id) is None:
        raise HTTPException(status_code=404, detail="knowledge base not found")

    source_key = idempotency_key or payload.source_key
    checksum = hashlib.sha256(payload.content.encode()).hexdigest()
    existing = await session.scalar(
        select(Document).where(
            Document.knowledge_base_id == knowledge_base_id,
            Document.source_key == source_key,
        )
    )
    if existing is not None:
        if existing.checksum != checksum:
            raise HTTPException(status_code=409, detail="source key was reused with new content")
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
            payload={"document_id": str(document.id)},
        )
    )
    await session.commit()
    await session.refresh(document)
    DOCUMENT_COUNTER.inc()
    return document


@app.get("/api/v1/documents/{document_id}", response_model=DocumentRead)
async def get_document(document_id: uuid.UUID, session: SessionDep) -> Document:
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    return document


@app.post("/api/v1/knowledge-bases/{knowledge_base_id}/query", response_model=QueryResponse)
async def query_knowledge_base(
    knowledge_base_id: uuid.UUID,
    payload: QueryRequest,
    session: SessionDep,
) -> QueryResponse:
    if await session.get(KnowledgeBase, knowledge_base_id) is None:
        raise HTTPException(status_code=404, detail="knowledge base not found")

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
    answer = await provider.answer(payload.question, [item.text for item in retrieved])
    latency_ms = int((time.perf_counter() - started) * 1_000)
    citations = [
        Citation(
            rank=index,
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
) -> Feedback:
    if payload.rating == 0:
        raise HTTPException(status_code=422, detail="rating must be -1 or 1")
    if await session.get(QueryTrace, trace_id) is None:
        raise HTTPException(status_code=404, detail="trace not found")
    feedback = Feedback(trace_id=trace_id, **payload.model_dump())
    session.add(feedback)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail="feedback already exists") from exc
    await session.refresh(feedback)
    return feedback


@app.get("/api/v1/knowledge-bases/{knowledge_base_id}/stats")
async def knowledge_base_stats(knowledge_base_id: uuid.UUID, session: SessionDep) -> dict[str, int]:
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
