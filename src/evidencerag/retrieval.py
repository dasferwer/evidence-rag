import re
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evidencerag.models import Chunk, Document, DocumentStatus

WORD_RE = re.compile(r"[\w-]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    document_id: UUID
    title: str
    source_key: str
    ordinal: int
    text: str
    score: float


def lexical_overlap(query: str, text: str) -> float:
    query_terms = set(WORD_RE.findall(query.casefold()))
    text_terms = set(WORD_RE.findall(text.casefold()))
    if not query_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)


def combined_score(query: str, text: str, distance: float) -> float:
    vector_score = max(0.0, 1.0 - distance)
    return round(0.75 * vector_score + 0.25 * lexical_overlap(query, text), 6)


async def retrieve(
    session: AsyncSession,
    *,
    knowledge_base_id: UUID,
    query: str,
    query_vector: list[float],
    candidates: int,
    top_k: int,
) -> list[RetrievedChunk]:
    distance = Chunk.embedding.cosine_distance(query_vector).label("distance")
    statement = (
        select(Chunk, Document, distance)
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Chunk.knowledge_base_id == knowledge_base_id,
            Document.status == DocumentStatus.ready,
        )
        .order_by(distance)
        .limit(candidates)
    )
    rows = (await session.execute(statement)).all()
    reranked = [
        RetrievedChunk(
            document_id=document.id,
            title=document.title,
            source_key=document.source_key,
            ordinal=chunk.ordinal,
            text=chunk.text,
            score=combined_score(query, chunk.text, float(vector_distance)),
        )
        for chunk, document, vector_distance in rows
    ]
    return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]
