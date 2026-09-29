import re
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from evidencerag.models import Chunk, Document, DocumentStatus

WORD_RE = re.compile(r"[\w-]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    document_id: UUID
    generation: int
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
    lexical = func.ts_rank_cd(
        func.to_tsvector("simple", Chunk.text), func.plainto_tsquery("simple", query)
    ).label("lexical_rank")
    base = (
        select(Chunk, Document, distance)
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Chunk.knowledge_base_id == knowledge_base_id,
            Document.status == DocumentStatus.ready,
            Chunk.generation == Document.generation,
            Document.indexed_generation == Document.generation,
        )
    )
    vector_rows = (await session.execute(base.order_by(distance).limit(candidates))).all()
    lexical_rows = (
        await session.execute(
            base.where(
                func.to_tsvector("simple", Chunk.text).op("@@")(
                    func.plainto_tsquery("simple", query)
                )
            )
            .order_by(lexical.desc(), Chunk.id)
            .limit(candidates)
        )
    ).all()
    merged = {
        chunk.id: (chunk, document, vector_distance)
        for chunk, document, vector_distance in [*vector_rows, *lexical_rows]
    }
    reranked = [
        RetrievedChunk(
            document_id=document.id,
            generation=document.generation,
            title=document.title,
            source_key=document.source_key,
            ordinal=chunk.ordinal,
            text=chunk.text,
            score=combined_score(query, chunk.text, float(vector_distance)),
        )
        for chunk, document, vector_distance in merged.values()
    ]
    return sorted(reranked, key=lambda item: (-item.score, str(item.document_id), item.ordinal))[
        :top_k
    ]
