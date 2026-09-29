import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from evidencerag.models import DocumentStatus


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=80)


class KnowledgeBaseRead(KnowledgeBaseCreate):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    owner_id: str
    created_at: datetime


class DocumentCreate(BaseModel):
    source_key: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=240)
    content: str = Field(min_length=40, max_length=500_000)


class DocumentRevision(BaseModel):
    title: str = Field(min_length=1, max_length=240)
    content: str = Field(min_length=40, max_length=500_000)


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    knowledge_base_id: uuid.UUID
    source_key: str
    title: str
    checksum: str
    generation: int
    indexed_generation: int | None
    status: DocumentStatus
    error: str | None
    created_at: datetime
    processed_at: datetime | None


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=10)


class Citation(BaseModel):
    rank: int
    generation: int
    document_id: uuid.UUID
    title: str
    source_key: str
    chunk_ordinal: int
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    trace_id: uuid.UUID
    answer: str
    citations: list[Citation]
    provider: str
    latency_ms: int


class FeedbackCreate(BaseModel):
    rating: int = Field(ge=-1, le=1)
    comment: str | None = Field(default=None, max_length=1_000)


class FeedbackRead(FeedbackCreate):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    trace_id: uuid.UUID
    created_at: datetime
