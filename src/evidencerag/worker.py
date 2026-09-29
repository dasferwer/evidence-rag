import asyncio
import json
import logging
import uuid
from datetime import timedelta

from aio_pika.abc import AbstractIncomingMessage
from sqlalchemy import delete, select

from evidencerag.ai import AIProvider
from evidencerag.broker import connect, declare_topology
from evidencerag.chunking import chunk_text
from evidencerag.config import get_settings
from evidencerag.db import session_factory
from evidencerag.models import Chunk, Document, DocumentStatus, OutboxEvent, utcnow

settings = get_settings()
provider = AIProvider(settings)
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


async def record_failure(
    document_id: uuid.UUID, generation: int, token: uuid.UUID, exc: Exception
) -> None:
    async with session_factory.begin() as session:
        document = await session.scalar(
            select(Document).where(Document.id == document_id).with_for_update()
        )
        if document is None or document.generation != generation or document.lease_token != token:
            return
        document.error = type(exc).__name__
        document.lease_token = document.lease_until = None
        if document.attempts >= settings.max_ingestion_attempts:
            document.status = DocumentStatus.failed
        else:
            document.status = DocumentStatus.queued
            document.next_attempt_at = utcnow() + timedelta(seconds=min(300, 2**document.attempts))


async def ingest_document(document_id: uuid.UUID) -> bool:
    async with session_factory() as session:
        document = await session.get(Document, document_id)
        if document is None:
            return False
        document = await session.scalar(
            select(Document)
            .where(Document.id == document_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        assert document is not None
        now = utcnow()
        if document.status in {DocumentStatus.ready, DocumentStatus.failed}:
            return False
        if document.lease_until is not None and document.lease_until > now:
            return False
        if document.next_attempt_at > now:
            return False
        generation, content = document.generation, document.content
        token = uuid.uuid4()
        document.lease_token = token
        document.lease_until = now + timedelta(seconds=settings.ingestion_lease_seconds)
        document.status = DocumentStatus.processing
        document.attempts += 1
        await session.commit()
        try:
            text_chunks = chunk_text(content)
            vectors = await provider.embed([item.text for item in text_chunks])
            if len(vectors) != len(text_chunks):
                raise ValueError("Количество векторов не совпадает с числом фрагментов")
        except Exception as exc:
            await record_failure(document_id, generation, token, exc)
            return False
        try:
            document = await session.scalar(
                select(Document)
                .where(Document.id == document_id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            if (
                document is None
                or document.generation != generation
                or document.lease_token != token
            ):
                await session.rollback()
                return False
            await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
            session.add_all(
                [
                    Chunk(
                        document_id=document_id,
                        knowledge_base_id=document.knowledge_base_id,
                        generation=generation,
                        ordinal=item.ordinal,
                        text=item.text,
                        token_count=item.token_count,
                        embedding=vector,
                    )
                    for item, vector in zip(text_chunks, vectors, strict=True)
                ]
            )
            document.status = DocumentStatus.ready
            document.indexed_generation = generation
            document.processed_at = utcnow()
            document.lease_token = document.lease_until = None
            document.error = None
            await session.commit()
            return True
        except Exception as exc:
            await session.rollback()
            await record_failure(document_id, generation, token, exc)
            return False


async def process_next_document() -> bool:
    async with session_factory() as session:
        now = utcnow()
        document_id = await session.scalar(
            select(Document.id)
            .where(
                Document.status.in_([DocumentStatus.queued, DocumentStatus.processing]),
                Document.next_attempt_at <= now,
                (Document.lease_until.is_(None) | (Document.lease_until <= now)),
            )
            .order_by(Document.next_attempt_at, Document.created_at)
            .limit(1)
        )
    if document_id is None:
        return False
    return await ingest_document(document_id)


async def handle_message(message: AbstractIncomingMessage) -> None:
    try:
        payload = json.loads(message.body)
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("document_id"), str)
            or not isinstance(payload.get("generation"), int)
            or isinstance(payload.get("generation"), bool)
        ):
            raise ValueError("Некорректное уведомление")
        uuid.UUID(payload["document_id"])
        if message.message_id is None:
            raise ValueError("Нет ID события")
        event_id = uuid.UUID(message.message_id)
        async with session_factory() as session:
            event = await session.get(OutboxEvent, event_id)
            if (
                event is None
                or event.event_type != "document.ingest.requested"
                or event.payload != payload
            ):
                raise ValueError("Уведомление не соответствует outbox")
    except (ValueError, TypeError, KeyError):
        await message.reject(requeue=False)
        return
    await message.ack()


async def run() -> None:
    while True:
        connection = None
        try:
            await process_next_document()
            connection = await connect()
            channel = await connection.channel(publisher_confirms=True)
            await channel.set_qos(prefetch_count=4)
            _, queue = await declare_topology(channel)
            await queue.consume(handle_message)
            while True:
                worked = await process_next_document()
                await asyncio.sleep(0.1 if worked else 1)
        except Exception as exc:
            logger.warning("ingest_retry kind=%s", type(exc).__name__)
            await asyncio.sleep(3)
        finally:
            if connection is not None:
                await connection.close()


if __name__ == "__main__":
    asyncio.run(run())
