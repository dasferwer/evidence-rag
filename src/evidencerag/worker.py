import asyncio
import json
import logging
import uuid

from aio_pika.abc import AbstractIncomingMessage
from sqlalchemy import delete, select

from evidencerag.ai import AIProvider
from evidencerag.broker import connect, declare_topology
from evidencerag.chunking import chunk_text
from evidencerag.config import get_settings
from evidencerag.db import session_factory
from evidencerag.models import Chunk, Document, DocumentStatus, utcnow

settings = get_settings()
provider = AIProvider(settings)
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


async def ingest_document(document_id: uuid.UUID) -> None:
    async with session_factory() as session:
        document = await session.scalar(
            select(Document).where(Document.id == document_id).with_for_update()
        )
        if document is None:
            logger.warning("document %s no longer exists", document_id)
            return
        if document.status == DocumentStatus.ready:
            return
        document.status = DocumentStatus.processing
        document.error = None
        await session.commit()

        try:
            text_chunks = chunk_text(document.content)
            vectors = await provider.embed([item.text for item in text_chunks])
            await session.execute(delete(Chunk).where(Chunk.document_id == document.id))
            session.add_all(
                [
                    Chunk(
                        document_id=document.id,
                        knowledge_base_id=document.knowledge_base_id,
                        ordinal=item.ordinal,
                        text=item.text,
                        token_count=item.token_count,
                        embedding=vector,
                    )
                    for item, vector in zip(text_chunks, vectors, strict=True)
                ]
            )
            document.status = DocumentStatus.ready
            document.processed_at = utcnow()
            await session.commit()
        except Exception as exc:
            await session.rollback()
            document = await session.get(Document, document_id)
            if document is not None:
                document.status = DocumentStatus.failed
                document.error = str(exc)[:2_000]
                await session.commit()
            raise


async def handle_message(message: AbstractIncomingMessage) -> None:
    async with message.process(requeue=False):
        payload = json.loads(message.body)
        document_id = uuid.UUID(payload["document_id"])
        await ingest_document(document_id)
        logger.info("document %s indexed", document_id)


async def run() -> None:
    connection = await connect()
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=4)
    _, queue = await declare_topology(channel)
    logger.info("ingestion worker started")
    await queue.consume(handle_message)
    try:
        await asyncio.Future()
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(run())
