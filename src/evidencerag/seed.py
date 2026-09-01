import asyncio
import hashlib

from sqlalchemy import select

from evidencerag.db import session_factory
from evidencerag.models import Document, KnowledgeBase, OutboxEvent

CONTENT = """Critical incidents have a fifteen minute acknowledgement SLA.

The incident commander opens a shared channel, appoints an owner and publishes
updates every thirty minutes. If a recent deployment caused the incident,
rollback is the preferred first mitigation. Customer communication must not
contain unverified recovery estimates."""


async def seed() -> None:
    async with session_factory() as session:
        knowledge_base = await session.scalar(
            select(KnowledgeBase).where(KnowledgeBase.slug == "operations-demo")
        )
        if knowledge_base is None:
            knowledge_base = KnowledgeBase(name="Operations handbook", slug="operations-demo")
            session.add(knowledge_base)
            await session.flush()

        document = await session.scalar(
            select(Document).where(
                Document.knowledge_base_id == knowledge_base.id,
                Document.source_key == "incident-response-v1",
            )
        )
        if document is None:
            document = Document(
                knowledge_base_id=knowledge_base.id,
                source_key="incident-response-v1",
                title="Incident response policy",
                content=CONTENT,
                checksum=hashlib.sha256(CONTENT.encode()).hexdigest(),
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
        print(f"knowledge_base_id={knowledge_base.id} document_id={document.id}")


if __name__ == "__main__":
    asyncio.run(seed())
