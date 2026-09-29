import asyncio
import hashlib

from sqlalchemy import select

from evidencerag.db import session_factory
from evidencerag.models import Document, KnowledgeBase, OutboxEvent

CONTENT = """Критический инцидент нужно подтвердить в течение пятнадцати минут.

Ответственный открывает общий канал, назначает исполнителя и публикует обновления
каждые тридцать минут. Если инцидент вызван недавним релизом, первым действием
служит откат изменений. Клиентам нельзя сообщать неподтверждённые сроки восстановления."""


async def seed() -> None:
    async with session_factory() as session:
        knowledge_base = await session.scalar(
            select(KnowledgeBase).where(
                KnowledgeBase.slug == "operations-demo", KnowledgeBase.owner_id == "demo"
            )
        )
        if knowledge_base is None:
            knowledge_base = KnowledgeBase(
                name="Демонстрационная база", slug="operations-demo", owner_id="demo"
            )
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
                title="Порядок реагирования на инциденты",
                content=CONTENT,
                checksum=hashlib.sha256(CONTENT.encode()).hexdigest(),
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
        print(f"knowledge_base_id={knowledge_base.id} document_id={document.id}")


if __name__ == "__main__":
    asyncio.run(seed())
