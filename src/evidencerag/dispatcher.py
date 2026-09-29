import asyncio
import json
import logging

import aio_pika
from sqlalchemy import select

from evidencerag.broker import ROUTING_KEY, connect, declare_topology
from evidencerag.config import get_settings
from evidencerag.db import session_factory
from evidencerag.models import OutboxEvent, utcnow

logging.basicConfig(level=get_settings().log_level)
logger = logging.getLogger(__name__)


async def dispatch_batch() -> int:
    connection = await connect()
    try:
        channel = await connection.channel(publisher_confirms=True, on_return_raises=True)
        exchange, _ = await declare_topology(channel)
        async with session_factory() as session, session.begin():
            statement = (
                select(OutboxEvent)
                .where(
                    OutboxEvent.published_at.is_(None),
                    OutboxEvent.available_at <= utcnow(),
                )
                .order_by(OutboxEvent.created_at)
                .limit(50)
                .with_for_update(skip_locked=True)
            )
            events = list((await session.scalars(statement)).all())
            for event in events:
                body = json.dumps(event.payload).encode()
                await exchange.publish(
                    aio_pika.Message(
                        body=body,
                        message_id=str(event.id),
                        content_type="application/json",
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    ),
                    routing_key=ROUTING_KEY,
                    mandatory=True,
                )
                event.published_at = utcnow()
                event.attempts += 1
            return len(events)
    finally:
        await connection.close()


async def run() -> None:
    settings = get_settings()
    logger.info("outbox dispatcher started")
    while True:
        try:
            count = await dispatch_batch()
            if count:
                logger.info("published %s outbox events", count)
        except Exception:
            logger.exception("outbox dispatch failed")
        await asyncio.sleep(settings.outbox_poll_seconds)


if __name__ == "__main__":
    asyncio.run(run())
