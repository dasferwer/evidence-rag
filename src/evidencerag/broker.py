import aio_pika
from aio_pika.abc import AbstractChannel, AbstractExchange, AbstractRobustConnection

from evidencerag.config import get_settings

EXCHANGE_NAME = "rag.events"
DLX_NAME = "rag.dlx"
QUEUE_NAME = "rag.document.ingest"
DLQ_NAME = "rag.document.ingest.dlq"
ROUTING_KEY = "document.ingest"


async def connect() -> AbstractRobustConnection:
    return await aio_pika.connect_robust(get_settings().rabbitmq_url)


async def declare_topology(
    channel: AbstractChannel,
) -> tuple[AbstractExchange, aio_pika.abc.AbstractQueue]:
    exchange = await channel.declare_exchange(
        EXCHANGE_NAME, aio_pika.ExchangeType.TOPIC, durable=True
    )
    dlx = await channel.declare_exchange(DLX_NAME, aio_pika.ExchangeType.TOPIC, durable=True)
    queue = await channel.declare_queue(
        QUEUE_NAME,
        durable=True,
        arguments={"x-dead-letter-exchange": DLX_NAME, "x-dead-letter-routing-key": ROUTING_KEY},
    )
    await queue.bind(exchange, ROUTING_KEY)
    dlq = await channel.declare_queue(DLQ_NAME, durable=True)
    await dlq.bind(dlx, ROUTING_KEY)
    return exchange, queue
