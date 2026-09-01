import math

from evidencerag.ai import AIProvider, local_embedding
from evidencerag.config import Settings


def test_local_embedding_is_deterministic_and_normalized() -> None:
    first = local_embedding("SLA breach escalation", 64)
    second = local_embedding("SLA breach escalation", 64)

    assert first == second
    assert math.isclose(math.sqrt(sum(item * item for item in first)), 1.0)


async def test_local_answer_has_citation() -> None:
    provider = AIProvider(Settings())
    answer = await provider.answer("What is the SLA?", ["Critical incidents have a 15 minute SLA."])

    assert "15 minute SLA" in answer
    assert "[1]" in answer


async def test_local_answer_refuses_without_context() -> None:
    provider = AIProvider(Settings())
    answer = await provider.answer("Unknown?", [])

    assert "недостаточно данных" in answer
