import json
import math

import httpx
import pytest

from evidencerag.ai import AIProvider, local_embedding
from evidencerag.config import Settings


def external_provider(monkeypatch, response):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=json.dumps(response).encode())
    )
    client = httpx.AsyncClient

    def make_client(**kwargs):
        return client(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    return AIProvider(Settings(llm_provider="openai", openai_api_key="test-placeholder"))


def embedding_item(index=0, embedding=None):
    return {"index": index, "embedding": [0.5] * 384 if embedding is None else embedding}


async def test_external_embeddings_follow_input_indexes(monkeypatch):
    provider = external_provider(
        monkeypatch,
        {"data": [embedding_item(1, [2.0] * 384), embedding_item(0, [1.0] * 384)]},
    )
    assert await provider.embed(["первый", "второй"]) == [[1.0] * 384, [2.0] * 384]


@pytest.mark.parametrize(
    "response",
    [
        [],
        True,
        {},
        {"data": {}},
        {"data": None},
        {"data": "invalid"},
        {"data": []},
        {"data": [embedding_item()]},
        {"data": [embedding_item(), embedding_item(1), embedding_item(2)]},
        {"data": [None, embedding_item(1)]},
        {"data": [{"embedding": [0.5] * 384}, embedding_item(1)]},
        {"data": [embedding_item(), embedding_item()]},
        *[{"data": [embedding_item(i), embedding_item(1)]} for i in (-1, 2, "0", 0.0, False)],
        {"data": [embedding_item(), {"index": 1}]},
        {"data": [embedding_item(), {"index": 1, "embedding": None}]},
        {"data": [embedding_item(), embedding_item(1, "bad")]},
        {"data": [embedding_item(), embedding_item(1, [])]},
        {"data": [embedding_item(), embedding_item(1, [0.0] * 383)]},
        {"data": [embedding_item(), embedding_item(1, [0.0] * 385)]},
        *[
            {"data": [embedding_item(), embedding_item(1, [value] + [0.0] * 383)]}
            for value in (float("nan"), float("inf"), -float("inf"), "1", True, None, 10**400)
        ],
    ],
)
async def test_external_embeddings_reject_invalid_batch(monkeypatch, response):
    provider = external_provider(monkeypatch, response)
    with pytest.raises(ValueError, match="Некорректный ответ embedding-провайдера"):
        await provider.embed(["первый", "второй"])


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
