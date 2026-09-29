from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid

BASE_URL = os.environ.get("EVIDENCERAG_SMOKE_URL", "http://localhost:8081")
API_KEY = os.environ.get("EVIDENCERAG_API_KEY", "local-demo-key-change-me")


def request(method: str, path: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE_URL + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + API_KEY},
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read())


def main() -> None:
    for _ in range(40):
        try:
            request("GET", "/ready")
            break
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.5)
    else:
        raise TimeoutError("API не стала доступной вовремя")

    suffix = uuid.uuid4().hex[:8]
    kb = request(
        "POST", "/api/v1/knowledge-bases", {"name": "Порядок реагирования", "slug": f"ops-{suffix}"}
    )
    document = request(
        "POST",
        f"/api/v1/knowledge-bases/{kb['id']}/documents",
        {
            "source_key": f"incident-policy-{suffix}",
            "title": "Порядок реагирования на инцидент",
            "content": (
                "Критический инцидент нужно подтвердить в течение пятнадцати минут. "
                "Ответственный открывает общий канал и назначает исполнителя. "
                "Если причиной стал релиз, первым действием служит откат изменений."
            ),
        },
    )
    for _ in range(40):
        document = request("GET", f"/api/v1/documents/{document['id']}")
        if document["status"] == "ready":
            break
        if document["status"] == "failed":
            raise RuntimeError(f"Индексация завершилась ошибкой: {document['error']}")
        time.sleep(0.5)
    else:
        raise TimeoutError("Документ не был проиндексирован вовремя")

    result = request(
        "POST",
        f"/api/v1/knowledge-bases/{kb['id']}/query",
        {"question": "За сколько минут нужно подтвердить критический инцидент?", "top_k": 3},
    )
    assert result["citations"], result
    assert "[1]" in str(result["answer"]), result
    feedback = request(
        "POST",
        f"/api/v1/traces/{result['trace_id']}/feedback",
        {"rating": 1, "comment": "Ответ опирается на документ"},
    )
    assert feedback["rating"] == 1
    print(json.dumps({"status": "ok", "trace_id": result["trace_id"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        print(exc.read().decode())
        raise
