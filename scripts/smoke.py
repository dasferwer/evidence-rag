from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid

BASE_URL = "http://localhost:8081"


def request(method: str, path: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE_URL + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read())


def main() -> None:
    suffix = uuid.uuid4().hex[:8]
    kb = request(
        "POST", "/api/v1/knowledge-bases", {"name": "Operations handbook", "slug": f"ops-{suffix}"}
    )
    document = request(
        "POST",
        f"/api/v1/knowledge-bases/{kb['id']}/documents",
        {
            "source_key": f"incident-policy-{suffix}",
            "title": "Incident response policy",
            "content": (
                "Critical incidents have a fifteen minute acknowledgement SLA. "
                "The incident commander must open a shared channel and assign an owner. "
                "If a deployment caused the incident, rollback is the preferred first action."
            ),
        },
    )
    for _ in range(40):
        document = request("GET", f"/api/v1/documents/{document['id']}")
        if document["status"] == "ready":
            break
        if document["status"] == "failed":
            raise RuntimeError(f"ingestion failed: {document['error']}")
        time.sleep(0.5)
    else:
        raise TimeoutError("document did not become ready")

    result = request(
        "POST",
        f"/api/v1/knowledge-bases/{kb['id']}/query",
        {"question": "What is the acknowledgement SLA for critical incidents?", "top_k": 3},
    )
    assert result["citations"], result
    assert "[1]" in str(result["answer"]), result
    feedback = request(
        "POST",
        f"/api/v1/traces/{result['trace_id']}/feedback",
        {"rating": 1, "comment": "grounded"},
    )
    assert feedback["rating"] == 1
    print(json.dumps({"status": "ok", "trace_id": result["trace_id"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        print(exc.read().decode())
        raise
