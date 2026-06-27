from fastapi.testclient import TestClient

from app.main import app


def test_health_and_flow():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        ingest = client.post(
            "/ingest",
            json={
                "documents": [
                    {
                        "id": "doc1",
                        "source": "Internal FAQ",
                        "text": (
                            "Our refund policy allows returns within 30 days. "
                            "Refunds are processed to the original payment method."
                        ),
                    }
                ]
            },
        )
        assert ingest.status_code == 200
        assert ingest.json()["ingested_documents"] == 1
        assert ingest.json()["total_chunks"] >= 1

        ask = client.post(
            "/ask", json={"question": "What is the refund policy?", "top_k": 3}
        )
        assert ask.status_code == 200
        body = ask.json()
        assert body["grounded"] is True
        assert len(body["citations"]) >= 1
        assert body["citations"][0]["source"] == "Internal FAQ"
