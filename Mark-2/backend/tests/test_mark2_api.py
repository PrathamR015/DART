import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND.parents[1]))

from dartweb.app import create_app  # noqa: E402
from dartweb.limiter import RateLimiter  # noqa: E402
from dartweb.settings import load_settings  # noqa: E402
from dartweb.store import MemoryStore  # noqa: E402


class FakeModel:
    """Stands in for the trained model: probabilities grow with the option's position."""

    device = "cpu"
    version = "vtest"

    def __init__(self):
        self.calls = []

    def decide(self, query, options):
        self.calls.append((query, list(options)))
        if "TOO LONG" in query:
            raise ValueError("prompt has 300 tokens; the limit is 256")
        weights = [i + 1 for i in range(len(options))]
        total = sum(weights)
        ranked = sorted(((o, w / total) for o, w in zip(options, weights)), key=lambda p: p[1], reverse=True)
        decision = {"id": "q1", "ranked": [{"option": o, "probability": p} for o, p in ranked],
                    "decision": ranked[0][0], "confidence": ranked[0][1]}
        return {"model": "D.A.R.T.", "decisions": [decision]}, 12.345


class BrokenStore(MemoryStore):
    async def save(self, record):
        raise RuntimeError("database is down")

    async def get(self, decision_id):
        raise RuntimeError("database is down")


@pytest.fixture()
def parts():
    return FakeModel(), MemoryStore()


@pytest.fixture()
def client(parts):
    model, store = parts
    with TestClient(create_app(settings=load_settings({}), model=model, store=store)) as test_client:
        yield test_client


def decide(client, query="Is this good?", options="yes, no, maybe"):
    return client.post("/api/decide", json={"query": query, "options": options})


def test_health_reports_the_model(client):
    body = client.get("/api/health").json()
    assert body == {"status": "ok", "model_loaded": True, "device": "cpu", "model_version": "vtest"}


def test_decide_returns_the_contract_with_latency(client):
    response = decide(client)
    assert response.status_code == 200
    body = response.json()
    decision = body["decisions"][0]
    probs = [item["probability"] for item in decision["ranked"]]
    assert body["model"] == "D.A.R.T." and body["latency_ms"] == 12.35
    assert probs == sorted(probs, reverse=True) and sum(probs) == pytest.approx(1.0)
    assert decision["decision"] == decision["ranked"][0]["option"] and decision["confidence"] == probs[0]
    assert body["options"] == ["yes", "no", "maybe"] and body["options_kind"] == "list"
    assert body["stored"] is True and isinstance(body["id"], str)


def test_the_whole_query_is_sent_to_the_model_trimmed(client, parts):
    decide(client, query="  Rate this movie review. It was wonderful.  ")
    assert parts[0].calls[0][0] == "Rate this movie review. It was wonderful."


def test_a_range_is_expanded_into_options(client, parts):
    body = decide(client, options="0-5").json()
    assert body["options_kind"] == "integer_range" and body["options"] == ["0", "1", "2", "3", "4", "5"]
    assert parts[0].calls[0][1] == ["0", "1", "2", "3", "4", "5"]


def test_a_decimal_range_is_reported_as_decimal(client):
    body = decide(client, options="0-5 step 0.5").json()
    assert body["options_kind"] == "decimal_range" and len(body["options"]) == 11


def test_the_input_and_output_are_stored(client, parts):
    body = decide(client, query="Pick one", options="a, b").json()
    record = parts[1].records[body["id"]]
    assert record["query"] == "Pick one" and record["options_input"] == "a, b" and record["options"] == ["a", "b"]
    assert record["decision"] == body["decisions"][0]["decision"] and record["model_version"] == "vtest"
    assert record["feedback"] is None and "ranked" in record and record["latency_ms"] == 12.35


@pytest.mark.parametrize("options", ["only", "0.0-5.0", "5-0", "", "0-100"])
def test_bad_options_give_a_clear_422(client, options):
    response = decide(client, options=options)
    assert response.status_code == 422


def test_too_many_values_message_suggests_a_step(client):
    assert "step" in decide(client, options="0.0-5.0").json()["detail"]


def test_empty_query_is_rejected(client):
    assert decide(client, query="   ").status_code == 422


def test_overlong_query_is_rejected(client):
    assert decide(client, query="x" * 2001).status_code == 422


def test_model_rejection_becomes_a_422_with_its_message(client):
    response = decide(client, query="TOO LONG")
    assert response.status_code == 422 and "256" in response.json()["detail"]


def test_a_broken_database_does_not_block_the_answer():
    with TestClient(create_app(settings=load_settings({}), model=FakeModel(), store=BrokenStore())) as client:
        body = decide(client).json()
    assert body["stored"] is False and body["id"] is None and body["decisions"][0]["decision"]


def test_after_a_database_failure_later_requests_skip_the_database_for_a_while():
    class CountingBrokenStore(BrokenStore):
        saves = 0

        async def save(self, record):
            type(self).saves += 1
            raise RuntimeError("database is down")

    with TestClient(create_app(settings=load_settings({}), model=FakeModel(), store=CountingBrokenStore())) as client:
        bodies = [decide(client).json() for _ in range(3)]
    assert CountingBrokenStore.saves == 1  # the first failure opens the breaker; it would otherwise wait every time
    assert all(body["stored"] is False and body["decisions"][0]["decision"] for body in bodies)


def test_feedback_saves_the_label(client, parts):
    decision_id = decide(client, options="a, b").json()["id"]
    response = client.post(f"/api/decisions/{decision_id}/feedback", json={"correct_option": "b"})
    assert response.status_code == 200 and response.json() == {"id": decision_id, "correct_option": "b"}
    assert parts[1].records[decision_id]["feedback"]["correct_option"] == "b"


def test_feedback_rejects_an_option_that_was_not_offered(client):
    decision_id = decide(client, options="a, b").json()["id"]
    assert client.post(f"/api/decisions/{decision_id}/feedback", json={"correct_option": "z"}).status_code == 422


@pytest.mark.parametrize("decision_id", ["nope", "0" * 24])
def test_feedback_for_an_unknown_decision_is_404(client, decision_id):
    assert client.post(f"/api/decisions/{decision_id}/feedback", json={"correct_option": "a"}).status_code == 404


def test_feedback_with_the_database_down_is_503():
    with TestClient(create_app(settings=load_settings({}), model=FakeModel(), store=BrokenStore())) as client:
        assert client.post("/api/decisions/abc/feedback", json={"correct_option": "a"}).status_code == 503


def test_rate_limit_returns_429():
    limiter = RateLimiter(per_minute=2)
    with TestClient(create_app(settings=load_settings({}), model=FakeModel(), store=MemoryStore(),
                               limiter=limiter)) as client:
        codes = [decide(client).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_cors_allows_the_configured_origin_only():
    settings = load_settings({"CORS_ORIGINS": "http://localhost:5173"})
    with TestClient(create_app(settings=settings, model=FakeModel(), store=MemoryStore())) as client:
        allowed = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
        other = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in other.headers


def test_settings_defaults_and_overrides():
    defaults = load_settings({})
    assert defaults.mongodb_uri == "mongodb://localhost:27017" and defaults.mongodb_db == "dart"
    assert defaults.cors_origins == ["http://localhost:5173"] and defaults.device is None
    custom = load_settings({"MONGODB_DB": "x", "CORS_ORIGINS": "https://a.app, https://b.app", "DART_DEVICE": "cpu",
                            "RATE_LIMIT_PER_MINUTE": "5"})
    assert custom.mongodb_db == "x" and custom.cors_origins == ["https://a.app", "https://b.app"]
    assert custom.device == "cpu" and custom.rate_limit_per_minute == 5
