import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND.parents[1]))

from dartweb.app import PARALLEL_MIN_QUESTIONS, ModelService, create_app, model_dtype  # noqa: E402
from dartweb.limiter import RateLimiter  # noqa: E402
from dartweb.settings import load_settings  # noqa: E402
from dartweb.store import MemoryStore  # noqa: E402


class FakeModel:
    """Stands in for the trained model: probabilities grow with the option's position."""

    device = "cpu"
    version = "vtest"
    parallel_min_questions = 2

    def __init__(self):
        self.calls = []

    @staticmethod
    def _decision(question):
        options = question["options"]
        weights = [i + 1 for i in range(len(options))]
        total = sum(weights)
        ranked = sorted(((o, w / total) for o, w in zip(options, weights)), key=lambda p: p[1], reverse=True)
        return {"id": question["id"], "ranked": [{"option": o, "probability": p} for o, p in ranked],
                "decision": ranked[0][0], "confidence": ranked[0][1]}

    def decide(self, context, questions):
        self.calls.append((context, [dict(q, options=list(q["options"])) for q in questions]))
        if any("TOO LONG" in q["query"] for q in questions):
            raise ValueError("prompt has 300 tokens; the limit is 256")
        parallel = bool(context) and len(questions) > 1
        return {"model": "D.A.R.T.", "decisions": [self._decision(q) for q in questions]}, 12.345, parallel


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


def decide_many(client, questions, context=""):
    return client.post("/api/decide", json={"context": context, "questions": questions})


def decide(client, query="Is this good?", options="yes, no, maybe", context=""):
    return decide_many(client, [{"query": query, "options": options}], context)


def record_id(body, index=0):
    return body["decisions"][index]["record_id"]


def test_health_reports_the_model(client):
    body = client.get("/api/health").json()
    assert body == {"status": "ok", "model_loaded": True, "device": "cpu", "model_version": "vtest",
                    "parallel_min_questions": 2}


def test_decide_returns_the_contract_with_latency(client):
    response = decide(client)
    assert response.status_code == 200
    body = response.json()
    decision = body["decisions"][0]
    probs = [item["probability"] for item in decision["ranked"]]
    assert body["model"] == "D.A.R.T." and body["latency_ms"] == 12.35 and body["parallel"] is False
    assert probs == sorted(probs, reverse=True) and sum(probs) == pytest.approx(1.0)
    assert decision["decision"] == decision["ranked"][0]["option"] and decision["confidence"] == probs[0]
    assert decision["options"] == ["yes", "no", "maybe"] and decision["options_kind"] == "list"
    assert body["stored"] is True and isinstance(decision["record_id"], str)


def test_the_context_and_question_are_sent_to_the_model_trimmed(client, parts):
    decide(client, query="  Was it wonderful?  ", context="  A movie review: it was wonderful.  ")
    context, questions = parts[0].calls[0]
    assert context == "A movie review: it was wonderful." and questions[0]["query"] == "Was it wonderful?"


def test_context_is_optional(client, parts):
    assert client.post("/api/decide", json={"questions": [{"query": "q?", "options": "a, b"}]}).status_code == 200
    assert parts[0].calls[0][0] == ""


def test_several_questions_share_the_context_and_run_together(client, parts):
    questions = [{"query": "Where does Tony live?", "options": "China, England"},
                 {"query": "How old is Tony?", "options": "10-20"},
                 {"query": "Is Tony an adult?", "options": "yes, no"}]
    body = decide_many(client, questions, context="Tony lives in England and is 16.").json()
    assert len(parts[0].calls) == 1  # one model call for every question
    context, sent = parts[0].calls[0]
    assert context == "Tony lives in England and is 16." and [q["id"] for q in sent] == ["q1", "q2", "q3"]
    assert sent[1]["options"] == [str(n) for n in range(10, 21)]
    assert body["parallel"] is True and [d["id"] for d in body["decisions"]] == ["q1", "q2", "q3"]
    assert [d["options_kind"] for d in body["decisions"]] == ["list", "integer_range", "list"]


def test_several_questions_without_a_context_are_not_reported_as_parallel(client):
    questions = [{"query": "a?", "options": "x, y"}, {"query": "b?", "options": "x, y"}]
    assert decide_many(client, questions).json()["parallel"] is False


def test_each_question_is_stored_with_the_shared_context_and_group(client, parts):
    questions = [{"query": "First?", "options": "a, b"}, {"query": "Second?", "options": "c, d"}]
    body = decide_many(client, questions, context="Shared text").json()
    first, second = (parts[1].records[record_id(body, i)] for i in range(2))
    assert first["context"] == second["context"] == "Shared text"
    assert first["group_id"] and first["group_id"] == second["group_id"]
    assert first["query"] == "First?" and second["options"] == ["c", "d"] and body["stored"] is True


def test_a_single_question_has_no_group(client, parts):
    body = decide(client).json()
    assert parts[1].records[record_id(body)]["group_id"] == ""


def test_a_range_is_expanded_into_options(client, parts):
    body = decide(client, options="0-5").json()
    assert body["decisions"][0]["options_kind"] == "integer_range"
    assert body["decisions"][0]["options"] == ["0", "1", "2", "3", "4", "5"]
    assert parts[0].calls[0][1][0]["options"] == ["0", "1", "2", "3", "4", "5"]


def test_options_can_be_sent_as_a_json_list(client, parts):
    body = decide(client, options=["hospital", "airport", "school, bank"]).json()
    assert body["decisions"][0]["options"] == ["hospital", "airport", "school, bank"]
    assert body["decisions"][0]["options_kind"] == "list"
    assert parts[1].records[record_id(body)]["options_input"] == ["hospital", "airport", "school, bank"]


@pytest.mark.parametrize("options", [["only"], [], ["a", "a"]])
def test_a_bad_json_list_gives_a_clear_422(client, options):
    response = decide(client, options=options)
    assert response.status_code == 422 and isinstance(response.json()["detail"], str)


def test_a_decimal_range_is_reported_as_decimal(client):
    decision = decide(client, options="0-5 step 0.5").json()["decisions"][0]
    assert decision["options_kind"] == "decimal_range" and len(decision["options"]) == 11


def test_the_input_and_output_are_stored(client, parts):
    body = decide(client, query="Pick one", options="a, b").json()
    record = parts[1].records[record_id(body)]
    assert record["query"] == "Pick one" and record["options_input"] == "a, b" and record["options"] == ["a", "b"]
    assert record["decision"] == body["decisions"][0]["decision"] and record["model_version"] == "vtest"
    assert record["feedback"] is None and "ranked" in record and record["latency_ms"] == 12.35


@pytest.mark.parametrize("options", ["only", "0.0-5.0", "5-0", "", "0-100"])
def test_bad_options_give_a_clear_422(client, options):
    response = decide(client, options=options)
    assert response.status_code == 422


def test_too_many_values_message_suggests_a_step(client):
    assert "step" in decide(client, options="0.0-5.0").json()["detail"]


def test_a_bad_option_list_names_its_question(client, parts):
    questions = [{"query": "a?", "options": "x, y"}, {"query": "b?", "options": "only"}]
    response = decide_many(client, questions)
    assert response.status_code == 422 and response.json()["detail"].startswith("Question 2: ")
    assert parts[0].calls == []  # nothing reaches the model when any question is invalid


def test_empty_query_is_rejected(client):
    assert decide(client, query="   ").status_code == 422


def test_overlong_query_and_context_are_rejected(client):
    assert decide(client, query="x" * 501).status_code == 422
    assert decide(client, context="x" * 2001).status_code == 422


@pytest.mark.parametrize("count", [0, 11])
def test_question_count_is_limited(client, count):
    assert decide_many(client, [{"query": "q?", "options": "a, b"}] * count).status_code == 422


def test_model_rejection_becomes_a_422_with_its_message(client):
    response = decide(client, query="TOO LONG")
    assert response.status_code == 422 and "256" in response.json()["detail"]


def test_a_broken_database_does_not_block_the_answer():
    with TestClient(create_app(settings=load_settings({}), model=FakeModel(), store=BrokenStore())) as client:
        body = decide(client).json()
    assert body["stored"] is False and record_id(body) is None and body["decisions"][0]["decision"]


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
    decision_id = record_id(decide(client, options="a, b").json())
    response = client.post(f"/api/decisions/{decision_id}/feedback", json={"correct_option": "b"})
    assert response.status_code == 200 and response.json() == {"id": decision_id, "correct_option": "b"}
    assert parts[1].records[decision_id]["feedback"]["correct_option"] == "b"


def test_feedback_rejects_an_option_that_was_not_offered(client):
    decision_id = record_id(decide(client, options="a, b").json())
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


class RecordingDart:
    """Records which inference path ModelService picks. `graphs` is set when the CUDA-graph fast path exists."""

    device = "cpu"

    def __init__(self, graphs):
        self.graphs = object() if graphs else None
        self.calls = []

    def decide(self, query, options, context="", question_id="q1"):
        self.calls.append("decide")
        return {"model": "D.A.R.T.", "decisions": [{"id": question_id}]}

    def decide_many(self, context, questions):
        self.calls.append("decide_many")
        return {"model": "D.A.R.T.", "decisions": [{"id": q["id"]} for q in questions]}


def questions_for(count):
    return [{"id": f"q{i + 1}", "query": f"{i}?", "options": ["x", "y"]} for i in range(count)]


@pytest.mark.parametrize("graphs,context,count,parallel", [
    (True, "ctx", 1, False),
    (True, "ctx", PARALLEL_MIN_QUESTIONS - 1, False),  # graphed single calls are still faster
    (True, "ctx", PARALLEL_MIN_QUESTIONS, True),
    (True, "", PARALLEL_MIN_QUESTIONS, False),  # nothing to share without a context
    (False, "ctx", 2, True),  # no graphs (CPU): sharing the context always wins
])
def test_model_service_picks_the_faster_path(graphs, context, count, parallel):
    dart = RecordingDart(graphs)
    service = ModelService(dart, "vtest")
    assert service.parallel_min_questions == (PARALLEL_MIN_QUESTIONS if graphs else 2)
    payload, _, used_parallel = service.decide(context, questions_for(count))
    assert used_parallel is parallel
    assert dart.calls == (["decide_many"] if parallel else ["decide"] * count)
    assert [d["id"] for d in payload["decisions"]] == [f"q{i + 1}" for i in range(count)]


def test_settings_defaults_and_overrides():
    defaults = load_settings({})
    assert defaults.mongodb_uri == "mongodb://localhost:27017" and defaults.mongodb_db == "dart"
    assert defaults.cors_origins == ["http://localhost:5173"] and defaults.device is None
    custom = load_settings({"MONGODB_DB": "x", "CORS_ORIGINS": "https://a.app, https://b.app", "DART_DEVICE": "cpu",
                            "RATE_LIMIT_PER_MINUTE": "5"})
    assert custom.mongodb_db == "x" and custom.cors_origins == ["https://a.app", "https://b.app"]
    assert custom.device == "cpu" and custom.rate_limit_per_minute == 5


@pytest.mark.parametrize("device,configured,expected", [
    ("cpu", None, "float32"),  # float16 on CPU is ~5x slower
    ("cuda", None, None),  # keep the exported float16 on a GPU
    ("cuda:0", None, None),
    ("cpu", "bfloat16", "bfloat16"),  # an explicit DART_DTYPE always wins
])
def test_model_dtype_defaults_to_float32_on_cpu(device, configured, expected):
    assert model_dtype(device, configured) == expected


def test_dtype_setting_is_validated():
    assert load_settings({}).dtype is None and load_settings({"DART_DTYPE": "float32"}).dtype == "float32"
    with pytest.raises(ValueError, match="DART_DTYPE"):
        load_settings({"DART_DTYPE": "fp8"})
