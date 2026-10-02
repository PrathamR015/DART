import asyncio
import sys
import uuid
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND.parents[1]))

from dartweb.limiter import RateLimiter  # noqa: E402
from dartweb.store import MemoryStore, MongoStore  # noqa: E402
from dartweb.training_export import build_rows, infer_decision_type, unlabeled  # noqa: E402
from data_prep.schema import validate_row  # noqa: E402

RECORD = {"query": "Pick one", "options": ["a", "b"], "decision": "a", "confidence": 0.6}


def run(coroutine):
    return asyncio.run(coroutine)


def test_memory_store_round_trip():
    store = MemoryStore()
    decision_id = run(store.save(dict(RECORD)))
    assert run(store.get(decision_id))["query"] == "Pick one"
    assert run(store.add_feedback(decision_id, "b")) is True
    assert run(store.get(decision_id))["feedback"]["correct_option"] == "b"
    assert run(store.add_feedback("missing", "b")) is False
    assert run(store.get("missing")) is None


def test_mongo_store_round_trip_with_a_real_database():
    """Uses a throw-away database on the local MongoDB, everything in one event loop (the client is loop-bound)."""

    async def scenario():
        database = f"dart_test_{uuid.uuid4().hex[:8]}"
        store = MongoStore("mongodb://localhost:27017", database)
        try:
            await store.client.admin.command("ping")
        except Exception:
            await store.close()
            pytest.skip("MongoDB is not running on localhost:27017")
        try:
            await store.setup()
            decision_id = await store.save(dict(RECORD))
            stored = await store.get(decision_id)
            assert stored["query"] == "Pick one" and stored["feedback"] is None and "created_at" in stored
            assert await store.add_feedback(decision_id, "b") is True
            assert (await store.get(decision_id))["feedback"]["correct_option"] == "b"
            assert await store.get("not-an-object-id") is None
            assert await store.add_feedback("not-an-object-id", "b") is False
            assert await store.add_feedback("0" * 24, "b") is False
        finally:
            await store.client.drop_database(database)
            await store.close()

    run(scenario())


def test_limiter_blocks_after_the_limit_and_recovers_after_the_window():
    now = [0.0]
    limiter = RateLimiter(per_minute=2, clock=lambda: now[0])
    assert [limiter.allow("a"), limiter.allow("a"), limiter.allow("a")] == [True, True, False]
    assert limiter.allow("b") is True  # another client has its own window
    now[0] = 61.0
    assert limiter.allow("a") is True


def test_limiter_with_zero_is_disabled():
    limiter = RateLimiter(per_minute=0)
    assert all(limiter.allow("a") for _ in range(100))


@pytest.mark.parametrize("options,expected", [
    (["yes", "no"], "binary"), (["Low", "Medium", "High"], "ordinal_lmh"), (["cat", "dog", "bird"], "categorical"),
    ([str(i) for i in range(6)], "scale_0_5"), ([str(i) for i in range(1, 6)], "scale_1_5"),
    ([str(i) for i in range(11)], "scale_0_10"), (["0.0", "0.5", "1.0"], "scale_other"),
    ([str(i) for i in range(3)], "scale_other"),
])
def test_decision_type_inference(options, expected):
    assert infer_decision_type(options) == expected


def test_only_labelled_records_become_training_rows():
    records = [
        {"_id": "1", **RECORD, "feedback": {"correct_option": "b"}},
        {"_id": "2", **RECORD, "feedback": None},
        {"_id": "3", **RECORD, "feedback": {"correct_option": "zzz"}},
    ]
    rows, skipped = build_rows(records)
    assert skipped == 2 and len(rows) == 1
    row = rows[0]
    assert row["id"] == "mark2_user_1" and row["label"] == 1 and row["target_option"] == "b"
    assert row["source"] == "mark2_user" and row["decision_type"] == "categorical" and validate_row(row) is None


def test_a_shared_context_record_keeps_its_context_and_group():
    record = {"_id": "1", **RECORD, "context": "Tony lives in England.", "group_id": "g1",
              "feedback": {"correct_option": "a"}}
    rows, _ = build_rows([record])
    assert rows[0]["context"] == "Tony lives in England." and rows[0]["group_id"] == "g1"


def test_unlabeled_lists_the_records_without_feedback():
    records = [{"_id": "1", **RECORD, "feedback": {"correct_option": "b"}}, {"_id": "2", **RECORD, "feedback": None}]
    assert unlabeled(records) == [{"id": "2", "context": "", "query": "Pick one", "options": ["a", "b"],
                                   "model_decision": "a"}]
