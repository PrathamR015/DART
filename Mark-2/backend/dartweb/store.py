"""Where decisions are stored: MongoDB in production, an in-memory stand-in for tests."""
import itertools
import logging
from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import AsyncMongoClient

logger = logging.getLogger(__name__)
SERVER_SELECTION_TIMEOUT_MS = 2000


def _now():
    return datetime.now(timezone.utc)


def _object_id(decision_id):
    try:
        return ObjectId(decision_id)
    except (InvalidId, TypeError):
        return None


class MongoStore:
    def __init__(self, uri, database):
        self.client = AsyncMongoClient(uri, serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS)
        self.collection = self.client[database]["decisions"]

    async def setup(self):
        try:
            await self.collection.create_index([("created_at", -1)])
        except Exception as error:  # the app still answers questions without the database
            logger.warning("could not prepare MongoDB (%s); decisions will not be stored", type(error).__name__)

    async def save(self, record):
        result = await self.collection.insert_one({**record, "created_at": _now(), "feedback": None})
        return str(result.inserted_id)

    async def get(self, decision_id):
        object_id = _object_id(decision_id)
        return None if object_id is None else await self.collection.find_one({"_id": object_id})

    async def add_feedback(self, decision_id, correct_option):
        object_id = _object_id(decision_id)
        if object_id is None:
            return False
        feedback = {"correct_option": correct_option, "at": _now()}
        result = await self.collection.update_one({"_id": object_id}, {"$set": {"feedback": feedback}})
        return result.matched_count == 1

    async def close(self):
        await self.client.close()


class MemoryStore:
    """Same interface as MongoStore, kept in a dict. Used by tests."""

    def __init__(self):
        self.records = {}
        self._ids = itertools.count(1)

    async def setup(self):
        return None

    async def save(self, record):
        decision_id = f"{next(self._ids):024x}"
        self.records[decision_id] = {**record, "_id": decision_id, "created_at": _now(), "feedback": None}
        return decision_id

    async def get(self, decision_id):
        return self.records.get(decision_id)

    async def add_feedback(self, decision_id, correct_option):
        if decision_id not in self.records:
            return False
        self.records[decision_id]["feedback"] = {"correct_option": correct_option, "at": _now()}
        return True

    async def close(self):
        return None
