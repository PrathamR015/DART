"""The D.A.R.T. web API: POST /api/decide, POST /api/decisions/{id}/feedback, GET /api/health.

Run from the project folder:
    .venv\\Scripts\\python.exe -m uvicorn dartweb.app:app --app-dir Mark-2/backend --port 8000
"""
import json
import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))  # so the trained-model code (`dart`, `data_prep`) can be imported

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from dartweb.limiter import RateLimiter  # noqa: E402
from dartweb.options import OptionsError, parse_options  # noqa: E402
from dartweb.schemas import (  # noqa: E402
    DecideRequest, DecideResponse, FeedbackRequest, FeedbackResponse, HealthResponse)
from dartweb.settings import Settings, load_settings  # noqa: E402
from dartweb.store import MongoStore  # noqa: E402

logger = logging.getLogger("dartweb")
STORE_RETRY_SECONDS = 30


class ModelService:
    """The trained model behind a lock, so concurrent requests run one at a time."""

    def __init__(self, dart, version):
        self.dart = dart
        self.version = version
        self.device = dart.device
        self._lock = threading.Lock()

    @classmethod
    def load(cls, settings: Settings):
        from dart.api import DART  # imported here so tests can run without the model stack

        config = json.loads((settings.model_dir / "config.json").read_text(encoding="utf8"))
        return cls(DART.from_pretrained(settings.model_dir, device=settings.device), config["version"])

    def warm_up(self):
        for _ in range(3):
            self.dart.decide("warm-up", ["a", "b"])

    def decide(self, query, options):
        """Returns (PRD 5.1 payload, model latency in milliseconds)."""
        with self._lock:
            start = time.perf_counter()
            payload = self.dart.decide(query, options)
            return payload, (time.perf_counter() - start) * 1000


def create_app(settings=None, model=None, store=None, limiter=None):
    settings = settings or load_settings()
    limiter = limiter or RateLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(app):
        app.state.model = model or ModelService.load(settings)
        if model is None:
            app.state.model.warm_up()
        app.state.store = store or MongoStore(settings.mongodb_uri, settings.mongodb_db)
        await app.state.store.setup()
        yield
        await app.state.store.close()

    app = FastAPI(title="D.A.R.T.", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])

    def enforce_rate_limit(request: Request):
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment and try again.")

    store_down_until = [0.0]

    async def save_safely(record):
        """Stores the decision, or returns None. After a failure the database is skipped for a while, so an
        outage costs one slow request (the connection timeout) and not one on every request."""
        if time.monotonic() < store_down_until[0]:
            return None
        try:
            return await app.state.store.save(record)
        except Exception as error:  # an unavailable database must never block an answer
            store_down_until[0] = time.monotonic() + STORE_RETRY_SECONDS
            logger.warning("could not store the decision (%s); skipping the database for %d s",
                           type(error).__name__, STORE_RETRY_SECONDS)
            return None

    @app.get("/api/health", response_model=HealthResponse)
    async def health():
        current = app.state.model
        return HealthResponse(status="ok", model_loaded=True, device=str(current.device),
                              model_version=current.version)

    @app.post("/api/decide", response_model=DecideResponse)
    async def decide(body: DecideRequest, request: Request):
        enforce_rate_limit(request)
        try:
            parsed = parse_options(body.options)
        except OptionsError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        try:
            payload, latency_ms = await run_in_threadpool(app.state.model.decide, body.query, parsed.options)
        except ValueError as error:  # for example a prompt over the model's 256-token limit
            raise HTTPException(status_code=422, detail=str(error)) from error
        decision = payload["decisions"][0]
        decision_id = await save_safely({
            "query": body.query, "options_input": body.options, "options": parsed.options,
            "options_kind": parsed.kind, "decision": decision["decision"], "confidence": decision["confidence"],
            "ranked": decision["ranked"], "latency_ms": round(latency_ms, 2),
            "model_version": app.state.model.version,
        })
        return DecideResponse(id=decision_id, model=payload["model"], decisions=payload["decisions"],
                              latency_ms=round(latency_ms, 2), options=parsed.options,
                              options_kind=parsed.kind, stored=decision_id is not None)

    @app.post("/api/decisions/{decision_id}/feedback", response_model=FeedbackResponse)
    async def feedback(decision_id: str, body: FeedbackRequest):
        try:
            record = await app.state.store.get(decision_id)
        except Exception as error:
            logger.warning("could not read the decision (%s)", type(error).__name__)
            raise HTTPException(status_code=503, detail="The database is unavailable.") from error
        if record is None:
            raise HTTPException(status_code=404, detail="Decision not found.")
        if body.correct_option not in record["options"]:
            raise HTTPException(status_code=422, detail="correct_option must be one of the decision's options.")
        await app.state.store.add_feedback(decision_id, body.correct_option)
        return FeedbackResponse(id=decision_id, correct_option=body.correct_option)

    return app


app = create_app()
