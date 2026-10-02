"""The D.A.R.T. web API: POST /api/decide, POST /api/decisions/{id}/feedback, GET /api/health.

Run from the project folder:
    .venv\\Scripts\\python.exe -m uvicorn dartweb.app:app --app-dir Mark-2/backend --port 8000
"""
import json
import logging
import sys
import threading
import time
import uuid
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
PARALLEL_MIN_QUESTIONS = 6  # measured on an RTX 3050 Laptop GPU: the break-even of the two paths


def model_dtype(device, configured=None):
    """The precision to load the model in. The export is float16, which suits a GPU, but CPUs run float16 (and
    bfloat16) matrix maths far slower: measured with 2 threads, one question took ~4.6 s in float16 and
    ~0.87 s in float32, with the same answers. None keeps the exported dtype."""
    if configured:
        return configured
    return None if str(device).startswith("cuda") else "float32"


class ModelService:
    """The trained model behind a lock, so concurrent requests run one at a time."""

    def __init__(self, dart, version):
        self.dart = dart
        self.version = version
        self.device = dart.device
        self._lock = threading.Lock()

    @classmethod
    def load(cls, settings: Settings):
        import torch  # imported here (with the model) so tests can run without the model stack
        from dart.api import DART

        config = json.loads((settings.model_dir / "config.json").read_text(encoding="utf8"))
        device = settings.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dart = DART.from_pretrained(settings.model_dir, device=device, dtype=model_dtype(device, settings.dtype))
        return cls(dart, config["version"])

    def warm_up(self):
        for _ in range(3):
            self.dart.decide("warm-up", ["a", "b"])
            self.dart.decide_many("warm-up", [{"query": "a?", "options": ["a", "b"]},
                                              {"query": "b?", "options": ["a", "b"]}])

    @property
    def parallel_min_questions(self):
        """From how many questions the shared-context pass (dart.parallel) is used. It is not CUDA-graphed, so on
        a GPU it costs a flat ~150 ms against ~25 ms per graphed question and only wins from
        PARALLEL_MIN_QUESTIONS questions. Without graphs (CPU) it wins whenever there are two or more."""
        return 2 if self.dart.graphs is None else PARALLEL_MIN_QUESTIONS

    def use_parallel(self, context, count):
        return bool(context) and count >= self.parallel_min_questions

    def decide(self, context, questions):
        """questions: [{"id", "query", "options"}, ...].

        Returns (PRD 5.1 payload, model latency in milliseconds, whether the shared-context pass was used).
        """
        parallel = self.use_parallel(context, len(questions))
        with self._lock:
            start = time.perf_counter()
            if parallel:
                payload = self.dart.decide_many(context, questions)
            else:
                singles = [self.dart.decide(q["query"], q["options"], context=context, question_id=q["id"])
                           for q in questions]
                payload = {**singles[0], "decisions": [s["decisions"][0] for s in singles]}
            return payload, (time.perf_counter() - start) * 1000, parallel


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
                              model_version=current.version, parallel_min_questions=current.parallel_min_questions)

    def parse_all(questions):
        parsed = []
        for number, question in enumerate(questions, start=1):
            try:
                parsed.append(parse_options(question.options))
            except OptionsError as error:
                prefix = f"Question {number}: " if len(questions) > 1 else ""
                raise HTTPException(status_code=422, detail=f"{prefix}{error}") from error
        return parsed

    @app.post("/api/decide", response_model=DecideResponse)
    async def decide(body: DecideRequest, request: Request):
        enforce_rate_limit(request)
        parsed = parse_all(body.questions)
        questions = [{"id": f"q{i + 1}", "query": q.query, "options": p.options}
                     for i, (q, p) in enumerate(zip(body.questions, parsed))]
        try:
            payload, latency_ms, parallel = await run_in_threadpool(app.state.model.decide, body.context, questions)
        except ValueError as error:  # for example a prompt over the model's 256-token limit
            raise HTTPException(status_code=422, detail=str(error)) from error
        latency_ms = round(latency_ms, 2)
        group_id = uuid.uuid4().hex if len(questions) > 1 else ""
        decisions = []
        for question, options, decision in zip(body.questions, parsed, payload["decisions"]):
            record_id = await save_safely({
                "context": body.context, "query": question.query, "group_id": group_id,
                "options_input": question.options, "options": options.options, "options_kind": options.kind,
                "decision": decision["decision"], "confidence": decision["confidence"], "ranked": decision["ranked"],
                "latency_ms": latency_ms, "model_version": app.state.model.version,
            })
            decisions.append({**decision, "options": options.options, "options_kind": options.kind,
                              "record_id": record_id})
        return DecideResponse(model=payload["model"], decisions=decisions, latency_ms=latency_ms,
                              parallel=parallel,
                              stored=all(d["record_id"] is not None for d in decisions))

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
