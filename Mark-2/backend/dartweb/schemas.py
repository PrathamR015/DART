"""Request and response models for the API."""
from pydantic import BaseModel, ConfigDict, Field

MAX_QUERY_CHARS = 2000
MAX_OPTIONS_CHARS = 500


class DecideRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    options: str = Field(min_length=1, max_length=MAX_OPTIONS_CHARS)


class RankedOption(BaseModel):
    option: str
    probability: float


class Decision(BaseModel):
    id: str
    ranked: list[RankedOption]
    decision: str
    confidence: float


class DecideResponse(BaseModel):
    """The PRD 5.1 payload plus latency, the resolved options and whether the input was stored."""

    id: str | None
    model: str
    decisions: list[Decision]
    latency_ms: float
    options: list[str]
    options_kind: str
    stored: bool


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    correct_option: str = Field(min_length=1, max_length=100)


class FeedbackResponse(BaseModel):
    id: str
    correct_option: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_version: str
