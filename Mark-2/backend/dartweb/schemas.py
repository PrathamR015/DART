"""Request and response models for the API."""
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

MAX_CONTEXT_CHARS = 2000
MAX_QUERY_CHARS = 500
MAX_OPTIONS_CHARS = 500
MAX_OPTION_ITEMS = 24  # a sanity cap on the JSON list; the parser explains the real limit of 12
MAX_QUESTIONS = 10

OptionsText = Annotated[str, Field(min_length=1, max_length=MAX_OPTIONS_CHARS)]
OptionsList = Annotated[list[Annotated[str, Field(max_length=MAX_OPTIONS_CHARS)]], Field(max_length=MAX_OPTION_ITEMS)]


class QuestionInput(BaseModel):
    """`options` is either text ("yes, no, maybe" or a range like "0-5") or a JSON list of option strings."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    options: OptionsText | OptionsList


class DecideRequest(BaseModel):
    """One optional shared context and 1 to MAX_QUESTIONS questions about it, each with its own options."""

    model_config = ConfigDict(str_strip_whitespace=True)

    context: str = Field(default="", max_length=MAX_CONTEXT_CHARS)
    questions: list[QuestionInput] = Field(min_length=1, max_length=MAX_QUESTIONS)


class RankedOption(BaseModel):
    option: str
    probability: float


class Decision(BaseModel):
    """The PRD 5.1 decision plus the resolved options and the stored record's id (for feedback)."""

    id: str
    ranked: list[RankedOption]
    decision: str
    confidence: float
    options: list[str]
    options_kind: str
    record_id: str | None


class DecideResponse(BaseModel):
    """The PRD 5.1 payload plus latency and how the questions were run.

    `parallel` is true when several questions shared one context pass; `stored` is true when every decision
    was saved.
    """

    model: str
    decisions: list[Decision]
    latency_ms: float
    parallel: bool
    stored: bool


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    correct_option: str = Field(min_length=1, max_length=100)


class FeedbackResponse(BaseModel):
    id: str
    correct_option: str


class HealthResponse(BaseModel):
    """`parallel_min_questions`: from this many questions over a shared context, the context is read once."""

    status: str
    model_loaded: bool
    device: str
    model_version: str
    parallel_min_questions: int
