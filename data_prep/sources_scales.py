"""Scale and low/medium/high converters built from rated public datasets."""
import re

import pandas as pd

from data_prep.schema import clean_text, make_row, word_count
from data_prep.sources_text import MAX_SEED, sample_frame

STAR_MENTION = re.compile(r"\b(\d|one|two|three|four|five)(\.\d)?[ -]*(stars?|/ ?5|out of 5)\b", re.IGNORECASE)
LMH = ["low", "medium", "high"]
SENTIMENT_BAND = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
HELPFULNESS_BAND = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}
MAX_SNIPPET_WORDS = 60
MAX_REVIEW_WORDS = 110
MAX_PROMPT_WORDS = 40
MAX_RESPONSE_WORDS = 90


def _balanced(frame, column, per_class, rng):
    parts = [sample_frame(part, per_class, rng) for _, part in frame.groupby(column)]
    return pd.concat(parts) if parts else frame.iloc[0:0]


def _halves(frame, rng):
    shuffled = frame.sample(frac=1, random_state=rng.randrange(MAX_SEED))
    middle = len(shuffled) // 2
    return shuffled.iloc[:middle], shuffled.iloc[middle:]


def _slice_for_index(index, count):
    return "hard_ambiguous" if count / 3 <= index < 2 * count / 3 else "core"


def _scale_row(source, decision_type, query, count, label, first_value, subtopic):
    options = [str(first_value + i) for i in range(count)]
    return make_row(source=source, decision_type=decision_type, query=query, options=options, label=label,
                    subtopic=subtopic, slice_=_slice_for_index(label, count))


def _lmh_row(source, query, band, subtopic):
    return make_row(source=source, decision_type="ordinal_lmh", query=query, options=LMH, label=band,
                    subtopic=subtopic, slice_="hard_ambiguous" if band == 1 else "core")


def _rating_rows(source, frame, rng, per_class, max_words, scale_query, lmh_query, subtopic):
    """Ratings 0-4 become a 1-5 scale question (first half) and a low/medium/high question (second half)."""
    frame = frame[frame.text.map(word_count) <= max_words].assign(text=lambda f: f.text.map(clean_text))
    frame = frame[~frame.text.str.contains(STAR_MENTION)]  # reviews that state their own rating leak the label
    scale_part, lmh_part = _halves(frame, rng)
    rows = [
        _scale_row(source, "scale_1_5", f'{scale_query} "{r.text}"', 5, int(r.label), 1, subtopic)
        for r in _balanced(scale_part, "label", per_class, rng).itertuples()
    ]
    banded = lmh_part.assign(band=lmh_part.label.map(SENTIMENT_BAND))
    rows += [
        _lmh_row(source, f'{lmh_query} "{r.text}"', int(r.band), subtopic)
        for r in _balanced(banded, "band", per_class, rng).itertuples()
    ]
    return rows


def sst5_rows(df, rng, per_class):
    return _rating_rows(
        "sst5", df, rng, per_class, MAX_SNIPPET_WORDS,
        "Rate the sentiment of this movie review snippet from 1 (very negative) to 5 (very positive):",
        "How positive is the sentiment of this movie review snippet?", "sentiment",
    )


def yelp_rows(df, rng, per_class):
    return _rating_rows(
        "yelp", df, rng, per_class, MAX_REVIEW_WORDS,
        "Rate the reviewer's satisfaction from 1 (very unhappy) to 5 (very happy):",
        "How satisfied is the reviewer with the business?", "customer reviews",
    )


def _round_half_up(value):
    return int(value + 0.5)


def stsb_rows(df, rng, per_class):
    scored = df[(df.label >= 0) & (df.label <= 5)]  # the GLUE test split has hidden labels (-1)
    five, ten = _halves(scored, rng)
    template = ('Rate how similar in meaning these two sentences are, from 0 (unrelated) to {top} (identical). '
                'Sentence 1: "{a}" Sentence 2: "{b}"')
    rows = []
    for part, top, decision_type, factor in ((five, 5, "scale_0_5", 1), (ten, 10, "scale_0_10", 2)):
        part = part.assign(score=part.label.map(lambda v: _round_half_up(v * factor)))
        for r in _balanced(part, "score", per_class, rng).itertuples():
            query = template.format(top=top, a=clean_text(r.sentence1), b=clean_text(r.sentence2))
            rows.append(_scale_row("stsb", decision_type, query, top + 1, int(r.score), 0, "semantic similarity"))
    return rows


def helpsteer_rows(df, rng, per_class):
    short = df[(df.prompt.map(word_count) <= MAX_PROMPT_WORDS) & (df.response.map(word_count) <= MAX_RESPONSE_WORDS)]
    quality_part, helpful_part = _halves(short, rng)
    quality = quality_part.assign(
        score=quality_part.apply(
            lambda r: min(10, _round_half_up((r.helpfulness + r.correctness + r.coherence) / 3 * 2.5)), axis=1))
    helpful = helpful_part.assign(band=helpful_part.helpfulness.map(HELPFULNESS_BAND))
    rows = [
        _scale_row("helpsteer2", "scale_0_10",
                   f'User prompt: "{clean_text(r.prompt)}" Response: "{clean_text(r.response)}" '
                   "Rate the overall quality of the response from 0 (unusable) to 10 (excellent).",
                   11, int(r.score), 0, "response quality")
        for r in _balanced(quality, "score", per_class, rng).itertuples()
    ]
    rows += [
        _lmh_row("helpsteer2",
                 f'User prompt: "{clean_text(r.prompt)}" Response: "{clean_text(r.response)}" '
                 "How helpful is the response to the prompt?", int(r.band), "response quality")
        for r in _balanced(helpful, "band", per_class, rng).itertuples()
    ]
    return rows
