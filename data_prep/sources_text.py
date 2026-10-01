"""Categorical converters: intent, topic, emotion and multiple-choice datasets."""
import re

from data_prep.schema import clean_text, make_row, sample_options, shuffle_options, word_count

NAMES_BLOCK = re.compile(r"names:\n((?:\s+'\d+': .+\n)+)")
NAME_LINE = re.compile(r"'\d+': (.+)")
AGNEWS_NAMES = ["World", "Sports", "Business", "Science and technology"]
GOEMOTIONS_NAMES = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
]
NEUTRAL_ID = 27
INTENT_QUESTIONS = ("What does the user want?", "Which intent best matches this request?",
                    "What is the user asking for?")
MAX_TEXT_WORDS = 110
MAX_QA_WORDS = 120
MAX_SEED = 2 ** 31


def parse_class_names(readme):
    """Class names from the first `names:` block of a dataset card, underscores as spaces."""
    block = NAMES_BLOCK.search(readme)
    if block is None:
        raise ValueError("no class names block found")
    return [NAME_LINE.match(line.strip()).group(1).replace("_", " ") for line in block.group(1).splitlines()]


def sample_frame(frame, limit, rng):
    return frame.sample(n=min(limit, len(frame)), random_state=rng.randrange(MAX_SEED))


def clinc_rows(df, names, rng, limit):
    rows = []
    for r in sample_frame(df, limit, rng).itertuples():
        options, label = sample_options(names[int(r.intent)], names, rng.randint(4, 8), rng)
        query = f'User said: "{clean_text(r.text)}" {rng.choice(INTENT_QUESTIONS)}'
        rows.append(make_row(source="clinc150", decision_type="categorical", query=query,
                             options=options, label=label, subtopic="intent routing"))
    return rows


def agnews_rows(df, rng, limit):
    frame = df[df.text.map(word_count) <= MAX_TEXT_WORDS]
    rows = []
    for r in sample_frame(frame, limit, rng).itertuples():
        options, label = shuffle_options(AGNEWS_NAMES, int(r.label), rng)
        query = f'Which news category fits this article? "{clean_text(r.text)}"'
        rows.append(make_row(source="agnews", decision_type="categorical", query=query,
                             options=options, label=label, subtopic="news topics"))
    return rows


def goemotions_rows(df, rng, limit, neutral_share=0.1):
    single = df[df.labels.map(len) == 1]
    single = single.assign(emotion=single.labels.map(lambda v: int(v[0])))
    neutral_n = int(limit * neutral_share)
    parts = [sample_frame(single[single.emotion != NEUTRAL_ID], limit - neutral_n, rng),
             sample_frame(single[single.emotion == NEUTRAL_ID], neutral_n, rng)]
    rows = []
    for part in parts:
        for r in part.itertuples():
            options, label = sample_options(GOEMOTIONS_NAMES[r.emotion], GOEMOTIONS_NAMES, rng.randint(4, 6), rng)
            query = f'Comment: "{clean_text(r.text)}" Which emotion does this comment mainly express?'
            rows.append(make_row(source="goemotions", decision_type="categorical", query=query,
                                 options=options, label=label, subtopic="emotion"))
    return rows


def mmlu_rows(df, rng, limit):
    rows = []
    for r in sample_frame(df, limit * 3, rng).itertuples():
        item = r.train
        options = [clean_text(c) for c in item["choices"]]
        if word_count(item["question"]) + sum(map(word_count, options)) > MAX_QA_WORDS:
            continue
        shuffled, label = shuffle_options(options, int(item["answer"]), rng)
        rows.append(make_row(source="mmlu_aux", decision_type="categorical", query=clean_text(item["question"]),
                             options=shuffled, label=label, subtopic="exam questions"))
        if len(rows) == limit:
            break
    return rows


def _choice_rows(df, rng, limit, source, subtopic, slice_):
    rows = []
    for r in sample_frame(df, limit, rng).itertuples():
        keys, texts = list(r.choices["label"]), [clean_text(t) for t in r.choices["text"]]
        if r.answerKey not in keys:
            continue
        options, label = shuffle_options(texts, keys.index(r.answerKey), rng)
        rows.append(make_row(source=source, decision_type="categorical", query=clean_text(r.question),
                             options=options, label=label, subtopic=subtopic, slice_=slice_))
    return rows


def arc_rows(df, rng, limit, challenge):
    return _choice_rows(df, rng, limit, "arc", "science exam", "hard_ambiguous" if challenge else "core")


def csqa_rows(df, rng, limit):
    return _choice_rows(df, rng, limit, "commonsenseqa", "common sense", "core")
