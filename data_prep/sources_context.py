"""Converters for datasets that come with a passage: yes/no reading and shared-context multi-question."""
import hashlib

from data_prep.schema import LETTERS, clean_text, make_row, word_count
from data_prep.sources_text import sample_frame

MAX_PASSAGE_WORDS = 140
MAX_PUBMED_WORDS = 160
MAX_ARTICLE_WORDS = 150
MAX_QUESTIONS_PER_ARTICLE = 5
YES_NO = ["yes", "no"]


def _question(text):
    text = clean_text(text)
    text = text[:1].upper() + text[1:]
    return text if text.endswith("?") else text + "?"


def _balanced_yes_no(frame, is_yes, per_class, rng):
    yes, no = frame[is_yes], frame[~is_yes]
    count = min(per_class, len(yes), len(no))
    return sample_frame(yes, count, rng), sample_frame(no, count, rng)


def _yes_no_rows(source, subtopic, yes, no, context_of):
    return [
        make_row(source=source, decision_type="binary", query=_question(r.question), options=YES_NO,
                 label=label, context=clean_text(context_of(r)), subtopic=subtopic)
        for frame, label in ((yes, 0), (no, 1)) for r in frame.itertuples()
    ]


def boolq_rows(df, rng, per_class):
    frame = df[df.passage.map(word_count) <= MAX_PASSAGE_WORDS]
    yes, no = _balanced_yes_no(frame, frame.answer.astype(bool), per_class, rng)
    return _yes_no_rows("boolq", "reading comprehension", yes, no, lambda r: r.passage)


def _abstract(row):
    return " ".join(row.context["contexts"])


def pubmedqa_rows(df, rng, per_class):
    frame = df[df.final_decision.isin(YES_NO)]
    frame = frame[frame.apply(lambda r: word_count(_abstract(r)) <= MAX_PUBMED_WORDS, axis=1)]
    yes, no = _balanced_yes_no(frame, frame.final_decision == "yes", per_class, rng)
    return _yes_no_rows("pubmedqa", "biomedical abstracts", yes, no, _abstract)


def race_rows(df, rng, limit_groups):
    """Several questions over one short article, kept together by group_id."""
    frame = df[df.article.map(word_count) <= MAX_ARTICLE_WORDS]
    groups = [(article, part) for article, part in frame.groupby("article", sort=False) if len(part) >= 2]
    rng.shuffle(groups)
    rows = []
    for article, part in groups[:limit_groups]:
        group_id = "race_" + hashlib.md5(article.encode("utf8")).hexdigest()[:10]
        for r in part.head(MAX_QUESTIONS_PER_ARTICLE).itertuples():
            options = [clean_text(o) for o in r.options]
            rows.append(make_row(source="race", decision_type="categorical", query=clean_text(r.question),
                                 options=options, label=LETTERS.index(r.answer), context=clean_text(article),
                                 group_id=group_id, subtopic="reading comprehension"))
    return rows
