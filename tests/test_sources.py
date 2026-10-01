import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_prep.schema import validate_row  # noqa: E402
from data_prep.sources_context import boolq_rows, race_rows  # noqa: E402
from data_prep.sources_scales import helpsteer_rows, sst5_rows, stsb_rows  # noqa: E402
from data_prep.sources_text import arc_rows, clinc_rows, parse_class_names  # noqa: E402


def rng():
    return random.Random(7)


def test_parse_class_names_reads_first_names_block():
    readme = "features:\n  class_label:\n    names:\n      '0': alpha\n      '1': beta_gamma\n  splits:\n"
    assert parse_class_names(readme) == ["alpha", "beta gamma"]


def test_clinc_rows_have_correct_label_and_varied_option_counts():
    names = [f"intent {i}" for i in range(30)]
    df = pd.DataFrame({"text": [f"utterance {i}" for i in range(40)], "intent": [i % 30 for i in range(40)]})
    rows = clinc_rows(df, names, rng(), limit=40)
    assert len(rows) == 40
    assert all(validate_row(r) is None for r in rows)
    assert len({len(r["options"]) for r in rows}) > 1
    assert all(r["decision_type"] == "categorical" for r in rows)


def test_arc_rows_map_numeric_answer_keys():
    df = pd.DataFrame({
        "question": ["Which is warm?", "Pick one"],
        "choices": [
            {"text": np.array(["ice", "sun", "snow"]), "label": np.array(["A", "B", "C"])},
            {"text": np.array(["w", "x", "y", "z"]), "label": np.array(["1", "2", "3", "4"])},
        ],
        "answerKey": ["B", "3"],
    })
    rows = arc_rows(df, rng(), limit=10, challenge=True)
    assert {r["target_option"] for r in rows} == {"sun", "y"}
    assert all(r["slice"] == "hard_ambiguous" for r in rows)


def test_stsb_rows_round_scores_and_use_both_scales():
    scores = [0.0, 1.2, 2.5, 3.8, 4.4, 5.0] * 10
    df = pd.DataFrame({"sentence1": ["a b c"] * 60, "sentence2": ["d e f"] * 60, "label": scores})
    rows = stsb_rows(df, rng(), per_class=100)
    by_type = {t: [r for r in rows if r["decision_type"] == t] for t in ("scale_0_5", "scale_0_10")}
    assert by_type["scale_0_5"] and by_type["scale_0_10"]
    assert all(len(r["options"]) == 6 for r in by_type["scale_0_5"])
    assert all(len(r["options"]) == 11 for r in by_type["scale_0_10"])
    assert all(validate_row(r) is None for r in rows)


def test_sst5_rows_map_bands_to_low_medium_high():
    df = pd.DataFrame({"text": [f"movie {i}" for i in range(100)], "label": [i % 5 for i in range(100)]})
    rows = sst5_rows(df, rng(), per_class=50)
    lmh = [r for r in rows if r["decision_type"] == "ordinal_lmh"]
    assert lmh and {r["target_option"] for r in lmh} == {"low", "medium", "high"}
    assert all(len(r["options"]) == 5 for r in rows if r["decision_type"] == "scale_1_5")


def test_rating_rows_drop_reviews_that_state_their_own_rating():
    texts = ["Great place but why 2 stars? staff", "I give it five stars", "Solid 4/5 experience", "Friendly staff"]
    df = pd.DataFrame({"text": texts * 25, "label": [i % 5 for i in range(100)]})
    rows = sst5_rows(df, rng(), per_class=100)
    assert rows and all("star" not in r["query"].lower() and "/5" not in r["query"] for r in rows)


def test_helpsteer_overall_quality_label_is_scaled_mean():
    df = pd.DataFrame({
        "prompt": ["hi"] * 40, "response": ["short answer"] * 40,
        "helpfulness": [4] * 40, "correctness": [4] * 40, "coherence": [4] * 40,
        "complexity": [1] * 40, "verbosity": [2] * 40,
    })
    rows = helpsteer_rows(df, rng(), per_class=50)
    quality = [r for r in rows if r["decision_type"] == "scale_0_10"]
    assert quality and all(r["target_option"] == "10" for r in quality)


def test_boolq_rows_balance_yes_no_and_drop_long_passages():
    df = pd.DataFrame({
        "question": [f"is thing {i} true" for i in range(30)],
        "answer": [i % 3 != 0 for i in range(30)],
        "passage": ["short passage"] * 29 + ["word " * 500],
    })
    rows = boolq_rows(df, rng(), per_class=50)
    yes = sum(r["target_option"] == "yes" for r in rows)
    assert yes == len(rows) - yes
    assert all(len(r["context"].split()) <= 140 for r in rows)


def test_race_rows_group_questions_by_article():
    article = "A short article about dogs."
    df = pd.DataFrame({
        "article": [article, article, "Another article."],
        "question": ["Q1?", "Q2?", "Q3?"],
        "options": [np.array(["a", "b", "c", "d"])] * 3,
        "answer": ["A", "C", "B"],
    })
    rows = race_rows(df, rng(), limit_groups=10)
    assert len(rows) == 2  # single-question articles are not shared-context groups
    assert len({r["group_id"] for r in rows}) == 1
    assert {r["target_option"] for r in rows} == {"a", "c"}
