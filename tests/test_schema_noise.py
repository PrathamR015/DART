import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_prep.noise import add_noise  # noqa: E402
from data_prep.schema import (  # noqa: E402
    clean_text,
    format_prompt,
    make_row,
    sample_options,
    shuffle_options,
    validate_row,
)


def test_make_row_sets_target_option_from_label():
    row = make_row(source="s", decision_type="binary", query="Q?", options=["yes", "no"], label=1)
    assert row["target_option"] == "no"
    assert row["slice"] == "core"


def test_make_row_rejects_label_out_of_range():
    with pytest.raises(ValueError):
        make_row(source="s", decision_type="binary", query="Q?", options=["yes", "no"], label=2)


def test_validate_row_accepts_good_row():
    row = make_row(source="s", decision_type="binary", query="Q?", options=["yes", "no"], label=0)
    assert validate_row(row) is None


@pytest.mark.parametrize(
    "options",
    [["only"], ["a", "A"], ["a", ""], [str(i) for i in range(13)]],
)
def test_validate_row_rejects_bad_option_lists(options):
    row = {"query": "Q?", "options": options, "label": 0, "target_option": options[0]}
    assert validate_row(row) is not None


def test_validate_row_rejects_empty_query():
    row = {"query": "  ", "options": ["a", "b"], "label": 0, "target_option": "a"}
    assert validate_row(row) is not None


def test_format_prompt_matches_template_with_and_without_context():
    row = make_row(source="s", decision_type="binary", query="Q?", options=["yes", "no"], label=0)
    assert format_prompt(row) == "Query: Q?\nOptions:\nA) yes\nB) no\nAnswer:"
    with_ctx = {**row, "context": "Some passage."}
    assert format_prompt(with_ctx).startswith("Context: Some passage.\nQuery: Q?")


def test_prefix_plus_suffix_equals_the_full_prompt():
    from data_prep.schema import format_prefix, format_suffix

    plain = make_row(source="s", decision_type="binary", query="Q?", options=["yes", "no"], label=0)
    with_ctx = {**plain, "context": "Some passage. More text."}
    for row in (plain, with_ctx):
        assert format_prefix(row["context"]) + format_suffix(row["query"], row["options"]) == format_prompt(row)
    assert format_prefix("") == ""
    assert format_prefix("Some passage.") == "Context: Some passage.\n"


def test_shuffle_options_keeps_the_correct_answer():
    options = ["red", "green", "blue", "gold"]
    for seed in range(20):
        shuffled, label = shuffle_options(options, 2, random.Random(seed))
        assert sorted(shuffled) == sorted(options)
        assert shuffled[label] == "blue"


def test_shuffle_options_leaves_positional_options_alone():
    options = ["Both A and B", "A only", "B only", "Neither"]
    shuffled, label = shuffle_options(options, 0, random.Random(1))
    assert shuffled == options and label == 0


def test_sample_options_includes_correct_and_requested_count():
    pool = [f"opt{i}" for i in range(20)]
    options, label = sample_options("opt3", pool, 5, random.Random(4))
    assert len(options) == 5 and options[label] == "opt3" and len(set(options)) == 5


def test_clean_text_collapses_whitespace_and_escaped_newlines():
    assert clean_text("a  b\\n c\n d ") == "a b c d"


def test_add_noise_is_deterministic_for_a_seed():
    text = "Please describe the patient with severe abdominal pain and vomiting"
    assert add_noise(text, random.Random(3)) == add_noise(text, random.Random(3))


def test_add_noise_changes_text_but_keeps_numbers():
    text = "The patient has a fever of 39 degrees and severe headache since yesterday"
    changed = [add_noise(text, random.Random(s)) for s in range(10)]
    assert any(c != text for c in changed)
    assert all("39" in c for c in changed)
