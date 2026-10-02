import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND.parents[1]))

from dartweb.options import OptionsError, parse_options  # noqa: E402


def values(text):
    return parse_options(text).options


def kind(text):
    return parse_options(text).kind


def test_comma_separated_list():
    parsed = parse_options("yes, no, maybe")
    assert parsed.options == ["yes", "no", "maybe"] and parsed.kind == "list"


def test_pipe_separator_allows_commas_inside_options():
    assert values("Paris, France | London, UK | Rome, Italy") == ["Paris, France", "London, UK", "Rome, Italy"]


def test_list_ignores_extra_spaces_and_empty_items():
    assert values(" a , , b ,") == ["a", "b"]


@pytest.mark.parametrize("text", ["only", "a, A", "", "  ", ",", ", ,"])
def test_bad_lists_are_rejected(text):
    with pytest.raises(OptionsError):
        parse_options(text)


def test_more_than_twelve_list_options_are_rejected():
    with pytest.raises(OptionsError, match="12"):
        parse_options(", ".join(str(i) for i in range(13)))


def test_option_that_is_too_long_is_rejected():
    with pytest.raises(OptionsError):
        parse_options("a, " + "x" * 200)


@pytest.mark.parametrize("text", ["0-5", "0 to 5", "0..5", "0–5", "  0 - 5  ", "0 TO 5"])
def test_integer_range_spellings(text):
    assert values(text) == ["0", "1", "2", "3", "4", "5"] and kind(text) == "integer_range"


def test_ranges_with_different_sizes():
    assert len(values("1-5")) == 5 and len(values("0-10")) == 11 and len(values("1-10")) == 10


def test_integer_range_with_integer_step():
    assert values("0 to 10 step 2") == ["0", "2", "4", "6", "8", "10"]
    assert kind("0 to 10 step 2") == "integer_range"


def test_decimal_endpoints_give_tenth_steps():
    assert values("0.0-1.0") == [f"0.{i}" for i in range(10)] + ["1.0"]
    assert kind("0.0-1.0") == "decimal_range"


def test_decimal_step_formats_values_consistently():
    assert values("0-5 step 0.5") == ["0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"]
    assert kind("0-5 step 0.5") == "decimal_range"


def test_decimal_values_are_exact_not_floating_point_noise():
    assert values("0.1-0.4 step 0.1") == ["0.1", "0.2", "0.3", "0.4"]


def test_negative_ranges_use_to():
    assert values("-1 to 1") == ["-1", "0", "1"]
    assert values("-1 to 1 step 0.5") == ["-1.0", "-0.5", "0.0", "0.5", "1.0"]


def test_step_that_does_not_divide_the_range_stops_before_the_end():
    assert values("0-5 step 2") == ["0", "2", "4"]


@pytest.mark.parametrize("text", ["0.0-5.0", "0-100", "0-20"])
def test_ranges_with_too_many_values_suggest_a_step(text):
    with pytest.raises(OptionsError, match="step"):
        parse_options(text)


@pytest.mark.parametrize("text", ["5-0", "3-3", "0-5 step 0", "0-5 step 10"])
def test_invalid_ranges_are_rejected(text):
    with pytest.raises(OptionsError):
        parse_options(text)


def test_text_that_only_looks_like_a_range_is_treated_as_a_list_item():
    with pytest.raises(OptionsError):  # one item only
        parse_options("well-known")
    assert values("well-known, lesser-known") == ["well-known", "lesser-known"]


@pytest.mark.parametrize("text,step", [("0-100", "10"), ("0.0-5.0", "0.5"), ("0-20", "2")])
def test_suggested_step_is_written_plainly(text, step):
    with pytest.raises(OptionsError, match=f"'{text} step {step}'"):
        parse_options(text)


def test_an_explicit_list_is_used_as_given_after_trimming():
    parsed = parse_options([" hospital ", "airport", "", "school, bank"])
    assert parsed.options == ["hospital", "airport", "school, bank"] and parsed.kind == "list"


@pytest.mark.parametrize("items", [["only"], ["a", "A"], [], [str(i) for i in range(13)], ["a", "x" * 101]])
def test_bad_explicit_lists_are_rejected(items):
    with pytest.raises(OptionsError):
        parse_options(items)
