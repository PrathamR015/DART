"""Turns the options field into a list of option strings: a comma/pipe list, or a numeric range.

Ranges:  0-5   0 to 5   0..5   0-5 step 0.5   -1 to 1
Integers by default. Decimal when an endpoint or the step has a decimal point. Without a step, the step is
one unit of the most precise endpoint (0.0-1.0 steps by 0.1). At most 12 values: the model has 12 option slots.
"""
import re
from dataclasses import dataclass
from decimal import Decimal

from data_prep.schema import MAX_OPTIONS

MAX_OPTION_LENGTH = 100
NUMBER = r"-?\d+(?:\.\d+)?"
RANGE = re.compile(
    rf"^\s*({NUMBER})\s*(?:to|\.\.|[-–—])\s*({NUMBER})(?:\s+step\s+({NUMBER}))?\s*$", re.IGNORECASE)


class OptionsError(ValueError):
    """The options field could not be turned into a usable option list."""


@dataclass(frozen=True)
class ParsedOptions:
    options: list
    kind: str  # "list", "integer_range" or "decimal_range"


def _decimal_places(text):
    return len(text.split(".")[1]) if "." in text else 0


def _check_count(options):
    if len(options) < 2:
        raise OptionsError("need at least 2 options")
    if len(options) > MAX_OPTIONS:
        raise OptionsError(f"at most {MAX_OPTIONS} options are supported, got {len(options)}")


def _parse_list(text):
    separator = "|" if "|" in text else ","
    options = [part.strip() for part in text.split(separator) if part.strip()]
    _check_count(options)
    if any(len(option) > MAX_OPTION_LENGTH for option in options):
        raise OptionsError(f"each option must be at most {MAX_OPTION_LENGTH} characters")
    if len({option.lower() for option in options}) != len(options):
        raise OptionsError("options must be distinct")
    return ParsedOptions(options, "list")


def _parse_range(match):
    start_text, end_text, step_text = match.groups()
    places = max(_decimal_places(start_text), _decimal_places(end_text), _decimal_places(step_text or ""))
    start, end = Decimal(start_text), Decimal(end_text)
    step = Decimal(step_text) if step_text else Decimal(1).scaleb(-places)
    if step <= 0:
        raise OptionsError("step must be greater than 0")
    if start >= end:
        raise OptionsError("the range must start below where it ends")
    count = int((end - start) / step) + 1
    if count > MAX_OPTIONS:
        raise OptionsError(f"that range has {count} values but at most {MAX_OPTIONS} are supported; "
                           f"add a larger step, for example '{start_text}-{end_text} step {_suggest_step(start, end)}'")
    options = [format(start + i * step, f".{places}f") for i in range(count)]
    _check_count(options)
    return ParsedOptions(options, "decimal_range" if places else "integer_range")


def _suggest_step(start, end):
    span = end - start
    for candidate in (Decimal("0.25"), Decimal("0.5"), Decimal(1), Decimal(2), Decimal(5), Decimal(10), Decimal(50)):
        if span / candidate + 1 <= MAX_OPTIONS:
            return candidate.normalize()
    return span.normalize()


def parse_options(text):
    text = (text or "").strip()
    if not text:
        raise OptionsError("enter the options, for example 'yes, no' or a range like '0-5'")
    match = RANGE.match(text)
    return _parse_range(match) if match else _parse_list(text)
