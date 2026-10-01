"""Shared row schema, prompt template and validation for the DART dataset."""
import random
import re

LETTERS = "ABCDEFGHIJKL"
MAX_OPTIONS = len(LETTERS)
MAX_PROMPT_TOKENS = 256
POSITIONAL_PHRASES = ("above", "both", "all of", "none of", "neither")
SPACES = re.compile(r"\s+")


def make_row(*, source, decision_type, query, options, label, domain="general", subtopic="",
             slice_="core", difficulty="medium", context="", group_id="", rationale=""):
    options = list(options)
    if not 0 <= label < len(options):
        raise ValueError(f"label {label} out of range for {len(options)} options")
    return {
        "id": "",
        "source": source,
        "group_id": group_id,
        "domain": domain,
        "subtopic": subtopic,
        "slice": slice_,
        "difficulty": difficulty,
        "decision_type": decision_type,
        "context": context,
        "query": query,
        "options": options,
        "label": label,
        "target_option": options[label],
        "rationale": rationale,
    }


def validate_row(row):
    """Return an error message, or None when the row is well formed."""
    options = row["options"]
    if not 2 <= len(options) <= MAX_OPTIONS:
        return f"option count {len(options)} outside 2..{MAX_OPTIONS}"
    if any(not str(o).strip() for o in options):
        return "empty option"
    if len({str(o).strip().lower() for o in options}) != len(options):
        return "duplicate options"
    if not isinstance(row["label"], int) or not 0 <= row["label"] < len(options):
        return "label out of range"
    if options[row["label"]] != row["target_option"]:
        return "target_option does not match label"
    if not row["query"].strip():
        return "empty query"
    return None


def format_prefix(context):
    """The part of the prompt shared by every question about one context (empty without a context)."""
    return f"Context: {context}\n" if context else ""


def format_suffix(query, options):
    """The part of the prompt specific to one question."""
    lines = [f"Query: {query}", "Options:"]
    lines.extend(f"{LETTERS[i]}) {option}" for i, option in enumerate(options))
    lines.append("Answer:")
    return "\n".join(lines)


def format_prompt(row):
    """The fixed plain-text prompt the model sees (no chat template)."""
    return format_prefix(row.get("context", "")) + format_suffix(row["query"], row["options"])


def shuffle_options(options, label, rng):
    """Shuffle options, keeping the label on the correct one. Positional options stay put."""
    if any(p in str(o).lower() for o in options for p in POSITIONAL_PHRASES):
        return list(options), label
    order = list(range(len(options)))
    rng.shuffle(order)
    return [options[i] for i in order], order.index(label)


def sample_options(correct, pool, count, rng):
    """Correct option plus random distractors from pool, shuffled. Returns (options, label)."""
    distractors = [o for o in dict.fromkeys(pool) if o != correct]
    chosen = rng.sample(distractors, min(count - 1, len(distractors))) + [correct]
    rng.shuffle(chosen)
    return chosen, chosen.index(correct)


def clean_text(text):
    return SPACES.sub(" ", str(text).replace("\\n", " ").replace("\\", " ")).strip()


def word_count(text):
    return len(str(text).split())


def with_seed(seed):
    return random.Random(seed)
