"""Realistic text noise (typos, slang, dropped punctuation) for the noisy slice."""
import re

ABBREVIATIONS = {
    "you": "u", "please": "pls", "because": "bc", "patient": "pt", "with": "w/",
    "without": "w/o", "and": "&", "people": "ppl", "message": "msg", "information": "info",
}
MIN_TYPO_LENGTH = 4
WORD_TYPO_RATE = 0.12
ABBREVIATION_RATE = 0.5
LOWERCASE_RATE = 0.4
STRIP_PUNCTUATION_RATE = 0.4
HAS_DIGIT = re.compile(r"\d")


def _typo(word, rng):
    if len(word) < MIN_TYPO_LENGTH or HAS_DIGIT.search(word):
        return word
    i = rng.randrange(1, len(word) - 1)
    kind = rng.choice(("swap", "drop", "dup"))
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind == "drop":
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def _noisy_word(word, rng):
    abbreviation = ABBREVIATIONS.get(word.lower())
    if abbreviation and rng.random() < ABBREVIATION_RATE:
        return abbreviation
    return _typo(word, rng) if rng.random() < WORD_TYPO_RATE else word


def add_noise(text, rng):
    """Return a noisy copy of text. Words containing digits are never altered."""
    original = text.split(" ")
    words = [_noisy_word(w, rng) for w in original]
    if words == original:
        longest = max(range(len(words)), key=lambda i: len(words[i]))
        words[longest] = _typo(words[longest], rng)
    noisy = " ".join(words)
    if rng.random() < LOWERCASE_RATE:
        noisy = noisy.lower()
    if rng.random() < STRIP_PUNCTUATION_RATE:
        noisy = noisy.rstrip(".?!")
    return noisy
