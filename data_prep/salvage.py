"""Phase 0, step 1: keep only the informative rows of the first synthetic dataset.

Drops the copy-task scale type, the generic templates whose labels are random,
near-copies of those templates (typo variants), and duplicate or conflicting
questions. Input rows are never mutated.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

SOURCE = "synthetic_v1"
DROPPED_TYPE = "scale_decimal"
MIN_GROUP = 15
MAX_MAJORITY = 0.6
NEAR_TEMPLATE_SIMILARITY = 0.8

REF_TAG = re.compile(r"\s*\[ref \d+\]")
NUMBER = re.compile(r"\d+(\.\d+)?")
NON_ALNUM = re.compile(r"[^a-z0-9 ]")
SPACES = re.compile(r"\s+")


def strip_ref(query):
    return REF_TAG.sub("", query).strip()


def normalize(query):
    return SPACES.sub(" ", NON_ALNUM.sub("", strip_ref(query).lower())).strip()


def skeleton(row):
    text = strip_ref(row["query"]).lower()
    for word in (row["domain"], row["subtopic"], row["subtopic"].replace("_", " ")):
        text = text.replace(word.lower(), "<x>")
    text = NUMBER.sub("<n>", text)
    return SPACES.sub(" ", text).strip(" .?")


def _key(row):
    return (row["decision_type"], skeleton(row))


def find_generic_skeletons(rows, min_group=MIN_GROUP, max_majority=MAX_MAJORITY):
    """Skeletons shared by many rows whose labels look random (no majority label)."""
    groups = defaultdict(list)
    for row in rows:
        groups[_key(row)].append(row["label"])
    generic = set()
    for key, labels in groups.items():
        if len(labels) < min_group:
            continue
        majority_share = Counter(labels).most_common(1)[0][1] / len(labels)
        if majority_share < max_majority:
            generic.add(key)
    return generic


def _near_template_rows(rows, generic_keys):
    """Indexes of rows whose skeleton is almost identical to a generic template."""
    templates = defaultdict(list)
    for decision_type, text in generic_keys:
        templates[decision_type].append(text)
    flagged = set()
    for decision_type, texts in templates.items():
        candidates = [(i, skeleton(r)) for i, r in enumerate(rows) if r["decision_type"] == decision_type]
        if not candidates:
            continue
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(texts)
        sims = (vectorizer.transform([t for _, t in candidates]) @ vectorizer.transform(texts).T).max(axis=1)
        for (index, _), sim in zip(candidates, sims.toarray().ravel()):
            if sim >= NEAR_TEMPLATE_SIMILARITY:
                flagged.add(index)
    return flagged


def dedupe_rows(rows):
    """Keep one row per identical question; drop every copy when their answers conflict."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["decision_type"], normalize(row["query"]), tuple(row["options"]))].append(row)
    kept, duplicates, conflicts = [], 0, 0
    for group in groups.values():
        if len({r["target_option"] for r in group}) > 1:
            conflicts += len(group)
            continue
        kept.append(group[0])
        duplicates += len(group) - 1
    order = {r["id"]: i for i, r in enumerate(rows)}
    return sorted(kept, key=lambda r: order[r["id"]]), duplicates, conflicts


def salvage(rows):
    typed = [r for r in rows if r["decision_type"] != DROPPED_TYPE]
    generic = find_generic_skeletons(typed)
    specific = [r for r in typed if _key(r) not in generic]
    near = _near_template_rows(specific, generic)
    informative = [r for i, r in enumerate(specific) if i not in near]
    cleaned = [{**r, "query": strip_ref(r["query"]), "source": SOURCE} for r in informative]
    kept, duplicates, conflicts = dedupe_rows(cleaned)
    report = {
        "input_rows": len(rows),
        "dropped_scale_decimal": len(rows) - len(typed),
        "dropped_generic_template": len(typed) - len(specific),
        "dropped_near_template": len(near),
        "dropped_duplicates": duplicates,
        "dropped_conflicts": conflicts,
        "kept": len(kept),
        "kept_by_type": dict(Counter(r["decision_type"] for r in kept)),
        "kept_by_slice": dict(Counter(r["slice"] for r in kept)),
    }
    return kept, report


def main(root=Path(__file__).resolve().parents[1] / "assets"):
    source = root / "datasets" / "dart_train_20k.jsonl"
    out_dir = root / "processed"
    with source.open(encoding="utf8") as handle:
        rows = [json.loads(line) for line in handle]
    kept, report = salvage(rows)
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "salvaged.jsonl").open("w", encoding="utf8") as handle:
        handle.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
    (out_dir / "salvage_report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
