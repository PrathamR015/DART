import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_prep.salvage import (  # noqa: E402
    dedupe_rows,
    find_generic_skeletons,
    salvage,
    skeleton,
    strip_ref,
)


def make_row(query, label=0, dtype="binary", options=None, domain="law", subtopic="evidence", **extra):
    options = options or ["yes", "no"]
    row = {
        "id": extra.pop("id", query[:20]),
        "domain": domain,
        "subtopic": subtopic,
        "slice": "core",
        "difficulty": "easy",
        "decision_type": dtype,
        "context": "",
        "query": query,
        "options": options,
        "label": label,
        "target_option": options[label],
        "rationale": "r",
    }
    row.update(extra)
    return row


def generic_rows(n=30):
    return [
        make_row(f"In law ops (evidencex), rate the risk level as low, medium or high. [ref {i}]",
                 label=i % 3, dtype="ordinal_lmh", options=["low", "medium", "high"],
                 subtopic="evidencex", id=f"g{i}")
        for i in range(n)
    ]


def test_strip_ref_removes_reference_tag():
    assert strip_ref("Is water wet? [ref 123]") == "Is water wet?"


def test_strip_ref_leaves_plain_query_alone():
    assert strip_ref("Is water wet?") == "Is water wet?"


def test_skeleton_ignores_domain_subtopic_numbers_and_ref():
    a = make_row("In law operations (evidence), rate risk 5. [ref 1]")
    b = make_row("In law operations (evidence), rate risk 9. [ref 2]")
    assert skeleton(a) == skeleton(b)


def test_find_generic_skeletons_flags_large_random_label_groups():
    generic = find_generic_skeletons(generic_rows(), min_group=15, max_majority=0.6)
    assert len(generic) == 1


def test_find_generic_skeletons_keeps_repeated_facts_with_consistent_label():
    rows = [make_row("Does an advantage point win the game in tennis? [ref %d]" % i, label=0, id=str(i))
            for i in range(30)]
    assert find_generic_skeletons(rows, min_group=15, max_majority=0.6) == set()


def test_dedupe_keeps_one_of_identical_questions():
    rows = [make_row("Same question? [ref 1]", id="a"), make_row("Same question? [ref 2]", id="b")]
    kept, dup, conflict = dedupe_rows(rows)
    assert len(kept) == 1 and dup == 1 and conflict == 0


def test_dedupe_drops_all_when_labels_conflict():
    rows = [make_row("Same question?", label=0, id="a"), make_row("Same question?", label=1, id="b")]
    kept, dup, conflict = dedupe_rows(rows)
    assert kept == [] and conflict == 2


def test_salvage_drops_scale_decimal_and_generic_and_strips_refs():
    decimal = [make_row("Index is 1.5, pick it", dtype="scale_decimal", options=["1.0", "1.5"], label=1, id="d1")]
    real = [make_row("Is a 500 FICO borrower eligible for prime mortgages? [ref 9]", label=1, id="r1")]
    kept, report = salvage(generic_rows() + decimal + real)
    assert [r["id"] for r in kept] == ["r1"]
    assert kept[0]["query"] == "Is a 500 FICO borrower eligible for prime mortgages?"
    assert kept[0]["source"] == "synthetic_v1"
    assert report["dropped_scale_decimal"] == 1
    assert report["dropped_generic_template"] == 30


def test_salvage_catches_typo_variants_of_generic_templates():
    noisy = make_row("in law opreations (evidencex), rate the rsik level as low, medium or high",
                     label=1, dtype="ordinal_lmh", options=["low", "medium", "high"], id="n1")
    kept, _ = salvage(generic_rows() + [noisy])
    assert kept == []
