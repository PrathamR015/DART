"""Quality report for the built dataset: coverage, balance, token lengths and leakage-style checks."""
import statistics
from collections import Counter, defaultdict

MIN_ROWS_FOR_BASELINE = 200


def _counts(rows, key):
    return dict(Counter(r[key] for r in rows).most_common())


def _label_distribution(rows):
    by_type = defaultdict(Counter)
    for r in rows:
        by_type[r["decision_type"]][r["label"]] += 1
    return {t: dict(sorted(c.items())) for t, c in by_type.items()}


def _majority_baseline(rows):
    """How well always guessing the most common label position does, per source and decision type.

    Scores near chance mean the label cannot be guessed from position or template alone.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[(r["source"], r["decision_type"])].append(r)
    report = {}
    for (source, decision_type), part in groups.items():
        if len(part) < MIN_ROWS_FOR_BASELINE:
            continue
        top = Counter(r["label"] for r in part).most_common(1)[0][1] / len(part)
        chance = statistics.mean(1 / len(r["options"]) for r in part)
        report[f"{source}/{decision_type}"] = {"rows": len(part), "majority": round(top, 3), "chance": round(chance, 3)}
    return report


def _stated_answer_rate(rows):
    """Share of categorical rows whose correct option text already appears in the query."""
    categorical = [r for r in rows if r["decision_type"] == "categorical" and len(r["target_option"]) >= 3]
    if not categorical:
        return 0.0
    stated = sum(r["target_option"].lower() in r["query"].lower() for r in categorical)
    return round(stated / len(categorical), 4)


def _token_stats(lengths):
    if not lengths:
        return {}
    ordered = sorted(lengths)
    return {"mean": round(statistics.mean(ordered), 1), "median": ordered[len(ordered) // 2],
            "p95": ordered[int(len(ordered) * 0.95)], "max": ordered[-1]}


def _split_summary(rows, lengths):
    grouped = [r for r in rows if r["group_id"]]
    return {
        "rows": len(rows),
        "by_decision_type": _counts(rows, "decision_type"),
        "by_source": _counts(rows, "source"),
        "by_slice": _counts(rows, "slice"),
        "option_counts": dict(sorted(Counter(len(r["options"]) for r in rows).items())),
        "label_distribution": _label_distribution(rows),
        "rows_in_shared_context_groups": len(grouped),
        "shared_context_groups": len({r["group_id"] for r in grouped}),
        "prompt_tokens": _token_stats(lengths),
        "stated_answer_rate_categorical": _stated_answer_rate(rows),
    }


def build_report(splits, token_lengths, stages):
    report = {"stages": stages, "splits": {}}
    for name, rows in splits.items():
        report["splits"][name] = _split_summary(rows, token_lengths.get(name, []))
    report["majority_baseline_train"] = _majority_baseline(splits["train"])
    return report
