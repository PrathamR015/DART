"""Evaluation records and summaries: accuracy, top-2, per-group accuracy and MAE for ordered types."""
from collections import Counter, defaultdict

from dart.loss import ORDERED_TYPES

GROUP_KEYS = ("decision_type", "source", "slice")


def make_records(rows, probabilities):
    """One record per row from a batch of probability vectors (rows and probabilities aligned)."""
    records = []
    for row, probs in zip(rows, probabilities):
        valid = [float(p) for p in probs[: len(row["options"])]]
        ranking = sorted(range(len(valid)), key=lambda i: valid[i], reverse=True)
        records.append({
            "decision_type": row["decision_type"], "source": row["source"], "slice": row["slice"],
            "correct": ranking[0] == row["label"], "top2": row["label"] in ranking[:2],
            "abs_error": abs(ranking[0] - row["label"]), "confidence": valid[ranking[0]],
            "pred": ranking[0], "n_options": len(valid),
        })
    return records


def _group_summary(records):
    n = len(records)
    summary = {"n": n, "accuracy": round(sum(r["correct"] for r in records) / n, 4),
               "top2": round(sum(r["top2"] for r in records) / n, 4)}
    ordered = [r for r in records if r["decision_type"] in ORDERED_TYPES]
    if ordered:
        summary["mae"] = round(sum(r["abs_error"] for r in ordered) / len(ordered), 4)
        summary["within1"] = round(sum(r["abs_error"] <= 1 for r in ordered) / len(ordered), 4)
    return summary


def _prediction_spread(records):
    """Which option positions were predicted, and which were never predicted (dead classes)."""
    predicted = Counter(r["pred"] for r in records)
    options = max(r["n_options"] for r in records)
    return {"predicted": dict(sorted(predicted.items())),
            "never_predicted": [i for i in range(options) if i not in predicted]}


def summarize(records):
    if not records:
        return {"overall": {"n": 0, "accuracy": 0.0, "top2": 0.0}}
    report = {"overall": _group_summary(records),
              "mean_confidence": round(sum(r["confidence"] for r in records) / len(records), 4)}
    for key in GROUP_KEYS:
        groups = defaultdict(list)
        for r in records:
            groups[r[key]].append(r)
        report[f"by_{key}"] = {name: _group_summary(part) for name, part in sorted(groups.items())}
    for name, summary in report["by_decision_type"].items():
        if name in ORDERED_TYPES:
            summary.update(_prediction_spread([r for r in records if r["decision_type"] == name]))
    return report
