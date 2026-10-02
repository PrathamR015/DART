"""Turns stored decisions into DART training rows (only the ones a user labelled)."""
from data_prep.schema import make_row, validate_row

SOURCE = "mark2_user"
DEFAULT_SLICE = "core"
SCALE_TYPES = {(6, 0): "scale_0_5", (5, 1): "scale_1_5", (11, 0): "scale_0_10"}


def _numbers(options):
    try:
        return [float(option) for option in options]
    except ValueError:
        return None


def infer_decision_type(options):
    lowered = {option.lower() for option in options}
    if lowered == {"yes", "no"}:
        return "binary"
    if lowered == {"low", "medium", "high"}:
        return "ordinal_lmh"
    numbers = _numbers(options)
    if numbers is None:
        return "categorical"
    integers = all(n == int(n) for n in numbers)
    steps_of_one = all(b - a == 1 for a, b in zip(numbers, numbers[1:]))
    if integers and steps_of_one:
        return SCALE_TYPES.get((len(numbers), int(numbers[0])), "scale_other")
    return "scale_other"


def build_rows(records):
    """Returns (rows, skipped). A record without feedback, or with an invalid row, is skipped."""
    rows, skipped = [], 0
    for record in records:
        feedback = record.get("feedback")
        options = record["options"]
        if not feedback or feedback["correct_option"] not in options:
            skipped += 1
            continue
        row = make_row(source=SOURCE, decision_type=infer_decision_type(options), query=record["query"],
                       options=options, label=options.index(feedback["correct_option"]),
                       subtopic="user input", slice_=DEFAULT_SLICE, context=record.get("context", ""),
                       group_id=record.get("group_id", ""))
        row["id"] = f"{SOURCE}_{record['_id']}"
        if validate_row(row) is not None:
            skipped += 1
            continue
        rows.append(row)
    return rows, skipped


def unlabeled(records):
    """Stored decisions nobody has labelled yet, in a small form for later labelling."""
    return [{"id": str(r["_id"]), "context": r.get("context", ""), "query": r["query"], "options": r["options"],
             "model_decision": r["decision"]}
            for r in records if not r.get("feedback")]
