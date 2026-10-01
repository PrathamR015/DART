"""Training targets and weighted loss for the masked decision head."""
import statistics
from collections import Counter

import torch

from data_prep.schema import MAX_OPTIONS

ORDERED_TYPES = frozenset({"ordinal_lmh", "scale_1_5", "scale_0_5", "scale_0_10"})
HARD_SLICE_WEIGHT = 1.5
SOURCE_WEIGHT_RANGE = (0.5, 2.0)
LABEL_WEIGHT_RANGE = (0.5, 2.0)


def soft_targets(labels, n_options, ordered, smoothing=0.05, neighbour=0.10, width=MAX_OPTIONS):
    """Target distribution over option slots.

    Every type gets label smoothing over the other valid options. Ordered types also put
    `neighbour` mass on the options next to the correct one.
    """
    positions = torch.arange(width, device=labels.device).unsqueeze(0)
    valid = positions < n_options.unsqueeze(1)
    is_label = positions == labels.unsqueeze(1)
    ordered_col = ordered.unsqueeze(1)
    adjacent = ((positions - labels.unsqueeze(1)).abs() == 1) & valid & ordered_col
    others = valid & ~is_label
    correct = 1.0 - smoothing - neighbour * ordered_col.float()
    n_adjacent = adjacent.sum(1, keepdim=True).clamp(min=1)
    n_others = others.sum(1, keepdim=True).clamp(min=1)
    return (is_label * correct + adjacent * (neighbour / n_adjacent) + others * (smoothing / n_others)).float()


def weighted_loss(logits, targets, weights):
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    per_row = -(targets * log_probs).sum(dim=-1)
    return (per_row * weights).sum() / weights.sum()


def _ordered_label_weights(rows):
    """Inverse-square-root weight per (ordered type, label): rare scale values count more.

    Fixes 'dead classes' (scale values the model never predicts). Unordered types are untouched.
    """
    counts = Counter((r["decision_type"], r["label"]) for r in rows if r["decision_type"] in ORDERED_TYPES)
    per_type = {}
    for (decision_type, _), count in counts.items():
        per_type.setdefault(decision_type, []).append(count)
    low, high = LABEL_WEIGHT_RANGE
    return {key: min(high, max(low, (statistics.mean(per_type[key[0]]) / count) ** 0.5))
            for key, count in counts.items()}


def sample_weights(rows):
    """Per-row weights: rarer decision types and sources count more, hard rows count more. Mean is 1."""
    types = Counter(r["decision_type"] for r in rows)
    sources = Counter(r["source"] for r in rows)
    mean_type = statistics.mean(types.values())
    median_source = statistics.median(sources.values())
    low, high = SOURCE_WEIGHT_RANGE
    label_weights = _ordered_label_weights(rows)
    raw = []
    for r in rows:
        type_weight = (mean_type / types[r["decision_type"]]) ** 0.5
        source_weight = min(high, max(low, (median_source / sources[r["source"]]) ** 0.5))
        slice_weight = HARD_SLICE_WEIGHT if r["slice"] == "hard_ambiguous" else 1.0
        label_weight = label_weights.get((r["decision_type"], r["label"]), 1.0)
        raw.append(type_weight * source_weight * slice_weight * label_weight)
    scale = len(raw) / sum(raw)
    return [w * scale for w in raw]
