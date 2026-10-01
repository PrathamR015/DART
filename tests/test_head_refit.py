import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart.head_refit import evaluate_head, fit_head  # noqa: E402
from data_prep.schema import make_row  # noqa: E402


def separable_data(n=300, dim=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 3, (n,), generator=generator)
    centres = torch.randn(3, dim, generator=generator) * 3
    features = centres[labels] + torch.randn(n, dim, generator=generator) * 0.3
    rows = [make_row(source="s", decision_type="categorical", query="q", options=["a", "b", "c"], label=int(y))
            for y in labels]
    return features, rows


def test_fit_head_improves_a_random_head_on_separable_features():
    features, rows = separable_data()
    init = nn.Linear(16, 12)
    best, history, base = fit_head(features, rows, features, rows, init, epochs=6, lr=1e-2, batch_size=64)
    final = evaluate_head(best, features, rows)["overall"]["accuracy"]
    assert final > base + 0.3 and final > 0.95
    assert len(history) == 6


def test_fit_head_never_returns_a_head_worse_than_the_starting_one_on_validation():
    features, rows = separable_data()
    init = nn.Linear(16, 12)
    with torch.no_grad():  # a head that is already very good
        init.weight.zero_()
        init.bias.zero_()
    start = evaluate_head(init, features, rows)["overall"]["accuracy"]
    best, _, _ = fit_head(features, rows, features, rows, init, epochs=2, lr=1e-9, batch_size=64)
    assert evaluate_head(best, features, rows)["overall"]["accuracy"] >= start


def test_fit_head_does_not_modify_the_head_it_was_given():
    features, rows = separable_data()
    init = nn.Linear(16, 12)
    before = init.weight.clone()
    fit_head(features, rows, features, rows, init, epochs=2, lr=1e-2, batch_size=64)
    assert torch.equal(init.weight, before)
