import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart.data import Collator, DecisionDataset, bucket_batches, option_mask  # noqa: E402
from dart.loss import sample_weights, soft_targets, weighted_loss  # noqa: E402
from dart.metrics import make_records, summarize  # noqa: E402
from dart.model import DecisionModel, ranked_options  # noqa: E402
from data_prep.schema import make_row  # noqa: E402


def row(decision_type="categorical", n=4, label=0, source="s", slice_="core", query="Which one?"):
    return make_row(source=source, decision_type=decision_type, query=query,
                    options=[f"opt{i}" for i in range(n)], label=label, slice_=slice_)


def tiny_backbone():
    from transformers import Qwen3Config, Qwen3Model

    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=1000, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512)
    return Qwen3Model(config).eval(), config.hidden_size


def test_soft_targets_sum_to_one_and_are_zero_on_invalid_slots():
    labels, n, ordered = torch.tensor([1, 0]), torch.tensor([4, 3]), torch.tensor([False, True])
    targets = soft_targets(labels, n, ordered)
    assert torch.allclose(targets.sum(1), torch.ones(2))
    assert targets[0, 4:].sum() == 0 and targets[1, 3:].sum() == 0


def test_soft_targets_put_extra_mass_next_to_the_label_for_ordered_types():
    targets = soft_targets(torch.tensor([2]), torch.tensor([5]), torch.tensor([True]))
    assert targets[0, 1] > targets[0, 4] and targets[0, 3] > targets[0, 0]
    assert targets[0, 2] == targets[0].max()


def test_weighted_loss_is_lower_when_logits_favour_the_label():
    targets = soft_targets(torch.tensor([0]), torch.tensor([3]), torch.tensor([False]))
    good = torch.tensor([[5.0, 0.0, 0.0] + [-1e4] * 9])
    bad = torch.tensor([[0.0, 5.0, 0.0] + [-1e4] * 9])
    weights = torch.ones(1)
    assert weighted_loss(good, targets, weights) < weighted_loss(bad, targets, weights)


def test_sample_weights_have_mean_one_and_favour_hard_rows_and_rare_types():
    rows = [row(slice_="core")] * 30 + [row(slice_="hard_ambiguous")] * 30 + [row("binary", 2)] * 5
    weights = sample_weights(rows)
    assert sum(weights) / len(weights) == pytest.approx(1.0)
    assert weights[30] > weights[0]
    assert weights[-1] > weights[0]


def test_sample_weights_favour_rare_labels_within_ordered_types_only():
    ordered = [row("scale_0_5", 6, label=0)] * 60 + [row("scale_0_5", 6, label=1)] * 20
    unordered = [row("categorical", 4, label=0)] * 60 + [row("categorical", 4, label=1)] * 20
    weights = sample_weights(ordered + unordered)
    assert weights[60] > weights[0]  # rarer scale label counts more
    assert weights[140] == pytest.approx(weights[80])  # categorical labels are not re-weighted


def test_option_mask_counts_valid_options():
    assert sum(option_mask(5)) == 5 and len(option_mask(5)) == 12


def test_collator_left_pads_and_last_token_is_real():
    from transformers import AutoTokenizer

    rows = [row(query="short"), row(query="a much longer question " * 8, n=6)]
    batch = Collator(AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"))([DecisionDataset(rows)[i] for i in (0, 1)])
    assert bool(batch["attention_mask"][:, -1].all())
    assert batch["option_mask"].sum(1).tolist() == [4, 6]
    assert batch["input_ids"].shape[0] == 2


def test_bucket_batches_cover_every_index_once():
    lengths = [random.Random(1).randint(5, 200) for _ in range(500)]
    batches = bucket_batches(lengths, 8, random.Random(2))
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(500)) and all(len(b) <= 8 for b in batches)


def test_decision_model_probabilities_are_valid_and_masked():
    backbone, hidden = tiny_backbone()
    model = DecisionModel(backbone, hidden).eval()
    ids = torch.randint(1, 1000, (2, 10))
    mask = torch.ones_like(ids)
    options = torch.tensor([option_mask(3), option_mask(7)])
    probs = model.predict(ids, mask, options)
    assert torch.allclose(probs.sum(1), torch.ones(2), atol=1e-5)
    assert probs[0, 3:].sum() < 1e-6 and probs[1, 7:].sum() < 1e-6


def test_batched_prediction_equals_single_prediction_with_left_padding():
    backbone, hidden = tiny_backbone()
    model = DecisionModel(backbone, hidden).eval()
    seqs = [torch.randint(1, 1000, (n,)) for n in (5, 9, 7)]
    options = torch.tensor([option_mask(4)] * 3)
    width = max(len(s) for s in seqs)
    ids = torch.stack([torch.cat([torch.zeros(width - len(s), dtype=torch.long), s]) for s in seqs])
    mask = torch.stack([torch.cat([torch.zeros(width - len(s), dtype=torch.long), torch.ones(len(s), dtype=torch.long)])
                        for s in seqs])
    batched = model.predict(ids, mask, options)
    for i, seq in enumerate(seqs):
        single = model.predict(seq.unsqueeze(0), torch.ones(1, len(seq), dtype=torch.long), options[:1])
        assert torch.allclose(batched[i], single[0], atol=1e-4)


def test_features_then_head_equals_forward():
    backbone, hidden = tiny_backbone()
    model = DecisionModel(backbone, hidden).eval()
    ids = torch.randint(1, 1000, (3, 8))
    mask = torch.ones_like(ids)
    options = torch.tensor([option_mask(4)] * 3)
    with torch.no_grad():
        expected = model(ids, mask, options)
        rebuilt = model.head(model.features(ids, mask)).masked_fill(~options, -1e4)
    assert torch.allclose(expected, rebuilt, atol=1e-6)


def test_ranked_options_sorted_descending():
    ranked = ranked_options([0.1, 0.6, 0.3], ["a", "b", "c"])
    assert [o for o, _ in ranked] == ["b", "c", "a"]


def test_summarize_reports_within_one_and_prediction_histogram_for_ordered_types():
    rows = [row("scale_0_5", 6, label=3), row("scale_0_5", 6, label=1), row("scale_0_5", 6, label=5)]
    one_hot = lambda i: [1 if j == i else 0 for j in range(6)] + [0] * 6  # noqa: E731
    report = summarize(make_records(rows, [one_hot(4), one_hot(4), one_hot(4)]))
    scale = report["by_decision_type"]["scale_0_5"]
    assert scale["within1"] == pytest.approx(2 / 3, abs=1e-3)  # errors of 1, 3 and 1 steps
    assert scale["predicted"] == {4: 3}
    assert scale["never_predicted"] == [0, 1, 2, 3, 5]


def test_summarize_reports_accuracy_and_mae_for_ordered_types():
    rows = [row("scale_0_5", 6, label=3), row("scale_0_5", 6, label=1), row("binary", 2, label=0)]
    probs = [[0, 0, 0, 1, 0, 0] + [0] * 6, [0, 0, 0, 1, 0, 0] + [0] * 6, [1, 0] + [0] * 10]
    report = summarize(make_records(rows, probs))
    assert report["overall"]["accuracy"] == pytest.approx(2 / 3, abs=1e-3)
    assert report["by_decision_type"]["scale_0_5"]["mae"] == 1.0
    assert "mae" not in report["by_decision_type"]["binary"]
