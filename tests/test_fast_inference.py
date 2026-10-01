import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart.data import option_mask  # noqa: E402
from dart.fast_inference import GraphedPredictor  # noqa: E402
from dart.model import DecisionModel  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")


def tiny_model():
    from transformers import Qwen3Config, Qwen3Model

    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=1000, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512)
    return DecisionModel(Qwen3Model(config), config.hidden_size).cuda().eval()


def left_padded(lengths, seed=1):
    generator = torch.Generator().manual_seed(seed)
    width = max(lengths)
    ids = torch.zeros(len(lengths), width, dtype=torch.long)
    mask = torch.zeros(len(lengths), width, dtype=torch.long)
    for row, n in enumerate(lengths):
        ids[row, width - n:] = torch.randint(1, 1000, (n,), generator=generator)
        mask[row, width - n:] = 1
    return ids.cuda(), mask.cuda()


def test_graphed_prediction_matches_eager_for_single_and_padded_batches():
    model = tiny_model()
    predictor = GraphedPredictor(model, batch_sizes=(1, 3), buckets=(16, 32))
    options = torch.tensor([option_mask(4)] * 3).cuda()
    for lengths in ([5], [20], [5, 9, 7], [12, 30, 20]):
        ids, mask = left_padded(lengths)
        expected = model.predict(ids, mask, options[: len(lengths)])
        actual = predictor.predict(ids, mask, options[: len(lengths)])
        assert torch.allclose(actual, expected, atol=1e-4)


def test_graphed_prediction_handles_different_option_counts_between_calls():
    model = tiny_model()
    predictor = GraphedPredictor(model, batch_sizes=(1,), buckets=(16,))
    ids, mask = left_padded([10])
    for n in (2, 7, 3):
        options = torch.tensor([option_mask(n)]).cuda()
        probs = predictor.predict(ids, mask, options)
        assert probs[0, n:].sum().item() < 1e-6 and probs.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_graphed_prediction_rejects_prompts_longer_than_every_bucket():
    predictor = GraphedPredictor(tiny_model(), batch_sizes=(1,), buckets=(16,))
    ids, mask = left_padded([20])
    with pytest.raises(ValueError):
        predictor.predict(ids, mask, torch.tensor([option_mask(3)]).cuda())
