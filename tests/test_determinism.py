"""NFR-4: the same input must give the same output, whatever was decided in between."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart.api import DART  # noqa: E402
from dart.data import option_mask  # noqa: E402
from dart.fast_inference import GraphedPredictor  # noqa: E402
from dart.model import DecisionModel  # noqa: E402

VOCAB = 151936


def tiny_model(device="cpu"):
    from transformers import Qwen3Config, Qwen3Model

    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512)
    return DecisionModel(Qwen3Model(config), config.hidden_size).to(device).eval()


@pytest.fixture(scope="module")
def dart():
    from transformers import AutoTokenizer

    return DART(tiny_model(), AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"), "cpu", fast=False)


def test_the_same_decision_twice_is_identical(dart):
    first = dart.decide("Where is Tony from?", ["China", "England", "Japan"], context="Tony is from England.")
    second = dart.decide("Where is Tony from?", ["China", "England", "Japan"], context="Tony is from England.")
    assert first == second


def test_decisions_in_between_do_not_change_a_later_answer(dart):
    args = ("Pick one", ["a", "b", "c"])
    before = dart.decide(*args)
    dart.decide("Something else entirely", ["x", "y"])
    dart.decide_many("A passage about cats.", [{"query": "Which animal?", "options": ["cat", "dog"]}])
    assert dart.decide(*args) == before


def test_parallel_decisions_are_repeatable(dart):
    questions = [{"id": "a", "query": "Where?", "options": ["here", "there"]},
                 {"id": "b", "query": "When?", "options": ["now", "later", "never"]}]
    context = "The meeting is here, later."
    assert dart.decide_many(context, questions) == dart.decide_many(context, questions)


def test_option_order_only_changes_probabilities_through_the_model_not_the_ranking_code(dart):
    options = ["a", "b", "c"]
    ranked = dart.decide("Pick one", options)["decisions"][0]["ranked"]
    assert sorted(item["option"] for item in ranked) == sorted(options)
    assert [item["probability"] for item in ranked] == sorted((i["probability"] for i in ranked), reverse=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_graph_replay_is_bitwise_repeatable_and_unaffected_by_other_inputs():
    predictor = GraphedPredictor(tiny_model("cuda"), batch_sizes=(1,), buckets=(16, 32))
    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(1, 1000, (1, 12), generator=generator).cuda()
    other = torch.randint(1, 1000, (1, 20), generator=generator).cuda()
    mask, other_mask = torch.ones_like(ids), torch.ones_like(other)
    options = torch.tensor([option_mask(4)]).cuda()
    reference = predictor.predict(ids, mask, options)
    for _ in range(20):
        predictor.predict(other, other_mask, torch.tensor([option_mask(2)]).cuda())
        assert torch.equal(predictor.predict(ids, mask, options), reference)
