import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart.data import option_mask  # noqa: E402
from dart.model import DecisionModel  # noqa: E402
from dart.parallel import ParallelDecider, decisions_payload, predict_shared  # noqa: E402
from data_prep.schema import format_prefix, format_prompt, format_suffix, make_row  # noqa: E402

VOCAB = 151936


def tiny_model():
    from transformers import Qwen3Config, Qwen3Model

    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512)
    return DecisionModel(Qwen3Model(config), config.hidden_size).eval()


def random_ids(n, seed):
    return torch.randint(1, 1000, (n,), generator=torch.Generator().manual_seed(seed))


def masks(counts):
    return torch.tensor([option_mask(n) for n in counts])


def test_shared_prefix_matches_standalone_full_prompts():
    model = tiny_model()
    prefix = random_ids(7, 1)
    suffixes = [random_ids(4, 2), random_ids(6, 3), random_ids(5, 4)]
    counts = [3, 4, 2]
    shared = predict_shared(model, prefix, suffixes, masks(counts))
    for k, suffix in enumerate(suffixes):
        full = torch.cat([prefix, suffix]).unsqueeze(0)
        alone = model.predict(full, torch.ones_like(full), masks([counts[k]]))
        assert torch.allclose(shared[k], alone[0], atol=1e-4)


def test_a_question_gets_the_same_answer_alone_or_batched_with_others():
    model = tiny_model()
    prefix = random_ids(9, 5)
    target = random_ids(5, 6)
    others = [random_ids(8, 7), random_ids(3, 8)]
    alone = predict_shared(model, prefix, [target], masks([4]))
    for position in range(3):
        batch = list(others)
        batch.insert(position, target)
        counts = [4 if s is target else 3 for s in batch]
        together = predict_shared(model, prefix, batch, masks(counts))
        assert torch.allclose(together[position], alone[0], atol=1e-4)


def test_repeated_calls_do_not_leak_state():
    model = tiny_model()
    prefix, suffixes = random_ids(6, 1), [random_ids(4, 2), random_ids(5, 3)]
    first = predict_shared(model, prefix, suffixes, masks([3, 3]))
    predict_shared(model, random_ids(10, 9), [random_ids(7, 10)], masks([5]))
    second = predict_shared(model, prefix, suffixes, masks([3, 3]))
    assert torch.equal(first, second)


def test_prompt_longer_than_the_token_limit_is_rejected():
    model = tiny_model()
    with pytest.raises(ValueError):
        predict_shared(model, random_ids(250, 1), [random_ids(20, 2)], masks([3]))


def test_prefix_and_suffix_tokenize_like_the_full_prompt():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    contexts = ["Tony is a boy. He is 16 years old.", "Shenzhen lies in South China. It is big.\nIt has parks.",
                "Hello! My name is Zhang Fei (\"Fei\"). I'm twelve!"]
    for context in contexts:
        row = make_row(source="s", decision_type="categorical", query="Where is Tony from?",
                       options=["China", "England", "Japan"], label=0, context=context)
        joined = (tokenizer(format_prefix(context), add_special_tokens=False)["input_ids"]
                  + tokenizer(format_suffix(row["query"], row["options"]), add_special_tokens=False)["input_ids"])
        assert joined == tokenizer(format_prompt(row), add_special_tokens=False)["input_ids"]


def test_decider_returns_sorted_probabilities_for_each_question():
    from transformers import AutoTokenizer

    decider = ParallelDecider(tiny_model(), AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"))
    questions = [{"query": "Where is Tony from?", "options": ["China", "England", "Japan"]},
                 {"query": "How old is he?", "options": ["12", "16"]}]
    ranked = decider.decide("Tony is a boy from England. He is 16 years old.", questions)
    assert len(ranked) == 2 and [len(r) for r in ranked] == [3, 2]
    for items in ranked:
        probs = [p for _, p in items]
        assert probs == sorted(probs, reverse=True) and sum(probs) == pytest.approx(1.0, abs=1e-5)


def test_decisions_payload_follows_the_output_contract():
    payload = decisions_payload(["q1"], [[("B", 0.8), ("A", 0.2)]])
    assert payload == {"model": "D.A.R.T.", "decisions": [
        {"id": "q1", "ranked": [{"option": "B", "probability": 0.8}, {"option": "A", "probability": 0.2}],
         "decision": "B", "confidence": 0.8}]}
