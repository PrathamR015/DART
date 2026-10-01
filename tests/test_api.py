import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart.api import DART, validate_options  # noqa: E402
from dart.data import option_mask  # noqa: E402
from dart.export import build_config, build_readme, save_artifact  # noqa: E402
from dart.model import DecisionModel  # noqa: E402
from data_prep.schema import format_prompt  # noqa: E402

VOCAB = 151936


def tiny_model():
    from transformers import Qwen3Config, Qwen3Model

    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512)
    return DecisionModel(Qwen3Model(config), config.hidden_size).eval()


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    from transformers import AutoTokenizer

    folder = tmp_path_factory.mktemp("artifact")
    model, tokenizer = tiny_model(), AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = build_config(model, "vtest", "float32")
    metrics = {"overall": {"accuracy": 0.5, "top2": 0.8}, "by_decision_type": {"binary": {"n": 10, "accuracy": 0.5}}}
    save_artifact(model, tokenizer, folder, config, build_readme(config, "artifact", metrics))
    return folder, model, tokenizer


def test_artifact_contains_every_expected_file(saved):
    folder, _, _ = saved
    for name in ("backbone", "head.pt", "config.json", "tokenizer", "README.md"):
        assert (folder / name).exists()
    config = json.loads((folder / "config.json").read_text())
    assert config["pooling"] == "last_token" and config["padding_side"] == "left" and config["max_options"] == 12
    assert "50.0%" in (folder / "README.md").read_text()


def test_loaded_model_reproduces_the_original_model_output(saved):
    folder, model, tokenizer = saved
    dart = DART.from_pretrained(folder, device="cpu", fast=False)
    options = ["China", "England", "Japan"]
    payload = dart.decide("Where is Tony from?", options, context="Tony is from England.")
    row = {"context": "Tony is from England.", "query": "Where is Tony from?", "options": options}
    ids = tokenizer(format_prompt(row), return_tensors="pt", add_special_tokens=False)["input_ids"]
    expected = model.predict(ids, torch.ones_like(ids), torch.tensor([option_mask(3)]))[0][:3]
    got = {item["option"]: item["probability"] for item in payload["decisions"][0]["ranked"]}
    for i, option in enumerate(options):
        assert got[option] == pytest.approx(float(expected[i]), abs=1e-5)


def test_payload_is_sorted_sums_to_one_and_names_the_decision(saved):
    dart = DART.from_pretrained(saved[0], device="cpu", fast=False)
    decision = dart.decide("Pick one", ["a", "b", "c", "d"], question_id="x")["decisions"][0]
    probs = [item["probability"] for item in decision["ranked"]]
    assert decision["id"] == "x" and probs == sorted(probs, reverse=True)
    assert sum(probs) == pytest.approx(1.0, abs=1e-5)
    assert decision["decision"] == decision["ranked"][0]["option"] and decision["confidence"] == probs[0]


def test_decide_many_equals_independent_decisions_over_the_same_context(saved):
    dart = DART.from_pretrained(saved[0], device="cpu", fast=False)
    context = "Tony is a boy from England. He is 16 years old and plays football."
    questions = [{"id": "country", "query": "Where is Tony from?", "options": ["China", "England", "Japan"]},
                 {"id": "age", "query": "How old is Tony?", "options": ["12", "16"]},
                 {"id": "sport", "query": "Which sport does he play?", "options": ["tennis", "football", "golf", "chess"]}]
    many = dart.decide_many(context, questions)["decisions"]
    for question, batched in zip(questions, many):
        alone = dart.decide(question["query"], question["options"], context=context)["decisions"][0]
        assert batched["id"] == question["id"]
        a = {i["option"]: i["probability"] for i in alone["ranked"]}
        b = {i["option"]: i["probability"] for i in batched["ranked"]}
        assert all(a[o] == pytest.approx(b[o], abs=1e-4) for o in a)


def test_decide_many_without_context_answers_each_question(saved):
    dart = DART.from_pretrained(saved[0], device="cpu", fast=False)
    result = dart.decide_many("", [{"query": "Q one?", "options": ["yes", "no"]},
                                   {"query": "Q two?", "options": ["low", "medium", "high"]}])
    assert [d["id"] for d in result["decisions"]] == ["q1", "q2"]
    assert [len(d["ranked"]) for d in result["decisions"]] == [2, 3]


@pytest.mark.parametrize("options", [["only"], ["a", "A"], ["a", ""], [str(i) for i in range(13)], "ab", [1, 2]])
def test_invalid_options_are_rejected(options):
    with pytest.raises(ValueError):
        validate_options(options)


def test_overlong_prompt_is_rejected(saved):
    dart = DART.from_pretrained(saved[0], device="cpu", fast=False)
    with pytest.raises(ValueError):
        dart.decide("word " * 400, ["a", "b"])


def test_decision_log_records_hash_and_ranked_output(saved, tmp_path):
    log = tmp_path / "decisions.jsonl"
    dart = DART.from_pretrained(saved[0], device="cpu", fast=False, log_path=log)
    dart.decide("Pick one", ["a", "b"])
    dart.decide("Pick one", ["a", "b"])
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(entries) == 2 and entries[0]["input_sha256"] == entries[1]["input_sha256"]
    assert len(entries[0]["input_sha256"]) == 64 and entries[0]["decisions"][0]["ranked"]
