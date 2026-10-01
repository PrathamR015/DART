"""Public API: DART.from_pretrained(path).decide(...) and .decide_many(...).

Every call is a single forward pass (no generation). Outputs follow the PRD 5.1 contract: every option with its
probability, sorted descending; the top option is the decision.
"""
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from data_prep.schema import MAX_OPTIONS, MAX_PROMPT_TOKENS, format_prompt
from dart.data import option_mask
from dart.fast_inference import GraphedPredictor
from dart.model import DecisionModel, ranked_options
from dart.parallel import ParallelDecider, decisions_payload

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def validate_options(options):
    if not isinstance(options, (list, tuple)) or not all(isinstance(o, str) for o in options):
        raise ValueError("options must be a list of strings")
    if not 2 <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"need between 2 and {MAX_OPTIONS} options, got {len(options)}")
    if any(not o.strip() for o in options):
        raise ValueError("options must not be empty")
    if len({o.strip().lower() for o in options}) != len(options):
        raise ValueError("options must be distinct")


class DART:
    def __init__(self, model, tokenizer, device, fast=False, log_path=None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.parallel = ParallelDecider(self.model, tokenizer)
        self.graphs = GraphedPredictor(self.model, batch_sizes=(1,)) if fast and device.startswith("cuda") else None
        self.log_path = Path(log_path) if log_path else None

    @classmethod
    def from_pretrained(cls, path, device=None, fast=True, dtype=None, log_path=None):
        path = Path(path)
        config = json.loads((path / "config.json").read_text(encoding="utf8"))
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        backbone = AutoModel.from_pretrained(path / "backbone", dtype=DTYPES[dtype or config["dtype"]]).to(device)
        model = DecisionModel(backbone, config["hidden_size"], config["max_options"]).to(device)
        model.head.load_state_dict(torch.load(path / "head.pt", map_location=device))
        return cls(model, AutoTokenizer.from_pretrained(path / "tokenizer"), device, fast, log_path)

    def _rank(self, query, options, context=""):
        validate_options(options)
        prompt = format_prompt({"context": context, "query": query, "options": options})
        ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        if ids.shape[1] > MAX_PROMPT_TOKENS:
            raise ValueError(f"prompt has {ids.shape[1]} tokens; the limit is {MAX_PROMPT_TOKENS}")
        masks = torch.tensor([option_mask(len(options))], device=self.device)
        predictor = self.graphs or self.model
        return ranked_options(predictor.predict(ids, torch.ones_like(ids), masks)[0].cpu(), options)

    def _log(self, request, payload):
        if self.log_path is None:
            return
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf8")).hexdigest()
        entry = {"input_sha256": digest, "decisions": payload["decisions"]}
        with self.log_path.open("a", encoding="utf8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def decide(self, query, options, context="", question_id="q1"):
        """One decision. Returns the PRD 5.1 payload with a single entry in `decisions`."""
        payload = decisions_payload([question_id], [self._rank(query, options, context)])
        self._log({"context": context, "query": query, "options": list(options)}, payload)
        return payload

    def decide_many(self, context, questions):
        """Several decisions over one shared context. questions: [{"id"?, "query", "options"}, ...]."""
        for question in questions:
            validate_options(question["options"])
        ids = [q.get("id", f"q{i + 1}") for i, q in enumerate(questions)]
        if context:
            ranked = self.parallel.decide(context, questions)
        else:
            ranked = [self._rank(q["query"], q["options"]) for q in questions]
        payload = decisions_payload(ids, ranked)
        self._log({"context": context, "questions": [{"query": q["query"], "options": list(q["options"])}
                                                     for q in questions]}, payload)
        return payload
