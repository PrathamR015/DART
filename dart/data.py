"""Dataset, collator and length-bucketed batching for DART training and evaluation."""
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from data_prep.schema import MAX_OPTIONS, MAX_PROMPT_TOKENS, format_prompt
from dart.loss import ORDERED_TYPES

CHUNK_FACTOR = 50


def load_rows(path, limit=None):
    rows = []
    with Path(path).open(encoding="utf8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def sample_rows(rows, limit, seed):
    """Reproducible random subset (files are ordered by source, so never just take the head)."""
    if not limit or limit >= len(rows):
        return rows
    return random.Random(seed).sample(rows, limit)


def option_mask(n_options):
    return [i < n_options for i in range(MAX_OPTIONS)]


class DecisionDataset(Dataset):
    def __init__(self, rows, weights=None):
        self.rows = rows
        self.prompts = [format_prompt(r) for r in rows]
        self.weights = weights or [1.0] * len(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "index": index,
            "prompt": self.prompts[index],
            "label": row["label"],
            "n_options": len(row["options"]),
            "ordered": row["decision_type"] in ORDERED_TYPES,
            "weight": self.weights[index],
        }


class Collator:
    """Tokenizes prompts with left padding so the last position is always a real token."""

    def __init__(self, tokenizer, max_length=MAX_PROMPT_TOKENS):
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, items):
        encoded = self.tokenizer([i["prompt"] for i in items], return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.max_length, add_special_tokens=False)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor([i["label"] for i in items]),
            "n_options": torch.tensor([i["n_options"] for i in items]),
            "ordered": torch.tensor([i["ordered"] for i in items]),
            "weights": torch.tensor([i["weight"] for i in items], dtype=torch.float32),
            "option_mask": torch.tensor([option_mask(i["n_options"]) for i in items]),
            "indices": [i["index"] for i in items],
        }


def token_lengths(tokenizer, prompts):
    return [len(ids) for ids in tokenizer(prompts, add_special_tokens=False)["input_ids"]]


def bucket_batches(lengths, batch_size, rng):
    """Shuffled batches of similar-length examples, to keep padding small."""
    order = list(range(len(lengths)))
    rng.shuffle(order)
    batches = []
    chunk = batch_size * CHUNK_FACTOR
    for start in range(0, len(order), chunk):
        part = sorted(order[start:start + chunk], key=lambda i: lengths[i])
        batches.extend(part[j:j + batch_size] for j in range(0, len(part), batch_size))
    rng.shuffle(batches)
    return batches
