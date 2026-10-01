"""Zero-shot baseline: score option letters with the LM head rows for those letters only (one forward pass)."""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_prep.schema import LETTERS
from dart.data import Collator, DecisionDataset, load_rows, sample_rows, token_lengths
from dart.metrics import make_records, summarize
from dart.model import MASK_VALUE, position_ids_from_mask

MODEL_NAME = "Qwen/Qwen3-0.6B"
ROOT = Path(__file__).resolve().parents[1]


def letter_token_ids(tokenizer):
    ids = []
    for letter in LETTERS:
        encoded = tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"' {letter}' is not a single token")
        ids.append(encoded[0])
    return ids


@torch.no_grad()
def zero_shot_probabilities(model, batch, letter_ids):
    hidden = model.model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        position_ids=position_ids_from_mask(batch["attention_mask"]), use_cache=False,
    ).last_hidden_state[:, -1]
    logits = (hidden @ model.lm_head.weight[letter_ids].T).float()
    return torch.softmax(logits.masked_fill(~batch["option_mask"], MASK_VALUE), dim=-1).cpu()


def _to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def evaluate_zero_shot(rows, model, tokenizer, device, batch_size):
    dataset, collator = DecisionDataset(rows), Collator(tokenizer)
    lengths = token_lengths(tokenizer, dataset.prompts)
    order = sorted(range(len(rows)), key=lengths.__getitem__)
    letter_ids = torch.tensor(letter_token_ids(tokenizer), device=device)
    records, start = [], time.time()
    for begin in range(0, len(order), batch_size):
        indices = order[begin:begin + batch_size]
        batch = _to_device(collator([dataset[i] for i in indices]), device)
        records += make_records([rows[i] for i in indices], zero_shot_probabilities(model, batch, letter_ids))
    elapsed = time.time() - start
    report = summarize(records)
    report["ms_per_decision_batched"] = round(1000 * elapsed / len(rows), 2)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "assets/processed/proxy_test.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default=str(ROOT / "runs/baseline_zero_shot.json"))
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype).to(device).eval()
    rows = sample_rows(load_rows(args.data), args.limit, seed=0)
    report = evaluate_zero_shot(rows, model, tokenizer, device, args.batch_size)
    report["device"] = device
    if device == "cuda":
        report["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps({k: report[k] for k in ("overall", "by_decision_type", "ms_per_decision_batched")}, indent=2))


if __name__ == "__main__":
    main()
