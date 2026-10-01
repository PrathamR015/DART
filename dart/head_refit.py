"""Re-fit only the decision head on cached backbone features (the backbone stays frozen).

Cheap experiment: it answers whether problems such as never-predicted scale values live in the head
(fixable here, in minutes) or in the backbone (needs more LoRA training).
"""
import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer

from dart.data import Collator, DecisionDataset, load_rows, option_mask, token_lengths
from dart.evaluate import comparison_table, load_model
from dart.loss import sample_weights, soft_targets, weighted_loss
from dart.metrics import make_records, summarize
from dart.model import MASK_VALUE
from dart.train import MODEL_NAME, to_device

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "proxy_test")


@torch.no_grad()
def extract_features(model, dataset, collator, lengths, device, batch_size):
    order = sorted(range(len(dataset)), key=lengths.__getitem__)
    features = torch.zeros(len(dataset), model.head.in_features)
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        batch = to_device(collator([dataset[i] for i in indices]), device)
        features[indices] = model.features(batch["input_ids"], batch["attention_mask"]).cpu()
    return features


def option_masks(rows):
    return torch.tensor([option_mask(len(r["options"])) for r in rows])


def evaluate_head(head, features, rows):
    with torch.no_grad():
        logits = head(features.to(next(head.parameters()).device))
        masked = logits.masked_fill(~option_masks(rows).to(logits.device), MASK_VALUE)
        probs = torch.softmax(masked, dim=-1).cpu()
    return summarize(make_records(rows, probs))


def _targets(rows, device):
    weights = sample_weights(rows)
    return {
        "labels": torch.tensor([r["label"] for r in rows], device=device),
        "n_options": torch.tensor([len(r["options"]) for r in rows], device=device),
        "ordered": torch.tensor([r["decision_type"] in ORDERED for r in rows], device=device),
        "weights": torch.tensor(weights, dtype=torch.float32, device=device),
        "mask": option_masks(rows).to(device),
    }


ORDERED = frozenset({"ordinal_lmh", "scale_1_5", "scale_0_5", "scale_0_10"})


def fit_head(train_features, train_rows, val_features, val_rows, init_head, epochs=8, lr=1e-3,
             batch_size=512, seed=13, smoothing=0.05, neighbour=0.10, device="cpu"):
    """Fit a copy of init_head on cached features. Returns (best_head, history, base_accuracy)."""
    torch.manual_seed(seed)
    head = copy.deepcopy(init_head).to(device)
    features, data = train_features.to(device), _targets(train_rows, device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    base_accuracy = evaluate_head(head, val_features, val_rows)["overall"]["accuracy"]
    best_head, best_accuracy, history = copy.deepcopy(head), base_accuracy, []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(features), device=device)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            logits = head(features[idx]).masked_fill(~data["mask"][idx], MASK_VALUE)
            targets = soft_targets(data["labels"][idx], data["n_options"][idx], data["ordered"][idx],
                                   smoothing, neighbour)
            loss = weighted_loss(logits, targets, data["weights"][idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        accuracy = evaluate_head(head, val_features, val_rows)["overall"]["accuracy"]
        history.append({"epoch": epoch, "val_accuracy": accuracy})
        if accuracy > best_accuracy:
            best_head, best_accuracy = copy.deepcopy(head), accuracy
    return best_head, history, base_accuracy


def cached_features(split, model, tokenizer, device, batch_size, cache_dir):
    rows = load_rows(ROOT / f"assets/processed/{split}.jsonl")
    path = cache_dir / f"{split}.pt"
    if path.exists():
        return rows, torch.load(path)
    dataset = DecisionDataset(rows)
    features = extract_features(model, dataset, Collator(tokenizer), token_lengths(tokenizer, dataset.prompts),
                                device, batch_size)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(features, path)
    return rows, features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=str(ROOT / "runs/lora_run1"))
    parser.add_argument("--out", default=str(ROOT / "runs/head_refit"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir, cache_dir = Path(args.out), ROOT / "runs/features"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(Path(args.run), device)
    model.backbone = model.backbone.merge_and_unload()
    data = {split: cached_features(split, model, tokenizer, device, args.batch_size, cache_dir) for split in SPLITS}
    old_head = model.head
    before = evaluate_head(old_head, data["proxy_test"][1], data["proxy_test"][0])
    best, history, base = fit_head(data["train"][1], data["train"][0], data["val"][1], data["val"][0], old_head,
                                   epochs=args.epochs, lr=args.lr, device=device)
    after = evaluate_head(best, data["proxy_test"][1], data["proxy_test"][0])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best.state_dict(), out_dir / "head.pt")
    (out_dir / "metrics.json").write_text(json.dumps({"history": history, "before": before, "after": after}, indent=2))
    print(f"val accuracy: original head {base:.4f} -> refit head history {[h['val_accuracy'] for h in history]}")
    print("\nproxy_test, ORIGINAL head vs REFIT head (same frozen backbone):")
    print(comparison_table(after, before))
    for name, part in after["by_decision_type"].items():
        if "within1" in part:
            old = before["by_decision_type"][name]
            print(f"  {name:12s} within1 {old['within1']:.3f} -> {part['within1']:.3f} | mae {old['mae']:.3f} -> {part['mae']:.3f}"
                  f" | never predicted {old['never_predicted']} -> {part['never_predicted']}")


if __name__ == "__main__":
    main()
