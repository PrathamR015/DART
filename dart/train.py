"""LoRA fine-tuning of the DART decision model (backbone adapters + masked decision head)."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from dart.data import Collator, DecisionDataset, bucket_batches, load_rows, sample_rows, token_lengths
from dart.loss import sample_weights, soft_targets, weighted_loss
from dart.metrics import make_records, summarize
from dart.model import DecisionModel

MODEL_NAME = "Qwen/Qwen3-0.6B"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
GRAD_CLIP = 1.0
WARMUP_FRACTION = 0.03
ROOT = Path(__file__).resolve().parents[1]


def build_model(rank, device, init_from=None):
    """Fresh LoRA model, or one resumed from a saved run directory (adapter/ and head.pt)."""
    backbone = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    hidden_size = backbone.config.hidden_size
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone.enable_input_require_grads()
    if init_from:
        backbone = PeftModel.from_pretrained(backbone, Path(init_from) / "adapter", is_trainable=True)
    else:
        lora = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=LORA_TARGETS, bias="none")
        backbone = get_peft_model(backbone, lora)
    model = DecisionModel(backbone, hidden_size).to(device)
    if init_from:
        model.head.load_state_dict(torch.load(Path(init_from) / "head.pt", map_location=device))
    return model


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def step_loss(model, batch, smoothing, neighbour):
    logits = model(batch["input_ids"], batch["attention_mask"], batch["option_mask"])
    targets = soft_targets(batch["labels"], batch["n_options"], batch["ordered"], smoothing, neighbour)
    return weighted_loss(logits, targets, batch["weights"])


@torch.no_grad()
def evaluate(model, dataset, collator, lengths, device, batch_size):
    model.eval()
    order = sorted(range(len(dataset)), key=lengths.__getitem__)
    records = []
    for begin in range(0, len(order), batch_size):
        indices = order[begin:begin + batch_size]
        batch = to_device(collator([dataset[i] for i in indices]), device)
        probs = model.predict(batch["input_ids"], batch["attention_mask"], batch["option_mask"]).cpu()
        records += make_records([dataset.rows[i] for i in indices], probs)
    model.train()
    return summarize(records)


def save_checkpoint(model, out_dir):
    model.backbone.save_pretrained(out_dir / "adapter")
    torch.save(model.head.state_dict(), out_dir / "head.pt")


def make_optimizer(model, lr, head_lr):
    adapters = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": adapters, "lr": lr}, {"params": list(model.head.parameters()), "lr": head_lr}]
    return torch.optim.AdamW(groups, weight_decay=0.01)


def run_training(args):
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = Collator(tokenizer)
    train_rows = sample_rows(load_rows(args.train), args.limit, args.seed)
    val_rows = sample_rows(load_rows(args.val), args.val_limit, args.seed)
    train_ds = DecisionDataset(train_rows, sample_weights(train_rows))
    val_ds = DecisionDataset(val_rows)
    train_len, val_len = (token_lengths(tokenizer, d.prompts) for d in (train_ds, val_ds))
    model = build_model(args.rank, device, args.init_from)
    model.train()
    optimizer = make_optimizer(model, args.lr, args.head_lr)
    per_epoch = math.ceil(len(train_ds) / args.batch_size / args.accum)
    total = args.max_steps or per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total * WARMUP_FRACTION), total)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf8")
    history, best, step, micro, running, start = [], -1.0, 0, 0, 0.0, time.time()
    for _ in range(args.epochs):
        for indices in bucket_batches(train_len, args.batch_size, rng):
            batch = to_device(collator([train_ds[i] for i in indices]), device)
            loss = step_loss(model, batch, args.smoothing, args.neighbour) / args.accum
            loss.backward()
            running += loss.item()
            micro += 1
            if micro % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                elapsed = time.time() - start
                print(f"step {step}/{total} loss {running / args.log_every:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                      f"{step * args.batch_size * args.accum / elapsed:.1f} samples/s "
                      f"elapsed {elapsed / 60:.1f}m eta {(total - step) * elapsed / step / 60:.1f}m", flush=True)
                running = 0.0
            if step % args.eval_every == 0 or step >= total:
                report = evaluate(model, val_ds, collator, val_len, device, args.eval_batch_size)
                history.append({"step": step, **report})
                accuracy = report["overall"]["accuracy"]
                print(f"  eval step {step}: val accuracy {accuracy:.4f} top2 {report['overall']['top2']:.4f}", flush=True)
                if accuracy > best:
                    best = accuracy
                    save_checkpoint(model, out_dir)
                (out_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf8")
            if step >= total:
                return history
    return history


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=str(ROOT / "assets/processed/train.jsonl"))
    p.add_argument("--val", default=str(ROOT / "assets/processed/val.jsonl"))
    p.add_argument("--out", default=str(ROOT / "runs/lora_run1"))
    p.add_argument("--limit", type=int, default=0, help="train on a random subset of this many rows")
    p.add_argument("--val-limit", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--init-from", default="", help="resume from a run directory with adapter/ and head.pt")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--smoothing", type=float, default=0.05)
    p.add_argument("--neighbour", type=float, default=0.10)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
