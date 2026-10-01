"""Evaluate a trained DART checkpoint (LoRA adapter + head) on a JSONL split, compare with the baseline, time it."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from dart.data import Collator, DecisionDataset, load_rows, sample_rows, token_lengths
from dart.model import DecisionModel
from dart.train import MODEL_NAME, evaluate, to_device

ROOT = Path(__file__).resolve().parents[1]
WARMUP_RUNS = 5
TIMED_RUNS = 50


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_model(run_dir, device, dtype=torch.bfloat16):
    backbone = AutoModel.from_pretrained(MODEL_NAME, dtype=dtype)
    hidden_size = backbone.config.hidden_size
    backbone = PeftModel.from_pretrained(backbone, run_dir / "adapter")
    model = DecisionModel(backbone, hidden_size).to(device)
    model.head.load_state_dict(torch.load(run_dir / "head.pt", map_location=device))
    return model.eval()


@torch.no_grad()
def single_decision_latency(model, dataset, collator, device):
    """Milliseconds for one decision at a time (batch of 1), using rows of median prompt length."""
    lengths = token_lengths(collator.tokenizer, dataset.prompts)
    median = sorted(lengths)[len(lengths) // 2]
    indices = [i for i, n in enumerate(lengths) if abs(n - median) <= 5][: WARMUP_RUNS + TIMED_RUNS]
    times = []
    for count, index in enumerate(indices):
        batch = to_device(collator([dataset[index]]), device)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model.predict(batch["input_ids"], batch["attention_mask"], batch["option_mask"])
        if device == "cuda":
            torch.cuda.synchronize()
        if count >= WARMUP_RUNS:
            times.append((time.perf_counter() - start) * 1000)
    ordered = sorted(times)
    return {"prompt_tokens": median, "mean_ms": round(statistics.mean(ordered), 2),
            "p95_ms": round(ordered[int(len(ordered) * 0.95)], 2)}


def comparison_table(tuned, baseline):
    lines = [f"{'decision type':16s} {'n':>5s} {'baseline':>9s} {'tuned':>7s} {'delta':>7s}"]
    for name, part in tuned["by_decision_type"].items():
        before = baseline["by_decision_type"][name]["accuracy"] if baseline else float("nan")
        lines.append(f"{name:16s} {part['n']:5d} {before:9.3f} {part['accuracy']:7.3f} {part['accuracy'] - before:+7.3f}")
    before = baseline["overall"]["accuracy"] if baseline else float("nan")
    lines.append(f"{'OVERALL':16s} {tuned['overall']['n']:5d} {before:9.3f} {tuned['overall']['accuracy']:7.3f} "
                 f"{tuned['overall']['accuracy'] - before:+7.3f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=str(ROOT / "runs/lora_run1"))
    parser.add_argument("--data", default=str(ROOT / "assets/processed/proxy_test.jsonl"))
    parser.add_argument("--baseline", default=str(ROOT / "runs/baseline_zero_shot.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--tag", default="", help="suffix for the metrics file name, e.g. fp16")
    args = parser.parse_args()
    run_dir, device = Path(args.run), "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = Collator(tokenizer)
    dataset = DecisionDataset(sample_rows(load_rows(args.data), args.limit, seed=0))
    model = load_model(run_dir, device, DTYPES[args.dtype])
    lengths = token_lengths(tokenizer, dataset.prompts)
    report = evaluate(model, dataset, collator, lengths, device, args.batch_size)
    report["latency_unmerged"] = single_decision_latency(model, dataset, collator, device)
    model.backbone = model.backbone.merge_and_unload()
    model.eval()
    report["latency_merged"] = single_decision_latency(model, dataset, collator, device)
    if device == "cuda":
        report["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf8")) if Path(args.baseline).exists() else None
    name = f"proxy_test_metrics_{args.tag}.json" if args.tag else "proxy_test_metrics.json"
    (run_dir / name).write_text(json.dumps(report, indent=2), encoding="utf8")
    print(comparison_table(report, baseline))
    print("\nordered types: within-1 accuracy and option positions never predicted")
    for name, part in report["by_decision_type"].items():
        if "within1" in part:
            print(f"  {name:12s} within1 {part['within1']:.3f}  mae {part['mae']:.3f}  never predicted: {part['never_predicted']}")
    print("\nby slice:", {k: v["accuracy"] for k, v in report["by_slice"].items()})
    print("by source:", {k: (v["n"], v["accuracy"]) for k, v in report["by_source"].items()})
    print("mean confidence:", report["mean_confidence"], "| top2:", report["overall"]["top2"], "| mae:", report["overall"].get("mae"))
    print("latency unmerged:", report["latency_unmerged"], "\nlatency merged:", report["latency_merged"],
          "\npeak vram gb:", report.get("peak_vram_gb"))


if __name__ == "__main__":
    main()
