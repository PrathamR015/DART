"""Export a trained checkpoint (LoRA adapter + head) as one self-contained artifact folder.

artifact/
  backbone/     merged Qwen3 weights (adapter folded in), half precision
  head.pt       decision head weights
  config.json   options, prompt template, pooling, padding, dtype, versions
  tokenizer/    tokenizer files
  README.md     usage, training summary, evaluation results
"""
import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

from data_prep.schema import MAX_OPTIONS, MAX_PROMPT_TOKENS
from dart.evaluate import load_model
from dart.fast_inference import DEFAULT_BUCKETS
from dart.train import MODEL_NAME

ROOT = Path(__file__).resolve().parents[1]
README_TEMPLATE = """# D.A.R.T. {version}

Decision Already Reached, Thanks. A single-forward-pass decision model: give it a query and a list of options
and it returns every option with its probability, sorted descending. The top option is the decision.

## Usage
```python
from dart.api import DART

dart = DART.from_pretrained("{folder}")
result = dart.decide("Which intent is this? 'my card was stolen'", ["report lost card", "weather", "transfer money"])
print(result["decisions"][0]["decision"], result["decisions"][0]["confidence"])

many = dart.decide_many("Tony is 16 and lives in England.",
                        [{{"id": "age", "query": "How old is Tony?", "options": ["12", "16"]}},
                         {{"id": "country", "query": "Where does Tony live?", "options": ["China", "England"]}}])
```

## Model
- Base: `{base_model}` (merged LoRA adapter, head trained on top), weights in {dtype}.
- Input: plain-text prompt (`Context / Query / Options / Answer:`), at most {max_tokens} tokens, 2 to {max_options} options.
- Output follows the PRD 5.1 contract. Probabilities are the raw softmax, not calibrated.
- Libraries: torch {torch_version}, transformers {transformers_version}.

## Evaluation (public proxy test set, NOT a hand-verified real set)
{table}

Overall accuracy {overall}. These numbers come from a proxy split drawn from the same public sources as training, so
they are optimistic. Domain coverage is general text only. Sign-off needs a hand-verified real held-out set.
"""


def build_config(model, version, dtype):
    return {
        "model": "D.A.R.T.", "version": version, "base_model": MODEL_NAME, "dtype": dtype,
        "hidden_size": model.head.in_features, "max_options": MAX_OPTIONS, "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "pooling": "last_token", "padding_side": "left", "graph_buckets": list(DEFAULT_BUCKETS),
        "prompt_template": {"prefix": "Context: {context}\n",
                            "suffix": "Query: {query}\nOptions:\nA) {option_a}\nB) {option_b}\n...\nAnswer:"},
        "torch": torch.__version__, "transformers": transformers.__version__,
    }


def build_readme(config, folder, metrics):
    lines = ["| decision type | rows | accuracy |", "|---|---|---|"]
    for name, part in metrics["by_decision_type"].items():
        lines.append(f"| {name} | {part['n']} | {part['accuracy']:.1%} |")
    return README_TEMPLATE.format(
        version=config["version"], folder=folder, base_model=config["base_model"], dtype=config["dtype"],
        max_tokens=config["max_prompt_tokens"], max_options=config["max_options"], torch_version=config["torch"],
        transformers_version=config["transformers"], table="\n".join(lines),
        overall=f"{metrics['overall']['accuracy']:.1%} (top-2 {metrics['overall']['top2']:.1%})")


def save_artifact(model, tokenizer, out_dir, config, readme):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(out / "backbone")
    torch.save(model.head.state_dict(), out / "head.pt")
    tokenizer.save_pretrained(out / "tokenizer")
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf8")
    (out / "README.md").write_text(readme, encoding="utf8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=str(ROOT / "runs/lora_run1_refit"))
    parser.add_argument("--out", default=str(ROOT / "artifacts/dart_v1.0"))
    parser.add_argument("--version", default="v1.0")
    parser.add_argument("--metrics", default=str(ROOT / "runs/lora_run1_refit/proxy_test_metrics_fp16.json"))
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(args.run), device, torch.float16)
    model.backbone = model.backbone.merge_and_unload()
    config = build_config(model, args.version, "float16")
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf8"))
    save_artifact(model, AutoTokenizer.from_pretrained(MODEL_NAME), args.out, config,
                  build_readme(config, Path(args.out).name, metrics))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
