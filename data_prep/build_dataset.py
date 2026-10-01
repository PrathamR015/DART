"""Phase 0: convert the public datasets and the salvaged synthetic rows into train / val / proxy_test files."""
import json
import random
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from data_prep import sources_context as ctx
from data_prep import sources_scales as scales
from data_prep import sources_text as text
from data_prep.dataset_report import build_report
from data_prep.noise import add_noise
from data_prep.salvage import normalize
from data_prep.schema import MAX_PROMPT_TOKENS, format_prompt, validate_row

ROOT = Path(__file__).resolve().parents[1] / "assets"
PUBLIC = ROOT / "public"
OUT = ROOT / "processed"
SEED = 13
NOISE_SHARE = 0.15
VAL_PERCENT = 3
EVAL_SCALE = 0.06
MIN_SIZE = 20
TOKENIZER_NAME = "Qwen/Qwen3-0.6B"
PUBMED_TRAIN_ROWS = 800


def _read(relative):
    path = PUBLIC / relative
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_json(path, lines=True)


def _pq(folder, split):
    return _read(f"{folder}/{split}-00000-of-00001.parquet")


def convert_split(split, scale, rng):
    """Convert every public source for one split ('train' or 'eval')."""
    train = split == "train"
    name = "train" if train else "test"
    validation = "train" if train else "validation"
    n = lambda base: max(MIN_SIZE, int(base * scale))  # noqa: E731
    intents = text.parse_class_names((PUBLIC / "clinc150" / "README.md").read_text(encoding="utf8"))
    pubmed = _read("pubmedqa/pqa_labeled/train-00000-of-00001.parquet")
    pubmed = pubmed.iloc[:PUBMED_TRAIN_ROWS] if train else pubmed.iloc[PUBMED_TRAIN_ROWS:]
    rows = []
    rows += text.clinc_rows(_pq("clinc150/plus", name), intents, rng, n(8000))
    rows += text.agnews_rows(_pq("agnews/data", name), rng, n(6000))
    rows += text.goemotions_rows(_pq("goemotions/simplified", name), rng, n(6000))
    rows += text.arc_rows(_pq("arc/ARC-Challenge", name), rng, n(1200), challenge=True)
    rows += text.arc_rows(_pq("arc/ARC-Easy", name), rng, n(2200), challenge=False)
    rows += text.csqa_rows(_pq("commonsenseqa/data", validation), rng, n(8000))
    rows += ctx.boolq_rows(_pq("boolq/data", validation), rng, n(3500))
    rows += ctx.pubmedqa_rows(pubmed, rng, n(300))
    rows += ctx.race_rows(_pq("race/all", name), rng, n(2500))
    rows += scales.sst5_rows(_read("sst5/train.jsonl" if train else "sst5/test.jsonl"), rng, n(900))
    rows += scales.yelp_rows(_pq("yelp/yelp_review_full", name), rng, n(1200))
    rows += scales.stsb_rows(_pq("stsb/stsb", "train" if train else "validation"), rng, n(500))
    helpsteer = "helpsteer2/train.jsonl.gz" if train else "helpsteer2/validation.jsonl.gz"
    rows += scales.helpsteer_rows(_read(helpsteer), rng, n(500))
    if train:
        rows += text.mmlu_rows(_pq("mmlu/auxiliary_train", "train"), rng, 12000)
    return rows


def load_salvaged():
    with (OUT / "salvaged.jsonl").open(encoding="utf8") as handle:
        return [{"group_id": "", **json.loads(line)} for line in handle]


def _key(row):
    return (normalize(row["query"]), normalize(row["context"])[:200], tuple(row["options"]))


def dedupe(rows):
    """Keep one row per identical question; drop every copy when answers conflict."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[_key(row)].append(index)
    drop, conflicts = set(), 0
    for indexes in groups.values():
        answers = {rows[i]["target_option"] for i in indexes}
        if len(answers) > 1:
            drop.update(indexes)
            conflicts += len(indexes)
        else:
            drop.update(indexes[1:])
    kept = [r for i, r in enumerate(rows) if i not in drop]
    return kept, len(drop) - conflicts, conflicts


def add_noisy_slice(rows, rng, share):
    clean = [i for i, r in enumerate(rows) if r["slice"] != "noisy"]
    chosen = set(rng.sample(clean, int(len(clean) * share)))
    out = []
    for index, row in enumerate(rows):
        if index in chosen:
            noisy = add_noise(row["query"], rng)
            if noisy != row["query"]:
                row = {**row, "query": noisy, "slice": "noisy"}
        out.append(row)
    return out


def assign_ids(rows):
    counters = Counter()
    out = []
    for row in rows:
        counters[row["source"]] += 1
        out.append({**row, "id": f"{row['source']}_{counters[row['source']]:06d}"})
    return out


def split_validation(rows, percent):
    """Group-aware split: rows sharing a context always land on the same side."""
    train, val = [], []
    for row in rows:
        bucket = zlib.crc32((row["group_id"] or row["id"]).encode("utf8")) % 100
        (val if bucket < percent else train).append(row)
    return train, val


def _tokenizer_counter():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return lambda prompts: [len(ids) for ids in tokenizer(prompts, add_special_tokens=False)["input_ids"]]


def within_token_budget(rows, count_tokens):
    lengths = count_tokens([format_prompt(r) for r in rows])
    kept = [(r, n) for r, n in zip(rows, lengths) if n <= MAX_PROMPT_TOKENS]
    return [r for r, _ in kept], [n for _, n in kept], len(rows) - len(kept)


def drop_invalid(rows):
    good = [r for r in rows if validate_row(r) is None]
    return good, len(rows) - len(good)


def prepare(rows, rng, count_tokens, banned_keys=frozenset()):
    """Validate, dedupe, add noise, budget-filter and id a list of rows."""
    stages = {"input": len(rows)}
    rows, stages["invalid"] = drop_invalid(rows)
    rows, stages["duplicates"], stages["conflicts"] = dedupe(rows)
    stages["leaked_into_other_split"] = sum(_key(r) in banned_keys for r in rows)
    rows = [r for r in rows if _key(r) not in banned_keys]
    rows = add_noisy_slice(rows, rng, NOISE_SHARE)
    rows, lengths, stages["over_token_budget"] = within_token_budget(rows, count_tokens)
    stages["kept"] = len(rows)
    return assign_ids(rows), lengths, stages


def write_jsonl(path, rows):
    with path.open("w", encoding="utf8") as handle:
        handle.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main():
    rng = random.Random(SEED)
    count_tokens = _tokenizer_counter()
    test_rows, test_lengths, test_stages = prepare(convert_split("eval", EVAL_SCALE, rng), rng, count_tokens)
    test_keys = frozenset(_key(r) for r in test_rows)
    raw_train = convert_split("train", 1.0, rng) + load_salvaged()
    rows, lengths, train_stages = prepare(raw_train, rng, count_tokens, test_keys)
    length_by_id = {r["id"]: n for r, n in zip(rows, lengths)}
    train, val = split_validation(rows, VAL_PERCENT)
    splits = {"train": train, "val": val, "proxy_test": test_rows}
    token_lengths = {"train": [length_by_id[r["id"]] for r in train],
                     "val": [length_by_id[r["id"]] for r in val], "proxy_test": test_lengths}
    OUT.mkdir(exist_ok=True)
    for name, part in splits.items():
        write_jsonl(OUT / f"{name}.jsonl", part)
    report = build_report(splits, token_lengths, {"train": train_stages, "proxy_test": test_stages})
    (OUT / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps({"stages": report["stages"],
                      "rows": {k: v["rows"] for k, v in report["splits"].items()}}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
