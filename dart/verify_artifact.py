"""Acceptance check for an exported artifact: does it stand on its own and reproduce the evaluation numbers?

What it proves:
  * the artifact loads from a copy in a temporary folder, with the `peft` library blocked (self-contained);
  * accuracy of every row through the public `decide()` call, with end-to-end latency (tokenizing included);
  * `decide_many` over every shared-context group agrees with deciding each question separately;
  * the same input gives the same output on the real model (determinism).
"""
import argparse
import collections
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.modules["peft"] = None  # any `import peft` now fails: the artifact must not need it

import torch  # noqa: E402

from dart.api import DART  # noqa: E402
from dart.data import load_rows  # noqa: E402

DETERMINISM_ROWS = 300


def decide_row(dart, row):
    start = time.perf_counter()
    payload = dart.decide(row["query"], row["options"], context=row["context"])
    return payload["decisions"][0], (time.perf_counter() - start) * 1000


def accuracy_and_latency(dart, rows):
    by_type = collections.defaultdict(lambda: [0, 0])
    latencies, outputs = [], []
    for row in rows:
        decision, ms = decide_row(dart, row)
        correct = decision["decision"] == row["target_option"]
        by_type[row["decision_type"]][0] += correct
        by_type[row["decision_type"]][1] += 1
        latencies.append(ms)
        outputs.append(decision)
    latencies.sort()
    return {
        "overall_accuracy": round(sum(c for c, _ in by_type.values()) / len(rows), 4),
        "by_type": {t: round(c / n, 4) for t, (c, n) in sorted(by_type.items())},
        "latency_ms": {"mean": round(statistics.mean(latencies), 2), "p50": round(latencies[len(rows) // 2], 2),
                       "p95": round(latencies[int(len(rows) * 0.95)], 2), "max": round(latencies[-1], 2)},
    }, outputs


def shared_context_agreement(dart, rows):
    groups = collections.defaultdict(list)
    for row in rows:
        if row["group_id"]:
            groups[row["group_id"]].append(row)
    groups = [g for g in groups.values() if len(g) >= 2]
    same = total = correct_many = correct_single = 0
    for group in groups:
        questions = [{"query": r["query"], "options": r["options"]} for r in group]
        many = dart.decide_many(group[0]["context"], questions)["decisions"]
        for row, parallel in zip(group, many):
            single, _ = decide_row(dart, row)
            same += parallel["decision"] == single["decision"]
            correct_many += parallel["decision"] == row["target_option"]
            correct_single += single["decision"] == row["target_option"]
            total += 1
    return {"groups": len(groups), "questions": total, "same_decision": same,
            "accuracy_parallel": round(correct_many / total, 4), "accuracy_separate": round(correct_single / total, 4)}


def determinism(dart, rows, first_outputs):
    identical = sum(decide_row(dart, row)[0] == first for row, first in zip(rows, first_outputs))
    return {"rows": len(rows), "identical_on_repeat": identical}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default=str(ROOT / "artifacts/dart_v1.0"))
    parser.add_argument("--data", default=str(ROOT / "assets/processed/proxy_test.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "runs/verify_artifact.json"))
    args = parser.parse_args()
    rows = load_rows(args.data)
    with tempfile.TemporaryDirectory() as folder:
        copy = Path(folder) / "artifact_copy"
        shutil.copytree(args.artifact, copy)
        dart = DART.from_pretrained(copy, fast=True)
        for _ in range(3):
            dart.decide("warm-up", ["a", "b"])
        report = {"peft_blocked": sys.modules["peft"] is None, "loaded_from_copy": True, "device": dart.device}
        report["single_decisions"], outputs = accuracy_and_latency(dart, rows)
        report["shared_context"] = shared_context_agreement(dart, rows)
        report["determinism"] = determinism(dart, rows[:DETERMINISM_ROWS], outputs[:DETERMINISM_ROWS])
        report["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2) if dart.device == "cuda" else None
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
