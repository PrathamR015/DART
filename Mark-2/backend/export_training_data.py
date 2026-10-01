"""Export the decisions users labelled in the web app as DART training rows.

Run from the project folder:
    .venv\\Scripts\\python.exe Mark-2\\backend\\export_training_data.py

Writes `mark2_labeled.jsonl` (rows with feedback, in the training format) and `mark2_unlabeled.jsonl` (stored
inputs nobody has labelled yet, for later labelling) into assets/processed/ by default.
"""
import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
ROOT = BACKEND.parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT))

from pymongo import MongoClient  # noqa: E402

from dartweb.settings import load_settings  # noqa: E402
from dartweb.training_export import build_rows, unlabeled  # noqa: E402

DEFAULT_OUT = ROOT / "assets" / "processed"


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def main():
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uri", default=settings.mongodb_uri)
    parser.add_argument("--db", default=settings.mongodb_db)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    client = MongoClient(args.uri, serverSelectionTimeoutMS=3000)
    records = list(client[args.db]["decisions"].find({}).sort("created_at", 1))
    rows, skipped = build_rows(records)
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "mark2_labeled.jsonl", rows)
    write_jsonl(out_dir / "mark2_unlabeled.jsonl", unlabeled(records))
    print(f"{len(records)} stored decisions: {len(rows)} labelled rows exported, {skipped} skipped "
          f"(no feedback or invalid) -> {out_dir}")


if __name__ == "__main__":
    main()
