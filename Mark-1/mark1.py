"""Mark-1: a small interactive program for trying the trained D.A.R.T. model.

Give it a question (and, if you like, a context) plus the options. It prints every option with its
probability, the decision, and how long the decision took.

Run from the project folder:
    .venv\\Scripts\\python.exe Mark-1\\mark1.py

Options can be typed separated by commas (yes, no, maybe) or by | when an option itself contains a comma.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dart.api import DART  # noqa: E402

DEFAULT_ARTIFACT = ROOT / "artifacts" / "dart_v1.0"
BAR_WIDTH = 30
OPTION_NAME_WIDTH = 40
WARMUP_CALLS = 3
QUIT_WORDS = {"q", "quit", "exit"}


def parse_options(text):
    separator = "|" if "|" in text else ","
    return [part.strip() for part in text.split(separator) if part.strip()]


def read_line(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        return "quit"


def timed_decide(dart, query, options, context):
    """Returns (decision dict, latency in milliseconds)."""
    start = time.perf_counter()
    payload = dart.decide(query, options, context=context)
    return payload["decisions"][0], (time.perf_counter() - start) * 1000


def format_result(decision, latency_ms):
    lines = []
    for rank, item in enumerate(decision["ranked"], start=1):
        bar = "#" * round(item["probability"] * BAR_WIDTH)
        name = item["option"][:OPTION_NAME_WIDTH]
        lines.append(f"  {rank}. {name:<{OPTION_NAME_WIDTH}} {item['probability'] * 100:5.1f}%  {bar}")
    lines.append(f"\n  Decision: {decision['decision']}  (confidence {decision['confidence'] * 100:.1f}%)")
    lines.append(f"  Latency:  {latency_ms:.1f} ms")
    return "\n".join(lines)


def run(dart):
    print("Type 'quit' at any prompt to exit.\n")
    while True:
        query = read_line("Question: ")
        if query.lower() in QUIT_WORDS:
            return
        if not query:
            print("  Please enter a question.\n")
            continue
        context = read_line("Context (optional, press Enter to skip): ")
        if context.lower() in QUIT_WORDS:
            return
        options = parse_options(read_line("Options (comma-separated, or use | between them): "))
        try:
            decision, latency_ms = timed_decide(dart, query, options, context)
        except ValueError as error:
            print(f"  Could not decide: {error}\n")
            continue
        print("\n" + format_result(decision, latency_ms) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Try the trained D.A.R.T. model interactively.")
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT), help="folder of the exported model")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--no-fast", action="store_true", help="skip the CUDA-graph fast path")
    args = parser.parse_args()

    print("Loading D.A.R.T. ...")
    start = time.perf_counter()
    dart = DART.from_pretrained(args.artifact, device=args.device, fast=not args.no_fast)
    print(f"Ready on {dart.device} in {time.perf_counter() - start:.1f} s "
          f"({'fast CUDA-graph path' if dart.graphs else 'standard path'}).")
    for _ in range(WARMUP_CALLS):
        dart.decide("warm-up", ["a", "b"])
    run(dart)


if __name__ == "__main__":
    main()
