"""Precompute a category's candidate x question Noul matrix with Jev.

Usage: uv run python -m scripts.build_matrix animals [--limit N]

Needs TYPESAFE_API_KEY. Writes twenty_q/data/<category>.matrix.json, which ships with the app
so gameplay makes no Jev calls.
"""

import argparse
import asyncio
import json
import time

from typesafe_sdk import AsyncTypeSafeClient

from twenty_q.engine import DATA_DIR, Question
from twenty_q.jev import build_matrix


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("category")
    parser.add_argument("--limit", type=int, help="only the first N candidates (for a cheap smoke test)")
    args = parser.parse_args()

    spec = json.loads((DATA_DIR / f"{args.category}.json").read_text())
    candidates = spec["candidates"][: args.limit]
    questions = [Question(**q) for q in spec["questions"]]
    print(f"{len(candidates)} candidates x {len(questions)} questions = {len(candidates) * len(questions)} nouls")

    started = time.perf_counter()
    async with AsyncTypeSafeClient() as client:
        matrix = await build_matrix(client, spec["noun"], candidates, questions)
    print(f"done in {time.perf_counter() - started:.1f}s")

    out = DATA_DIR / f"{args.category}.matrix.json"
    out.write_text(json.dumps(matrix, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
