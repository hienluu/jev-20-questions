"""Fair evaluation of Jev's matrix against answers that did not come from Jev.

Usage: uv run python -m scripts.evaluate animals [--games logs/games.jsonl]

1. Facts: how often Jev's P(yes) lands on the right side of 0.5, per the independent answer key
   (<category>.truth.json), plus a calibration table and the worst disagreements to spot-check.
2. Simulated games: the player answers from the answer key (optionally with random mistakes) while
   the engine plays on Jev's matrix. A "perfect knowledge" engine that plays on the answer key itself
   gives the ceiling, so the gap is what Jev's errors cost.
3. Real games (if a log exists): how often people's answers agree with Jev about the actual animal.
"""

import argparse
import json
import random
import statistics
from dataclasses import replace
from pathlib import Path

from twenty_q import engine
from twenty_q.engine import Category, Turn

CODE_TO_ANSWER = {"y": "yes", "n": "no", "s": "sort_of"}
# Probabilities used when the answer key itself drives the engine (the ceiling).
CODE_TO_P = {"y": 0.95, "n": 0.05, "s": 0.5}
NOISE_LEVELS = (0.0, 0.05, 0.10)
TRIALS = 5


def load_truth(category: Category) -> dict[str, dict[str, str]]:
    """The answer key, limited to the category's current animals and questions.

    Questions you add to the category that the key doesn't cover are treated as "don't know" in
    simulated games and skipped in the fact report.
    """
    raw = json.loads((engine.DATA_DIR / f"{category.name}.truth.json").read_text())
    ids = raw["question_ids"]
    current = {q.id for q in category.questions}
    return {
        c: {q: code for q, code in zip(ids, codes.replace(" ", ""), strict=True) if q in current}
        for c, codes in raw["answers"].items()
        if c in category.matrix
    }


def fact_report(category: Category, truth: dict[str, dict[str, str]]) -> None:
    pairs = [(category.matrix[c][q], code, c, q) for c, row in truth.items() for q, code in row.items()]
    decided = [(p, code, c, q) for p, code, c, q in pairs if code != "s"]
    agree = sum((p > 0.5) == (code == "y") for p, code, *_ in decided)
    print(f"\n## Facts: Jev vs answer key ({len(decided)} yes/no pairs, {len(pairs) - len(decided)} 'sort of' skipped)")
    print(f"agreement: {agree}/{len(decided)} = {agree / len(decided):.1%}")

    print("\ncalibration (when Jev says p, how often is the key 'yes'?)")
    print("  Jev p      pairs  key yes")
    for lo, hi in [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
        bucket = [code for p, code, *_ in decided if lo <= p < hi]
        if bucket:
            print(f"  {lo:.1f}-{min(hi, 1):.1f}   {len(bucket):6}  {sum(c == 'y' for c in bucket) / len(bucket):6.0%}")

    texts = {q.id: q.text for q in category.questions}
    worst = sorted(decided, key=lambda d: -abs(d[0] - (1.0 if d[1] == "y" else 0.0)))[:20]
    print("\nworst disagreements (spot-check these: either Jev or the key is wrong)")
    for p, code, c, q in worst:
        print(f"  {c:13} {texts[q]:45} key={CODE_TO_ANSWER[code]:4} jev={p:.2f}")


def play(category: Category, truth_row: dict[str, str], secret: str, noise: float, rng: random.Random) -> dict:
    """One game with the same rules as the UI: wrong guesses are excluded and play continues."""
    history: list[Turn] = []
    excluded: set[str] = set()
    guesses = 0
    while True:
        step = engine.next_step(category, history, frozenset(excluded))
        if step.should_guess:
            guesses += 1
            if step.ranking[0].candidate == secret:
                return {"won": True, "questions": len(history), "guesses": guesses}
            excluded.add(step.ranking[0].candidate)
            if len(history) >= engine.MAX_QUESTIONS - 1 or len(excluded) == len(category.candidates):
                return {"won": False, "questions": len(history), "guesses": guesses}
            continue
        answer = CODE_TO_ANSWER.get(truth_row.get(step.question.id, ""), "unknown")
        if answer in ("yes", "no") and rng.random() < noise:
            answer = "no" if answer == "yes" else "yes"
        history.append(Turn(step.question.id, answer))


def game_report(category: Category, truth: dict[str, dict[str, str]]) -> None:
    ceiling = replace(
        category,
        candidates=list(truth),
        matrix={c: {q.id: CODE_TO_P[row[q.id]] if q.id in row else 0.5 for q in category.questions} for c, row in truth.items()},
    )
    print(f"\n## Simulated games: player answers from the key, {len(truth)} animals x {TRIALS} trials per noise level")
    print("  wrong answers | engine on      | win    | 1st-guess | avg questions")
    for noise in NOISE_LEVELS:
        for label, cat in (("Jev matrix", category), ("answer key", ceiling)):
            rng = random.Random(42)
            games = [
                play(cat, truth[secret], secret, noise, rng)
                for _ in range(TRIALS if noise else 1)
                for secret in truth
            ]
            win = sum(g["won"] for g in games) / len(games)
            first = sum(g["won"] and g["guesses"] == 1 for g in games) / len(games)
            avg_q = statistics.mean(g["questions"] for g in games)
            print(f"  {noise:12.0%}  | {label:14} | {win:6.1%} | {first:9.1%} | {avg_q:5.1f}")


def human_report(category: Category, log: Path) -> None:
    games = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    games = [g for g in games if g["category"] == category.name]
    if not games:
        return
    agree = total = 0
    for g in games:
        row = category.matrix.get(g["actual"])
        for t in g["history"] if row else []:
            if t["answer"] in ("yes", "no"):
                total += 1
                agree += (row[t["question_id"]] > 0.5) == (t["answer"] == "yes")
    wins = sum(g["won"] for g in games)
    print(f"\n## Real games ({log})")
    print(f"games: {len(games)}, computer won: {wins}/{len(games)}")
    if total:
        print(f"player answers agreeing with Jev about the actual animal: {agree}/{total} = {agree / total:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("category")
    parser.add_argument("--games", type=Path, default=Path("logs/games.jsonl"))
    args = parser.parse_args()

    category = engine.load_category(args.category)
    if (engine.DATA_DIR / f"{args.category}.truth.json").exists():
        truth = load_truth(category)
        fact_report(category, truth)
        game_report(category, truth)
    else:
        print(f"No answer key ({args.category}.truth.json), so skipping the fact and simulated-game reports.")
    if args.games.exists():
        human_report(category, args.games)


if __name__ == "__main__":
    main()
