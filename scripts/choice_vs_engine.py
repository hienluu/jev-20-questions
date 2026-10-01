"""Who should make the final guess: the engine's Bayes update, or a Jev Choice over the transcript?

Usage: uv run --env-file .env python -m scripts.choice_vs_engine animals [--noise 0.1] [--shortlist 5]

Plays one simulated game per animal (the player answers from the independent answer key, with
optional random mistakes), stops when the engine is ready to guess, then asks Jev to pick from the
engine's shortlist by reading the Q&A transcript. Prints both accuracies.

This is LAB.md exercise 3: edit CHOICE_INSTRUCTIONS / build_state() below and try to beat the engine.
Cost: one Choice request per animal (62 for the animals category).
"""

import argparse
import asyncio
import random

from typesafe_sdk import AsyncTypeSafeClient, Choice

from scripts.evaluate import CODE_TO_ANSWER, load_truth
from twenty_q import engine
from twenty_q.engine import Category, Turn

CHOICE_INSTRUCTIONS = "Which animal best fits all of the yes/no answers in `answers`?"


def build_state(category: Category, history: list[Turn]) -> dict:
    """What Jev reads. Try: dropping 'sort of' answers, adding the engine's odds, rephrasing."""
    return {
        "answers": [
            {"question": category.question(t.question_id).text, "answer": t.answer.replace("_", " ")}
            for t in history
            if t.answer != "unknown"
        ]
    }


def play_until_guess(category: Category, truth_row: dict[str, str], noise: float, rng: random.Random) -> list[Turn]:
    history: list[Turn] = []
    while not (step := engine.next_step(category, history)).should_guess:
        answer = CODE_TO_ANSWER.get(truth_row.get(step.question.id, ""), "unknown")
        if answer in ("yes", "no") and rng.random() < noise:
            answer = "no" if answer == "yes" else "yes"
        history.append(Turn(step.question.id, answer))
    return history


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("category")
    parser.add_argument("--noise", type=float, default=0.0, help="chance each yes/no answer is flipped")
    parser.add_argument("--shortlist", type=int, default=5)
    args = parser.parse_args()

    category = engine.load_category(args.category)
    truth = load_truth(category)
    rng = random.Random(42)
    games = {secret: play_until_guess(category, truth[secret], args.noise, rng) for secret in truth}

    semaphore = asyncio.Semaphore(8)

    async def judge(client: AsyncTypeSafeClient, secret: str, history: list[Turn]) -> tuple[str, str, str, float]:
        shortlist = [r.candidate for r in engine.next_step(category, history, top_k=args.shortlist).ranking]
        async with semaphore:
            response = await client.system_one(
                state=build_state(category, history),
                questions={"guess": Choice(instructions=CHOICE_INSTRUCTIONS, criteria={c: None for c in shortlist})},
            )
        answer = response.choices["guess"]
        return secret, shortlist[0], answer.choice, answer.confidence

    async with AsyncTypeSafeClient() as client:
        results = await asyncio.gather(*(judge(client, s, h) for s, h in games.items()))

    n = len(results)
    engine_right = sum(secret == eng for secret, eng, _, _ in results)
    jev_right = sum(secret == pick for secret, _, pick, _ in results)
    print(f"{n} games, noise {args.noise:.0%}, shortlist {args.shortlist}")
    print(f"engine top candidate correct: {engine_right}/{n}")
    print(f"Jev Choice correct:           {jev_right}/{n}")
    misses = [(s, e, p, c) for s, e, p, c in results if p != s]
    if misses:
        print("\nJev Choice misses (secret -> engine / Jev, Jev confidence):")
        for secret, eng, pick, conf in misses:
            print(f"  {secret:13} -> {eng:13} / {pick:13} {conf:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
