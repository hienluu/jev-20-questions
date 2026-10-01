"""TypeSafe (Jev) calls: the candidate x question Noul matrix and the after-a-miss answer check."""

import asyncio
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score, constants

from twenty_q.engine import Category, Question, Turn

# Questions per request when building the matrix; one request per candidate chunk.
CHUNK_SIZE = 30
# How far a player's answer must disagree with Jev before we call it a contradiction.
CONTRADICTION_MARGIN = 0.3


def _noul(noun: str, question: Question) -> Noul:
    return Noul(instructions=f"Answer about the {noun} named in the state: {question.text}")


def request_payload(noun: str, candidate: str, questions: list[Question]) -> dict:
    """The JSON body the SDK sends for one candidate: what the "under the hood" view shows."""
    return {
        "model": os.environ.get(constants.DEFAULT_MODEL_ENV, constants.DEFAULT_MODEL),
        "state": candidate,
        "questions": {q.id: _noul(noun, q).model_dump() for q in questions},
    }


@dataclass(frozen=True)
class LiveNoul:
    request: dict
    response: dict
    latency_ms: float


async def ask_noul(client: AsyncTypeSafeClient, noun: str, candidate: str, question: Question) -> LiveNoul:
    """One live Noul call, same shape as the matrix build, timed end to end."""
    started = time.perf_counter()
    response = await client.system_one(state=candidate, questions={question.id: _noul(noun, question)})
    latency_ms = (time.perf_counter() - started) * 1000
    return LiveNoul(
        request=request_payload(noun, candidate, [question]),
        response={
            "model": response.model,
            "answers": {name: answer.model_dump() for name, answer in response.answers.items()},
            "usage": response.usage.model_dump(),
        },
        latency_ms=latency_ms,
    )


async def candidate_row(
    client: AsyncTypeSafeClient, noun: str, candidate: str, questions: list[Question]
) -> dict[str, float]:
    """P(yes) for every question about one candidate."""
    row: dict[str, float] = {}
    for start in range(0, len(questions), CHUNK_SIZE):
        chunk = questions[start : start + CHUNK_SIZE]
        response = await client.system_one(state=candidate, questions={q.id: _noul(noun, q) for q in chunk})
        row.update({q.id: response.nouls[q.id].noul for q in chunk})
    return row


async def build_matrix(
    client: AsyncTypeSafeClient, noun: str, candidates: list[str], questions: list[Question], concurrency: int = 8
) -> dict[str, dict[str, float]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(candidate: str) -> tuple[str, dict[str, float]]:
        async with semaphore:
            return candidate, await candidate_row(client, noun, candidate, questions)

    return dict(await asyncio.gather(*(one(c) for c in candidates)))


@dataclass(frozen=True)
class Contradiction:
    question: str
    player_said: str
    p_yes: float


async def answer_row(
    client_factory: Callable[[], AsyncTypeSafeClient], category: Category, history: list[Turn], actual: str
) -> dict[str, float]:
    """P(yes) for the asked questions about `actual`: from the matrix when known, otherwise live from Jev."""
    if actual in category.matrix:
        return category.matrix[actual]
    asked = [category.question(t.question_id) for t in history]
    return await candidate_row(client_factory(), category.noun, actual, asked)


def contradictions(category: Category, history: list[Turn], row: dict[str, float]) -> list[Contradiction]:
    """After a miss: which of the player's answers disagree with what Jev knows about the real answer?"""
    found = []
    for turn in history:
        p_yes = row[turn.question_id]
        if (turn.answer == "yes" and p_yes < CONTRADICTION_MARGIN) or (turn.answer == "no" and p_yes > 1 - CONTRADICTION_MARGIN):
            found.append(Contradiction(category.question(turn.question_id).text, turn.answer, p_yes))
    return found


# The same decision asked three ways, to compare Noul, Choice, and Score side by side.
_ABOUT = "About the animal named in the state: "
PRIMITIVE_DEMOS: dict[str, dict] = {
    "size": {
        "title": "How big is it?",
        "nouls": {
            "bigger_than_cat": "Is it bigger than a house cat?",
            "bigger_than_human": "Is it bigger than an adult human?",
            "bigger_than_horse": "Is it bigger than a horse?",
        },
        "choice": {
            "instructions": "Which size class fits it best?",
            "criteria": {
                "tiny": "Fits in the palm of your hand",
                "small": "Bigger than that, up to a house cat",
                "medium": "Bigger than a cat, up to an adult human",
                "large": "Bigger than a human, up to a horse",
                "huge": "Bigger than a horse",
            },
        },
        "score": {
            "instructions": "How big is it?",
            "criteria": [
                "Fits in the palm of your hand",
                "Up to a house cat",
                "Up to an adult human",
                "Up to a horse",
                "Bigger than a horse",
            ],
        },
    },
    "diet": {
        "title": "What does it eat?",
        "nouls": {"eats_meat": "Does it mainly eat meat?", "eats_plants": "Does it mainly eat plants?"},
        "choice": {
            "instructions": "Which best describes what it eats?",
            "criteria": {
                "herbivore": "Mostly plants, fruit, or seeds",
                "carnivore": "Mostly meat or other animals",
                "omnivore": "A real mix of plants and animals",
                "filter_feeder": "Strains tiny food out of water",
            },
        },
        "score": {
            "instructions": "Where does its diet sit between plants and meat?",
            "criteria": ["Only plants", "Mostly plants", "Both about equally", "Mostly meat", "Only meat"],
        },
    },
    "danger": {
        "title": "How dangerous is it to people?",
        "nouls": {
            "can_injure": "Could it seriously injure an adult human?",
            "can_kill": "Could it kill an adult human?",
        },
        "choice": {
            "instructions": "How dangerous is it to an adult human?",
            "criteria": {
                "harmless": "No real risk",
                "minor": "Bites, scratches, or stings that hurt but heal",
                "serious": "Can seriously injure",
                "deadly": "Can kill",
            },
        },
        "score": {
            "instructions": "How dangerous is it to an adult human?",
            "criteria": ["Harmless", "Minor injuries at worst", "Can seriously injure", "Can kill"],
        },
    },
}


def demo_questions(demo: str) -> dict:
    spec = PRIMITIVE_DEMOS[demo]
    questions: dict = {qid: Noul(instructions=_ABOUT + text) for qid, text in spec["nouls"].items()}
    questions["as_choice"] = Choice(instructions=_ABOUT + spec["choice"]["instructions"], criteria=spec["choice"]["criteria"])
    questions["as_score"] = Score(instructions=_ABOUT + spec["score"]["instructions"], criteria=spec["score"]["criteria"])
    return questions


async def compare_primitives(client: AsyncTypeSafeClient, candidate: str, demo: str) -> LiveNoul:
    """One request mixing all three primitives about the same decision."""
    questions = demo_questions(demo)
    started = time.perf_counter()
    response = await client.system_one(state=candidate, questions=questions)
    latency_ms = (time.perf_counter() - started) * 1000
    return LiveNoul(
        request={
            "model": os.environ.get(constants.DEFAULT_MODEL_ENV, constants.DEFAULT_MODEL),
            "state": candidate,
            "questions": {name: q.model_dump() for name, q in questions.items()},
        },
        response={
            "model": response.model,
            "answers": {name: answer.model_dump() for name, answer in response.answers.items()},
            "usage": response.usage.model_dump(),
        },
        latency_ms=latency_ms,
    )
