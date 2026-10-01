"""FastAPI server for the 20 Questions demo.

Stateless: the browser sends the full answer history with every request, and the
server recomputes the posterior from it. Run locally with:

    uv run uvicorn app:app --reload
"""

import json
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from functools import cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typesafe_sdk import AsyncTypeSafeClient, TypeSafeError

from twenty_q import engine, jev

STATIC_DIR = Path(__file__).parent / "static"
# Finished games, for scripts/evaluate.py. Local disk only: on Cloud Run/Modal this is ephemeral.
GAMES_LOG = Path(os.environ.get("GAMES_LOG", "logs/games.jsonl"))
# Cap on live (billed) Jev requests per client per hour. In-memory, so per server instance.
LIVE_BUDGET_PER_HOUR = int(os.environ.get("LIVE_BUDGET_PER_HOUR", "1000"))


@cache
def categories() -> dict[str, engine.Category]:
    names = sorted(p.name.removesuffix(".matrix.json") for p in engine.DATA_DIR.glob("*.matrix.json"))
    return {name: engine.load_category(name) for name in names}


_client: AsyncTypeSafeClient | None = None


def client() -> AsyncTypeSafeClient:
    """Created on first use so the game still runs (without Jev features) when no API key is set."""
    global _client
    if _client is None:
        _client = AsyncTypeSafeClient()
    return _client


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    if _client is not None:
        await _client.aclose()


app = FastAPI(title="20 Questions with Jev", lifespan=lifespan)


class TurnIn(BaseModel):
    question_id: str
    answer: Literal["yes", "no", "sort_of", "unknown"]


class GameIn(BaseModel):
    category: str
    history: list[TurnIn] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list, description="candidates already guessed wrong")

    def resolve(self) -> tuple[engine.Category, list[engine.Turn]]:
        category = _category(self.category)
        known = {q.id for q in category.questions}
        if bad := [t.question_id for t in self.history if t.question_id not in known]:
            raise HTTPException(422, f"unknown question ids {bad}")
        if len(set(self.excluded) & set(category.candidates)) >= len(category.candidates):
            raise HTTPException(422, "every candidate is excluded")
        return category, [engine.Turn(t.question_id, t.answer) for t in self.history]


class RevealIn(GameIn):
    actual: str = Field(min_length=1, max_length=60)


def _category(name: str) -> engine.Category:
    category = categories().get(name)
    if category is None:
        raise HTTPException(404, f"unknown category {name!r}")
    return category


_spent: dict[str, deque[tuple[float, int]]] = defaultdict(deque)


def _charge(request: Request, cost: int) -> None:
    """Refuse a live call that would take this client past its hourly budget of Jev requests."""
    key = request.client.host if request.client else "unknown"
    window, now = _spent[key], time.time()
    while window and window[0][0] < now - 3600:
        window.popleft()
    used = sum(c for _, c in window)
    if used + cost > LIVE_BUDGET_PER_HOUR:
        raise HTTPException(429, f"Live Jev budget used up ({used}/{LIVE_BUDGET_PER_HOUR} requests this hour). Try again later.")
    window.append((now, cost))


def _ranking(step: engine.NextStep) -> list[dict]:
    return [{"candidate": r.candidate, "probability": r.probability} for r in step.ranking]


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/categories")
async def list_categories() -> list[dict]:
    return [{"name": c.name, "title": c.title, "prompt": c.prompt, "size": len(c.candidates)} for c in categories().values()]


@app.post("/api/step")
async def step(game: GameIn) -> dict:
    category, history = game.resolve()
    excluded = frozenset(game.excluded)
    result = engine.next_step(category, history, excluded)
    return {
        "ranking": _ranking(result),
        "question": None if result.question is None else {"id": result.question.id, "text": result.question.text},
        "gain_bits": result.gain_bits,
        "should_guess": result.should_guess,
        "asked": len(history),
        "max_questions": engine.MAX_QUESTIONS,
        "under_the_hood": _under_the_hood(category, history, excluded, result),
    }


def _under_the_hood(
    category: engine.Category, history: list[engine.Turn], excluded: frozenset[str], result: engine.NextStep
) -> dict:
    """Where Jev's numbers enter the game this turn, for the UI's "under the hood" panel."""
    out: dict = {"next": None, "last": None}
    leaders = [r.candidate for r in result.ranking]
    if result.question:
        out["next"] = {
            "question": result.question.text,
            # The precomputed Noul answers the engine used to pick this question.
            "nouls": [{"candidate": c, "p_yes": category.matrix[c][result.question.id]} for c in leaders],
            "request": jev.request_payload(category.noun, leaders[0], [result.question]),
        }
    if history:
        last = history[-1]
        before = engine.posterior(category, history[:-1], excluded)
        after = engine.posterior(category, history, excluded)
        top = sorted(after, key=after.get, reverse=True)[:6]
        out["last"] = {
            "question": category.question(last.question_id).text,
            "answer": last.answer,
            "rows": [
                {
                    "candidate": c,
                    "before": before[c],
                    "p_yes": category.matrix[c][last.question_id],
                    "likelihood": engine.likelihood(category.matrix[c][last.question_id], last.answer),
                    "after": after[c],
                }
                for c in top
            ],
        }
    return out


class LiveNoulIn(BaseModel):
    category: str
    candidate: str = Field(min_length=1, max_length=60)
    question: str = Field(min_length=3, max_length=160)


@app.post("/api/jev/noul")
async def live_noul(body: LiveNoulIn, request: Request) -> dict:
    """Ask Jev one Noul live: the playground in the "under the hood" panel. Billed per call."""
    category = _category(body.category)
    _charge(request, 1)
    question = engine.Question(id="question", text=body.question.strip())
    try:
        live = await jev.ask_noul(client(), category.noun, body.candidate.strip().lower(), question)
    except TypeSafeError as error:
        raise HTTPException(503, f"Jev unavailable: {error}") from error
    return {"request": live.request, "response": live.response, "latency_ms": live.latency_ms}


class CompareIn(BaseModel):
    category: str
    candidate: str = Field(min_length=1, max_length=60)
    demo: Literal["size", "diet", "danger"] = "size"


@app.get("/api/jev/demos")
async def demos() -> dict:
    return {name: {"title": d["title"], "choice": d["choice"]["criteria"], "score": d["score"]["criteria"]} for name, d in jev.PRIMITIVE_DEMOS.items()}


@app.post("/api/jev/compare")
async def compare(body: CompareIn, request: Request) -> dict:
    """The same decision as Nouls, a Choice, and a Score, in one request."""
    _category(body.category)
    _charge(request, 1)
    try:
        live = await jev.compare_primitives(client(), body.candidate.strip().lower(), body.demo)
    except TypeSafeError as error:
        raise HTTPException(503, f"Jev unavailable: {error}") from error
    return {"request": live.request, "response": live.response, "latency_ms": live.latency_ms}


@app.post("/api/guess")
async def guess(game: GameIn) -> dict:
    """The engine's top candidate. Jev's own Choice over the transcript was tested here and guessed worse."""
    category, history = game.resolve()
    top = engine.next_step(category, history, frozenset(game.excluded), top_k=1).ranking[0]
    return {"candidate": top.candidate, "probability": top.probability}


@app.post("/api/reveal")
async def reveal(game: RevealIn, request: Request) -> dict:
    category, history = game.resolve()
    actual = game.actual.strip().lower()
    if actual not in category.matrix:
        _charge(request, 1 + len(history) // jev.CHUNK_SIZE)
    try:
        row = await jev.answer_row(client, category, history, actual)
    except TypeSafeError as error:
        raise HTTPException(503, f"Jev unavailable: {error}") from error
    found = jev.contradictions(category, history, row)
    return {
        "actual": actual,
        "known": actual in category.matrix,
        "contradictions": [{"question": c.question, "player_said": c.player_said, "p_yes": c.p_yes} for c in found],
    }


class ResultIn(GameIn):
    actual: str = Field(min_length=1, max_length=60)
    won: bool


@app.post("/api/result")
async def result(game: ResultIn) -> dict:
    """Record a finished game so real players' answers can be scored against Jev's matrix."""
    category, _ = game.resolve()
    GAMES_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.time(), **game.model_dump(), "category": category.name, "actual": game.actual.strip().lower()}
    with GAMES_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return {"ok": True}
