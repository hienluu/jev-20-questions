"""Game logic for 20 Questions: Bayesian updates and question selection.

Pure functions with no I/O. A category's Noul matrix holds P(yes) for every
(candidate, question) pair; the player's answers update a posterior over
candidates, and the next question is the one with the highest expected
information gain.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Answer = Literal["yes", "no", "sort_of", "unknown"]

DATA_DIR = Path(__file__).parent / "data"
MAX_QUESTIONS = 20
GUESS_THRESHOLD = 0.6
MIN_GAIN_BITS = 0.01
# Keep likelihoods away from 0/1 so one mistaken answer can't eliminate the right candidate.
NOISE_FLOOR = 0.05


@dataclass(frozen=True)
class Question:
    id: str
    text: str


@dataclass(frozen=True)
class Category:
    name: str
    title: str
    prompt: str
    noun: str
    candidates: list[str]
    questions: list[Question]
    # matrix[candidate][question_id] = P(candidate answers yes)
    matrix: dict[str, dict[str, float]]

    def question(self, question_id: str) -> Question:
        return next(q for q in self.questions if q.id == question_id)


@dataclass(frozen=True)
class Turn:
    question_id: str
    answer: Answer


@dataclass(frozen=True)
class Ranked:
    candidate: str
    probability: float


@dataclass(frozen=True)
class NextStep:
    ranking: list[Ranked]
    question: Question | None
    gain_bits: float
    should_guess: bool


def load_category(name: str, data_dir: Path = DATA_DIR) -> Category:
    spec = json.loads((data_dir / f"{name}.json").read_text())
    matrix_path = data_dir / f"{name}.matrix.json"
    if not matrix_path.exists():
        raise FileNotFoundError(f"{matrix_path} is missing; run `uv run python -m scripts.build_matrix {name}` first")
    return Category(
        name=spec["name"],
        title=spec["title"],
        prompt=spec["prompt"],
        noun=spec["noun"],
        candidates=spec["candidates"],
        questions=[Question(**q) for q in spec["questions"]],
        matrix=json.loads(matrix_path.read_text()),
    )


def likelihood(p_yes: float, answer: Answer) -> float:
    p = min(max(p_yes, NOISE_FLOOR), 1 - NOISE_FLOOR)
    match answer:
        case "yes":
            return p
        case "no":
            return 1 - p
        case "sort_of":
            # Favors candidates the model is unsure about, without punishing the others much.
            return 1 - abs(p - 0.5)
        case "unknown":
            return 1.0


def posterior(category: Category, history: list[Turn], excluded: frozenset[str] = frozenset()) -> dict[str, float]:
    """Excluded candidates (wrong guesses) are dropped entirely."""
    log_weights = {c: 0.0 for c in category.candidates if c not in excluded}
    for turn in history:
        for c in log_weights:
            log_weights[c] += math.log(likelihood(category.matrix[c][turn.question_id], turn.answer))
    peak = max(log_weights.values())
    weights = {c: math.exp(w - peak) for c, w in log_weights.items()}
    total = sum(weights.values())
    return {c: w / total for c, w in weights.items()}


def entropy(probs: list[float]) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)


def information_gain(category: Category, post: dict[str, float], question_id: str) -> float:
    """Expected reduction in entropy (bits) from asking a yes/no question."""
    p_yes = sum(post[c] * category.matrix[c][question_id] for c in post)
    p_no = 1 - p_yes
    if p_yes <= 0 or p_no <= 0:
        return 0.0
    after_yes = [post[c] * category.matrix[c][question_id] / p_yes for c in post]
    after_no = [post[c] * (1 - category.matrix[c][question_id]) / p_no for c in post]
    return entropy(list(post.values())) - (p_yes * entropy(after_yes) + p_no * entropy(after_no))


def next_step(
    category: Category, history: list[Turn], excluded: frozenset[str] = frozenset(), top_k: int = 8
) -> NextStep:
    post = posterior(category, history, excluded)
    ranking = [Ranked(c, p) for c, p in sorted(post.items(), key=lambda kv: kv[1], reverse=True)]
    asked = {t.question_id for t in history}
    gains = {q.id: information_gain(category, post, q.id) for q in category.questions if q.id not in asked}
    best_id = max(gains, key=gains.__getitem__, default=None)
    best_gain = gains.get(best_id, 0.0) if best_id else 0.0
    out_of_questions = len(history) >= MAX_QUESTIONS - 1 or best_id is None or best_gain < MIN_GAIN_BITS
    return NextStep(
        ranking=ranking[:top_k],
        question=None if best_id is None else category.question(best_id),
        gain_bits=best_gain,
        should_guess=ranking[0].probability >= GUESS_THRESHOLD or out_of_questions,
    )
