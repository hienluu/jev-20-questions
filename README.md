# 20 Questions with Jev

Think of an animal; the computer guesses it. Jev (TypeSafe's System One model) supplies
common-sense facts as probabilities, and ordinary Python code plays the game.

## How Jev is used

### The idea in one paragraph

Jev is not a chatbot and does not "play" the game. It answers narrow, typed questions about a
piece of input and returns a calibrated probability. This app uses one kind of question, a
**Noul** (a yes/no question answered with P(yes)): *"Would a penguin answer yes to 'Can it fly?'"*
→ `0.04`. All the reasoning, from combining answers and choosing the next question to deciding when to
guess, is deterministic code that consumes those probabilities. Think of Jev as a **fuzzy lookup
function** `p_yes(animal, question) -> float` and the game as a normal program built on top of it.

### When Jev is called

| When | What is called | How often | Code |
|---|---|---|---|
| **Build time** (once, before deploy) | A Noul for every animal × question: 62 × 60 = 3,720 | Once per category, about 2 s, results saved to JSON | `scripts/build_matrix.py`, `twenty_q/jev.py: build_matrix` |
| **During play** | Nothing. The server reads the saved table | 0 calls per game | `twenty_q/engine.py` |
| **After a loss, "What was it?"** | Nouls for the questions asked, about the animal the player typed, *only if it isn't in the table* | 0 or 1 request per lost game | `twenty_q/jev.py: answer_row` |
| **Playground: "Ask Jev one question"** in the Under the hood drawer | One Noul, with any animal and question the developer types | Per click (billed) | `POST /api/jev/noul`, `twenty_q/jev.py: ask_noul` |
| **Playground: "One decision, three primitives"** | One request mixing Nouls, a Choice and a Score about the same decision | Per click (billed) | `POST /api/jev/compare`, `twenty_q/jev.py: compare_primitives` |

The expensive, model-backed work happens **offline**. The online path is a dictionary lookup plus
arithmetic, so a turn takes well under a millisecond and a game costs nothing in API calls.

### Which of Jev's three primitives are used

TypeSafe offers three question types: **Noul** (yes/no → probability), **Choice** (pick one option
from a set → per-option probabilities and confidence), and **Score** (rate against ordered levels).

| Primitive | Used? | Where |
|---|---|---|
| **Noul** | **Yes, everywhere Jev is called** | Building the table, the "What was it?" check, the Playground |
| **Choice** | In the Playground's primitives comparison; tried and removed as the final guess | The final guess did worse than the code (see *Why it's split this way*) |
| **Score** | In the Playground's primitives comparison | Not used by the game logic; see ideas below |

**Noul: how it's used.** Each question in the bank becomes a Noul whose instructions point at the
animal in `state`. One `system_one` call answers a batch of them for one animal
(`twenty_q/jev.py`):

```python
from typesafe_sdk import AsyncTypeSafeClient, Noul

def _noul(noun, question):
    return Noul(instructions=f"Answer about the {noun} named in the state: {question.text}")

async def candidate_row(client: AsyncTypeSafeClient, noun, candidate, questions):
    """P(yes) for every question about one candidate."""
    row = {}
    for start in range(0, len(questions), CHUNK_SIZE):           # 30 questions per request
        chunk = questions[start : start + CHUNK_SIZE]
        response = await client.system_one(state=candidate, questions={q.id: _noul(noun, q) for q in chunk})
        row.update({q.id: response.nouls[q.id].noul for q in chunk})  # e.g. {"fly": 0.04, "cold": 0.94}
    return row
```

`build_matrix` runs `candidate_row` for all 62 animals concurrently, using the async client and a
semaphore of 8, and the result is the table the game reads.

**Choice: what was tried.** The final guess once asked Jev to pick from the engine's top 5, reading
the whole Q&A transcript:

```python
response = await client.system_one(
    state={"answers": [{"question": "Can it fly?", "answer": "no"}, ...]},
    questions={"guess": Choice(
        instructions="Which animal best fits all of the yes/no answers in `answers`?",
        criteria={"penguin": None, "seagull": None, "puffin": None, "duck": None, "ostrich": None},
    )},
)
response.choices["guess"].choice, response.choices["guess"].confidence  # e.g. "penguin", 0.78
```

It works, but combining 7–10 answers in one judgment is multi-step reasoning, where code is more
accurate. It was removed in favour of the engine's top candidate.

**Where Choice and Score could fit next:** an open-ended mode where the player types any animal.
Use a **Choice** to place a new animal into a known group (bird / mammal / fish …) and seed its
odds, and a **Score** (obscure → household name) to set how likely people are to pick it in the
first place.

### What a request looks like

One request per animal, batching up to 30 questions (`CHUNK_SIZE`). The animal is the `state`;
each question is a literal yes/no instruction:

```json
{
  "model": "jev-latest",
  "state": "penguin",
  "questions": {
    "fly":  { "type": "noul", "instructions": "Answer about the animal named in the state: Can it fly?" },
    "cold": { "type": "noul", "instructions": "Answer about the animal named in the state: Does it live somewhere cold?" }
  }
}
```

The response carries one probability per question, e.g. `"fly": {"type": "noul", "noul": 0.04}`.
The build script stores these as `matrix[animal][question_id] = p_yes` in
`twenty_q/data/animals.matrix.json`, which ships with the app.

### What happens on each answer

Every time the player answers, the browser sends the **full answer history** to `POST /api/step`.
The server is stateless, so it recomputes from scratch:

1. **Start uniform.** Every animal begins equally likely (1/62).
2. **Score each animal against each answer.** Look up Jev's `p_yes` for that animal and question,
   and convert it to "how well does this animal fit the player's answer?":
   - answer *yes* → score = `p_yes`; answer *no* → score = `1 - p_yes`
   - *sort of* → favours animals Jev was unsure about; *don't know* → no effect
   - scores are clamped to `[0.05, 0.95]`, so one wrong answer lowers the right animal but never
     eliminates it
3. **Multiply and normalise.** Each animal's probability is multiplied by its score, then all are
   rescaled to sum to 1. This is a **Bayes update**: prior × likelihood, normalised. (Done in log
   space for numerical stability: `engine.posterior`.)
4. **Pick the next question.** For every unasked question, compute how much it is expected to
   shrink the uncertainty (expected information gain, in bits), using the same table. Ask the best
   one. A question that splits the remaining odds 50/50 is worth about 1 bit.
5. **Guess or keep going.** Guess when the leader passes 60%, at question 20, or when no question
   is informative any more. A wrong guess excludes that animal and play continues.

Worked example with three animals left, player answers **yes** to "Can it fly?":

| Animal | Before | Jev P(yes) | Score | Before × score | After (normalised) |
|---|---|---|---|---|---|
| Eagle | 1/3 | 0.97 | 0.95 | 0.317 | **48.7%** |
| Owl | 1/3 | 0.95 | 0.95 | 0.317 | **48.7%** |
| Penguin | 1/3 | 0.04 | 0.05 | 0.017 | **2.6%** |

Because the server replays the whole history on each request, the player can change an earlier
answer and the odds simply recompute. The cost is at most 62 animals × 20 answers multiplications.

### Why it's split this way

- **Jev is good at narrow, literal facts.** Against an independent answer key, its table agrees on
  96.8% of clear yes/no facts, and its probabilities are well calibrated (when it says >90%, the
  key says yes 100% of the time). See `scripts/evaluate.py`.
- **Jev is worse at combining many facts.** We tried letting Jev make the final guess with a
  Choice question over the whole Q&A transcript. It was right 52/62 times, against 62/62 for the
  code's Bayes update on the same games. With noisy players it fell further (36/62 vs 53/62).
  This matches TypeSafe's guidance: give the model narrow decisions and keep multi-step reasoning in
  code. See the [Jev 1.13 jaggedness notes](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md).
- **Precomputing keeps the hot path cheap and deterministic.** The questions don't depend on the
  player, so there's no reason to ask them per game. Live calls are reserved for inputs we couldn't
  precompute (an animal not in the list).

### The general pattern

This shape applies well beyond games: **use the model offline to turn unstructured knowledge into
a table of calibrated probabilities, then run ordinary, testable code over that table online.**
The same split works for tagging a product catalogue, triaging tickets against a fixed rubric, or
scoring documents against a checklist. Swap the candidate list and question bank, keep the engine.

### See it while you play

In the game, press **H** (or the "Under the hood" button) to open a drawer from the right. It has two tabs:

1. **This turn:** the exact Noul request behind the current question, Jev's P(yes) for the leading
   animals, and the update table above for your last answer (before, Jev's P(yes), score, after).
2. **Playground**, a sandbox that doesn't touch your game:
   - **Ask Jev one question:** any animal + any question, with the raw request and response.
   - **One decision, three primitives:** one decision ("How big is it?", "What does it eat?", "How dangerous is it?") asked
   as Nouls, a Choice and a Score in one request, side by side with probabilities and confidence.

Every live result has **Show code**: a Python SDK snippet and a `curl` command that reproduce it.

For a guided, hands-on session, work through **[LAB.md](LAB.md)** (about 90 minutes).

## Smallest example

New to the SDK? [`examples/ticket_triage.py`](examples/ticket_triage.py) uses all three primitives
(Choice, Score, Noul) on one support ticket in about 30 lines:

```sh
uv run --env-file .env python examples/ticket_triage.py
```

## Run locally

```sh
# TYPESAFE_API_KEY lives in .env
uv run --env-file .env python -m scripts.build_matrix animals --limit 3   # cheap smoke test
uv run --env-file .env python -m scripts.build_matrix animals             # 62 × 60 = 3,720 nouls, ~2s
uv run --env-file .env uvicorn app:app --reload                           # http://localhost:8000
uv run pytest
```

Gameplay needs no API key once the matrix is built. Live calls (the "what was it?" check for an
animal not in the list, and the drawer's Playground) are capped at
`LIVE_BUDGET_PER_HOUR` Jev requests per client (default 1,000; in memory, per server instance).

## Evaluate

```sh
uv run python -m scripts.evaluate animals
```

Scores Jev's matrix against an independent answer key (`twenty_q/data/animals.truth.json`, written
by Claude, not human-verified), simulates games where the player answers from that key with 0/5/10%
mistakes, and, once people have played, scores their logged games (`logs/games.jsonl`).

## Layout

```
twenty_q/engine.py        pure game logic
twenty_q/jev.py           TypeSafe calls
twenty_q/data/*.json      categories (+ generated *.matrix.json)
scripts/build_matrix.py   precompute a category's matrix
scripts/evaluate.py       score Jev's matrix and simulate games against an answer key
scripts/choice_vs_engine.py  reproduce the "who makes the final guess" experiment (LAB.md part 4)
app.py                    FastAPI API + serves static/index.html
static/index.html         single-page UI
```

The server is stateless: the browser sends the full answer history with each request.

## Deploy

**Modal:** `modal secret create typesafe TYPESAFE_API_KEY=...` then `modal deploy modal_app.py`.

**Cloud Run:** `gcloud run deploy twenty-questions --source . --set-secrets TYPESAFE_API_KEY=typesafe-key:latest`

Build the matrix before deploying; it is packaged into the image.
