# Jev lab: learn TypeSafe's System One model by playing and building

About 90 minutes, in five parts. You'll use the 20 Questions game to see where Jev fits in, write your
own Jev questions, compare the three primitives, test a design decision, and build a new category.

Read **"How Jev is used"** in the [README](README.md) first (5 minutes). The short version: Jev
answers narrow, typed questions with calibrated probabilities, and ordinary code does the reasoning.

## 0 · Setup (5 min)

```sh
uv sync
echo "TYPESAFE_API_KEY=..." > .env         # never commit this; .env is gitignored
uv run --env-file .env uvicorn app:app --reload --port 8001
```

Open http://localhost:8001 and pick **Animals**.

> **Cost.** Gameplay makes no API calls. Things marked **(live)** below do, and are billed. The server
> caps live calls at `LIVE_BUDGET_PER_HOUR` Jev requests per client (default 1,000).

## 1 · Watch Jev in the game (15 min)

Play a game with **Under the hood** open (press **H**). Stay on the **This turn** tab.

- Card 1 shows the exact request that produced the numbers behind the current question. Find the
  `state` and the `instructions`. What does Jev see, and what doesn't it see?
- Card 2 shows the update after your last answer. Answer **Sort of** once and **No idea** once.
  What happens to the *× match* column in each case, and why?
- Give one deliberately wrong answer. Does the right animal recover? Which number in `engine.py`
  makes that possible? (Hint: `NOISE_FLOOR`.)

**Takeaway:** the model's output is a number your code consumes. Everything you see move is plain code.

## 2 · Question design in the Playground (15 min, live)

Open the **Playground** tab. **Ask Jev one question** sends one Noul (any animal, any question) and
shows the raw request and response. Use it to see how Jev reads your wording:

1. Ask something vague about a borderline animal: *"Is it big?"* for **wolf**, **penguin** and
   **kangaroo**. Then make it literal: *"Is it bigger than an adult human?"* Which phrasing gives
   probabilities nearer 0 or 1, and which hovers near 0.5 (Jev unsure)?
2. Ask a negation: *"Is it unable to fly?"* against *"Can it fly?"* for **penguin**, **bat** and
   **chicken**. Do the two add up to about 1? See
   [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md) on negation.
3. Try an animal that isn't in the game (*moose*, *axolotl*). Jev doesn't need a list: the
   candidate is just the `state`.
4. Open **Show code** on a result and run the Python snippet from your terminal
   (`uv run --env-file .env python snippet.py`).

**Takeaway:** Jev answers the words you wrote, literally. Good questions are specific, single-fact,
and phrased positively.

## 3 · Noul vs Choice vs Score (15 min, live)

Still in the **Playground**, use **One decision, three primitives**. Each run asks one decision three
ways in a **single request**.

1. *How big is it?* for **penguin**, then **lion**, then **octopus**. Where do the three primitives
   agree? Where is confidence low, and is that reasonable?
2. *What does it eat?* for **platypus**, **bear** and **whale**. Which primitive handles "it's a mix"
   best? Notice how a Score lands *between* levels.
3. Look at the raw response. Confidence exists only on Choice and Score. It's computed from how
   peaked the probabilities are (see [Confidence](https://docs.typesafe.ai/confidence.md)).
   Sketch the code you'd write to act at high confidence, ask a human at medium, and refuse at low.

**Takeaway:** Noul gives independent facts to combine yourself; Choice picks one option with a
confidence; Score places things on an ordered scale. Pick the one whose output your code needs.

## 4 · Who should make the final guess? (20 min, live)

The game's final guess comes from code (the Bayes update), not from Jev. Reproduce why:

```sh
uv run --env-file .env python -m scripts.choice_vs_engine animals
uv run --env-file .env python -m scripts.choice_vs_engine animals --noise 0.1
```

Each run plays one simulated game per animal, then asks Jev to pick from the engine's top 5 by
reading the whole Q&A transcript (62 Choice requests, about 2 s). Expect roughly engine 62/62 against
Jev in the low 50s.

Now try to close the gap by editing `scripts/choice_vs_engine.py`:

- Rewrite `CHOICE_INSTRUCTIONS`.
- Change `build_state()`: drop "sort of" answers, or include the engine's odds in the state.
- Try `--shortlist 2` or `--shortlist 10`.
- Add descriptions to the Choice criteria instead of `None`.

Look at the misses and their confidence. Would a confidence threshold ("only trust Jev above 0.7")
help?

**Takeaway:** combining many facts in one judgment is multi-step reasoning, and code does it more
reliably. Use Jev for the narrow decisions, and code for the combination.

## 5 · Build your own category (30 min, live)

1. Create `twenty_q/data/foods.json`, copying the shape of `animals.json`: `name`, `title`, `prompt`,
   `noun` ("food"), 20–30 `candidates`, and 20–30 `questions` with short `id`s.
2. Build the table, small first:
   ```sh
   uv run --env-file .env python -m scripts.build_matrix foods --limit 3
   uv run --env-file .env python -m scripts.build_matrix foods
   ```
   Open `twenty_q/data/foods.matrix.json` and spot-check a few numbers.
3. **Restart the server** (the category list is loaded once), then play Foods. When the game guesses
   wrong, use the Playground to try questions that would have told the leaders apart, add the best
   ones to `foods.json`, and rebuild.
4. Optional: write `foods.truth.json` (see `animals.truth.json` for the format) and run
   `uv run python -m scripts.evaluate foods` to measure Jev's agreement and the simulated win rate.

**Takeaway:** the pattern generalises. Model offline → table of probabilities → ordinary code online.
Swap the candidates and questions and the engine still works.

## Stretch · Use a Score as the starting odds

Every animal starts equally likely, but people think of *dog* far more often than *platypus*. Add a
Score question per candidate, such as *"How likely is someone to pick this animal in a guessing
game?"* with levels from "almost nobody" to "very often". Store it next to the matrix, and use it as
the prior in `engine.posterior`, where `log_weights` currently starts at `0.0` for everyone. Then rerun
`scripts.evaluate` to see whether average questions per game drop.

## Where to look in the code

| File | What to read |
|---|---|
| `twenty_q/jev.py` | Every Jev call: `_noul`, `candidate_row`, `ask_noul`, `compare_primitives` |
| `twenty_q/engine.py` | `likelihood`, `posterior` (Bayes), `information_gain`, `next_step` |
| `app.py` | API endpoints; `_under_the_hood` builds the drawer's data |
| `scripts/build_matrix.py` | The offline step |
| `scripts/evaluate.py`, `scripts/choice_vs_engine.py` | How we measured all of this |
