import json

import pytest

from twenty_q import engine
from twenty_q.engine import Turn

SPEC = {
    "name": "toy",
    "title": "Toy",
    "prompt": "Think of an animal.",
    "noun": "animal",
    "candidates": ["penguin", "eagle", "dog", "shark"],
    "questions": [
        {"id": "bird", "text": "Is it a bird?"},
        {"id": "fly", "text": "Can it fly?"},
        {"id": "water", "text": "Does it live in water?"},
        {"id": "always_yes", "text": "Is it an animal?"},
    ],
}
MATRIX = {
    "penguin": {"bird": 0.99, "fly": 0.02, "water": 0.8, "always_yes": 0.99},
    "eagle": {"bird": 0.99, "fly": 0.98, "water": 0.05, "always_yes": 0.99},
    "dog": {"bird": 0.01, "fly": 0.01, "water": 0.05, "always_yes": 0.99},
    "shark": {"bird": 0.01, "fly": 0.01, "water": 0.99, "always_yes": 0.99},
}


@pytest.fixture
def toy(tmp_path):
    (tmp_path / "toy.json").write_text(json.dumps(SPEC))
    (tmp_path / "toy.matrix.json").write_text(json.dumps(MATRIX))
    return engine.load_category("toy", tmp_path)


def test_missing_matrix_explains_how_to_build(tmp_path):
    (tmp_path / "toy.json").write_text(json.dumps(SPEC))
    with pytest.raises(FileNotFoundError, match="scripts.build_matrix toy"):
        engine.load_category("toy", tmp_path)


def test_uniform_prior(toy):
    post = engine.posterior(toy, [])
    assert post == pytest.approx({c: 0.25 for c in SPEC["candidates"]})


def test_answers_concentrate_posterior(toy):
    post = engine.posterior(toy, [Turn("bird", "yes"), Turn("fly", "no")])
    assert max(post, key=post.get) == "penguin"
    assert post["penguin"] > 0.85


def test_one_wrong_answer_does_not_eliminate(toy):
    post = engine.posterior(toy, [Turn("bird", "no")])
    assert post["penguin"] > 0


def test_unknown_answer_changes_nothing(toy):
    assert engine.posterior(toy, [Turn("fly", "unknown")]) == pytest.approx(engine.posterior(toy, []))


def test_uninformative_question_has_no_gain(toy):
    post = engine.posterior(toy, [])
    assert engine.information_gain(toy, post, "always_yes") == pytest.approx(0, abs=1e-6)
    assert engine.information_gain(toy, post, "bird") > 0.9  # splits 4 candidates in half: ~1 bit


def test_next_step_skips_asked_and_guesses_when_sure(toy):
    first = engine.next_step(toy, [])
    assert first.question.id != "always_yes"
    assert not first.should_guess

    done = engine.next_step(toy, [Turn("bird", "yes"), Turn("fly", "no"), Turn("water", "yes")])
    assert done.ranking[0].candidate == "penguin"
    assert done.should_guess
    assert done.question.id == "always_yes"  # only one left


def test_excluded_candidates_are_dropped(toy):
    post = engine.posterior(toy, [Turn("bird", "yes")], excluded=frozenset({"penguin"}))
    assert "penguin" not in post
    assert max(post, key=post.get) == "eagle"

