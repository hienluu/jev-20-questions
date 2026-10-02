import json

import pytest
from fastapi.testclient import TestClient

import app as app_module
from tests.test_engine import toy  # noqa: F401  (fixture)


@pytest.fixture
def client(toy, monkeypatch):  # noqa: F811
    monkeypatch.setattr(app_module, "categories", lambda: {"toy": toy})
    monkeypatch.setattr(app_module, "_client", None)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    return TestClient(app_module.app)


def test_index_and_categories(client):
    assert "20 Questions" in client.get("/").text
    assert client.get("/api/categories").json() == [{"name": "toy", "title": "Toy", "prompt": "Think of an animal.", "size": 4}]


def test_step_flow(client):
    first = client.post("/api/step", json={"category": "toy"}).json()
    assert first["asked"] == 0 and first["question"] and not first["should_guess"]

    done = client.post(
        "/api/step",
        json={
            "category": "toy",
            "history": [
                {"question_id": "bird", "answer": "yes"},
                {"question_id": "fly", "answer": "no"},
                {"question_id": "water", "answer": "yes"},
            ],
        },
    ).json()
    assert done["ranking"][0]["candidate"] == "penguin"
    assert done["should_guess"]


def test_excluded_and_validation(client):
    body = {"category": "toy", "history": [{"question_id": "bird", "answer": "yes"}], "excluded": ["penguin"]}
    assert client.post("/api/step", json=body).json()["ranking"][0]["candidate"] == "eagle"
    assert client.post("/api/step", json={"category": "nope"}).status_code == 404
    assert client.post("/api/step", json={"category": "toy", "history": [{"question_id": "x", "answer": "yes"}]}).status_code == 422


def test_guess_is_engine_top_candidate(client):
    body = {"category": "toy", "history": [{"question_id": "bird", "answer": "yes"}, {"question_id": "fly", "answer": "no"}]}
    guess = client.post("/api/guess", json=body).json()
    assert guess["candidate"] == "penguin" and guess["probability"] > 0.85


def test_reveal_known_candidate_uses_matrix(client):
    # "eagle" is in the matrix, so no Jev call is needed even without a key.
    body = {"category": "toy", "history": [{"question_id": "fly", "answer": "no"}], "actual": "Eagle"}
    r = client.post("/api/reveal", json=body).json()
    assert r["known"] and r["contradictions"][0]["question"] == "Can it fly?"


def test_reveal_unknown_candidate_needs_jev(client):
    body = {"category": "toy", "history": [{"question_id": "fly", "answer": "no"}], "actual": "platypus"}
    assert client.post("/api/reveal", json=body).status_code == 503


def test_result_is_logged(client, tmp_path, monkeypatch):
    log = tmp_path / "games.jsonl"
    monkeypatch.setattr(app_module, "GAMES_LOG", log)
    body = {"category": "toy", "history": [{"question_id": "bird", "answer": "yes"}], "actual": " Penguin ", "won": True}
    assert client.post("/api/result", json=body).json() == {"ok": True}
    record = json.loads(log.read_text())
    assert record["actual"] == "penguin" and record["won"] and record["history"][0]["question_id"] == "bird"


def test_step_explains_where_jev_numbers_come_from(client):
    hood = client.post("/api/step", json={"category": "toy"}).json()["under_the_hood"]
    assert hood["last"] is None
    assert hood["next"]["request"]["questions"][next(iter(hood["next"]["request"]["questions"]))]["type"] == "noul"
    assert {n["candidate"] for n in hood["next"]["nouls"]} == {"penguin", "eagle", "dog", "shark"}

    body = {"category": "toy", "history": [{"question_id": "bird", "answer": "yes"}]}
    last = client.post("/api/step", json=body).json()["under_the_hood"]["last"]
    penguin = next(r for r in last["rows"] if r["candidate"] == "penguin")
    assert last["answer"] == "yes" and penguin["before"] == pytest.approx(0.25)
    assert penguin["likelihood"] == pytest.approx(0.95) and penguin["after"] > penguin["before"]


def test_live_noul_needs_jev(client):
    body = {"category": "toy", "candidate": "moose", "question": "Does it have antlers?"}
    assert client.post("/api/jev/noul", json=body).status_code == 503


def test_live_budget_caps_billed_calls(client, monkeypatch):
    monkeypatch.setattr(app_module, "LIVE_BUDGET_PER_HOUR", 2)
    monkeypatch.setattr(app_module, "_spent", app_module.defaultdict(app_module.deque))
    body = {"category": "toy", "candidate": "moose", "question": "Does it have antlers?"}
    for _ in range(2):
        assert client.post("/api/jev/noul", json=body).status_code == 503  # charged, then no API key
    r = client.post("/api/jev/noul", json=body)
    assert r.status_code == 429 and "budget" in r.json()["detail"]


def test_compare_lists_demos_and_needs_jev(client):
    assert set(client.get("/api/jev/demos").json()) == {"size", "diet", "danger"}
    assert client.post("/api/jev/compare", json={"category": "toy", "candidate": "penguin"}).status_code == 503


def test_config_exposes_analytics_code_only_when_valid(client, monkeypatch):
    assert client.get("/api/config").json()["goatcounter"] is None
    monkeypatch.setattr(app_module, "GOATCOUNTER_CODE", "jev-20q")
    assert client.get("/api/config").json() == {"goatcounter": "jev-20q", "goatcounter_allow_local": False}
    monkeypatch.setattr(app_module, "GOATCOUNTER_ALLOW_LOCAL", True)
    assert client.get("/api/config").json()["goatcounter_allow_local"] is True
    monkeypatch.setattr(app_module, "GOATCOUNTER_CODE", "bad code/../x")
    assert client.get("/api/config").json()["goatcounter"] is None
