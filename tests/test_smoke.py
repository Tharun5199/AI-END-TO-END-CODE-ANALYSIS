"""End-to-end smoke tests through the Flask app.

The real ChatGroq client talks to a local mock of the Groq API (see
conftest.py), and embeddings are faked, so this needs no API key and no
model download -- but everything else (Flask routes, GitHub clone, chunking,
Chroma, the LCEL chain, ChatGroq's request/response handling) is real.

Run with:  python -m pytest -q          (or:  python tests/test_smoke.py)
"""
import time

import pytest

from conftest import SAMPLE_REPO, needs_github

import app as app_module
from codeanalyzer.config import Config
from codeanalyzer.pipeline import CodeAnalyzerPipeline


@pytest.fixture
def client():
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


def _use(monkeypatch, cfg, pipeline=None):
    monkeypatch.setattr(app_module, "config", cfg)
    monkeypatch.setattr(app_module, "pipeline", pipeline or CodeAnalyzerPipeline(cfg))


def test_basic_routes(client, tmp_config, monkeypatch):
    _use(monkeypatch, tmp_config)
    assert client.get("/").status_code == 200
    assert client.get("/health").get_json() == {"status": "ok"}
    status = client.get("/status").get_json()
    assert status["llm_configured"] is True and status["repo"] is None


def test_missing_key_fails_fast_before_cloning(client, tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    cfg = Config()
    cfg.repos_dir = str(tmp_path / "repos")
    cfg.chroma_persist_dir = str(tmp_path / "vs")
    _use(monkeypatch, cfg)

    assert client.get("/status").get_json()["llm_configured"] is False

    started = time.perf_counter()
    res = client.post("/ingest", json={"github_url": SAMPLE_REPO})
    assert res.status_code == 400
    assert "GROQ_API_KEY is not set" in res.get_json()["error"]
    assert "Error occurred in script" not in res.get_json()["error"]  # no internal paths leak to the browser
    assert time.perf_counter() - started < 2
    assert not (tmp_path / "repos").exists()  # nothing was cloned


def test_input_validation(client, tmp_config, monkeypatch):
    _use(monkeypatch, tmp_config)
    assert client.post("/ingest", json={}).status_code == 400
    bad = client.post("/ingest", json={"github_url": "https://gitlab.com/a/b"})
    assert bad.status_code == 400 and "GitHub repository link" in bad.get_json()["error"]
    assert client.post("/chat", json={}).status_code == 400
    not_ready = client.post("/chat", json={"question": "hi"})
    assert not_ready.status_code == 400 and "Analyze repo" in not_ready.get_json()["error"]


@needs_github
def test_full_flow_with_real_chatgroq_client(client, tmp_config, mock_groq, fake_embeddings, monkeypatch):
    pipeline = CodeAnalyzerPipeline(tmp_config, llm_base_url=mock_groq.base_url, embeddings=fake_embeddings)
    _use(monkeypatch, tmp_config, pipeline)

    # Ingest a real repo (URL pasted as a "tree" link on purpose).
    res = client.post("/ingest", json={"github_url": SAMPLE_REPO + "/tree/master/sample"})
    assert res.status_code == 200, res.get_json()
    info = res.get_json()
    assert info["repo_name"] == "kennethreitz/samplemod" and info["num_files"] > 0 and info["num_chunks"] > 0

    # Single-turn question: one request to Groq, carrying retrieved code as context.
    mock_groq.answers = ["`get_answer()` in `sample/helpers.py` returns **True**."]
    res = client.post("/chat", json={"question": "What does get_answer return?", "history": []})
    assert res.status_code == 200, res.get_json()
    body = res.get_json()
    assert "returns **True**" in body["answer"] and body["sources"]
    request = mock_groq.requests[-1]
    assert request["path"].endswith("/openai/v1/chat/completions")
    assert request["body"]["model"] == tmp_config.groq_model
    assert request["headers"]["Authorization"] == f"Bearer {tmp_config.groq_api_key}"
    system_prompt = request["body"]["messages"][0]["content"]
    assert any(src in system_prompt for src in body["sources"])  # retrieved files reached the LLM

    # Follow-up with history: condense request + answer request.
    before = len(mock_groq.requests)
    mock_groq.answers = ["Where is get_answer defined in samplemod?", "In sample/helpers.py."]
    res = client.post("/chat", json={
        "question": "where is it defined?",
        "history": [["human", "What does get_answer return?"], ["ai", body["answer"]]],
    })
    assert res.status_code == 200
    assert res.get_json()["standalone_question"] == "Where is get_answer defined in samplemod?"
    assert len(mock_groq.requests) - before == 2

    # Status now reports the loaded repo; re-ingesting doesn't duplicate chunks.
    assert client.get("/status").get_json()["repo"]["repo_name"] == "kennethreitz/samplemod"
    again = client.post("/ingest", json={"github_url": SAMPLE_REPO}).get_json()
    assert again["num_chunks"] == info["num_chunks"]


@needs_github
def test_invalid_groq_key_gives_actionable_error(client, tmp_config, mock_groq, fake_embeddings, monkeypatch):
    pipeline = CodeAnalyzerPipeline(tmp_config, llm_base_url=mock_groq.base_url, embeddings=fake_embeddings)
    _use(monkeypatch, tmp_config, pipeline)
    assert client.post("/ingest", json={"github_url": SAMPLE_REPO}).status_code == 200

    mock_groq.status = 401
    res = client.post("/chat", json={"question": "What does this repo do?"})
    assert res.status_code == 500
    assert "rejected your API key" in res.get_json()["error"]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    tests_dir = str(Path(__file__).resolve().parent)
    sys.exit(pytest.main(["-q", "-W", "ignore::pytest.PytestAssertRewriteWarning", tests_dir]))
