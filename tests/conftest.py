"""Shared pytest fixtures.

Nothing here needs a Groq API key or the Hugging Face model download:
  * `FakeEmbeddings` / `FakeListChatModel` stand in where only the wiring matters.
  * `mock_groq` runs a tiny local HTTP server that speaks Groq's (OpenAI-compatible)
    chat-completions API, so the REAL `ChatGroq` class is exercised end to end.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PRELOAD_EMBEDDINGS", "false")  # don't download models while importing app.py

from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402

SAMPLE_REPO = "https://github.com/kennethreitz/samplemod"


def github_reachable() -> bool:
    import subprocess

    try:
        subprocess.run(
            ["git", "ls-remote", "--heads", SAMPLE_REPO + ".git"],
            check=True, capture_output=True, timeout=20,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        return True
    except Exception:  # noqa: BLE001
        return False


needs_github = pytest.mark.skipif(not github_reachable(), reason="github.com not reachable")


@pytest.fixture
def fake_embeddings():
    return DeterministicFakeEmbedding(size=64)


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """A Config whose storage lives in a throwaway temp folder."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_for_mock_server_only")
    from codeanalyzer.config import Config

    cfg = Config()
    cfg.repos_dir = str(tmp_path / "repos")
    cfg.chroma_persist_dir = str(tmp_path / "vectorstore")
    return cfg


class _MockGroq:
    def __init__(self):
        self.requests = []
        self.status = 200
        self.answers = []

    def next_answer(self) -> str:
        return self.answers.pop(0) if self.answers else "Mock answer: `helpers.get_answer()` returns True."


@pytest.fixture
def mock_groq():
    """Local stand-in for https://api.groq.com (OpenAI-compatible schema)."""
    state = _MockGroq()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep test output clean
            pass

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            state.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
            if state.status != 200:
                payload = {"error": {"message": "Invalid API Key", "type": "invalid_request_error",
                                     "code": "invalid_api_key"}}
            else:
                payload = {
                    "id": "chatcmpl-mock", "object": "chat.completion", "created": 0,
                    "model": body.get("model"),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": state.next_answer()}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            raw = json.dumps(payload).encode()
            self.send_response(state.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield state
    finally:
        server.shutdown()
