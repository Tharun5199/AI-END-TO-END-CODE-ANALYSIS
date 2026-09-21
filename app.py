"""Flask front-end for the AutoCode Analyzer.

Endpoints:
  GET  /         chat UI
  GET  /health   liveness probe (Docker HEALTHCHECK / load balancer)
  GET  /status   is the LLM configured? which repo is currently loaded?
  POST /ingest   { "github_url": "..." }                           -> clone + index a repo
  POST /chat     { "question": "...", "history": [[role, text]] }  -> answer + source files

This demo keeps a single pipeline instance in memory (one analyzed repo at a
time). That's why the Docker image runs gunicorn with ONE worker process and
several threads -- separate worker processes would each have their own
in-memory pipeline, so a question could land on a worker that never saw the
repo. See the README's "Known limitations" section for how to extend this.
"""
import os
import sys
import threading
from pathlib import Path

if __name__ == "__main__":
    print("Starting AutoCode Analyzer -- loading AI libraries. The first start after a reboot can take\n"
          "1-2 minutes on Windows; it is NOT frozen. Wait for 'Running on http://127.0.0.1:8080'.", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from flask import Flask, jsonify, render_template, request  # noqa: E402
from git import GitCommandError  # noqa: E402

from codeanalyzer.config import config  # noqa: E402
from codeanalyzer.exceptions import CodeAnalyzerError, friendly_message  # noqa: E402
from codeanalyzer.logger import get_logger  # noqa: E402
from codeanalyzer.pipeline import CodeAnalyzerPipeline  # noqa: E402

logger = get_logger(__name__)

app = Flask(__name__)
app.secret_key = config.flask_secret_key
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1MB is plenty for a question + history

pipeline = CodeAnalyzerPipeline(config)

_startup_problem = config.groq_key_problem()
if _startup_problem:
    logger.warning(f"LLM not configured: {_startup_problem}")


def _warm_up_embeddings() -> None:
    """Load the embedding model in the background so the first 'Analyze' click is fast."""
    try:
        from codeanalyzer.embeddings.embedding_manager import get_embedding_model

        get_embedding_model(config.embedding_model)
        logger.info("Embedding model loaded and ready")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Embedding model warm-up failed (will retry on first ingest): {exc}")


if os.getenv("PRELOAD_EMBEDDINGS", "true").lower() in ("1", "true", "yes"):
    threading.Thread(target=_warm_up_embeddings, name="embeddings-warmup", daemon=True).start()


def _error_response(exc: Exception):
    """Log full detail server-side; send only a clean, actionable message to the browser."""
    logger.error(str(exc))
    if isinstance(exc, CodeAnalyzerError):
        original = exc.original
        status = 400 if isinstance(original, (ValueError, GitCommandError)) else 500
        return jsonify(error=exc.user_message), status
    return jsonify(error=friendly_message(exc)), 500


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(status="ok"), 200


@app.get("/status")
def status():
    problem = config.groq_key_problem()
    return jsonify(
        llm_configured=problem is None,
        config_problem=problem,
        model=config.groq_model,
        embedding_model=config.embedding_model,
        repo=pipeline.last_result.to_dict() if pipeline.ready and pipeline.last_result else None,
    )


@app.post("/ingest")
def ingest():
    payload = request.get_json(silent=True) or {}
    github_url = str(payload.get("github_url") or "").strip()
    if not github_url:
        return jsonify(error="Please paste a GitHub repository URL."), 400

    try:
        result = pipeline.ingest(github_url)
        return jsonify(result.to_dict())
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question") or "").strip()
    history = payload.get("history") or []
    if not question:
        return jsonify(error="Please type a question."), 400
    if not isinstance(history, list):
        history = []

    try:
        result = pipeline.ask(question, chat_history=history)
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


if __name__ == "__main__":
    logger.info(f"Open http://localhost:{config.port} in your browser")
    # threaded=True is the default; use_reloader=False keeps the in-memory index
    # from being wiped by a surprise restart when debug mode is on.
    app.run(host="0.0.0.0", port=config.port, debug=config.debug, use_reloader=False)
