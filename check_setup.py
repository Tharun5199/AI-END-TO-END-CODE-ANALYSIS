"""Diagnose your local setup in one command:

    python check_setup.py            # full check, including a real Groq API call
    python check_setup.py --offline  # skip the network checks

Each line prints [ OK ], [WARN] or [FAIL] with a fix. Exits non-zero if
anything that would stop the app from working is broken.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PRELOAD_EMBEDDINGS", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")  # keep the report readable

failures = 0


def report(status: str, what: str, fix: str = "") -> None:
    global failures
    if status == "FAIL":
        failures += 1
    tag = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[status]
    print(f"{tag} {what}")
    if fix:
        print(f"       -> {fix}")


def check_python() -> None:
    v = sys.version_info
    if v >= (3, 10):
        report("OK", f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        report("FAIL", f"Python {v.major}.{v.minor} is too old", "Install Python 3.10 or newer (3.11/3.12 recommended).")


def check_packages() -> bool:
    print("       Loading AI libraries (ONNX embeddings, no torch). The first run after a reboot can take")
    print("       30-60 seconds on Windows -- it is NOT frozen, please don't press Ctrl+C.", flush=True)
    missing = []
    for module in ("flask", "git", "langchain_core", "langchain_groq", "langchain_chroma",
                   "langchain_text_splitters", "chromadb", "onnxruntime", "tokenizers", "dotenv"):
        print(f"       ... {module}", flush=True)
        try:
            __import__(module)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{module} ({type(exc).__name__}: {str(exc)[:80]})")
    if missing:
        report("FAIL", "Missing/broken packages: " + ", ".join(missing),
               "Activate your virtualenv, then: pip install -r requirements.txt && pip install -e .")
        return False
    report("OK", "All required Python packages import")
    return True


def check_git() -> None:
    git = shutil.which("git")
    if not git:
        report("FAIL", "git is not on PATH", "Install Git from https://git-scm.com and reopen your terminal.")
        return
    out = subprocess.run([git, "--version"], capture_output=True, text=True).stdout.strip()
    report("OK", out)


def check_env():
    from codeanalyzer.config import ENV_FILE, PROJECT_ROOT, Config

    if ENV_FILE.exists():
        report("OK", f".env found at {ENV_FILE}")
    elif (PROJECT_ROOT / ".env.txt").exists():
        report("FAIL", "Found .env.txt instead of .env",
               "Rename it to exactly '.env' (Notepad adds .txt; in Explorer enable 'File name extensions').")
    else:
        report("FAIL", f"No .env file in {PROJECT_ROOT}",
               "Run: copy .env.example .env   then put your Groq key in it.")

    cfg = Config()
    problem = cfg.groq_key_problem()
    if problem:
        report("FAIL", "GROQ_API_KEY problem", problem)
    elif not cfg.groq_api_key.startswith("gsk_"):
        report("WARN", "GROQ_API_KEY doesn't start with 'gsk_' -- Groq keys normally do",
               "Double-check you copied the whole key from https://console.groq.com/keys")
    else:
        report("OK", f"GROQ_API_KEY is set (gsk_...{cfg.groq_api_key[-4:]}), model: {cfg.groq_model}")
    return cfg


def check_groq(cfg) -> None:
    if cfg.groq_key_problem():
        report("WARN", "Skipping Groq API call (fix GROQ_API_KEY first)")
        return
    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=cfg.groq_model, api_key=cfg.groq_api_key, temperature=0,
                       max_tokens=300, timeout=30, max_retries=0)
        reply = llm.invoke("Reply with exactly the word: OK")
        text = (reply.content or "").strip()
        report("OK", f"Groq API works with {cfg.groq_model} (replied: {text[:40]!r})")
    except Exception as exc:  # noqa: BLE001
        from codeanalyzer.exceptions import friendly_message

        report("FAIL", f"Groq API call failed: {type(exc).__name__}", friendly_message(exc))


def check_github() -> None:
    git = shutil.which("git")
    if not git:
        return
    try:
        subprocess.run([git, "ls-remote", "--heads", "https://github.com/kennethreitz/samplemod.git"],
                       check=True, capture_output=True, timeout=30,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"})
        report("OK", "github.com is reachable for cloning")
    except Exception as exc:  # noqa: BLE001
        report("FAIL", f"Can't reach github.com with git ({type(exc).__name__})",
               "Check your internet connection / proxy / firewall.")


def check_embeddings(cfg) -> None:
    try:
        from codeanalyzer.embeddings.embedding_manager import get_embedding_model

        vector = get_embedding_model(cfg.embedding_model).embed_query("def hello(): return 'world'")
        report("OK", f"Embedding model '{cfg.embedding_model}' loads ({len(vector)} dimensions)")
    except Exception as exc:  # noqa: BLE001
        msg = getattr(exc, "user_message", str(exc))
        report("FAIL", "Embedding model failed to load", msg[:300])


def check_chroma() -> None:
    try:
        from langchain_core.documents import Document
        from langchain_core.embeddings import DeterministicFakeEmbedding

        from codeanalyzer.vectorstore.chroma_store import CodeVectorStore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            store = CodeVectorStore(tmp, "repo-selfcheck-00000000", DeterministicFakeEmbedding(size=16))
            store.build_from_documents([Document(page_content="x", metadata={"source": "x.py"})])
            assert store.count() == 1
        report("OK", "Chroma vector store can write and read")
    except Exception as exc:  # noqa: BLE001
        report("FAIL", f"Chroma vector store failed: {exc}", "Try: pip install --force-reinstall chromadb")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="skip Groq, GitHub and model-download checks")
    args = parser.parse_args()

    print(f"AutoCode Analyzer setup check  (project: {ROOT})\n")
    check_python()
    if not check_packages():
        return 1
    check_git()
    cfg = check_env()
    check_chroma()
    if not args.offline:
        check_github()
        check_embeddings(cfg)
        check_groq(cfg)

    print()
    if failures:
        print(f"{failures} problem(s) found -- fix the [FAIL] lines above, then run this again.")
        return 1
    print("All good! Start the app with:  python app.py   then open http://localhost:8080")
    return 0


if __name__ == "__main__":
    sys.exit(main())
