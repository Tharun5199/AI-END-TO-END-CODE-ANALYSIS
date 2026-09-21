"""Application configuration, loaded from environment variables / .env.

Everything is resolved relative to the project root (the folder that holds
app.py and .env), so the app, the notebook, the tests and an IDE "Run"
button all find the same .env, write to the same repos/ and vectorstore/
folders, and log to the same logs/ folder -- no matter which directory
they're launched from.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import find_dotenv, load_dotenv

# src/codeanalyzer/config.py -> parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

# Real environment variables (e.g. `docker run -e GROQ_API_KEY=...`, GitHub
# Actions secrets) always win over values in .env (override=False).
if ENV_FILE.exists():
    load_dotenv(ENV_FILE, override=False)
else:
    load_dotenv(find_dotenv(usecwd=True), override=False)

# Quieter, safer third-party defaults. setdefault() never overrides a value
# you've set yourself.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")          # Chroma usage telemetry
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")        # HF tokenizers fork warning
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # noisy on Windows


def _get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except ValueError:
        return default


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    # Tolerate stray whitespace/quotes, a common copy-paste mistake in .env files.
    value = value.strip().strip('"').strip("'").strip()
    return value or default


def _get_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (PROJECT_ROOT / path))


DEFAULT_ALLOWED_EXTENSIONS = [
    ".py", ".js", ".jsx", ".mjs", ".ts", ".tsx", ".java", ".go", ".rs", ".rb",
    ".php", ".c", ".cpp", ".cc", ".h", ".hpp", ".cs", ".kt", ".kts",
    ".swift", ".scala", ".sql", ".sh", ".yaml", ".yml", ".json",
    ".md", ".rst", ".html", ".css", ".toml", ".cfg", ".ini",
    "Dockerfile", "Makefile",
]

DEFAULT_IGNORED_DIRS = [
    ".git", "__pycache__", ".venv", "venv", "env", "node_modules",
    "dist", "build", ".next", ".idea", ".vscode", "target", "vendor",
    ".pytest_cache", ".mypy_cache", ".tox", "coverage", "site-packages",
]

# Placeholder values people commonly leave in .env by accident.
_PLACEHOLDER_MARKERS = ("your", "here", "<", ">", "xxx", "changeme", "change-me", "placeholder")


@dataclass
class Config:
    # LLM (Groq -- free tier, no cost)
    groq_api_key: str = field(default_factory=lambda: _get_str("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: _get_str("GROQ_MODEL", "openai/gpt-oss-120b"))
    llm_temperature: float = field(default_factory=lambda: _get_float("LLM_TEMPERATURE", 0.2))
    llm_timeout_seconds: int = field(default_factory=lambda: _get_int("LLM_TIMEOUT_SECONDS", 60))

    # Embeddings (local sentence-transformers -- free, no API key, runs on CPU)
    embedding_model: str = field(
        default_factory=lambda: _get_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # Storage (relative paths are resolved against the project root)
    chroma_persist_dir: str = field(
        default_factory=lambda: _resolve_path(_get_str("CHROMA_PERSIST_DIR", "vectorstore"))
    )
    repos_dir: str = field(default_factory=lambda: _resolve_path(_get_str("REPOS_DIR", "repos")))

    # Chunking / retrieval
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 1500))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 200))
    top_k: int = field(default_factory=lambda: _get_int("TOP_K", 5))
    max_file_size_kb: int = field(default_factory=lambda: _get_int("MAX_FILE_SIZE_KB", 512))
    max_files: int = field(default_factory=lambda: _get_int("MAX_FILES", 1500))
    max_history_turns: int = field(default_factory=lambda: _get_int("MAX_HISTORY_TURNS", 6))

    # File filtering
    allowed_extensions: List[str] = field(
        default_factory=lambda: _get_list("ALLOWED_EXTENSIONS", DEFAULT_ALLOWED_EXTENSIONS)
    )
    ignored_dirs: List[str] = field(
        default_factory=lambda: _get_list("IGNORED_DIRS", DEFAULT_IGNORED_DIRS)
    )

    # Flask
    flask_secret_key: str = field(default_factory=lambda: _get_str("FLASK_SECRET_KEY", "dev-secret-change-me"))
    port: int = field(default_factory=lambda: _get_int("PORT", 8080))
    debug: bool = field(default_factory=lambda: _get_bool("FLASK_DEBUG", False))

    def __post_init__(self) -> None:
        # Guard against settings that would make the text splitter throw.
        if self.chunk_size < 100:
            self.chunk_size = 100
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = self.chunk_size // 5
        if self.top_k < 1:
            self.top_k = 1

    def groq_key_problem(self) -> Optional[str]:
        """Return a human-readable problem with GROQ_API_KEY, or None if it looks usable."""
        key = self.groq_api_key
        if not key:
            hint = ""
            if not ENV_FILE.exists():
                if (PROJECT_ROOT / ".env.txt").exists():
                    hint = (
                        " Found '.env.txt' instead of '.env' -- Windows Notepad added a .txt "
                        "extension. Rename it to exactly '.env'."
                    )
                else:
                    hint = f" No .env file was found at {ENV_FILE}. Copy .env.example to .env first."
            return (
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
                "and put it in your .env file as GROQ_API_KEY=gsk_..., then restart the app." + hint
            )
        lowered = key.lower()
        # Real Groq keys start with "gsk_"; only flag placeholder text on keys that don't.
        if not lowered.startswith("gsk_") and any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
            return (
                "GROQ_API_KEY still contains the placeholder text from .env.example. Replace it with "
                "your real key from https://console.groq.com/keys (it starts with 'gsk_'), then restart the app."
            )
        return None


config = Config()
