"""Local, free embedding model (no API key, no per-call cost).

Uses a sentence-transformers model running on CPU via langchain-huggingface.
The model weights (~90MB for the default all-MiniLM-L6-v2) are downloaded
once from the Hugging Face Hub on first use and cached locally afterwards
(in ~/.cache/huggingface).
"""
import sys
import threading
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from codeanalyzer.exceptions import CodeAnalyzerError
from codeanalyzer.logger import get_logger

logger = get_logger(__name__)
_load_lock = threading.Lock()


@lru_cache(maxsize=4)
def _load(model_name: str) -> HuggingFaceEmbeddings:
    logger.info(f"Loading local embedding model: {model_name} (first run downloads it once)")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance for `model_name`."""
    try:
        with _load_lock:  # two concurrent first requests must not load the model twice
            return _load(model_name)
    except Exception as exc:  # noqa: BLE001
        raise CodeAnalyzerError(
            exc,
            sys,
            user_message=(
                f"Could not load the embedding model '{model_name}'. The first run needs internet "
                "access to download it from huggingface.co (about 90MB); after that it's cached. "
                f"Details: {str(exc)[:200]}"
            ),
        ) from exc
