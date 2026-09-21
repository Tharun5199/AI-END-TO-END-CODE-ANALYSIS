"""Groq-hosted LLM (free tier, extremely fast inference)."""
import sys
from typing import Optional

from langchain_groq import ChatGroq

from codeanalyzer.config import Config
from codeanalyzer.exceptions import CodeAnalyzerError
from codeanalyzer.logger import get_logger

logger = get_logger(__name__)


class MissingAPIKeyError(ValueError):
    """GROQ_API_KEY is missing or still a placeholder."""


def ensure_groq_key(config: Config) -> None:
    """Fail fast -- before any cloning/embedding work -- if the key is unusable."""
    problem = config.groq_key_problem()
    if problem:
        raise CodeAnalyzerError(MissingAPIKeyError(problem), sys, user_message=problem)


def get_llm(config: Config, base_url: Optional[str] = None) -> ChatGroq:
    """Build a ChatGroq client from the app config.

    `base_url` is only used by the test-suite to point the real ChatGroq
    class at a local mock server.
    """
    ensure_groq_key(config)
    try:
        logger.info(f"Initializing Groq LLM: {config.groq_model}")
        kwargs = dict(
            model=config.groq_model,
            api_key=config.groq_api_key,
            temperature=config.llm_temperature,
            timeout=config.llm_timeout_seconds,
            max_retries=2,
        )
        if base_url:
            kwargs["base_url"] = base_url
        return ChatGroq(**kwargs)
    except Exception as exc:  # noqa: BLE001
        raise CodeAnalyzerError(exc, sys) from exc
