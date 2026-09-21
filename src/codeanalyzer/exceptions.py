"""Project-wide custom exception with rich traceback context.

Every error carries two messages:
  * `error_message` -- detailed (file + line + original error) for the log file.
  * `user_message`  -- short and actionable, safe to show in the browser
                       (no internal file paths or stack details).
"""
import sys
from typing import Optional


def _error_message_detail(error: BaseException, error_detail) -> str:
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is None:
        return f"Error: {error}"
    # Walk to the innermost frame -- that's where the error actually happened.
    while exc_tb.tb_next is not None:
        exc_tb = exc_tb.tb_next
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error occurred in script [{file_name}] at line [{line_number}]: {error}"


def friendly_message(error: BaseException) -> str:
    """Translate common low-level failures into something a user can act on."""
    if isinstance(error, CodeAnalyzerError):
        return error.user_message

    name = type(error).__name__
    text = str(error)
    lowered = text.lower()

    # Groq SDK errors (matched by class name so this module doesn't import groq)
    if name == "AuthenticationError" or "invalid api key" in lowered or "error code: 401" in lowered:
        return ("Groq rejected your API key (401). Check GROQ_API_KEY in your .env -- copy a fresh key "
                "from https://console.groq.com/keys -- then restart the app.")
    if name == "RateLimitError" or "rate limit" in lowered or "error code: 429" in lowered:
        return ("Groq's free-tier rate limit was hit. Wait a minute and try again "
                "(or switch GROQ_MODEL in .env to another free model).")
    if name == "PermissionDeniedError" or (name == "NotFoundError" and "model" in lowered):
        return (f"Groq can't use the model configured in GROQ_MODEL ({text[:200]}). "
                "Pick a model your account can access from https://console.groq.com/docs/models.")
    if name in ("APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout"):
        return "Could not reach the Groq API. Check your internet connection and try again."

    return text[:500] if text else name


class CodeAnalyzerError(Exception):
    """Raised for any expected failure inside the codeanalyzer package."""

    def __init__(self, error_message: BaseException, error_detail=sys, user_message: Optional[str] = None):
        super().__init__(str(error_message))
        self.original = error_message
        self.error_message = _error_message_detail(error_message, error_detail)
        self.user_message = user_message or friendly_message(error_message)

    def __str__(self) -> str:
        return self.error_message
