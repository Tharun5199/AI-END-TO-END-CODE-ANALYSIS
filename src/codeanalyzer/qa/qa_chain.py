"""Retrieval-augmented Q&A over a codebase, built with LangChain Expression
Language (LCEL): retriever -> prompt -> LLM -> parser, with a small
history-condensing step so follow-up questions ("what does it return?")
resolve against the prior turn.
"""
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import VectorStoreRetriever

from codeanalyzer.exceptions import CodeAnalyzerError
from codeanalyzer.logger import get_logger

logger = get_logger(__name__)

_CONDENSE_SYSTEM_PROMPT = (
    "Given a chat history and a follow-up question, rewrite the follow-up "
    "question as a standalone question that can be understood without the "
    "chat history. Do not answer the question, only rewrite it. If it is "
    "already standalone, return it unchanged. Reply with the question only."
)

_ANSWER_SYSTEM_PROMPT = (
    "You are AutoCode Analyzer, an assistant that explains a codebase to "
    "developers. Answer the question using ONLY the code excerpts given in "
    "the context below. Reference specific file paths and function/class "
    "names when relevant, and use Markdown (with fenced code blocks) for "
    "code. If the answer is not contained in the context, say you don't have "
    "enough information from the indexed code rather than guessing.\n\n"
    "Context:\n{context}"
)

_ROLE_MAP = {"human": "human", "user": "human", "ai": "ai", "assistant": "ai"}
_MAX_HISTORY_MESSAGE_CHARS = 2000


def _format_docs(docs: List[Document]) -> str:
    if not docs:
        return "(no matching code found)"
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        idx, total = doc.metadata.get("chunk_index"), doc.metadata.get("total_chunks")
        where = f" (part {idx + 1} of {total})" if isinstance(idx, int) and total else ""
        parts.append(f"--- {source}{where} ---\n{doc.page_content}")
    return "\n\n".join(parts)


def sanitize_history(
    history: Optional[Iterable[Sequence[Any]]], max_turns: int = 6
) -> List[Tuple[str, str]]:
    """Keep only well-formed (role, text) pairs with known roles, most recent last.

    Unknown roles (which would make LangChain raise) are dropped, very long
    messages are truncated, and only the last `max_turns` exchanges are kept
    so a long conversation can't blow past the model's context window.
    """
    cleaned: List[Tuple[str, str]] = []
    for item in history or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        role, text = item
        role = _ROLE_MAP.get(str(role).strip().lower())
        if role is None or not isinstance(text, str) or not text.strip():
            continue
        cleaned.append((role, text.strip()[:_MAX_HISTORY_MESSAGE_CHARS]))
    return cleaned[-2 * max_turns:] if max_turns > 0 else []


class CodeQAChain:
    """Ties a retriever and an LLM together into an ask() method."""

    def __init__(self, retriever: VectorStoreRetriever, llm: BaseChatModel, max_history_turns: int = 6):
        self.retriever = retriever
        self.llm = llm
        self.max_history_turns = max_history_turns

        condense_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _CONDENSE_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "Follow-up question: {question}"),
            ]
        )
        self._condense_chain = condense_prompt | llm | StrOutputParser()

        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _ANSWER_SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )
        self._answer_chain = answer_prompt | llm | StrOutputParser()

    def ask(self, question: str, chat_history: Optional[Iterable[Sequence[Any]]] = None) -> Dict[str, Any]:
        """Answer `question`, optionally aware of prior (role, text) turns.

        Returns {"answer": str, "sources": list[str], "standalone_question": str}.
        """
        question = (question or "").strip()
        if not question:
            raise CodeAnalyzerError(ValueError("Question is empty."), sys, user_message="Please type a question.")

        try:
            history = sanitize_history(chat_history, self.max_history_turns)

            standalone_question = question
            if history:
                rewritten = self._condense_chain.invoke({"question": question, "chat_history": history})
                rewritten = (rewritten or "").strip().strip('"').strip()
                if rewritten:
                    standalone_question = rewritten

            docs = self.retriever.invoke(standalone_question)
            answer = self._answer_chain.invoke(
                {"question": standalone_question, "context": _format_docs(docs)}
            ).strip()
            if not answer:
                answer = "The model returned an empty answer. Please try rephrasing your question."

            # Preserve retrieval order (most relevant first) while de-duplicating.
            sources = list(dict.fromkeys(doc.metadata.get("source", "unknown") for doc in docs))

            return {
                "answer": answer,
                "sources": sources,
                "standalone_question": standalone_question,
            }
        except CodeAnalyzerError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise CodeAnalyzerError(exc, sys) from exc
