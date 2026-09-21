"""End-to-end orchestration: GitHub URL in, a ready-to-query codebase out."""
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional, Sequence

from codeanalyzer.config import Config
from codeanalyzer.data_ingestion.repo_loader import (
    InvalidRepoURLError,
    clone_repo,
    load_repo_files,
    parse_github_url,
    repo_slug,
)
from codeanalyzer.embeddings.embedding_manager import get_embedding_model
from codeanalyzer.exceptions import CodeAnalyzerError
from codeanalyzer.llm.groq_llm import ensure_groq_key, get_llm
from codeanalyzer.logger import get_logger
from codeanalyzer.qa.qa_chain import CodeQAChain
from codeanalyzer.text_splitter.code_splitter import split_documents
from codeanalyzer.vectorstore.chroma_store import CodeVectorStore

logger = get_logger(__name__)


@dataclass
class IngestResult:
    repo_url: str
    repo_name: str
    collection_name: str
    num_files: int
    num_chunks: int
    truncated: bool
    seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def collection_name_for(github_url: str) -> str:
    # Chroma names must be 3-512 chars of [a-zA-Z0-9._-], starting and ending
    # alphanumeric; "repo-" + slug (which ends in a hex digest) always is.
    return f"repo-{repo_slug(github_url)}"


class CodeAnalyzerPipeline:
    """One instance == one analyzed repository, ready to answer questions.

    Thread-safe: the Flask dev server and gunicorn both serve requests on
    multiple threads, so ingest/ask are serialized with a lock to avoid
    answering from a half-built index.
    """

    def __init__(self, config: Config, llm_base_url: Optional[str] = None, embeddings=None):
        self.config = config
        self._qa_chain: Optional[CodeQAChain] = None
        self.last_result: Optional[IngestResult] = None
        self._lock = threading.RLock()
        # Test hooks: point the real ChatGroq at a mock server / use fake embeddings.
        self._llm_base_url = llm_base_url
        self._embeddings_override = embeddings

    @property
    def ready(self) -> bool:
        return self._qa_chain is not None

    def ingest(self, github_url: str) -> IngestResult:
        """Clone the repo, split it into chunks, and embed it into Chroma."""
        started = time.perf_counter()

        # 1. Cheap checks first, so a bad URL or a missing API key fails in
        #    milliseconds instead of after cloning + embedding the whole repo.
        try:
            repo = parse_github_url(github_url)
        except InvalidRepoURLError as exc:
            raise CodeAnalyzerError(exc, sys, user_message=str(exc)) from exc
        ensure_groq_key(self.config)

        with self._lock:
            try:
                repo_path = clone_repo(repo.web_url, base_path=self.config.repos_dir)
                documents = load_repo_files(
                    repo_path,
                    allowed_extensions=self.config.allowed_extensions,
                    ignored_dirs=self.config.ignored_dirs,
                    max_file_size_kb=self.config.max_file_size_kb,
                    max_files=self.config.max_files,
                )
                if not documents:
                    msg = ("No analyzable source files were found in this repository "
                           "(it may be empty, or only contain file types not listed in ALLOWED_EXTENSIONS).")
                    raise CodeAnalyzerError(ValueError(msg), sys, user_message=msg)

                chunks = split_documents(
                    documents, chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap
                )

                embeddings = self._embeddings_override or get_embedding_model(self.config.embedding_model)
                collection_name = collection_name_for(repo.web_url)
                store = CodeVectorStore(
                    persist_directory=self.config.chroma_persist_dir,
                    collection_name=collection_name,
                    embedding_function=embeddings,
                ).build_from_documents(chunks)

                llm = get_llm(self.config, base_url=self._llm_base_url)
                self._qa_chain = CodeQAChain(
                    retriever=store.as_retriever(k=self.config.top_k),
                    llm=llm,
                    max_history_turns=self.config.max_history_turns,
                )

                self.last_result = IngestResult(
                    repo_url=repo.web_url,
                    repo_name=f"{repo.owner}/{repo.name}",
                    collection_name=collection_name,
                    num_files=len(documents),
                    num_chunks=store.count(),
                    truncated=bool(self.config.max_files) and len(documents) >= self.config.max_files,
                    seconds=round(time.perf_counter() - started, 1),
                )
                logger.info(f"Ingestion complete: {self.last_result}")
                return self.last_result
            except CodeAnalyzerError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise CodeAnalyzerError(exc, sys) from exc

    def ask(self, question: str, chat_history: Optional[Iterable[Sequence[Any]]] = None) -> dict:
        with self._lock:
            if self._qa_chain is None:
                msg = "No repository has been analyzed yet. Paste a GitHub URL and click 'Analyze repo' first."
                raise CodeAnalyzerError(ValueError(msg), sys, user_message=msg)
            return self._qa_chain.ask(question, chat_history)
