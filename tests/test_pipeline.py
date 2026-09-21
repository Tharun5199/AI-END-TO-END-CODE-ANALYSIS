"""Regression tests for every bug fixed in the pipeline.

Run with:  python -m pytest -q
"""
import errno
import os
import stat
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from conftest import SAMPLE_REPO, needs_github

import codeanalyzer.data_ingestion.repo_loader as rl
from codeanalyzer.config import Config
from codeanalyzer.exceptions import CodeAnalyzerError, friendly_message
from codeanalyzer.qa.qa_chain import CodeQAChain, sanitize_history
from codeanalyzer.text_splitter.code_splitter import split_documents
from codeanalyzer.vectorstore.chroma_store import CodeVectorStore


# --------------------------------------------------------------------------
# Config / API key
# --------------------------------------------------------------------------
@pytest.mark.parametrize("key,ok", [
    ("", False),
    ("your_groq_api_key_here", False),
    ("<paste-key>", False),
    ("gsk_abcDEF123yourhere456", True),   # real-looking key that happens to contain "your"/"here"
    ("gsk_" + "a" * 52, True),
])
def test_groq_key_problem_detection(monkeypatch, key, ok):
    monkeypatch.setenv("GROQ_API_KEY", key)
    assert (Config().groq_key_problem() is None) == ok


def test_config_paths_are_absolute_and_env_values_are_cleaned(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", '  "gsk_quoted"  ')
    monkeypatch.setenv("CHUNK_SIZE", "not-a-number")
    cfg = Config()
    assert cfg.groq_api_key == "gsk_quoted"
    assert cfg.chunk_size == 1500
    assert Path(cfg.repos_dir).is_absolute() and Path(cfg.chroma_persist_dir).is_absolute()


# --------------------------------------------------------------------------
# GitHub URL handling
# --------------------------------------------------------------------------
@pytest.mark.parametrize("url", [
    "https://github.com/Tharun5199/job-hunt-agent.git",
    "https://github.com/Tharun5199/job-hunt-agent/",
    "github.com/Tharun5199/job-hunt-agent",
    "https://github.com/Tharun5199/job-hunt-agent/tree/main/src",
    "git@github.com:Tharun5199/job-hunt-agent.git",
])
def test_url_variants_normalize_to_same_repo(url):
    repo = rl.parse_github_url(url)
    assert (repo.owner, repo.name) == ("Tharun5199", "job-hunt-agent")
    assert rl.repo_slug(url) == rl.repo_slug("https://github.com/Tharun5199/job-hunt-agent")


@pytest.mark.parametrize("url", ["", "hello", "https://gitlab.com/a/b", "https://github.com/only-owner"])
def test_invalid_urls_rejected(url):
    with pytest.raises(rl.InvalidRepoURLError):
        rl.parse_github_url(url)


# --------------------------------------------------------------------------
# Windows read-only .git files (the "already exists and is not an empty
# directory" bug)
# --------------------------------------------------------------------------
def test_rmtree_removes_readonly_files_with_windows_semantics(tmp_path, monkeypatch):
    target = tmp_path / "clone" / ".git" / "objects" / "pack"
    target.mkdir(parents=True)
    packfile = target / "pack-1.pack"
    packfile.write_bytes(b"data")
    packfile.chmod(stat.S_IREAD)  # git makes pack files read-only

    real_unlink = os.unlink

    def windows_like_unlink(path, *args, **kwargs):
        st = os.stat(path, dir_fd=kwargs.get("dir_fd"), follow_symlinks=False)
        if not st.st_mode & stat.S_IWRITE:
            raise PermissionError(errno.EACCES, "Access is denied", str(path))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", windows_like_unlink)
    rl._rmtree(tmp_path / "clone")
    assert not (tmp_path / "clone").exists()


# --------------------------------------------------------------------------
# File loading filters
# --------------------------------------------------------------------------
def _make_fake_repo(root: Path) -> Path:
    files = {
        "app.py": "def main():\n    return 1\n",
        "README.MD": "# Title\n",                      # upper-case extension
        "Dockerfile": "FROM python:3.11\n",
        "package-lock.json": '{"lock": true}',          # lockfile -> skip
        "static/app.min.js": "var a=1;",                # minified -> skip
        "node_modules/lib/index.js": "x",               # ignored dir -> skip
        ".github/workflows/ci.yml": "on: push",         # hidden dir -> skip
        "src/pkg.egg-info/PKG-INFO": "x",               # egg-info -> skip
        "empty.py": "   \n",                            # blank -> skip
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "logo.py").write_bytes(b"\x89PNG\x00\x00binary")  # binary content -> skip
    return root


def test_loader_filters(tmp_path):
    cfg = Config()
    repo = _make_fake_repo(tmp_path / "repo")
    docs = rl.load_repo_files(repo, cfg.allowed_extensions, cfg.ignored_dirs, cfg.max_file_size_kb)
    assert sorted(d.metadata["source"] for d in docs) == ["Dockerfile", "README.MD", "app.py"]


def test_loader_respects_max_files(tmp_path):
    cfg = Config()
    repo = tmp_path / "repo"
    repo.mkdir()
    for i in range(10):
        (repo / f"m{i}.py").write_text(f"x = {i}\n")
    docs = rl.load_repo_files(repo, cfg.allowed_extensions, cfg.ignored_dirs, 512, max_files=4)
    assert len(docs) == 4


# --------------------------------------------------------------------------
# Chunking + vector store
# --------------------------------------------------------------------------
def test_chunk_index_is_per_file():
    docs = [
        Document(page_content="def a():\n    pass\n" * 200, metadata={"source": "a.py", "extension": ".py"}),
        Document(page_content="def b():\n    pass\n" * 200, metadata={"source": "b.py", "extension": ".py"}),
    ]
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)
    for source in ("a.py", "b.py"):
        indexes = [c.metadata["chunk_index"] for c in chunks if c.metadata["source"] == source]
        assert indexes == list(range(len(indexes)))  # 0..n-1 within EACH file


def test_reingesting_same_repo_does_not_duplicate_chunks(tmp_path, fake_embeddings):
    docs = [Document(page_content=f"chunk {i}", metadata={"source": f"f{i}.py", "chunk_index": 0})
            for i in range(19)]
    for _ in range(5):  # the user analyzed the same repo 5 times
        store = CodeVectorStore(str(tmp_path / "vs"), "repo-test-1234abcd", fake_embeddings)
        store.build_from_documents(docs)
    assert store.count() == 19
    hits = store.as_retriever(k=5).invoke("chunk 3")
    assert len({h.page_content for h in hits}) == 5  # 5 DIFFERENT chunks, not one chunk 5x


# --------------------------------------------------------------------------
# QA chain
# --------------------------------------------------------------------------
def test_history_sanitization():
    raw = [["human", "q1"], ["ai", "a1"], ["system", "evil"], ["user", "q2"], ["assistant", "a2"],
           "garbage", ["human"], ["ai", ""], ["human", "x" * 5000]]
    cleaned = sanitize_history(raw, max_turns=2)
    # valid messages: q1, a1, q2(user->human), a2(assistant->ai), long; keep the last 2 turns = 4 messages
    assert cleaned[:3] == [("ai", "a1"), ("human", "q2"), ("ai", "a2")]
    assert len(cleaned) == 4
    assert cleaned[-1][0] == "human" and len(cleaned[-1][1]) == 2000


def test_qa_chain_single_and_multi_turn(tmp_path, fake_embeddings):
    docs = [Document(page_content="def get_answer():\n    return True", metadata={"source": "sample/helpers.py"})]
    store = CodeVectorStore(str(tmp_path / "vs"), "repo-qa-1234abcd", fake_embeddings).build_from_documents(docs)

    qa = CodeQAChain(store.as_retriever(k=1), FakeListChatModel(responses=["It returns True."]))
    result = qa.ask("What does get_answer return?")
    assert result["answer"] == "It returns True."
    assert result["sources"] == ["sample/helpers.py"]

    qa2 = CodeQAChain(store.as_retriever(k=1), FakeListChatModel(responses=["What does get_answer return?", "True."]))
    result2 = qa2.ask("and that one?", chat_history=[["human", "tell me about helpers"], ["ai", "ok"]])
    assert result2["standalone_question"] == "What does get_answer return?"
    assert result2["answer"] == "True."


def test_empty_question_rejected(tmp_path, fake_embeddings):
    store = CodeVectorStore(str(tmp_path / "vs"), "repo-e-1234abcd", fake_embeddings).build_from_documents(
        [Document(page_content="x", metadata={"source": "x.py"})])
    qa = CodeQAChain(store.as_retriever(k=1), FakeListChatModel(responses=["unused"]))
    with pytest.raises(CodeAnalyzerError) as info:
        qa.ask("   ")
    assert "type a question" in info.value.user_message.lower()


def test_friendly_messages_for_groq_errors():
    class AuthenticationError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    assert "rejected your api key" in friendly_message(AuthenticationError("Error code: 401")).lower()
    assert "rate limit" in friendly_message(RateLimitError("Error code: 429")).lower()


# --------------------------------------------------------------------------
# Real network paths (skipped automatically when offline)
# --------------------------------------------------------------------------
@needs_github
def test_clone_twice_and_missing_repo(tmp_path):
    first = rl.clone_repo(SAMPLE_REPO, base_path=str(tmp_path))
    second = rl.clone_repo(SAMPLE_REPO + ".git", base_path=str(tmp_path))  # different spelling, same repo
    assert first == second and (second / "setup.py").exists()

    with pytest.raises(CodeAnalyzerError) as info:
        rl.clone_repo("https://github.com/Tharun5199/definitely-not-a-real-repo-zz9", base_path=str(tmp_path))
    assert "doesn't exist or is private" in info.value.user_message
