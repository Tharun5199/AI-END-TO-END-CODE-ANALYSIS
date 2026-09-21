"""Clone a GitHub repository and load its source files as LangChain Documents."""
import hashlib
import os
import re
import shutil
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from git import GitCommandError, Repo
from langchain_core.documents import Document

from codeanalyzer.exceptions import CodeAnalyzerError
from codeanalyzer.logger import get_logger

logger = get_logger(__name__)

# Accepts all the ways people paste a repo link:
#   https://github.com/owner/repo            github.com/owner/repo
#   https://github.com/owner/repo.git        https://www.github.com/owner/repo/
#   https://github.com/owner/repo/tree/main/src   (a folder/branch link)
#   https://github.com/owner/repo/blob/main/app.py (a file link)
#   git@github.com:owner/repo.git
_GITHUB_URL_RE = re.compile(
    r"^(?:(?:https?://)?(?:www\.)?github\.com/|git@github\.com:)"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/"
    r"(?P<repo>[A-Za-z0-9._-]+?)"
    r"(?:\.git)?"
    r"(?:/.*)?$",
    re.IGNORECASE,
)

# Files that are pure noise for code understanding (and often huge).
_SKIP_FILE_NAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "pipfile.lock",
    "composer.lock", "cargo.lock", "go.sum", "gemfile.lock",
}
_SKIP_FILE_SUFFIXES = (".min.js", ".min.css", ".map", ".bundle.js")

# Never block a web request waiting for a GitHub username/password prompt
# (terminal prompt, or the Git Credential Manager login popup on Windows).
_GIT_ENV = {"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}


@dataclass(frozen=True)
class GitHubRepo:
    owner: str
    name: str

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}.git"

    @property
    def web_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}"


class InvalidRepoURLError(ValueError):
    pass


def parse_github_url(url: str) -> GitHubRepo:
    """Normalize any common GitHub repo link into (owner, repo)."""
    cleaned = (url or "").strip().split("?")[0].split("#")[0].rstrip("/")
    match = _GITHUB_URL_RE.match(cleaned)
    if not match:
        raise InvalidRepoURLError(
            "That doesn't look like a GitHub repository link. Use a URL like "
            "https://github.com/owner/repo"
        )
    return GitHubRepo(owner=match.group("owner"), name=match.group("repo"))


def repo_slug(github_url: str) -> str:
    """Short, filesystem-safe and stable folder/collection name for a repo.

    Every way of writing the same repo URL maps to the same slug.
    """
    repo = parse_github_url(github_url)
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "-", repo.name).strip("-_")[:40] or "repo"
    digest = hashlib.sha1(f"{repo.owner}/{repo.name}".lower().encode("utf-8")).hexdigest()[:8]
    return f"{safe_name}-{digest}"


def _on_rm_error(func, path, _exc) -> None:
    """rmtree error hook: clear the read-only bit and retry.

    Git marks files under .git/objects read-only. On Windows that makes a
    plain delete raise PermissionError, which previously left the folder
    half-deleted -- and the next clone failed with "destination path already
    exists and is not an empty directory".
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except FileNotFoundError:
        pass


def _rmtree(path: Path) -> None:
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_on_rm_error)  # `onerror` is deprecated from 3.12
    else:
        shutil.rmtree(path, onerror=_on_rm_error)


def _friendly_git_error(exc: Exception, repo: GitHubRepo) -> str:
    stderr = (getattr(exc, "stderr", "") or str(exc)).lower()
    if any(s in stderr for s in ("repository not found", "could not read username",
                                  "terminal prompts disabled", "authentication failed", "403")):
        return (f"Couldn't clone {repo.web_url}: the repository doesn't exist or is private. "
                "Only public repositories can be analyzed.")
    if any(s in stderr for s in ("could not resolve host", "unable to access", "timed out",
                                  "failed to connect")):
        return "Couldn't reach github.com. Check your internet connection and try again."
    if "not found" in stderr and "git" in stderr and "executable" in stderr:
        return "Git is not installed or not on your PATH. Install it from https://git-scm.com and restart."
    return f"git clone failed for {repo.web_url}: {(getattr(exc, 'stderr', '') or str(exc)).strip()[:300]}"


def clone_repo(github_url: str, base_path: str = "repos", force_refresh: bool = True) -> Path:
    """Shallow-clone `github_url` into `base_path/<repo-slug>` and return the local path.

    With force_refresh=True (the default) an existing copy is deleted and
    re-cloned, so analysis always reflects the repo's current default branch.
    """
    try:
        repo = parse_github_url(github_url)
    except InvalidRepoURLError as exc:
        raise CodeAnalyzerError(exc, sys, user_message=str(exc)) from exc

    try:
        base = Path(base_path)
        base.mkdir(parents=True, exist_ok=True)
        target_dir = base / repo_slug(github_url)

        if target_dir.exists():
            if not force_refresh:
                logger.info(f"Reusing previously cloned repo at {target_dir}")
                return target_dir
            logger.info(f"Removing previously cloned copy at {target_dir}")
            try:
                _rmtree(target_dir)
            except OSError as rm_exc:
                # E.g. a file inside is open in an editor / scanned by antivirus.
                # Don't fail the whole request -- clone next to it instead.
                fallback = target_dir.with_name(f"{target_dir.name}-{int(time.time())}")
                logger.warning(f"Could not remove {target_dir} ({rm_exc}); cloning into {fallback}")
                target_dir = fallback

        logger.info(f"Cloning {repo.clone_url} -> {target_dir}")
        try:
            Repo.clone_from(repo.clone_url, target_dir, depth=1, single_branch=True, env=_GIT_ENV)
        except GitCommandError:
            # A leftover partial clone (e.g. from a killed process) can make even
            # the first attempt fail with "not an empty directory". Clear it and
            # retry once; any other failure is re-raised as-is.
            if target_dir.exists() and any(target_dir.iterdir()):
                logger.warning(f"Clone failed, clearing {target_dir} and retrying once")
                _rmtree(target_dir)
                Repo.clone_from(repo.clone_url, target_dir, depth=1, single_branch=True, env=_GIT_ENV)
            else:
                raise
        return target_dir
    except GitCommandError as exc:
        raise CodeAnalyzerError(exc, sys, user_message=_friendly_git_error(exc, repo)) from exc
    except CodeAnalyzerError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CodeAnalyzerError(exc, sys) from exc


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            return b"\x00" in handle.read(8192)
    except OSError:
        return True


def load_repo_files(
    repo_path: Path,
    allowed_extensions: List[str],
    ignored_dirs: List[str],
    max_file_size_kb: int = 512,
    max_files: Optional[int] = None,
) -> List[Document]:
    """Walk `repo_path` and load every allowed source file as a Document.

    Skipped: ignored/hidden directories (.git, node_modules, venv, build
    output...), lockfiles, minified bundles, binary files, oversized files.
    """
    try:
        repo_path = Path(repo_path)
        max_bytes = max_file_size_kb * 1024
        allowed = {ext.lower() for ext in allowed_extensions}
        ignored = {d.lower() for d in ignored_dirs}
        documents: List[Document] = []

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = sorted(
                d for d in dirs
                if d.lower() not in ignored and not d.startswith(".") and not d.endswith(".egg-info")
            )

            for filename in sorted(files):
                lower_name = filename.lower()
                if lower_name in _SKIP_FILE_NAMES or lower_name.endswith(_SKIP_FILE_SUFFIXES):
                    continue

                file_path = Path(root) / filename
                extension = file_path.suffix.lower()
                if extension not in allowed and lower_name not in allowed:
                    continue

                try:
                    if file_path.stat().st_size > max_bytes or _is_binary(file_path):
                        continue
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue

                if not text.strip():
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path.relative_to(repo_path).as_posix(),
                            "file_name": filename,
                            "extension": extension,
                        },
                    )
                )

                if max_files and len(documents) >= max_files:
                    logger.warning(f"Reached MAX_FILES={max_files}; remaining files are skipped")
                    logger.info(f"Loaded {len(documents)} source files from {repo_path}")
                    return documents

        logger.info(f"Loaded {len(documents)} source files from {repo_path}")
        return documents
    except Exception as exc:  # noqa: BLE001
        raise CodeAnalyzerError(exc, sys) from exc
