"""Language-aware chunking: split each file using a splitter tuned for its
own programming language, so chunks respect function/class boundaries
instead of cutting code in half at an arbitrary character count.
"""
import sys
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from codeanalyzer.exceptions import CodeAnalyzerError
from codeanalyzer.logger import get_logger

logger = get_logger(__name__)

_EXTENSION_TO_LANGUAGE = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".mjs": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".c": Language.C,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".h": Language.CPP,
    ".hpp": Language.CPP,
    ".cs": Language.CSHARP,
    ".kt": Language.KOTLIN,
    ".kts": Language.KOTLIN,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".sol": Language.SOL,
    ".md": Language.MARKDOWN,
    ".rst": Language.RST,
    ".html": Language.HTML,
}


def _splitter_for_extension(extension: str, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    language = _EXTENSION_TO_LANGUAGE.get(extension)
    if language is not None:
        return RecursiveCharacterTextSplitter.from_language(
            language=language, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def split_documents(
    documents: List[Document], chunk_size: int = 1500, chunk_overlap: int = 200
) -> List[Document]:
    """Split documents into chunks, using a language-aware splitter per file type."""
    try:
        splitters = {}
        chunks: List[Document] = []
        for doc in documents:
            extension = doc.metadata.get("extension", "").lower()
            if extension not in splitters:
                splitters[extension] = _splitter_for_extension(extension, chunk_size, chunk_overlap)
            file_chunks = splitters[extension].split_documents([doc])
            total = len(file_chunks)
            for i, chunk in enumerate(file_chunks):
                # Numbered per file (not per extension), so "chunk 2 of 5 in app.py" is meaningful
                # and (source, chunk_index) is a unique, stable key for the vector store.
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = total
            chunks.extend(file_chunks)

        logger.info(f"Split {len(documents)} files into {len(chunks)} chunks")
        return chunks
    except Exception as exc:  # noqa: BLE001
        raise CodeAnalyzerError(exc, sys) from exc
