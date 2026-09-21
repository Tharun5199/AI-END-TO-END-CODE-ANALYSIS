"""
Project scaffolding script.

Run this once (`python template.py`) to (re)create the full folder/file
skeleton for the project. It never overwrites a file that already has
content, so it is safe to re-run at any time -- it only fills in anything
that's missing.
"""
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

PROJECT_NAME = "codeanalyzer"

list_of_files = [
    ".github/workflows/cicd.yaml",
    f"src/{PROJECT_NAME}/__init__.py",
    f"src/{PROJECT_NAME}/logger.py",
    f"src/{PROJECT_NAME}/exceptions.py",
    f"src/{PROJECT_NAME}/config.py",
    f"src/{PROJECT_NAME}/pipeline.py",
    f"src/{PROJECT_NAME}/data_ingestion/__init__.py",
    f"src/{PROJECT_NAME}/data_ingestion/repo_loader.py",
    f"src/{PROJECT_NAME}/text_splitter/__init__.py",
    f"src/{PROJECT_NAME}/text_splitter/code_splitter.py",
    f"src/{PROJECT_NAME}/embeddings/__init__.py",
    f"src/{PROJECT_NAME}/embeddings/embedding_manager.py",
    f"src/{PROJECT_NAME}/vectorstore/__init__.py",
    f"src/{PROJECT_NAME}/vectorstore/chroma_store.py",
    f"src/{PROJECT_NAME}/llm/__init__.py",
    f"src/{PROJECT_NAME}/llm/groq_llm.py",
    f"src/{PROJECT_NAME}/qa/__init__.py",
    f"src/{PROJECT_NAME}/qa/qa_chain.py",
    "notebook/experiments.ipynb",
    "templates/index.html",
    "static/css/style.css",
    "static/js/script.js",
    "tests/conftest.py",
    "tests/test_pipeline.py",
    "tests/test_smoke.py",
    "app.py",
    "check_setup.py",
    "setup.py",
    "requirements.txt",
    "requirements-dev.txt",
    ".env.example",
    "Dockerfile",
    ".dockerignore",
    "README.md",
]


def create_project_structure() -> None:
    for file_path in list_of_files:
        path = Path(file_path)
        file_dir, file_name = os.path.split(path)

        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)
            logging.info(f"Created directory: {file_dir}")

        if (not path.exists()) or path.stat().st_size == 0:
            with open(path, "w"):
                pass
            logging.info(f"Created empty file: {path}")
        else:
            logging.info(f"File already exists, skipping: {path}")


if __name__ == "__main__":
    create_project_structure()
