from setuptools import find_packages, setup

setup(
    name="codeanalyzer",
    version="0.1.0",
    description="AI-powered GitHub repository analyzer (RAG over source code).",
    author="Your Name",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
