"""Local, free embeddings -- no API key, no PyTorch.

Uses the ONNX build of `all-MiniLM-L6-v2` that ships with Chroma (via
onnxruntime). It's the same model sentence-transformers would load (same
weights, mean pooling, L2-normalized 384-dim vectors), but without the
~1 GB PyTorch stack -- which is what lets the whole app run inside a
512 MB free-tier container (e.g. Render). The model (~80 MB) is downloaded
once on first use and cached in ~/.cache/chroma/onnx_models.
"""
import gc
import os
import sys
import threading
from functools import cached_property
from typing import List

from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
from langchain_core.embeddings import Embeddings

from codeanalyzer.exceptions import CodeAnalyzerError
from codeanalyzer.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_MODELS = {"all-minilm-l6-v2", "sentence-transformers/all-minilm-l6-v2"}
_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))  # small batches keep peak RAM low


class _TunedONNXMiniLM(ONNXMiniLM_L6_V2):
    """Chroma's ONNX MiniLM, with a configurable CPU thread count.

    On a tiny cloud instance (e.g. 0.1 vCPU) letting onnxruntime spin up one
    thread per *host* core wastes the CPU quota; EMBEDDING_THREADS=1 fixes that.
    """

    @cached_property
    def model(self):  # type: ignore[override]
        options = self.ort.SessionOptions()
        options.log_severity_level = 3
        options.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        threads = int(os.getenv("EMBEDDING_THREADS", "0") or 0)
        if threads > 0:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        return self.ort.InferenceSession(
            os.path.join(self.DOWNLOAD_PATH, self.EXTRACTED_FOLDER_NAME, "model.onnx"),
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )


class OnnxMiniLMEmbeddings(Embeddings):
    """LangChain `Embeddings` adapter around the ONNX MiniLM model."""

    def __init__(self) -> None:
        self._fn = _TunedONNXMiniLM()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = [t if t.strip() else " " for t in texts[start:start + _BATCH_SIZE]]
            vectors.extend(v.tolist() for v in self._fn(batch))
        gc.collect()
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self._fn([text or " "])[0].tolist()


_lock = threading.Lock()
_instance: OnnxMiniLMEmbeddings = None  # type: ignore[assignment]


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> OnnxMiniLMEmbeddings:
    """Return the shared embedding model, downloading it on first use."""
    global _instance
    if model_name and model_name.strip().lower() not in SUPPORTED_MODELS:
        logger.warning(
            f"EMBEDDING_MODEL={model_name!r} isn't supported by the lightweight ONNX backend; "
            "using all-MiniLM-L6-v2 instead."
        )
    try:
        with _lock:  # two concurrent first requests must not load the model twice
            if _instance is None:
                logger.info("Loading local embedding model all-MiniLM-L6-v2 (ONNX; first run downloads ~80MB)")
                candidate = OnnxMiniLMEmbeddings()
                candidate.embed_query("warm-up")  # triggers the one-time download + session start
                _instance = candidate
            return _instance
    except Exception as exc:  # noqa: BLE001
        raise CodeAnalyzerError(
            exc,
            sys,
            user_message=(
                "Could not load the embedding model. The first run needs internet access to download "
                f"it (about 80MB); after that it's cached. Details: {str(exc)[:200]}"
            ),
        ) from exc
