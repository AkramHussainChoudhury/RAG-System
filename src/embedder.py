from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from src.tracer import get_logger

logger = get_logger("embedder")

MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level singleton so the model loads only once per session
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded")
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Return a (N, D) float32 numpy array of embeddings for the given texts."""
    model = _get_model()
    if not texts:
        dim = model.get_sentence_embedding_dimension()
        return np.empty((0, dim), dtype=np.float32)

    embeddings = np.array(
        model.encode(texts, convert_to_numpy=True, show_progress_bar=False),
        dtype=np.float32,
    )
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    logger.info(f"Embedded {len(texts)} texts → shape {embeddings.shape}")
    return embeddings
