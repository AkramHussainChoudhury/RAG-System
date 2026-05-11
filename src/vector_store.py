import numpy as np
from src.tracer import get_logger

logger = get_logger("vector_store")


class VectorStore:
    def __init__(self):
        self.chunks: list[str] = []
        self.embeddings: np.ndarray = np.empty((0,))

    def add(self, chunks: list[str], embeddings: np.ndarray) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        logger.info(f"Stored {len(chunks)} chunks, embedding dim={embeddings.shape[1]}")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        chunk_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        denom = chunk_norms * query_norm
        scores = np.where(denom > 0, self.embeddings @ query_embedding / denom, 0.0)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.chunks[i], float(scores[i])) for i in top_indices]
        logger.info(f"Top-{top_k} scores: {[round(s, 3) for _, s in results]}")
        return results
