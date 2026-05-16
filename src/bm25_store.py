from rank_bm25 import BM25Okapi
from src.tracer import get_logger

logger = get_logger("bm25_store")


class BM25Store:
    def __init__(self):
        self._bm25 = None
        self._chunks = []

    def build(self, chunks: list[str]) -> None:
        self._chunks = chunks
        tokenised = [chunk.lower().split() for chunk in chunks]
        self._bm25 = BM25Okapi(tokenised)
        logger.info(f"BM25 index built from {len(chunks)} chunks")

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = [(self._chunks[i], float(scores[i])) for i in top_indices]
        logger.info(f"BM25 top-{top_k} scores: {[round(s, 3) for _, s in results]}")
        return results
