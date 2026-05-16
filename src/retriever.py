from langsmith import traceable
from src.embedder import embed
from src.vector_store import VectorStore
from src.bm25_store import BM25Store
from src.tracer import get_logger

logger = get_logger("retriever")


def _reciprocal_rank_fusion(
    semantic_results: list[tuple[str, float]],
    keyword_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores = {}
    for rank, (chunk, _) in enumerate(semantic_results):
        scores[chunk] = scores.get(chunk, 0) + 1 / (k + rank + 1)
    for rank, (chunk, _) in enumerate(keyword_results):
        scores[chunk] = scores.get(chunk, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


@traceable
def retrieve(query: str, store: VectorStore, bm25: BM25Store, top_k: int = 5) -> list[tuple[str, float]]:
    query_embedding = embed([query])[0]
    semantic_results = store.search(query_embedding, top_k=top_k)
    keyword_results = bm25.search(query, top_k=top_k)

    fused = _reciprocal_rank_fusion(semantic_results, keyword_results)[:top_k]

    for i, (chunk, score) in enumerate(fused):
        logger.info(f"  [{i+1}] rrf_score={score:.4f} | {chunk[:80].strip()}...")

    return fused
