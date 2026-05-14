from langsmith import traceable
from src.embedder import embed
from src.vector_store import VectorStore
from src.tracer import get_logger

logger = get_logger("retriever")


@traceable
def retrieve(query: str, store: VectorStore, top_k: int = 3) -> list[tuple[str, float]]:
    query_embedding = embed([query])[0]
    results = store.search(query_embedding, top_k=top_k)
    for i, (chunk, score) in enumerate(results):
        logger.info(f"  [{i+1}] score={score:.3f} | {chunk[:80].strip()}...")
    return results
