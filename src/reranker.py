from sentence_transformers import CrossEncoder
from src.tracer import get_logger

logger = get_logger("reranker")

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        logger.info(f"Loading reranker model: {MODEL_NAME}")
        _model = CrossEncoder(MODEL_NAME)
        logger.info("Reranker model loaded")
    return _model


def rerank(query: str, chunks: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    model = _get_model()
    pairs = [(query, chunk) for chunk in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    results = ranked[:top_k]
    logger.info(f"Reranked {len(chunks)} → top {top_k} scores: {[round(float(s), 3) for _, s in results]}")
    return [(chunk, float(score)) for chunk, score in results]
