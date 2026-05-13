import nltk
import numpy as np
from src.embedder import embed
from src.tracer import get_logger

nltk.download("punkt_tab", quiet=True)

logger = get_logger("chunker")

SIMILARITY_THRESHOLD = 0.5  # split when similarity between consecutive sentences drops below this
MAX_CHUNK_CHARS = 1000       # safety cap so no single chunk grows too large


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def chunk_text(text: str, threshold: float = SIMILARITY_THRESHOLD, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return []

    logger.info(f"Tokenized {len(sentences)} sentences")

    # Embed all sentences in one batch — reuses the singleton model from embedder.py
    embeddings = embed(sentences)

    chunks = []
    current_sentences = [sentences[0]]
    current_chars = len(sentences[0])

    for i in range(1, len(sentences)):
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
        too_large = current_chars + len(sentences[i]) > max_chars

        if sim < threshold or too_large:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentences[i]]
            current_chars = len(sentences[i])
            logger.info(f"  Split at sentence {i + 1} — similarity={sim:.3f}")
        else:
            current_sentences.append(sentences[i])
            current_chars += len(sentences[i])

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    logger.info(f"Semantic chunking → {len(chunks)} chunks (threshold={threshold})")
    return chunks
