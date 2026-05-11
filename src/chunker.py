from src.tracer import get_logger

logger = get_logger("chunker")

# chunk_size: how many characters per chunk
# overlap: how many characters the next chunk re-reads from the previous one
#   overlap prevents an answer from being cut across two chunks and lost
CHUNK_SIZE = 512
OVERLAP = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Split text into fixed-size overlapping chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    logger.info(f"Chunked into {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks
