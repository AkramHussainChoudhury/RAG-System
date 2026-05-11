import time
from src.loader import load_pdf
from src.chunker import chunk_text
from src.embedder import embed
from src.vector_store import VectorStore
from src.retriever import retrieve
from src.generator import generate
from src.tracer import get_logger

logger = get_logger("pipeline")


def build_index(pdf_path: str) -> VectorStore:
    t0 = time.time()
    logger.info(f"=== Building index: {pdf_path} ===")

    text = load_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = embed(chunks)

    store = VectorStore()
    store.add(chunks, embeddings)
    logger.info(f"=== Index ready in {time.time() - t0:.2f}s ===")
    return store


def query(question: str, store: VectorStore, top_k: int = 3) -> str:
    logger.info(f"=== Query: {question} ===")
    t0 = time.time()

    results = retrieve(question, store, top_k=top_k)
    context_chunks = [chunk for chunk, _ in results]
    answer = generate(question, context_chunks)

    logger.info(f"=== Done in {time.time() - t0:.2f}s ===")
    return answer
