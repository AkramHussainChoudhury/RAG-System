import ollama
from src.tracer import get_logger

logger = get_logger("generator")

MODEL = "llama3.2:3b"


def generate(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    prompt = (
        "Use only the context below to answer the question. "
        "If the answer is not in the context, say 'I don't know based on the provided document.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    logger.info(f"Sending {len(context_chunks)} chunks to {MODEL}")
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]
