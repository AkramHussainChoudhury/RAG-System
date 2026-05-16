from typing import Optional
from langchain_core.tools import tool
from src.vector_store import VectorStore
from src.retriever import retrieve

_store: Optional[VectorStore] = None


def set_store(store: VectorStore) -> None:
    global _store
    _store = store


@tool
def search_document(query: str) -> str:
    """Search the loaded document for information relevant to the query."""
    if _store is None:
        return "No document loaded. Run: python main.py load <path_to_pdf>"
    results = retrieve(query, _store, top_k=5)
    if not results:
        return "No relevant chunks found."
    return "\n\n---\n\n".join(
        f"[Chunk {i+1} | score={score:.3f}]\n{chunk}"
        for i, (chunk, score) in enumerate(results)
    )
