from typing import Optional
from langchain_core.tools import tool
from src.vector_store import VectorStore
from src.bm25_store import BM25Store
from src.retriever import retrieve

_store: Optional[VectorStore] = None
_bm25: Optional[BM25Store] = None


def set_store(store: VectorStore, bm25: BM25Store) -> None:
    global _store, _bm25
    _store = store
    _bm25 = bm25


@tool
def search_document(query: str) -> str:
    """Search the loaded document for information relevant to the query."""
    if _store is None or _bm25 is None:
        return "No document loaded. Run: python main.py load <path_to_pdf>"
    results = retrieve(query, _store, _bm25, top_k=5)
    if not results:
        return "No relevant chunks found."
    return "\n\n---\n\n".join(
        f"[Chunk {i+1} | score={score:.4f}]\n{chunk}"
        for i, (chunk, score) in enumerate(results)
    )
