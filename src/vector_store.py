import numpy as np
import chromadb
from src.tracer import get_logger

logger = get_logger("vector_store")

PERSIST_DIR = ".chroma"


class VectorStore:
    def __init__(self, collection_name: str = "rag"):
        self._client = chromadb.PersistentClient(path=PERSIST_DIR)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB collection '{collection_name}' — {self._collection.count()} chunks on disk")

    def is_populated(self) -> bool:
        return self._collection.count() > 0

    def add(self, chunks: list[str], embeddings: np.ndarray) -> None:
        ids = [str(i) for i in range(len(chunks))]
        self._collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
        )
        logger.info(f"Stored {len(chunks)} chunks → ChromaDB at '{PERSIST_DIR}'")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "distances"],
        )
        # ChromaDB returns cosine distance (0=identical, 2=opposite); convert to similarity
        docs = results["documents"][0]
        distances = results["distances"][0]
        output = [(doc, 1.0 - dist) for doc, dist in zip(docs, distances)]
        logger.info(f"Top-{top_k} scores: {[round(s, 3) for _, s in output]}")
        return output
