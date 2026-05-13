import sys
import chromadb
from pathlib import Path
from src.pipeline import build_index, query
from src.vector_store import VectorStore, PERSIST_DIR


def cmd_load(pdf_path: str) -> None:
    build_index(pdf_path)
    collection = Path(pdf_path).stem.lower()
    print(f"\nLoaded. To query, run: python main.py query {collection}\n")


def cmd_query(collection_name: str) -> None:
    store = VectorStore(collection_name)
    if not store.is_populated():
        print(f"No index found for '{collection_name}'. Run: python main.py load <path_to_pdf>")
        sys.exit(1)

    print(f"\nRAG ready ({collection_name}). Ask questions (type 'quit' to exit).\n")
    while True:
        question = input("Q: ").strip()
        if not question or question.lower() in ("quit", "exit", "q"):
            break
        answer = query(question, store)
        print(f"\nA: {answer}\n")


def cmd_list() -> None:
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collections = client.list_collections()
    if not collections:
        print("No collections found. Run: python main.py load <path_to_pdf>")
    else:
        print("\nAvailable collections:")
        for col in collections:
            print(f"  {col.name}")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py load <path_to_pdf>")
        print("  python main.py query <collection_name>")
        print("  python main.py list")
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        cmd_list()
    elif len(sys.argv) < 3:
        print("Usage:")
        print("  python main.py load <path_to_pdf>")
        print("  python main.py query <collection_name>")
        print("  python main.py list")
        sys.exit(1)
    elif command == "load":
        cmd_load(sys.argv[2])
    elif command == "query":
        cmd_query(sys.argv[2])
    else:
        print(f"Unknown command '{command}'. Use 'load', 'query', or 'list'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
