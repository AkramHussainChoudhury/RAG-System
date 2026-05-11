import sys
from src.pipeline import build_index, query


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    store = build_index(pdf_path)

    print("\nRAG ready. Ask questions about your PDF (type 'quit' to exit).\n")
    while True:
        question = input("Q: ").strip()
        if not question or question.lower() in ("quit", "exit", "q"):
            break
        answer = query(question, store)
        print(f"\nA: {answer}\n")


if __name__ == "__main__":
    main()
