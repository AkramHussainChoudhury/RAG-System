import json
from pathlib import Path
from src.vector_store import VectorStore
from src.bm25_store import BM25Store
from src.retriever import retrieve
from src.pipeline import query
from src.tracer import get_logger

logger = get_logger("evaluator")


def _hit(candidates: list[tuple[str, float]], expected: str) -> bool:
    return any(expected.lower() in chunk.lower() for chunk, _ in candidates)


def _reciprocal_rank(candidates: list[tuple[str, float]], expected: str) -> float:
    for rank, (chunk, _) in enumerate(candidates):
        if expected.lower() in chunk.lower():
            return 1 / (rank + 1)
    return 0.0


def _exact_match(answer: str, expected: str) -> bool:
    return expected.lower() in answer.lower()


def run_eval(store: VectorStore, bm25: BM25Store, dataset_path: str) -> None:
    dataset = json.loads(Path(dataset_path).read_text())

    hit_rates, reciprocal_ranks, exact_matches = [], [], []

    print(f"\nRunning eval on {len(dataset)} questions...\n")

    for item in dataset:
        question = item["question"]
        expected = item["expected_answer"]

        candidates = retrieve(question, store, bm25, top_k=10)
        hit = _hit(candidates, expected)
        rr = _reciprocal_rank(candidates, expected)
        hit_rates.append(hit)
        reciprocal_ranks.append(rr)

        answer = query(question, store, bm25)
        match = _exact_match(answer, expected)
        exact_matches.append(match)

        status = "✓" if match else "✗"
        print(f"  [{status}] {question}")
        print(f"       Expected : {expected}")
        print(f"       Got      : {answer[:120].strip()}")
        print(f"       Hit={hit} | RR={rr:.2f} | Match={match}\n")

    n = len(dataset)
    print("=== Evaluation Results ===")
    print(f"Questions   : {n}")
    print(f"Hit Rate    : {sum(hit_rates)/n:.1%}")
    print(f"MRR         : {sum(reciprocal_ranks)/n:.3f}")
    print(f"Answer Acc  : {sum(exact_matches)/n:.1%}")
