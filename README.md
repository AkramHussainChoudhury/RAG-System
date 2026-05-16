# RAG System

A Retrieval-Augmented Generation (RAG) pipeline built from scratch for learning purposes. Each component is isolated so the concepts are easy to understand and swap out independently.

## What is RAG?

RAG is a technique that lets a language model answer questions about your own documents. Instead of relying on what the model was trained on, it:

1. **Retrieves** the most relevant chunks from your documents
2. **Augments** the prompt with those chunks as context
3. **Generates** an answer grounded in your actual content

```
Your PDF → chunks → embeddings → vector store
                                      ↑
Your question → embed → search ───────┘ → top chunks → reranker → LLM → answer
```

## Project Structure

```
RAG-System/
├── data/               ← drop your PDFs here (gitignored)
├── src/
│   ├── tracer.py       ← console logging across all steps
│   ├── loader.py       ← extract text from PDFs (pypdf)
│   ├── chunker.py      ← split text into semantic chunks (nltk + cosine similarity)
│   ├── embedder.py     ← convert text to vectors (sentence-transformers)
│   ├── vector_store.py ← persist embeddings and search (ChromaDB)
│   ├── bm25_store.py   ← keyword search index (BM25)
│   ├── retriever.py    ← hybrid search with reciprocal rank fusion
│   ├── reranker.py     ← cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
│   ├── generator.py    ← send retrieved chunks + question to Ollama
│   ├── tools.py        ← wraps retriever as a LangChain tool for the agent
│   ├── agent.py        ← LangGraph ReAct agent with document search tool
│   └── pipeline.py     ← wires all steps together
├── main.py             ← entry point (load / query / agent / list commands)
└── requirements.txt
```

## Stack

| Component | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 | Phase 6 |
|-----------|---------|---------|---------|---------|---------|---------|
| Chunking | Fixed-size | Semantic | Semantic | Semantic | Semantic | Semantic |
| Vector store | numpy in-memory | ChromaDB | ChromaDB | ChromaDB | ChromaDB | ChromaDB |
| Retrieval | Semantic only | Semantic only | Semantic only | Semantic only | Hybrid (semantic + BM25 + RRF) | Hybrid + Reranking |
| Reranking | — | — | — | — | — | `ms-marco-MiniLM-L-6-v2` |
| Generation | `llama3.2:1b` | `llama3.2:3b` | `llama3.2:3b` | `llama3.2:3b` | `llama3.2:3b` | `llama3.2:3b` |
| LLM interface | `ollama` direct | `ollama` direct | `langchain-ollama` | `langchain-ollama` | `langchain-ollama` | `langchain-ollama` |
| Tracing | Python `logging` | Python `logging` | LangSmith | LangSmith | LangSmith | LangSmith |
| Agent | — | — | — | LangGraph ReAct | LangGraph ReAct | LangGraph ReAct |

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running with `llama3.2:3b` pulled:

```bash
ollama pull llama3.2:3b
```

- [LangSmith](https://smith.langchain.com) account and API key. Add to a `.env` file in the project root:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_api_key_here
LANGCHAIN_PROJECT=rag-system
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/AkramHussainChoudhury/RAG-System.git
cd RAG-System

# 2. Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Running

```bash
# Index a PDF (run once per document)
python main.py load data/your-file.pdf

# Ask questions (direct RAG pipeline)
python main.py query your-file

# Ask questions (ReAct agent — decides when to search)
python main.py agent your-file

# See all indexed documents
python main.py list
```

> On first run, `sentence-transformers` downloads embedding and reranker models (~170MB total) from Hugging Face automatically. No account or token needed.

## Known Limitations

- **Scanned / image-based PDFs** — pypdf can only read text-based PDFs
- **Single-document queries** — each session searches one collection; cross-document search is a future upgrade
- **Agent tool calling** — small local models (3b) can be unreliable at tool calling; a larger model or cloud API (Groq) improves reliability
