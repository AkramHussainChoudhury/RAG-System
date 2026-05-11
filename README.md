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
Your question → embed → search ───────┘ → top chunks → LLM → answer
```

## Project Structure

```
RAG-System/
├── data/               ← drop your PDFs here (gitignored)
├── src/
│   ├── tracer.py       ← console logging across all steps
│   ├── loader.py       ← extract text from PDFs (pypdf)
│   ├── chunker.py      ← split text into fixed-size overlapping chunks
│   ├── embedder.py     ← convert text to vectors (sentence-transformers)
│   ├── vector_store.py ← store embeddings, search by cosine similarity
│   ├── retriever.py    ← embed query and find top-k matching chunks
│   ├── generator.py    ← send retrieved chunks + question to Ollama
│   └── pipeline.py     ← wires all steps together
├── main.py             ← entry point (interactive Q&A loop)
└── requirements.txt
```

## Phase 1 Stack

| Component | Tool | Why |
|-----------|------|-----|
| PDF parsing | `pypdf` | Lightweight, pure Python |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Free, fully local, no API key |
| Vector store | `numpy` (in-memory) | Makes cosine similarity math transparent |
| Generation | `Ollama` (`llama3.2:1b`) | Free, runs locally, no API subscription needed |
| Tracing | Python `logging` | Zero dependencies, see every pipeline step |

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running with `llama3.2:1b` pulled:

```bash
ollama pull llama3.2:1b
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

> On first run, `sentence-transformers` will automatically download the `all-MiniLM-L6-v2` model (~90 MB) from Hugging Face. No account or token needed.

## Running

Drop any text-based PDF into the `data/` folder, then:

```bash
python main.py data/your-file.pdf
```

Example session:

```
10:32:01 [pipeline] === Building index: data/paper.pdf ===
10:32:01 [loader]   Total extracted: 42,301 chars from 12 pages
10:32:01 [chunker]  Chunked into 87 chunks (size=512, overlap=50)
10:32:04 [embedder] Embedded 87 texts → shape (87, 384)
10:32:04 [pipeline] === Index ready in 3.42s ===

RAG ready. Ask questions about your PDF (type 'quit' to exit).

Q: What is the main contribution of this paper?
A: The main contribution is ...

Q: quit
```

## Known Limitations (Phase 1)

- **Scanned / image-based PDFs** — pypdf can only read text-based PDFs. OCR support is planned for a later phase.
- **Fixed-size chunking** — can split sentences mid-thought. Semantic chunking comes in a later phase.
- **In-memory store** — index is rebuilt every run. Persistent vector DB comes in a later phase.
- **Small model** — `llama3.2:1b` is fast but limited in quality. Swap the model name in `src/generator.py` for a larger Ollama model.
