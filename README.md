# epstein-search 🔍

[![PyPI](https://img.shields.io/pypi/v/epstein-search)](https://pypi.org/project/epstein-search)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Semantic search over the publicly available Epstein Files — court documents, FBI reports, and DOJ publications — using AI-powered vector search.

Built with [zvec](https://github.com/alibaba/zvec) and pre-computed embeddings from [devankit7873/EpsteinFiles-Vector-Embeddings-ChromaDB](https://huggingface.co/datasets/devankit7873/EpsteinFiles-Vector-Embeddings-ChromaDB).

**No API key needed for search. Runs entirely locally.**

---

## ☕ Support This Project

This is free, open-source research tooling. If you find it useful, please consider supporting:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-simulationship-yellow?style=for-the-badge&logo=buy-me-a-coffee)](https://buymeacoffee.com/simulationship)

Or tip via crypto:

| Chain | Address |
|-------|---------|
| **SOL** | `76fCU6va3cGrbak4i9mwFfdYx1QsJJrqrxViFDUTsXUL` |
| **ETH** | `0xE808754a18A893A3eeFE01780D822C902680d1B7` |
| **BASE** | `0xE808754a18A893A3eeFE01780D822C902680d1B7` |

---

## Quick Start

```bash
pip install epstein-search

# One-time setup — downloads pre-built index (~100K+ document chunks)
epstein-search setup

# Search — free, local, no API key needed
epstein-search search "flight logs to the island"

# Filter by document type
epstein-search search "testimony" --doc-type deposition

# AI-powered answers (local Ollama = free, or any LLM API)
epstein-search ask "Who appears most in the flight logs?" --model ollama/llama3
```

## Commands

| Command | Description | API Key? |
|---------|-------------|----------|
| `epstein-search setup` | Download & build search index | No |
| `epstein-search chat` | **Interactive mode (recommended)** | Only for cloud LLMs |
| `epstein-search search "query"` | One-off semantic search | **No** |
| `epstein-search ask "question"` | One-off RAG answer | Only for cloud LLMs |
| `epstein-search info` | Show index stats | No |
| `epstein-search ingest` | Build from scratch (advanced) | No |

## Interactive Mode (Recommended)

The easiest way to use epstein-search:

```bash
epstein-search chat
```

Type questions naturally. No flags needed. Commands inside chat:

| Command | Description |
|---------|-------------|
| `/search` | Switch to search-only mode (no LLM) |
| `/ask` | Switch to RAG mode (LLM answers) |
| `/model openai/llama3` | Change LLM on the fly |
| `/topk 5` | Change number of results |
| `/info` | Show current settings |
| `/quit` | Exit |

**Tip:** Set your model once in `.env` so you never need to specify it:
```
EPSTEIN_LLM_MODEL=openai/your-model-name
```

## Search Options

```bash
epstein-search search "financial records" --top-k 5
epstein-search search "testimony" --source "FBI"
epstein-search search "flight logs" --doc-type flight_log
epstein-search search "court order" --json-output
```

### Document Type Filters

| Filter | Description |
|--------|-------------|
| `court_filing` | Court motions, orders, filings |
| `deposition` | Sworn testimony, Q&A transcripts |
| `fbi_report` | FBI investigation reports |
| `flight_log` | Aircraft flight records |
| `financial` | Bank records, wire transfers |
| `other` | Miscellaneous documents |

## Ask (RAG)

The `ask` command retrieves relevant documents and generates an answer using any LLM via [LiteLLM](https://github.com/BerriAI/litellm).

**Free options (no API key):**
```bash
# Run Ollama locally — completely free
epstein-search ask "Key individuals mentioned" --model ollama/llama3
epstein-search ask "What do the flight logs show?" --model ollama/mistral
```

**LM Studio (local, no API key):**

1. Download [LM Studio](https://lmstudio.ai), load a model, start the local server (default: port 1234)
2. Add to your `.env` file:
```
OPENAI_API_BASE=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio
EPSTEIN_LLM_MODEL=openai/your-model-name
```
3. Then just run:
```bash
epstein-search chat   # interactive mode, no flags needed
```

**Cloud LLM options (API key required):**
```bash
# Gemini (requires GEMINI_API_KEY)
epstein-search ask "Summarize the findings" --model gemini/gemini-3-flash-preview

# OpenAI (requires OPENAI_API_KEY)
epstein-search ask "Timeline of events" --model gpt-4o

# Anthropic (requires ANTHROPIC_API_KEY)
epstein-search ask "Financial connections" --model anthropic/claude-sonnet-4-20250514
```

```bash
# Show source documents alongside the answer
epstein-search ask "Flight patterns" --show-sources
```

## How It Works

```
pip install epstein-search
         ↓
epstein-search setup
  → Downloads 100K+ pre-computed embeddings (all-MiniLM-L6-v2)
  → Imports into local zvec vector database
         ↓
epstein-search search "your query"
  → Embeds query locally (sentence-transformers, no API key)
  → Vector similarity search via zvec
  → Returns matching document chunks
```

## Dataset

Embeddings: [devankit7873/EpsteinFiles-Vector-Embeddings-ChromaDB](https://huggingface.co/datasets/devankit7873/EpsteinFiles-Vector-Embeddings-ChromaDB) — 100K+ chunks, 384-dim vectors, based on the [Epstein Files 20K](https://huggingface.co/datasets/teyler/epstein-search-20k) corpus.

All content is from publicly available sources:
- U.S. Department of Justice Epstein Library
- House Oversight Committee releases
- Unsealed federal court documents
- FBI reports and DOJ publications

## License

MIT — see [LICENSE](LICENSE)
