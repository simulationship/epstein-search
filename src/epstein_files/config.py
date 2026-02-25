"""Configuration constants and default paths."""

import os
from pathlib import Path

# --- Paths ---
DEFAULT_DATA_DIR = Path(os.environ.get(
    "EPSTEIN_DATA_DIR",
    Path.home() / ".epstein-search"
))
INDEX_DIR = DEFAULT_DATA_DIR / "index"
CHROMA_DIR = DEFAULT_DATA_DIR / "chroma_db"

# --- Embedding ---
# Uses sentence-transformers/all-MiniLM-L6-v2 (same as pre-made ChromaDB embeddings)
# This runs 100% locally — NO API key needed for search!
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
EMBEDDING_BATCH_SIZE = 256  # Local model can handle larger batches

# --- zvec ---
COLLECTION_NAME = "epstein_docs"
ZVEC_DB_PATH = INDEX_DIR / "epstein.zvec"

# --- Pre-made embeddings source ---
CHROMA_HF_REPO = "devankit7873/EpsteinFiles-Vector-Embeddings-ChromaDB"

# --- Chunking (only used for fresh ingest, not for ChromaDB import) ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --- RAG ---
DEFAULT_LLM_MODEL = os.environ.get("EPSTEIN_LLM_MODEL", "gemini/gemini-3-flash-preview")
RAG_SYSTEM_PROMPT = """\
You are a legal research assistant analyzing the Epstein Files — a collection of \
publicly available court documents, FBI reports, depositions, flight logs, and DOJ \
publications related to Jeffrey Epstein.

Rules:
1. Answer ONLY based on the provided document excerpts. Do not use outside knowledge.
2. ALWAYS cite sources using the exact source filename in parentheses, e.g. (HOUSE_OVERSIGHT_010850.txt). Never use generic labels like "Document 1" or "Document 2".
3. Quote exact phrases from the documents when making claims.
4. Distinguish between allegations, testimony, and established facts.
5. Note any redactions or gaps in the provided documents.
6. If the documents don't contain enough information, say so clearly.
7. Use precise legal language where appropriate."""

DEFAULT_TOP_K = 10

# Valid document types for filtering
DOC_TYPES = [
    "court_filing",
    "deposition",
    "fbi_report",
    "flight_log",
    "financial",
    "other",
]
