"""Ingest pipeline — stream HF dataset, embed, store in zvec.

This is the MAINTAINER-SIDE script. Run once to build the index.
End users download the pre-built index instead.

Optimized for legal document search:
- Paragraph-aware chunking (preserves legal context boundaries)
- Court document artifact cleaning (headers, footers, redactions)
- Rich metadata (source, page_num, doc_type, chunk_index)
"""

import json
import re
from pathlib import Path
from typing import Optional

import zvec
from datasets import load_dataset
from tqdm import tqdm

from . import config
from .embeddings import embed_texts


# --- Legal document type detection ---

DOC_TYPE_PATTERNS = {
    "court_filing": [
        r"(?i)united states district court",
        r"(?i)case no\.",
        r"(?i)plaintiff.*v[s]?\..*defendant",
        r"(?i)court of appeals",
        r"(?i)motion to",
        r"(?i)order granting",
        r"(?i)docket no\.",
    ],
    "deposition": [
        r"(?i)deposition of",
        r"(?i)q\.\s+.*\n\s*a\.\s+",
        r"(?i)direct examination",
        r"(?i)cross.?examination",
        r"(?i)the witness",
    ],
    "fbi_report": [
        r"(?i)federal bureau of investigation",
        r"(?i)fbi\b",
        r"(?i)field office",
        r"(?i)case id",
        r"(?i)investigat(ion|ive|ing)",
    ],
    "flight_log": [
        r"(?i)flight log",
        r"(?i)passenger",
        r"(?i)tail number",
        r"(?i)aircraft",
        r"(?i)departure.*arrival",
    ],
    "financial": [
        r"(?i)wire transfer",
        r"(?i)bank statement",
        r"(?i)account number",
        r"(?i)transaction",
        r"(?i)invoice",
    ],
}


def detect_doc_type(text: str) -> str:
    """Detect legal document type from content patterns."""
    scores = {}
    for doc_type, patterns in DOC_TYPE_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, text[:3000]))
        if score > 0:
            scores[doc_type] = score

    if scores:
        return max(scores, key=scores.get)
    return "other"


# --- Legal-specific text cleaning ---

def clean_text(text: str) -> str:
    """Clean extracted PDF text with legal-document-specific handling."""
    if not text:
        return ""

    # Remove PDF artifacts and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

    # Remove repeated page headers/footers (common in court docs)
    # Pattern: "Page X of Y" or "X" at start/end of lines
    text = re.sub(r'\n\s*Page \d+ of \d+\s*\n', '\n', text)
    text = re.sub(r'\n\s*-\s*\d+\s*-\s*\n', '\n', text)

    # Clean up court document line numbers (left margin numbers)
    text = re.sub(r'^\s*\d{1,2}\s{2,}', '', text, flags=re.MULTILINE)

    # Normalize redaction markers
    text = re.sub(r'\[REDACTED\]|\[SEALED\]|\*{3,}|_{5,}', '[REDACTED]', text)

    # Remove excessive whitespace while preserving paragraph breaks
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r' {3,}', ' ', text)
    text = re.sub(r'\t+', ' ', text)

    # Remove Bates stamp numbers (common in legal discovery)
    text = re.sub(r'\b[A-Z]{2,6}-?\d{4,8}\b', '', text)

    return text.strip()


# --- Paragraph-aware chunking ---

def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks, respecting paragraph boundaries.

    Legal documents have meaningful paragraph structure — splitting mid-paragraph
    loses legal context. This chunker tries to break at paragraph boundaries,
    falling back to sentence boundaries, then word boundaries.
    """
    if not text or not text.strip():
        return []

    # Split into paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return []

    # Estimate tokens (~0.75 words per token)
    def est_tokens(t: str) -> int:
        return int(len(t.split()) / 0.75)

    # If entire text fits in one chunk, return as-is
    if est_tokens(text) <= chunk_size:
        return [text.strip()]

    chunks = []
    current_chunk_parts = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = est_tokens(para)

        # If single paragraph exceeds chunk size, split it by sentences
        if para_tokens > chunk_size:
            # Flush current chunk first
            if current_chunk_parts:
                chunks.append("\n\n".join(current_chunk_parts))
                current_chunk_parts = []
                current_tokens = 0

            # Split long paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sent_chunk_parts = []
            sent_tokens = 0

            for sent in sentences:
                s_tokens = est_tokens(sent)
                if sent_tokens + s_tokens > chunk_size and sent_chunk_parts:
                    chunks.append(" ".join(sent_chunk_parts))
                    # Keep overlap sentences
                    overlap_tokens = 0
                    overlap_parts = []
                    for s in reversed(sent_chunk_parts):
                        overlap_tokens += est_tokens(s)
                        if overlap_tokens >= overlap:
                            break
                        overlap_parts.insert(0, s)
                    sent_chunk_parts = overlap_parts
                    sent_tokens = sum(est_tokens(s) for s in sent_chunk_parts)

                sent_chunk_parts.append(sent)
                sent_tokens += s_tokens

            if sent_chunk_parts:
                chunks.append(" ".join(sent_chunk_parts))
            continue

        # Would adding this paragraph exceed chunk size?
        if current_tokens + para_tokens > chunk_size and current_chunk_parts:
            chunks.append("\n\n".join(current_chunk_parts))

            # Keep last paragraph(s) for overlap
            overlap_tokens = 0
            overlap_parts = []
            for p in reversed(current_chunk_parts):
                overlap_tokens += est_tokens(p)
                if overlap_tokens >= overlap:
                    break
                overlap_parts.insert(0, p)
            current_chunk_parts = overlap_parts
            current_tokens = sum(est_tokens(p) for p in current_chunk_parts)

        current_chunk_parts.append(para)
        current_tokens += para_tokens

    # Don't forget the last chunk
    if current_chunk_parts:
        chunks.append("\n\n".join(current_chunk_parts))

    # Filter out tiny chunks (< 20 chars)
    chunks = [c for c in chunks if len(c) >= 20]

    return chunks


# --- zvec collection ---

def create_collection(db_path: Path = config.ZVEC_DB_PATH) -> zvec.Collection:
    """Create or open a zvec collection with legal-document schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    schema = zvec.CollectionSchema(
        name=config.COLLECTION_NAME,
        vectors=zvec.VectorSchema(
            "embedding",
            zvec.DataType.VECTOR_FP32,
            config.EMBEDDING_DIMENSIONS,
        ),
        fields=[
            zvec.FieldSchema("text", zvec.DataType.STRING),
            zvec.FieldSchema("source", zvec.DataType.STRING),
            zvec.FieldSchema("doc_type", zvec.DataType.STRING),
            zvec.FieldSchema("page_num", zvec.DataType.INT64),
            zvec.FieldSchema("chunk_index", zvec.DataType.INT64),
        ],
    )

    return zvec.create_and_open(path=str(db_path), schema=schema)


# --- Main ingest pipeline ---

def ingest(
    db_path: Path = config.ZVEC_DB_PATH,
    max_rows: Optional[int] = None,
    batch_size: int = 50,
    resume_from: int = 0,
) -> dict:
    """Run the full ingest pipeline.

    Args:
        db_path: Path to store the zvec database.
        max_rows: Optional limit on rows to process (for testing).
        batch_size: Documents to embed at once.
        resume_from: Row index to resume from (for crash recovery).

    Returns:
        dict with stats: total_rows, total_chunks, total_errors, doc_types.
    """
    collection = create_collection(db_path)

    # Stream the dataset — no full download needed
    dataset = load_dataset(config.HF_DATASET_NAME, split="train", streaming=True)

    stats = {
        "total_rows": 0,
        "total_chunks": 0,
        "total_errors": 0,
        "total_skipped": 0,
        "doc_types": {},
    }
    pending_docs = []
    pending_texts = []

    progress = tqdm(desc="Ingesting", unit=" rows")

    for row_idx, row in enumerate(dataset):
        if row_idx < resume_from:
            continue
        if max_rows is not None and stats["total_rows"] >= max_rows:
            break

        stats["total_rows"] += 1
        progress.update(1)

        # Extract text from the row
        text = row.get("text", "") or ""
        text = clean_text(text)
        if not text or len(text) < 50:  # Skip empty/tiny entries
            stats["total_skipped"] += 1
            continue

        # Get metadata
        source = str(row.get("source", row.get("filename", f"row_{row_idx}")))
        page_num = int(row.get("page_num", row.get("page", 0)) or 0)

        # Detect document type
        doc_type = detect_doc_type(text)
        stats["doc_types"][doc_type] = stats["doc_types"].get(doc_type, 0) + 1

        # Chunk the text (paragraph-aware)
        chunks = chunk_text(text)

        for chunk_idx, chunk in enumerate(chunks):
            doc_id = f"{row_idx}_{chunk_idx}"
            pending_docs.append({
                "id": doc_id,
                "text": chunk,
                "source": source,
                "doc_type": doc_type,
                "page_num": page_num,
                "chunk_index": chunk_idx,
            })
            pending_texts.append(chunk)

            # Process batch when full
            if len(pending_texts) >= batch_size:
                _insert_batch(collection, pending_docs, pending_texts, stats)
                pending_docs = []
                pending_texts = []

    # Process remaining
    if pending_texts:
        _insert_batch(collection, pending_docs, pending_texts, stats)

    # Optimize for search performance
    progress.set_description("Optimizing index")
    collection.optimize()
    progress.close()

    # Save stats
    stats_path = db_path.parent / "ingest_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def _insert_batch(
    collection: zvec.Collection,
    docs: list[dict],
    texts: list[str],
    stats: dict,
) -> None:
    """Embed and insert a batch of documents."""
    try:
        embeddings = embed_texts(texts)

        zvec_docs = []
        for doc, embedding in zip(docs, embeddings):
            zvec_docs.append(
                zvec.Doc(
                    id=doc["id"],
                    vectors={"embedding": embedding},
                    fields={
                        "text": doc["text"],
                        "source": doc["source"],
                        "doc_type": doc["doc_type"],
                        "page_num": doc["page_num"],
                        "chunk_index": doc["chunk_index"],
                    },
                )
            )

        results = collection.insert(zvec_docs)
        stats["total_chunks"] += len(zvec_docs)

        # Count any insertion errors
        for r in results:
            if isinstance(r, dict) and r.get("code", 0) != 0:
                stats["total_errors"] += 1

    except Exception as e:
        stats["total_errors"] += len(docs)
        tqdm.write(f"Error inserting batch: {e}")
