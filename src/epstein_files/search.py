"""Search module — query the zvec index for relevant Epstein Files documents.

Optimized for legal document search:
- Document type filtering (court filings, depositions, FBI reports, etc.)
- Source document filtering
- Score-based result re-ranking with deduplication

No API key needed — uses local sentence-transformers for query embedding.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import zvec

from . import config
from .embeddings import embed_query


@dataclass
class SearchResult:
    """A single search result."""
    text: str
    source: str
    score: float
    chunk_index: int
    doc_id: str
    doc_type: str = "other"
    page_num: int = 0


def open_collection(db_path: Path = config.ZVEC_DB_PATH) -> zvec.Collection:
    """Open an existing zvec collection."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"Index not found at {db_path}.\n"
            "Run 'epstein-search setup' to download and import the index,\n"
            "or 'epstein-search ingest' to build from scratch."
        )
    return zvec.open(path=str(db_path))


def search(
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    source_filter: Optional[str] = None,
    doc_type_filter: Optional[str] = None,
    db_path: Path = config.ZVEC_DB_PATH,
    deduplicate: bool = True,
) -> list[SearchResult]:
    """Search the Epstein Files index.

    No API key needed — query embedding runs locally.

    Args:
        query: Natural language search query.
        top_k: Number of results to return.
        source_filter: Optional filter by source document name (substring match).
        doc_type_filter: Optional filter by doc type (court_filing, deposition, etc).
        db_path: Path to the zvec database.
        deduplicate: Remove near-duplicate chunks from same source.

    Returns:
        List of SearchResult objects sorted by relevance.
    """
    collection = open_collection(db_path)

    # Embed the query locally (no API key needed!)
    query_vector = embed_query(query)

    # Build the vector query
    vector_query = zvec.VectorQuery("embedding", vector=query_vector)

    # Build filter string
    filters = []
    if source_filter:
        filters.append(f"source LIKE '%{source_filter}%'")
    if doc_type_filter:
        filters.append(f"doc_type = '{doc_type_filter}'")

    filter_str = " AND ".join(filters) if filters else None

    # Fetch more results than needed for deduplication
    fetch_k = top_k * 3 if deduplicate else top_k

    # Execute search
    if filter_str:
        results = collection.query(vector_query, topk=fetch_k, filter=filter_str)
    else:
        results = collection.query(vector_query, topk=fetch_k)

    # Convert to SearchResult objects
    search_results = []
    for r in results:
        fields = r.fields if hasattr(r, 'fields') else {}
        search_results.append(
            SearchResult(
                text=fields.get("text", ""),
                source=fields.get("source", "unknown"),
                score=r.score if hasattr(r, 'score') else 0.0,
                chunk_index=fields.get("chunk_index", 0),
                doc_id=r.id if hasattr(r, 'id') else "",
                doc_type=fields.get("doc_type", "other"),
                page_num=fields.get("page_num", 0),
            )
        )

    # Deduplicate: remove chunks from the same source that are very similar
    if deduplicate and len(search_results) > top_k:
        search_results = _deduplicate(search_results, top_k)
    else:
        search_results = search_results[:top_k]

    return search_results


def _deduplicate(results: list[SearchResult], top_k: int) -> list[SearchResult]:
    """Remove near-duplicate results from same source."""
    seen = set()
    deduped = []

    for r in results:
        content_key = (r.source, r.text[:100])
        if content_key in seen:
            continue
        seen.add(content_key)
        deduped.append(r)
        if len(deduped) >= top_k:
            break

    return deduped
