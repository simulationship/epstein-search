"""RAG module — search + LLM answer generation via LiteLLM.

Optimized for legal document analysis:
- Structured context with document metadata
- Legal-specific system prompt with citation requirements
- Source grouping for coherent answer generation
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import litellm

from . import config
from .search import SearchResult, search


@dataclass
class Answer:
    """A RAG-generated answer with source citations."""
    text: str
    model: str
    sources: list[SearchResult] = field(default_factory=list)


def ask(
    question: str,
    model: str = config.DEFAULT_LLM_MODEL,
    top_k: int = config.DEFAULT_TOP_K,
    source_filter: Optional[str] = None,
    doc_type_filter: Optional[str] = None,
    db_path: Path = config.ZVEC_DB_PATH,
) -> Answer:
    """Ask a question about the Epstein Files with RAG.

    Search is free (local embeddings). Only the LLM answer requires an API key.

    Args:
        question: Natural language question.
        model: LiteLLM model string (e.g., "gemini/gemini-2.0-flash", "gpt-4o").
        top_k: Number of document chunks to retrieve.
        source_filter: Optional source document filter.
        doc_type_filter: Optional document type filter.
        db_path: Path to the zvec database.

    Returns:
        Answer with generated text and source citations.
    """
    # Step 1: Retrieve relevant documents (no API key needed!)
    results = search(
        query=question,
        top_k=top_k,
        source_filter=source_filter,
        doc_type_filter=doc_type_filter,
        db_path=db_path,
    )

    if not results:
        return Answer(
            text="No relevant documents found for your question.",
            model=model,
            sources=[],
        )

    # Step 2: Build structured context, grouped by source
    context = _build_legal_context(results)

    # Step 3: Generate answer via LiteLLM (this is the only part needing an API key)
    messages = [
        {"role": "system", "content": config.RAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Based on the following document excerpts from the Epstein Files, "
                f"answer this question:\n\n"
                f"**Question:** {question}\n\n"
                f"**Retrieved Documents:**\n\n{context}"
            ),
        },
    ]

    response = litellm.completion(model=model, messages=messages)
    answer_text = response.choices[0].message.content

    return Answer(
        text=answer_text,
        model=model,
        sources=results,
    )


def _build_legal_context(results: list[SearchResult]) -> str:
    """Build structured context from search results, grouped by source."""
    source_groups: dict[str, list[SearchResult]] = {}
    for r in results:
        if r.source not in source_groups:
            source_groups[r.source] = []
        source_groups[r.source].append(r)

    parts = []
    for source, group in source_groups.items():
        group.sort(key=lambda x: x.chunk_index)

        doc_type_label = group[0].doc_type.replace("_", " ").title()
        header = f"═══ SOURCE: {source} (Type: {doc_type_label}) ═══"

        chunk_texts = []
        for r in group:
            page_info = f"[Page {r.page_num}] " if r.page_num > 0 else ""
            chunk_texts.append(f"[Source: {source}] {page_info}{r.text}")

        parts.append(f"{header}\n\n" + "\n\n[...]\n\n".join(chunk_texts))

    return "\n\n" + "\n\n".join(parts)
