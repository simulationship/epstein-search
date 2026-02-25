"""Embedding module — handles text-to-vector conversion.

Uses sentence-transformers/all-MiniLM-L6-v2 (local, free, no API key).
Same model used by the pre-made ChromaDB embeddings, so vectors are compatible.
"""

from typing import Optional

from . import config

# Lazy-loaded model singleton
_model = None


def _get_model():
    """Load the sentence-transformers model (lazy, cached)."""
    global _model
    if _model is None:
        import os
        import sys
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required.\n"
                "Install with: pip install sentence-transformers"
            )
        # Suppress BertModel LOAD REPORT (printed by C extension, bypasses Python)
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        try:
            _model = SentenceTransformer(config.EMBEDDING_MODEL)
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(devnull)
            os.close(old_stdout)
            os.close(old_stderr)
    return _model


def embed_texts(
    texts: list[str],
    batch_size: int = config.EMBEDDING_BATCH_SIZE,
    show_progress: bool = False,
) -> list[list[float]]:
    """Embed a list of texts using the local sentence-transformers model.

    Args:
        texts: List of text strings to embed.
        batch_size: Number of texts per batch.
        show_progress: Show progress bar during embedding.

    Returns:
        List of embedding vectors (list of floats).
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single search query.

    Uses the same model as document embedding for consistent results.
    Runs locally — no API key needed.
    """
    results = embed_texts([query])
    return results[0]
