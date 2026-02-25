"""Import pre-made ChromaDB embeddings into zvec.

Downloads the pre-computed ChromaDB vector store from HuggingFace,
extracts all vectors + text + metadata, and imports into zvec.
This skips the expensive embedding step entirely.
"""

import json
from pathlib import Path
from typing import Optional

import zvec
from huggingface_hub import snapshot_download
from rich.console import Console
from tqdm import tqdm

from . import config
from .ingest import create_collection, detect_doc_type

console = Console()


def download_chroma(
    repo_id: str = config.CHROMA_HF_REPO,
    dest_dir: Path = config.CHROMA_DIR,
    force: bool = False,
) -> Path:
    """Download the pre-made ChromaDB from HuggingFace."""
    if dest_dir.exists() and not force:
        console.print(f"[green]✓[/green] ChromaDB already downloaded at {dest_dir}")
        return dest_dir

    console.print(f"[bold]Downloading pre-made ChromaDB from {repo_id}...[/bold]")

    downloaded_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest_dir),
    )

    console.print(f"[green]✓[/green] Downloaded to {dest_dir}")
    return Path(downloaded_path)


def import_from_chroma(
    chroma_dir: Path = config.CHROMA_DIR,
    zvec_path: Path = config.ZVEC_DB_PATH,
    batch_size: int = 500,
    max_docs: Optional[int] = None,
) -> dict:
    """Import ChromaDB vectors + metadata into zvec.

    Reads the ChromaDB persistence directory, extracts all embeddings
    and metadata, and inserts into a zvec collection.

    Args:
        chroma_dir: Path to ChromaDB persistence directory.
        zvec_path: Path to store the zvec database.
        batch_size: Documents to insert per batch.
        max_docs: Optional limit for testing.

    Returns:
        dict with import statistics.
    """
    try:
        import chromadb
    except ImportError:
        raise ImportError(
            "chromadb is required for import.\n"
            "Install with: pip install chromadb"
        )

    # Find the chroma_db subdirectory if it exists
    chroma_db_path = chroma_dir / "chroma_db"
    if not chroma_db_path.exists():
        chroma_db_path = chroma_dir

    # Open ChromaDB
    console.print(f"[dim]Opening ChromaDB at {chroma_db_path}...[/dim]")
    client = chromadb.PersistentClient(path=str(chroma_db_path))

    # Get the collection (should be the only one)
    collections = client.list_collections()
    if not collections:
        raise ValueError(f"No collections found in ChromaDB at {chroma_db_path}")

    collection_name = collections[0].name
    chroma_collection = client.get_collection(collection_name)
    total_count = chroma_collection.count()

    console.print(f"[green]Found collection '{collection_name}' with {total_count:,} documents[/green]")

    # Create zvec collection
    zvec_collection = create_collection(zvec_path)

    stats = {
        "total_docs": 0,
        "total_imported": 0,
        "total_errors": 0,
        "total_skipped": 0,
        "doc_types": {},
    }

    # Read from ChromaDB in batches
    limit = min(total_count, max_docs) if max_docs else total_count
    progress = tqdm(total=limit, desc="Importing", unit=" docs")

    offset = 0
    while offset < limit:
        current_batch = min(batch_size, limit - offset)

        # Get batch from ChromaDB with embeddings
        results = chroma_collection.get(
            limit=current_batch,
            offset=offset,
            include=["documents", "embeddings", "metadatas"],
        )

        ids = results.get("ids", [])
        documents = results.get("documents", [])
        embeddings = results.get("embeddings", [])
        metadatas = results.get("metadatas", [])

        if not ids:
            break

        zvec_docs = []
        for i, doc_id in enumerate(ids):
            text = documents[i] if documents is not None and i < len(documents) else ""
            embedding = embeddings[i] if embeddings is not None and i < len(embeddings) else None
            metadata = metadatas[i] if metadatas is not None and i < len(metadatas) else {}

            if embedding is None or not text:
                stats["total_skipped"] += 1
                continue

            # Convert numpy array to list if needed
            emb_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

            # Get source from metadata
            source = metadata.get("source", "unknown")

            # Detect document type from text content
            doc_type = detect_doc_type(text)
            stats["doc_types"][doc_type] = stats["doc_types"].get(doc_type, 0) + 1

            # Get chunk index from chunk_id if available
            chunk_id = metadata.get("chunk_id", "")
            try:
                chunk_index = int(chunk_id.split("_")[-1]) if chunk_id else 0
            except (ValueError, IndexError):
                chunk_index = 0

            zvec_docs.append(
                zvec.Doc(
                    id=str(doc_id),
                    vectors={"embedding": emb_list},
                    fields={
                        "text": text,
                        "source": source,
                        "doc_type": doc_type,
                        "page_num": int(metadata.get("page_num", 0)),
                        "chunk_index": chunk_index,
                    },
                )
            )

        # Insert batch into zvec
        if zvec_docs:
            try:
                insert_results = zvec_collection.insert(zvec_docs)
                stats["total_imported"] += len(zvec_docs)
                for r in insert_results:
                    if isinstance(r, dict) and r.get("code", 0) != 0:
                        stats["total_errors"] += 1
            except Exception as e:
                stats["total_errors"] += len(zvec_docs)
                tqdm.write(f"Error inserting batch: {e}")

        stats["total_docs"] += len(ids)
        offset += len(ids)
        progress.update(len(ids))

    # Optimize the zvec index
    progress.set_description("Optimizing zvec index")
    zvec_collection.optimize()
    progress.close()

    # Save stats
    stats_path = zvec_path.parent / "ingest_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats
