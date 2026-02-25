"""Index download/management — fetch pre-built zvec index from HuggingFace."""

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from rich.console import Console
from rich.progress import Progress

from . import config

console = Console()


def download_index(
    repo_id: str = config.HF_INDEX_REPO,
    dest_dir: Path = config.INDEX_DIR,
    force: bool = False,
) -> Path:
    """Download the pre-built zvec index from HuggingFace.

    Args:
        repo_id: HuggingFace repo containing the index.
        dest_dir: Local directory to store the index.
        force: If True, re-download even if index exists.

    Returns:
        Path to the downloaded index directory.
    """
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    if config.ZVEC_DB_PATH.exists() and not force:
        console.print(
            f"[green]✓[/green] Index already exists at {config.ZVEC_DB_PATH}"
        )
        return dest_dir

    console.print(f"[bold]Downloading pre-built index from {repo_id}...[/bold]")
    console.print("[dim]This is a one-time download (~2GB)[/dim]\n")

    try:
        # Download all files from the repo
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest_dir),
        )
        console.print(f"\n[green]✓[/green] Index downloaded to {dest_dir}")
        return Path(downloaded_path)

    except Exception as e:
        console.print(f"\n[red]✗[/red] Download failed: {e}")
        console.print(
            "[dim]You can build the index locally with: "
            "epstein-search ingest[/dim]"
        )
        raise


def index_exists(db_path: Path = config.ZVEC_DB_PATH) -> bool:
    """Check if the zvec index exists locally."""
    return db_path.exists()


def index_info(db_path: Path = config.ZVEC_DB_PATH) -> dict:
    """Get information about the local index."""
    import zvec
    import json

    info = {
        "path": str(db_path),
        "exists": db_path.exists(),
    }

    if db_path.exists():
        # Get directory size
        total_size = sum(f.stat().st_size for f in db_path.rglob("*") if f.is_file())
        info["size_mb"] = round(total_size / (1024 * 1024), 1)

        # Check for ingest stats
        stats_path = db_path.parent / "ingest_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                info["ingest_stats"] = json.load(f)

    info["embedding_model"] = config.EMBEDDING_MODEL
    info["embedding_dims"] = config.EMBEDDING_DIMENSIONS
    info["hf_dataset"] = config.HF_DATASET_NAME
    info["chroma_source"] = config.CHROMA_HF_REPO

    return info


def delete_index(db_path: Path = config.ZVEC_DB_PATH) -> None:
    """Delete the local index."""
    if db_path.exists():
        shutil.rmtree(db_path)
        console.print(f"[yellow]Deleted index at {db_path}[/yellow]")
    else:
        console.print("[dim]No index to delete[/dim]")
