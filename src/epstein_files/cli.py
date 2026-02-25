"""CLI for epstein-search — semantic search over the Epstein Files."""

# Auto-load .env file if present (no error if python-dotenv not installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from . import __version__, config

console = Console()


@click.group()
@click.version_option(__version__, prog_name="epstein-search")
def cli():
    """🔍 Semantic search over the Epstein Files.

    Search publicly available court documents, FBI reports, and DOJ publications
    related to Jeffrey Epstein using AI-powered vector search.

    No API key needed for search. Only the 'ask' command (RAG) requires an LLM API key.
    """
    pass


@cli.command()
@click.option("--force", is_flag=True, help="Re-download even if already exists.")
def setup(force):
    """Download pre-made embeddings and build the search index.

    Downloads ~100K pre-computed vector embeddings from HuggingFace
    and imports them into a local zvec index. One-time setup.
    """
    from .chroma_import import download_chroma, import_from_chroma

    try:
        # Step 1: Download ChromaDB
        download_chroma(force=force)

        # Step 2: Import into zvec
        stats_file = config.ZVEC_DB_PATH.parent / "ingest_stats.json"
        if stats_file.exists() and not force:
            console.print(f"[green]✓[/green] zvec index already exists at {config.ZVEC_DB_PATH}")
        else:
            # Remove any empty/stale index
            import shutil
            if config.ZVEC_DB_PATH.exists():
                shutil.rmtree(config.ZVEC_DB_PATH)
            console.print("\n[bold]Importing into zvec index...[/bold]")
            stats = import_from_chroma()

            table = Table(title="Import Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Total docs", f"{stats['total_docs']:,}")
            table.add_row("Imported", f"{stats['total_imported']:,}")
            table.add_row("Skipped", f"{stats.get('total_skipped', 0):,}")
            table.add_row("Errors", str(stats["total_errors"]))
            if stats.get("doc_types"):
                for dt, count in sorted(stats["doc_types"].items(), key=lambda x: -x[1]):
                    table.add_row(f"  └ {dt}", f"{count:,}")
            console.print(table)

        console.print("\n[green bold]✓ Setup complete![/green bold]")
        console.print("Run [cyan]epstein-search search \"your query\"[/cyan] to start searching.")
        console.print("[dim]No API key needed for search![/dim]")

    except Exception as e:
        console.print(f"\n[red]Error during setup: {e}[/red]")
        raise SystemExit(1)


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=config.DEFAULT_TOP_K, help="Number of results.")
@click.option("--source", "-s", default=None, help="Filter by source document name.")
@click.option("--doc-type", "-d", default=None, type=click.Choice(config.DOC_TYPES), help="Filter by document type.")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON.")
def search(query, top_k, source, doc_type, json_output):
    """Search the Epstein Files for relevant documents.

    No API key needed — runs entirely locally.

    \b
    Examples:
      epstein-search search "flight logs to the island"
      epstein-search search "testimony" --doc-type deposition
      epstein-search search "financial records" --source "FBI"
    """
    from .search import search as do_search

    try:
        results = do_search(
            query=query,
            top_k=top_k,
            source_filter=source,
            doc_type_filter=doc_type,
        )
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    if json_output:
        import json
        output = [
            {
                "text": r.text,
                "source": r.source,
                "score": r.score,
                "doc_type": r.doc_type,
                "page_num": r.page_num,
                "chunk_index": r.chunk_index,
                "doc_id": r.doc_id,
            }
            for r in results
        ]
        click.echo(json.dumps(output, indent=2))
        return

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(f"\n[bold]Found {len(results)} results for:[/bold] {query}\n")

    for i, r in enumerate(results, 1):
        display_text = r.text[:300] + "..." if len(r.text) > 300 else r.text

        panel = Panel(
            display_text,
            title=f"[bold cyan]#{i}[/bold cyan] [dim]{r.source}[/dim]",
            subtitle=f"[dim]Score: {r.score:.4f} | {r.doc_type} | Chunk: {r.chunk_index}[/dim]",
            border_style="blue",
            padding=(1, 2),
        )
        console.print(panel)

    console.print("[dim]Found this useful? ☕ buymeacoffee.com/simulationship[/dim]")
    console.print("[dim]Crypto: SOL 76fCU6va3cGrbak4i9mwFfdYx1QsJJrqrxViFDUTsXUL · ETH/BASE 0xE808754a18A893A3eeFE01780D822C902680d1B7[/dim]")


@cli.command()
@click.argument("question")
@click.option("--model", "-m", default=config.DEFAULT_LLM_MODEL, help="LLM model (LiteLLM format).")
@click.option("--top-k", "-k", default=config.DEFAULT_TOP_K, help="Number of docs to retrieve.")
@click.option("--source", "-s", default=None, help="Filter by source document name.")
@click.option("--doc-type", "-d", default=None, type=click.Choice(config.DOC_TYPES), help="Filter by document type.")
@click.option("--show-sources", is_flag=True, help="Show source documents with the answer.")
def ask(question, model, top_k, source, doc_type, show_sources):
    """Ask a question about the Epstein Files (RAG).

    Search is free (local). Only the LLM answer needs an API key.

    \b
    Examples:
      epstein-search ask "Who visited the island most frequently?"
      epstein-search ask "What do the flight logs show?" --model gpt-4o
      epstein-search ask "Financial connections" --model ollama/llama3
    """
    from .rag import ask as do_ask

    with console.status("[bold cyan]Searching and generating answer...[/bold cyan]"):
        try:
            answer = do_ask(
                question=question,
                model=model,
                top_k=top_k,
                source_filter=source,
                doc_type_filter=doc_type,
            )
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            raise SystemExit(1)

    # Display the answer
    console.print()
    console.print(Panel(
        Markdown(answer.text),
        title="[bold green]Answer[/bold green]",
        subtitle=f"[dim]Model: {answer.model} | Sources: {len(answer.sources)}[/dim]",
        border_style="green",
        padding=(1, 2),
    ))

    if show_sources and answer.sources:
        console.print("\n[bold]📄 Source Documents:[/bold]\n")
        for i, r in enumerate(answer.sources, 1):
            display_text = r.text[:200] + "..." if len(r.text) > 200 else r.text
            console.print(f"  [cyan]{i}.[/cyan] [dim]{r.source}[/dim] ({r.doc_type}, score: {r.score:.3f})")
            console.print(f"     {display_text}\n")

    console.print("[dim]Found this useful? ☕ buymeacoffee.com/simulationship[/dim]")


@cli.command()
@click.option("--max-rows", "-n", default=None, type=int, help="Limit rows to process.")
@click.option("--resume-from", "-r", default=0, type=int, help="Resume from row index.")
@click.option("--batch-size", "-b", default=50, type=int, help="Embedding batch size.")
def ingest(max_rows, resume_from, batch_size):
    """Build the search index from scratch (advanced).

    Streams the full HuggingFace dataset and embeds locally.
    Most users should run 'epstein-search setup' instead.
    """
    from .ingest import ingest as do_ingest

    console.print("[bold]🔨 Building index from HuggingFace dataset...[/bold]")
    console.print(f"[dim]Dataset: {config.HF_DATASET_NAME}[/dim]")
    console.print(f"[dim]Embedding: {config.EMBEDDING_MODEL} ({config.EMBEDDING_DIMENSIONS}d)[/dim]")
    if max_rows:
        console.print(f"[dim]Limiting to {max_rows} rows[/dim]")
    console.print()

    try:
        stats = do_ingest(
            max_rows=max_rows,
            resume_from=resume_from,
            batch_size=batch_size,
        )
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    console.print(f"\n[green bold]✓ Ingest complete![/green bold]")

    table = Table(title="Ingest Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Rows processed", str(stats["total_rows"]))
    table.add_row("Chunks indexed", str(stats["total_chunks"]))
    table.add_row("Skipped (empty)", str(stats.get("total_skipped", 0)))
    table.add_row("Errors", str(stats["total_errors"]))
    if stats.get("doc_types"):
        for dt, count in sorted(stats["doc_types"].items(), key=lambda x: -x[1]):
            table.add_row(f"  └ {dt}", str(count))
    console.print(table)


@cli.command()
def info():
    """Show index information and configuration."""
    from .index import index_info

    info_data = index_info()

    table = Table(title="📊 epstein-search info")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Index path", info_data["path"])
    table.add_row("Index exists", "✓" if info_data["exists"] else "✗")
    if info_data.get("size_mb"):
        table.add_row("Index size", f"{info_data['size_mb']} MB")
    table.add_row("Embedding model", info_data["embedding_model"])
    table.add_row("Embedding dims", str(info_data["embedding_dims"]))
    table.add_row("Embeddings source", info_data.get("chroma_source", "N/A"))
    table.add_row("API key needed", "No (search) / Yes (ask)")

    if info_data.get("ingest_stats"):
        stats = info_data["ingest_stats"]
        table.add_row("Total docs", str(stats.get("total_docs", stats.get("total_rows", "?"))))
        table.add_row("Total imported", str(stats.get("total_imported", stats.get("total_chunks", "?"))))

    console.print(table)


DONATE_MSG = "[dim]☕ buymeacoffee.com/simulationship[/dim]\n[dim]Crypto: SOL 76fCU6va3cGrbak4i9mwFfdYx1QsJJrqrxViFDUTsXUL · ETH/BASE 0xE808754a18A893A3eeFE01780D822C902680d1B7[/dim]"


@cli.command()
@click.option("--model", "-m", default=config.DEFAULT_LLM_MODEL, help="LLM model (LiteLLM format).")
@click.option("--top-k", "-k", default=config.DEFAULT_TOP_K, help="Number of docs to retrieve.")
@click.option("--search-only", is_flag=True, help="Only search, don't generate answers.")
def chat(model, top_k, search_only):
    """Interactive mode — search and ask questions in a loop.

    \b
    Commands inside chat:
      /search    Switch to search-only mode
      /ask       Switch to ask (RAG) mode
      /model     Change the LLM model
      /info      Show current settings
      /quit      Exit

    \b
    Examples:
      epstein-search chat
      epstein-search chat --model openai/deepseek-r1-distill-qwen-1.5b
      epstein-search chat --search-only
    """
    from .search import search as do_search
    from .rag import ask as do_ask

    mode = "search" if search_only else "ask"

    console.print()
    console.print("[bold cyan]🔍 Epstein Files — Interactive Mode[/bold cyan]")
    console.print(f"[dim]Model: {model} | Mode: {mode} | Top-K: {top_k}[/dim]")
    console.print("[dim]Type a query, or /help for commands. /quit to exit.[/dim]")
    console.print()

    while True:
        try:
            query = console.input("[bold green]> [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query:
            continue

        # Handle slash commands
        if query.startswith("/"):
            cmd = query.lower().split()[0]
            if cmd in ("/quit", "/exit", "/q"):
                console.print("[dim]Goodbye![/dim]")
                break
            elif cmd == "/search":
                mode = "search"
                console.print("[cyan]Switched to search mode[/cyan]")
                continue
            elif cmd == "/ask":
                mode = "ask"
                console.print("[cyan]Switched to ask (RAG) mode[/cyan]")
                continue
            elif cmd == "/model":
                parts = query.split(maxsplit=1)
                if len(parts) > 1:
                    new_model = parts[1].strip()
                else:
                    console.print(f"[dim]Current model: {model}[/dim]")
                    try:
                        new_model = console.input("[dim]New model (or Enter to keep): [/dim]").strip()
                    except (EOFError, KeyboardInterrupt):
                        new_model = ""
                if new_model:
                    # Auto-prepend provider if missing (most local models use OpenAI-compatible API)
                    if "/" not in new_model:
                        new_model = f"openai/{new_model}"
                        console.print(f"[dim]Auto-detected as: {new_model}[/dim]")
                    model = new_model
                    console.print(f"[cyan]Model set to: {model}[/cyan]")
                continue
            elif cmd == "/topk":
                parts = query.split(maxsplit=1)
                if len(parts) > 1:
                    try:
                        top_k = int(parts[1].strip())
                        console.print(f"[cyan]Top-K set to: {top_k}[/cyan]")
                    except ValueError:
                        console.print("[red]Invalid number[/red]")
                else:
                    console.print(f"[dim]Current top-k: {top_k}[/dim]")
                continue
            elif cmd in ("/info", "/settings"):
                console.print(f"[dim]Mode: {mode} | Model: {model} | Top-K: {top_k}[/dim]")
                continue
            elif cmd in ("/help", "/?"):
                console.print("[dim]/search   — search only (no LLM)[/dim]")
                console.print("[dim]/ask      — search + LLM answer[/dim]")
                console.print("[dim]/model X  — change LLM model[/dim]")
                console.print("[dim]/topk N   — change number of results[/dim]")
                console.print("[dim]/info     — show current settings[/dim]")
                console.print("[dim]/quit     — exit[/dim]")
                continue
            else:
                console.print(f"[yellow]Unknown command: {cmd}. Type /help[/yellow]")
                continue

        # Execute query
        try:
            if mode == "search":
                results = do_search(query=query, top_k=top_k)
                if not results:
                    console.print("[yellow]No results found.[/yellow]\n")
                    continue
                console.print(f"\n[bold]{len(results)} results:[/bold]\n")
                for i, r in enumerate(results, 1):
                    text = r.text[:250] + "..." if len(r.text) > 250 else r.text
                    panel = Panel(
                        text,
                        title=f"[bold cyan]#{i}[/bold cyan] [dim]{r.source}[/dim]",
                        subtitle=f"[dim]{r.score:.4f} | {r.doc_type}[/dim]",
                        border_style="blue",
                        padding=(0, 1),
                    )
                    console.print(panel)
            else:
                with console.status("[cyan]Thinking...[/cyan]"):
                    answer = do_ask(question=query, model=model, top_k=top_k)
                console.print()
                console.print(Panel(
                    Markdown(answer.text),
                    title="[bold green]Answer[/bold green]",
                    subtitle=f"[dim]{answer.model} | {len(answer.sources)} sources[/dim]",
                    border_style="green",
                    padding=(1, 2),
                ))
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

        console.print(f"\n{DONATE_MSG}\n")


if __name__ == "__main__":
    cli()
