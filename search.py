#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p python312 python312Packages.lancedb python312Packages.sentence-transformers python312Packages.pyyaml python312Packages.pyarrow python312Packages.pandas

"""
Creative Writing Vector Search

Semantic search across indexed creative writing content.
"""

import argparse
from pathlib import Path

import yaml
import lancedb
from sentence_transformers import SentenceTransformer

TOOLKIT_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = TOOLKIT_DIR / "config.yaml"


def load_config():
    """Load configuration from config.yaml."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def get_default_content_root():
    """Get content root from config, resolved relative to toolkit directory."""
    config = load_config()
    content_root = config.get("content_root", "../fairy_project")
    return (TOOLKIT_DIR / content_root).resolve()


# Module-level config
_content_root = None
_db_path = None
_table_name = None


def configure(root: str | None = None, table: str | None = None):
    """Configure search paths."""
    global _content_root, _db_path, _table_name
    if root:
        _content_root = Path(root).resolve()
    else:
        _content_root = get_default_content_root()
    _db_path = _content_root / "vector_db"
    _table_name = table or _content_root.name


def search(
    query: str,
    n_results: int = 5,
    folder: str | None = None,
    character: str | None = None,
    topic: str | None = None,
):
    """Search the vector database."""
    model = SentenceTransformer('all-MiniLM-L6-v2')

    try:
        db = lancedb.connect(str(_db_path))
        table = db.open_table(_table_name)
    except Exception as e:
        print(f"Error: Could not open database at {_db_path}. Run index.py first.")
        print(f"  ({e})")
        return

    # Embed query
    query_embedding = model.encode([query])[0].tolist()

    # Build search with optional SQL-style filter
    filters = []
    if folder:
        filters.append(f"folders LIKE '%{folder}%'")
    if character:
        filters.append(f"characters LIKE '%{character}%'")
    if topic:
        filters.append(f"topics LIKE '%{topic}%'")

    where_clause = " AND ".join(filters) if filters else None

    # Execute search
    search_query = table.search(query_embedding)
    if where_clause:
        search_query = search_query.where(where_clause)

    results = search_query.limit(n_results).to_list()

    # Display results
    if not results:
        print("No results found.")
        return

    print(f"\n{'='*60}")
    print(f"Search: \"{query}\"")
    if folder:
        print(f"Filter: folder={folder}")
    if character:
        print(f"Filter: character={character}")
    if topic:
        print(f"Filter: topic={topic}")
    print(f"{'='*60}\n")

    for i, row in enumerate(results):
        # LanceDB returns distance in _distance field
        dist = row.get('_distance', 0)
        score = 1 - dist if dist < 1 else 1 / (1 + dist)  # Convert to similarity

        folders = row.get('folders', '')
        folder_display = folders if folders else "root"

        print(f"[{i+1}] {row['filename']} (chunk {row['chunk_idx']})")
        print(f"    Type: {folder_display} | Score: {score:.3f}")
        if row.get('characters'):
            print(f"    Characters: {row['characters']}")
        if row.get('topics'):
            print(f"    Topics: {row['topics']}")
        print(f"    ---")
        # Show preview (first 300 chars)
        text = row.get('text', '')
        preview = text[:300].replace('\n', ' ')
        if len(text) > 300:
            preview += "..."
        print(f"    {preview}")
        print()


def interactive_mode():
    """Interactive search REPL."""
    print("Fairy Project Search (LanceDB)")
    print("Commands: :quit, :folder <name>, :char <character>, :topic <topic>, :clear")
    print()

    filters = {"folder": None, "character": None, "topic": None}

    while True:
        try:
            query = input("search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue

        if query == ":quit":
            break
        elif query.startswith(":folder "):
            filters["folder"] = query[8:].strip() or None
            print(f"Filter set: folder={filters['folder']}")
            continue
        elif query.startswith(":char "):
            filters["character"] = query[6:].strip() or None
            print(f"Filter set: character={filters['character']}")
            continue
        elif query.startswith(":topic "):
            filters["topic"] = query[7:].strip() or None
            print(f"Filter set: topic={filters['topic']}")
            continue
        elif query == ":clear":
            filters = {"folder": None, "character": None, "topic": None}
            print("Filters cleared.")
            continue

        search(query, **filters)


def main():
    parser = argparse.ArgumentParser(description="Search indexed content")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--root", metavar="DIR", help="Root directory with vector_db (default: fairy_project)")
    parser.add_argument("--table", metavar="NAME", help="Table name (default: directory name)")
    parser.add_argument("-n", "--num", type=int, default=5, help="Number of results")
    parser.add_argument("-t", "--folder", dest="folder", help="Filter by folder (e.g., stories, worldbuilding, non-canon)")
    parser.add_argument("-c", "--char", dest="character", help="Filter by character")
    parser.add_argument("--topic", help="Filter by topic")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    # Configure paths if --root specified
    configure(args.root, args.table)

    if args.interactive or not args.query:
        interactive_mode()
    else:
        search(
            args.query,
            n_results=args.num,
            folder=args.folder,
            character=args.character,
            topic=args.topic,
        )


if __name__ == "__main__":
    main()
