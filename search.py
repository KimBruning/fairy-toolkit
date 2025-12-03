#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p python312 python312Packages.chromadb python312Packages.sentence-transformers python312Packages.pyyaml

"""
Creative Writing Vector Search

Semantic search across indexed creative writing content.
"""

import argparse
from pathlib import Path

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"  # Disable ChromaDB telemetry (belt)

import yaml
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Suspenders
CHROMA_SETTINGS = Settings(anonymized_telemetry=False)

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


# Module-level state for configurable paths
_content_root = None
_db_path = None
_collection_name = None


def configure(root: Path | None = None, collection: str | None = None):
    """Configure database path and collection name."""
    global _content_root, _db_path, _collection_name

    if root:
        _content_root = Path(root).resolve()
    else:
        _content_root = get_default_content_root()

    _db_path = _content_root / "vector_db"
    _collection_name = collection or _content_root.name


def search(
    query: str,
    n_results: int = 5,
    folder: str | None = None,
    character: str | None = None,
    topic: str | None = None,
):
    """Search the vector database."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path=str(_db_path), settings=CHROMA_SETTINGS)

    try:
        collection = client.get_collection(_collection_name)
    except Exception:
        print(f"Error: No indexed data found at {_db_path}. Run index.py first.")
        return

    # Embed query and search
    # Fetch extra results if we need to post-filter
    needs_filter = folder or character or topic
    fetch_n = n_results * 5 if needs_filter else n_results
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"],
    )

    # Post-filter for folder/character/topic (substring match on comma-separated fields)
    if needs_filter:
        filtered = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if folder and folder.lower() not in meta.get("folders", "").lower():
                continue
            if character and character.lower() not in meta.get("characters", "").lower():
                continue
            if topic and topic.lower() not in meta.get("topics", "").lower():
                continue
            filtered["documents"][0].append(doc)
            filtered["metadatas"][0].append(meta)
            filtered["distances"][0].append(dist)
            if len(filtered["documents"][0]) >= n_results:
                break
        results = filtered

    # Display results
    if not results["documents"][0]:
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

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        score = 1 - dist  # Convert distance to similarity
        folders = meta.get('folders', '')
        folder_display = folders if folders else "root"
        print(f"[{i+1}] {meta['filename']} (chunk {meta['chunk_idx']})")
        print(f"    Type: {folder_display} | Score: {score:.3f}")
        if meta.get('characters'):
            print(f"    Characters: {meta['characters']}")
        if meta.get('topics'):
            print(f"    Topics: {meta['topics']}")
        print(f"    ---")
        # Show preview (first 300 chars)
        preview = doc[:300].replace('\n', ' ')
        if len(doc) > 300:
            preview += "..."
        print(f"    {preview}")
        print()


def interactive_mode():
    """Interactive search REPL."""
    print("Fairy Project Search")
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
    parser.add_argument("--collection", metavar="NAME", help="Collection name (default: directory name)")
    parser.add_argument("-n", "--num", type=int, default=5, help="Number of results")
    parser.add_argument("-t", "--folder", dest="folder", help="Filter by folder (e.g., stories, worldbuilding, non-canon)")
    parser.add_argument("-c", "--char", dest="character", help="Filter by character")
    parser.add_argument("--topic", help="Filter by topic")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    # Configure paths if --root specified
    configure(args.root, args.collection)

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
