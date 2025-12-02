#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p python312 python312Packages.chromadb python312Packages.sentence-transformers python312Packages.pyyaml

"""
Creative Writing Vector Search

Semantic search across indexed creative writing content.
"""

import argparse
from pathlib import Path

import yaml
import chromadb
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
    source_type: str | None = None,
    character: str | None = None,
    topic: str | None = None,
):
    """Search the vector database."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path=str(_db_path))

    try:
        collection = client.get_collection(_collection_name)
    except Exception:
        print(f"Error: No indexed data found at {_db_path}. Run index.py first.")
        return

    # Build where filter (only source_type uses ChromaDB filter)
    where = {"source_type": source_type} if source_type else None

    # Embed query and search
    # Fetch extra results if we need to post-filter by character/topic
    fetch_n = n_results * 5 if (character or topic) else n_results
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=fetch_n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # Post-filter for character/topic (ChromaDB doesn't support substring match)
    if character or topic:
        filtered = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
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
    if source_type:
        print(f"Filter: source_type={source_type}")
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
        print(f"[{i+1}] {meta['filename']} (chunk {meta['chunk_idx']})")
        print(f"    Type: {meta['source_type']} | Score: {score:.3f}")
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
    print("Commands: :quit, :type <type>, :char <character>, :topic <topic>, :clear")
    print()

    filters = {"source_type": None, "character": None, "topic": None}

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
        elif query.startswith(":type "):
            filters["source_type"] = query[6:].strip() or None
            print(f"Filter set: source_type={filters['source_type']}")
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
            filters = {"source_type": None, "character": None, "topic": None}
            print("Filters cleared.")
            continue

        search(query, **filters)


def main():
    parser = argparse.ArgumentParser(description="Search indexed content")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--root", metavar="DIR", help="Root directory with vector_db (default: fairy_project)")
    parser.add_argument("--collection", metavar="NAME", help="Collection name (default: directory name)")
    parser.add_argument("-n", "--num", type=int, default=5, help="Number of results")
    parser.add_argument("-t", "--type", dest="source_type", help="Filter by source type")
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
            source_type=args.source_type,
            character=args.character,
            topic=args.topic,
        )


if __name__ == "__main__":
    main()
