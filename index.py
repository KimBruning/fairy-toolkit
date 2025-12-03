#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p python312 python312Packages.chromadb python312Packages.sentence-transformers python312Packages.pyyaml

"""
Creative Writing Vector DB Indexer

Indexes content into ChromaDB for semantic search.
Supports incremental updates - only indexes new/modified files.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Generator

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


# Known characters for auto-tagging (customize in config.yaml)
CHARACTERS = [
    "Gwyneth", "Caoimhe", "Caiomhe", "Eirwen", "Rhia", "Eilis", "Brigid",
    "Underhill", "Sylvarum", "Marion", "Selenis"
]

# Topic keywords for auto-tagging
TOPIC_KEYWORDS = {
    "magic": ["magic", "spell", "geas", "enchant", "ward"],
    "flight": ["flight", "flying", "wings", "hover", "glide"],
    "politics": ["baron", "domain", "council", "treaty", "ambassador"],
    "physiology": ["metabolism", "vision", "anatomy", "scale", "size"],
    "shadow-walking": ["shadow", "realm", "portal", "dimension"],
    "family": ["mother", "sister", "cousin", "grandmother", "aunt", "family"],
}


def extract_characters(text: str) -> list[str]:
    """Extract mentioned characters from text."""
    found = []
    text_lower = text.lower()
    for char in CHARACTERS:
        if char.lower() in text_lower:
            found.append(char)
    return found


def extract_topics(text: str) -> list[str]:
    """Extract topics based on keyword matching."""
    found = []
    text_lower = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(topic)
    return found


def get_folders(filepath: Path, root: Path) -> list[str]:
    """Get all folder names from file path as a list."""
    try:
        rel = filepath.relative_to(root)
        parts = rel.parts
    except ValueError:
        parts = filepath.parts

    # All folders (everything except the filename)
    if len(parts) > 1:
        return list(parts[:-1])
    return []


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> Generator[tuple[str, int], None, None]:
    """
    Split text into overlapping chunks.
    Yields (chunk_text, chunk_index).
    """
    if len(text) <= chunk_size:
        yield text, 0
        return

    start = 0
    chunk_idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at paragraph or sentence boundary
        if end < len(text):
            para_break = chunk.rfind('\n\n')
            if para_break > chunk_size // 2:
                chunk = chunk[:para_break]
                end = start + para_break
            else:
                sent_break = max(chunk.rfind('. '), chunk.rfind('.\n'))
                if sent_break > chunk_size // 2:
                    chunk = chunk[:sent_break + 1]
                    end = start + sent_break + 1

        yield chunk.strip(), chunk_idx
        chunk_idx += 1
        start = end - overlap


def collect_documents(root: Path, extensions: set = None) -> Generator[tuple[Path, str], None, None]:
    """Collect all indexable documents."""
    if extensions is None:
        extensions = {'.md', '.txt'}
    skip_dirs = {'py', 'vector_db', '__pycache__', '.git', 'node_modules'}

    for filepath in root.rglob('*'):
        if filepath.is_file() and filepath.suffix in extensions:
            if not any(skip in filepath.parts for skip in skip_dirs):
                try:
                    content = filepath.read_text(encoding='utf-8')
                    if content.strip():
                        yield filepath, content
                except Exception as e:
                    print(f"Warning: Could not read {filepath}: {e}")


def create_doc_id(filepath: Path, chunk_idx: int, root: Path) -> str:
    """Create a unique document ID based on relative path."""
    try:
        rel_path = filepath.relative_to(root)
    except ValueError:
        rel_path = filepath
    return hashlib.md5(f"{rel_path}:{chunk_idx}".encode()).hexdigest()


def file_hash(filepath: Path) -> str:
    """Get a hash representing file content."""
    stat = filepath.stat()
    return f"{stat.st_mtime}:{stat.st_size}"


def load_manifest(manifest_path: Path) -> dict:
    """Load the index manifest tracking indexed files."""
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {"files": {}, "version": 1}


def save_manifest(manifest: dict, manifest_path: Path):
    """Save the index manifest."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))


def get_files_to_index(root: Path, manifest: dict, force: bool = False) -> tuple[list[Path], list[Path], list[str]]:
    """
    Determine which files need indexing.
    Returns: (new_files, modified_files, deleted_file_keys)
    """
    current_files = {}
    for filepath, _ in collect_documents(root):
        try:
            rel = str(filepath.relative_to(root))
        except ValueError:
            rel = str(filepath)
        current_files[rel] = (filepath, file_hash(filepath))

    indexed = manifest.get("files", {})

    new_files = []
    modified_files = []
    deleted_keys = []

    # Find new and modified files
    for rel_path, (filepath, fhash) in current_files.items():
        if rel_path not in indexed:
            new_files.append(filepath)
        elif indexed[rel_path]["hash"] != fhash or force:
            modified_files.append(filepath)

    # Find deleted files
    for rel_path in indexed:
        if rel_path not in current_files:
            deleted_keys.append(rel_path)

    return new_files, modified_files, deleted_keys


def index_files(
    files: list[Path],
    collection,
    model: SentenceTransformer,
    manifest: dict,
    root: Path,
    verbose: bool = True,
    batch_size: int = 500
):
    """Index a list of files into the collection."""
    if not files:
        return 0

    docs_to_index = []

    for filepath in files:
        try:
            content = filepath.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")
            continue

        try:
            rel_path = str(filepath.relative_to(root))
        except ValueError:
            rel_path = str(filepath)

        folders = get_folders(filepath, root)
        chunk_ids = []

        for chunk, chunk_idx in chunk_text(content):
            doc_id = create_doc_id(filepath, chunk_idx, root)
            characters = extract_characters(chunk)
            topics = extract_topics(chunk)
            chunk_ids.append(doc_id)

            docs_to_index.append({
                "id": doc_id,
                "text": chunk,
                "metadata": {
                    "filename": rel_path,
                    "folders": ",".join(folders) if folders else "",
                    "characters": ",".join(characters) if characters else "",
                    "topics": ",".join(topics) if topics else "",
                    "chunk_idx": chunk_idx,
                }
            })

        # Update manifest
        manifest["files"][rel_path] = {
            "hash": file_hash(filepath),
            "chunk_ids": chunk_ids,
        }

    if not docs_to_index:
        return 0

    # Batch embed and add
    for i in range(0, len(docs_to_index), batch_size):
        batch = docs_to_index[i:i + batch_size]
        texts = [d["text"] for d in batch]
        embeddings = model.encode(texts).tolist()

        collection.add(
            ids=[d["id"] for d in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[d["metadata"] for d in batch],
        )
        if verbose:
            print(f"  Indexed {min(i + batch_size, len(docs_to_index))}/{len(docs_to_index)} chunks")

    return len(docs_to_index)


def remove_file_chunks(rel_path: str, collection, manifest: dict):
    """Remove all chunks for a file from the collection."""
    file_info = manifest["files"].get(rel_path)
    if file_info and file_info.get("chunk_ids"):
        try:
            collection.delete(ids=file_info["chunk_ids"])
        except Exception:
            pass  # Chunks might not exist
    if rel_path in manifest["files"]:
        del manifest["files"][rel_path]


def index_incremental(
    root: Path,
    db_path: Path,
    manifest_path: Path,
    collection_name: str,
    verbose: bool = True,
    batch_size: int = 500
):
    """Incremental index - only process new/modified files."""
    if verbose:
        print("Initializing embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    if verbose:
        print(f"Setting up ChromaDB at {db_path}...")
    db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_path), settings=CHROMA_SETTINGS)
    manifest = load_manifest(manifest_path)

    # Get or create collection
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": f"Indexed content from {root.name}"}
        )

    # Find what needs indexing
    new_files, modified_files, deleted_keys = get_files_to_index(root, manifest)

    if not new_files and not modified_files and not deleted_keys:
        if verbose:
            print("Index is up to date. No changes detected.")
        return

    if verbose:
        if new_files:
            print(f"New files: {len(new_files)}")
        if modified_files:
            print(f"Modified files: {len(modified_files)}")
        if deleted_keys:
            print(f"Deleted files: {len(deleted_keys)}")

    # Handle deletions and modifications (remove old chunks)
    for rel_path in deleted_keys:
        remove_file_chunks(rel_path, collection, manifest)

    for filepath in modified_files:
        try:
            rel_path = str(filepath.relative_to(root))
        except ValueError:
            rel_path = str(filepath)
        remove_file_chunks(rel_path, collection, manifest)

    # Index new and modified files
    all_files = new_files + modified_files
    if all_files:
        if verbose:
            print(f"Indexing {len(all_files)} files...")
        chunk_count = index_files(all_files, collection, model, manifest, root, verbose, batch_size)
        if verbose:
            print(f"Indexed {chunk_count} chunks from {len(all_files)} files.")

    save_manifest(manifest, manifest_path)
    if verbose:
        print("Manifest saved.")


def index_rebuild(
    root: Path,
    db_path: Path,
    manifest_path: Path,
    collection_name: str,
    verbose: bool = True,
    batch_size: int = 500
):
    """Full rebuild - delete and recreate index."""
    if verbose:
        print("Initializing embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    if verbose:
        print(f"Setting up ChromaDB at {db_path}...")
    db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_path), settings=CHROMA_SETTINGS)

    # Delete existing collection
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": f"Indexed content from {root.name}"}
    )

    # Fresh manifest
    manifest = {"files": {}, "version": 1}

    # Collect all files
    all_files = [fp for fp, _ in collect_documents(root)]

    if verbose:
        print(f"Indexing {len(all_files)} files...")

    chunk_count = index_files(all_files, collection, model, manifest, root, verbose, batch_size)
    save_manifest(manifest, manifest_path)

    if verbose:
        print(f"\nDone! Indexed {chunk_count} chunks from {len(all_files)} files.")
        print(f"Database stored at: {db_path}")


def index_add(
    paths: list[str],
    root: Path,
    db_path: Path,
    manifest_path: Path,
    collection_name: str,
    verbose: bool = True,
    batch_size: int = 500
):
    """Add specific files or directories to the index."""
    if verbose:
        print("Initializing embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path), settings=CHROMA_SETTINGS)
    manifest = load_manifest(manifest_path)

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": f"Indexed content from {root.name}"}
        )

    files_to_add = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"Warning: {p} does not exist, skipping")
            continue

        if path.is_file():
            files_to_add.append(path)
        elif path.is_dir():
            for fp, _ in collect_documents(path):
                files_to_add.append(fp)

    if not files_to_add:
        print("No files to add.")
        return

    # Remove existing chunks for these files (in case of update)
    for filepath in files_to_add:
        try:
            rel_path = str(filepath.relative_to(root))
        except ValueError:
            rel_path = str(filepath)
        if rel_path in manifest["files"]:
            remove_file_chunks(rel_path, collection, manifest)

    if verbose:
        print(f"Adding {len(files_to_add)} files...")

    chunk_count = index_files(files_to_add, collection, model, manifest, root, verbose, batch_size)
    save_manifest(manifest, manifest_path)

    if verbose:
        print(f"Added {chunk_count} chunks from {len(files_to_add)} files.")


def show_status(manifest_path: Path, db_path: Path):
    """Show index status."""
    manifest = load_manifest(manifest_path)
    files = manifest.get("files", {})

    total_chunks = sum(len(f.get("chunk_ids", [])) for f in files.values())

    print(f"Index location: {db_path}")
    print(f"Files indexed: {len(files)}")
    print(f"Total chunks: {total_chunks}")

    if files:
        print("\nIndexed files:")
        for rel_path in sorted(files.keys()):
            chunk_count = len(files[rel_path].get("chunk_ids", []))
            print(f"  {rel_path} ({chunk_count} chunks)")


def get_paths(root: Path | None, collection: str | None) -> tuple[Path, Path, Path, str]:
    """Get configured paths based on root/collection args."""
    if root:
        root_path = Path(root).resolve()
    else:
        root_path = get_default_content_root()

    coll_name = collection or root_path.name
    db_path = root_path / "vector_db"
    manifest_path = db_path / "manifest.json"
    return root_path, db_path, manifest_path, coll_name


def main():
    parser = argparse.ArgumentParser(
        description="Index content into ChromaDB for semantic search"
    )
    parser.add_argument(
        "--root", metavar="DIR",
        help="Root directory to index (default: fairy_project)"
    )
    parser.add_argument(
        "--collection", metavar="NAME",
        help="Collection name (default: directory name)"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force full rebuild of the index"
    )
    parser.add_argument(
        "--add", nargs="+", metavar="PATH",
        help="Add specific files or directories to the index"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show index status"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be indexed without actually indexing"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--batch-size", type=int, default=500, metavar="N",
        help="Batch size for DB writes (default: 500, larger = fewer fsyncs)"
    )
    args = parser.parse_args()

    verbose = not args.quiet
    root_path, db_path, manifest_path, coll_name = get_paths(args.root, args.collection)

    if args.status:
        show_status(manifest_path, db_path)
    elif args.dry_run:
        manifest = load_manifest(manifest_path)
        new_files, modified_files, deleted_keys = get_files_to_index(root_path, manifest)
        print(f"Root: {root_path}")
        print(f"Collection: {coll_name}")
        print(f"\nNew files ({len(new_files)}):")
        for f in new_files:
            try:
                print(f"  + {f.relative_to(root_path)}")
            except ValueError:
                print(f"  + {f}")
        print(f"\nModified files ({len(modified_files)}):")
        for f in modified_files:
            try:
                print(f"  ~ {f.relative_to(root_path)}")
            except ValueError:
                print(f"  ~ {f}")
        print(f"\nDeleted files ({len(deleted_keys)}):")
        for k in deleted_keys:
            print(f"  - {k}")
        if not new_files and not modified_files and not deleted_keys:
            print("\nIndex is up to date.")
    elif args.add:
        index_add(args.add, root=root_path, db_path=db_path, manifest_path=manifest_path,
                  collection_name=coll_name, verbose=verbose, batch_size=args.batch_size)
    elif args.rebuild:
        index_rebuild(root=root_path, db_path=db_path, manifest_path=manifest_path,
                      collection_name=coll_name, verbose=verbose, batch_size=args.batch_size)
    else:
        # Default: incremental update
        index_incremental(root=root_path, db_path=db_path, manifest_path=manifest_path,
                          collection_name=coll_name, verbose=verbose, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
