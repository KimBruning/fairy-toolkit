#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p python312 python312Packages.lancedb python312Packages.sentence-transformers python312Packages.pyyaml python312Packages.pyarrow python312Packages.pandas

"""
Creative Writing Vector DB Indexer

Indexes content into LanceDB for semantic search.
Supports incremental updates - only indexes new/modified files.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Generator

import yaml
import lancedb
import pyarrow as pa
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


# Known characters for auto-tagging
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

    if len(parts) > 1:
        return list(parts[:-1])
    return []


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> Generator[tuple[str, int], None, None]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        yield text, 0
        return

    start = 0
    chunk_idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

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
    return {"files": {}, "version": 2}


def save_manifest(manifest: dict, manifest_path: Path):
    """Save the index manifest."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))


def get_files_to_index(root: Path, manifest: dict, force: bool = False) -> tuple[list[Path], list[Path], list[str]]:
    """Determine which files need indexing."""
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

    for rel_path, (filepath, fhash) in current_files.items():
        if rel_path not in indexed:
            new_files.append(filepath)
        elif indexed[rel_path]["hash"] != fhash or force:
            modified_files.append(filepath)

    for rel_path in indexed:
        if rel_path not in current_files:
            deleted_keys.append(rel_path)

    return new_files, modified_files, deleted_keys


def prepare_documents(
    files: list[Path],
    model: SentenceTransformer,
    manifest: dict,
    root: Path,
    verbose: bool = True
) -> list[dict]:
    """Prepare documents with embeddings for indexing."""
    docs = []

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

            docs.append({
                "id": doc_id,
                "text": chunk,
                "filename": rel_path,
                "folders": ",".join(folders) if folders else "",
                "characters": ",".join(characters) if characters else "",
                "topics": ",".join(topics) if topics else "",
                "chunk_idx": chunk_idx,
            })

        manifest["files"][rel_path] = {
            "hash": file_hash(filepath),
            "chunk_ids": chunk_ids,
        }

    # Batch embed all texts
    if docs:
        if verbose:
            print(f"  Embedding {len(docs)} chunks...")
        texts = [d["text"] for d in docs]
        embeddings = model.encode(texts, show_progress_bar=verbose)
        for doc, emb in zip(docs, embeddings):
            doc["vector"] = emb.tolist()

    return docs


def index_rebuild(
    root: Path,
    db_path: Path,
    manifest_path: Path,
    table_name: str,
    verbose: bool = True,
):
    """Full rebuild - delete and recreate index."""
    if verbose:
        print("Initializing embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    if verbose:
        print(f"Setting up LanceDB at {db_path}...")
    db_path.mkdir(parents=True, exist_ok=True)

    db = lancedb.connect(str(db_path))

    # Drop existing table if it exists
    try:
        db.drop_table(table_name)
    except Exception:
        pass

    # Fresh manifest
    manifest = {"files": {}, "version": 2}

    # Collect all files
    all_files = [fp for fp, _ in collect_documents(root)]

    if verbose:
        print(f"Indexing {len(all_files)} files...")

    docs = prepare_documents(all_files, model, manifest, root, verbose)

    if docs:
        if verbose:
            print(f"  Writing {len(docs)} chunks to database...")
        db.create_table(table_name, docs)

    save_manifest(manifest, manifest_path)

    if verbose:
        print(f"\nDone! Indexed {len(docs)} chunks from {len(all_files)} files.")
        print(f"Database stored at: {db_path}")


def index_incremental(
    root: Path,
    db_path: Path,
    manifest_path: Path,
    table_name: str,
    verbose: bool = True,
):
    """Incremental index - only process new/modified files."""
    if verbose:
        print("Initializing embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    if verbose:
        print(f"Setting up LanceDB at {db_path}...")
    db_path.mkdir(parents=True, exist_ok=True)

    db = lancedb.connect(str(db_path))
    manifest = load_manifest(manifest_path)

    # Check if table exists
    try:
        table = db.open_table(table_name)
        table_exists = True
    except Exception:
        table_exists = False

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

    # Handle deletions and modifications
    if table_exists and (deleted_keys or modified_files):
        ids_to_delete = []
        for rel_path in deleted_keys:
            file_info = manifest["files"].get(rel_path)
            if file_info and file_info.get("chunk_ids"):
                ids_to_delete.extend(file_info["chunk_ids"])
            if rel_path in manifest["files"]:
                del manifest["files"][rel_path]

        for filepath in modified_files:
            try:
                rel_path = str(filepath.relative_to(root))
            except ValueError:
                rel_path = str(filepath)
            file_info = manifest["files"].get(rel_path)
            if file_info and file_info.get("chunk_ids"):
                ids_to_delete.extend(file_info["chunk_ids"])
            if rel_path in manifest["files"]:
                del manifest["files"][rel_path]

        if ids_to_delete:
            # Delete by ID using SQL-style filter
            id_list = ", ".join(f"'{id}'" for id in ids_to_delete)
            table.delete(f"id IN ({id_list})")

    # Index new and modified files
    all_files = new_files + modified_files
    if all_files:
        if verbose:
            print(f"Indexing {len(all_files)} files...")
        docs = prepare_documents(all_files, model, manifest, root, verbose)

        if docs:
            if table_exists:
                table.add(docs)
            else:
                db.create_table(table_name, docs)

            if verbose:
                print(f"Indexed {len(docs)} chunks from {len(all_files)} files.")

    save_manifest(manifest, manifest_path)
    if verbose:
        print("Manifest saved.")


def index_add(
    paths: list[str],
    root: Path,
    db_path: Path,
    manifest_path: Path,
    table_name: str,
    verbose: bool = True,
):
    """Add specific files or directories to the index."""
    if verbose:
        print("Initializing embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    db_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_path))
    manifest = load_manifest(manifest_path)

    try:
        table = db.open_table(table_name)
        table_exists = True
    except Exception:
        table_exists = False

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

    # Remove existing chunks for these files
    if table_exists:
        ids_to_delete = []
        for filepath in files_to_add:
            try:
                rel_path = str(filepath.relative_to(root))
            except ValueError:
                rel_path = str(filepath)
            file_info = manifest["files"].get(rel_path)
            if file_info and file_info.get("chunk_ids"):
                ids_to_delete.extend(file_info["chunk_ids"])
            if rel_path in manifest["files"]:
                del manifest["files"][rel_path]

        if ids_to_delete:
            id_list = ", ".join(f"'{id}'" for id in ids_to_delete)
            table.delete(f"id IN ({id_list})")

    if verbose:
        print(f"Adding {len(files_to_add)} files...")

    docs = prepare_documents(files_to_add, model, manifest, root, verbose)

    if docs:
        if table_exists:
            table.add(docs)
        else:
            db.create_table(table_name, docs)

    save_manifest(manifest, manifest_path)

    if verbose:
        print(f"Added {len(docs)} chunks from {len(files_to_add)} files.")


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

    table_name = collection or root_path.name
    db_path = root_path / "vector_db"
    manifest_path = db_path / "manifest.json"
    return root_path, db_path, manifest_path, table_name


def main():
    parser = argparse.ArgumentParser(
        description="Index content into LanceDB for semantic search"
    )
    parser.add_argument(
        "--root", metavar="DIR",
        help="Root directory to index (default: fairy_project)"
    )
    parser.add_argument(
        "--collection", metavar="NAME",
        help="Table name (default: directory name)"
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
    args = parser.parse_args()

    verbose = not args.quiet
    root_path, db_path, manifest_path, table_name = get_paths(args.root, args.collection)

    if args.status:
        show_status(manifest_path, db_path)
    elif args.dry_run:
        manifest = load_manifest(manifest_path)
        new_files, modified_files, deleted_keys = get_files_to_index(root_path, manifest)
        print(f"Root: {root_path}")
        print(f"Table: {table_name}")
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
                  table_name=table_name, verbose=verbose)
    elif args.rebuild:
        index_rebuild(root=root_path, db_path=db_path, manifest_path=manifest_path,
                      table_name=table_name, verbose=verbose)
    else:
        index_incremental(root=root_path, db_path=db_path, manifest_path=manifest_path,
                          table_name=table_name, verbose=verbose)


if __name__ == "__main__":
    main()
