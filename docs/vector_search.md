# Vector Search Documentation

Semantic search system for the Gwyneth fairy universe content using ChromaDB and sentence-transformers.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Content Files  │────▶│    index.py      │────▶│  ChromaDB   │
│  (.md, .txt)    │     │  (embedding +    │     │  vector_db/ │
└─────────────────┘     │   chunking)      │     └─────────────┘
                        └──────────────────┘            │
                                                        │
┌─────────────────┐     ┌──────────────────┐            │
│   User Query    │────▶│    search.py     │◀───────────┘
│                 │     │  (embed + search │
└─────────────────┘     │   + filter)      │
                        └──────────────────┘
```

## index.py

Indexes all markdown and text files in the fairy_project directory.

### Usage

```bash
./py/index.py              # Build/rebuild index
./py/index.py --dry-run    # Show what would be indexed
./py/index.py --help       # Show help
```

### Chunking Strategy

- **Chunk size**: ~1000 characters
- **Overlap**: 200 characters
- **Break points**: Prefers paragraph (`\n\n`) or sentence (`. `) boundaries
- **Minimum**: Single chunk if content < 1000 chars

### Auto-Tagging

Each chunk is automatically tagged with:

| Field | Source | Example |
|-------|--------|---------|
| `filename` | File path relative to project root | `worldbuilding/Fairy_Flight.md` |
| `source_type` | Directory name | `worldbuilding`, `story`, `document`, `note` |
| `characters` | Keyword match in text | `Gwyneth,Eirwen` |
| `topics` | Keyword match in text | `magic,flight` |
| `chunk_idx` | Sequential index | `0`, `1`, `2`... |

### Character Detection

Scans for: Gwyneth, Caoimhe, Caiomhe, Eirwen, Rhia, Eilis, Brigid, Underhill, Sylvarum, Marion, Selenis

### Topic Detection

| Topic | Keywords |
|-------|----------|
| magic | magic, spell, geas, enchant, ward |
| flight | flight, flying, wings, hover, glide |
| politics | baron, domain, council, treaty, ambassador |
| physiology | metabolism, vision, anatomy, scale, size |
| shadow-walking | shadow, realm, portal, dimension |
| family | mother, sister, cousin, grandmother, aunt, family |

## search.py

Semantic search with optional filtering.

### Usage

```bash
./py/search.py "query"                      # Basic search
./py/search.py "query" -n 10                # Return 10 results
./py/search.py "query" -t worldbuilding     # Filter by type
./py/search.py "query" -c Gwyneth           # Filter by character
./py/search.py "query" --topic magic        # Filter by topic
./py/search.py -i                           # Interactive REPL
./py/search.py --help                       # Show help
```

### Interactive Mode Commands

| Command | Action |
|---------|--------|
| `:quit` | Exit |
| `:type <type>` | Set source_type filter |
| `:char <name>` | Set character filter |
| `:topic <topic>` | Set topic filter |
| `:clear` | Clear all filters |

### Result Format

```
[1] worldbuilding/Fairy_Flight.md (chunk 2)
    Type: worldbuilding | Score: 0.438
    Characters: Gwyneth
    Topics: magic,flight
    ---
    Preview text of the matching chunk...
```

## Embedding Model

Uses `all-MiniLM-L6-v2` from sentence-transformers:
- 384-dimensional embeddings
- Good balance of speed and quality
- ~100MB download on first run
- Cached in `~/.cache/huggingface/`

## Database Location

ChromaDB persistent storage at `fairy_project/py/vector_db/`

To reset: delete the `vector_db/` directory and re-run `index.py`.

## Extending

### Adding New Content

1. Add `.md` or `.txt` files to `worldbuilding/`, `stories/`, `documents/`, or `notes/`
2. Re-run `./py/index.py`

### Adding Characters/Topics

Edit `CHARACTERS` and `TOPIC_KEYWORDS` lists in `index.py`, then re-index.

### Custom Source Types

Add new directories - `get_source_type()` in `index.py` will detect them if you add a condition.
