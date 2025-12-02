# Creative Writing Toolkit

Semantic search and indexing tools for creative writing projects. Find stories, worldbuilding docs, and notes by meaning rather than keywords.

## Quick Start

1. Clone this repo alongside your content directory
2. Edit `config.yaml` to point at your content:
   ```yaml
   content_root: ../my-writing-project
   ```
3. Index your content:
   ```bash
   ./index.py --rebuild
   ```
4. Search:
   ```bash
   ./search.py "character discovers secret"
   ```

## Directory Structure

Your content directory should have subdirectories like:
```
my-writing-project/
├── stories/           # Story files (.md, .txt)
├── worldbuilding/     # Reference docs
├── documents/         # Other documents
├── notes/             # Working notes
└── vector_db/         # Created by index.py
```

The toolkit auto-tags content based on directory (`stories/` → type:story, etc.)

## Tools

### index.py

Indexes all `.md` and `.txt` files into a vector database.

```bash
./index.py              # Incremental update (only new/changed files)
./index.py --rebuild    # Full rebuild
./index.py --status     # Show what's indexed
./index.py --dry-run    # Preview without indexing
```

### search.py

Semantic search across indexed content.

```bash
./search.py "query"                 # Basic search
./search.py "query" -n 10           # More results
./search.py "query" -t story        # Filter by type
./search.py "query" -c CharName     # Filter by character
./search.py -i                      # Interactive mode
```

## Configuration

Edit `config.yaml`:

```yaml
content_root: ../fairy_project    # Path to content (relative to toolkit)
```

CLI `--root` overrides config:
```bash
./search.py --root /other/project "query"
```

## Dependencies

Uses nix-shell for dependencies (chromadb, sentence-transformers, pyyaml).

Or install manually:
```bash
pip install chromadb sentence-transformers pyyaml
```

## Auto-Tagging

Content is automatically tagged with:
- **source_type**: Based on directory (stories → story, worldbuilding → worldbuilding)
- **characters**: Detected character names in text
- **topics**: Keyword-based topic detection (magic, flight, politics, etc.)

Customize character/topic lists in `index.py`.

## See Also

- `docs/vector_search.md` - Detailed architecture documentation
