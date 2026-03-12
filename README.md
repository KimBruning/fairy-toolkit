# Creative Writing Toolkit

Semantic search and indexing tools for creative writing projects. Find stories, worldbuilding docs, and notes by meaning rather than keywords.

**Built for NixOS** - uses nix-shell shebangs for automatic dependency management. No virtualenv needed.

## Installation

```bash
# Clone the toolkit
git clone https://github.com/KimBruning/fairy-toolkit.git
cd fairy-toolkit

# Set up a new project (creates sibling directory + parent CLAUDE.md)
./setup.py --dirname ../my-project
```

This creates:
```
parent/
├── CLAUDE.md              ← Symlinked from toolkit
├── fairy-toolkit/         ← This repo
└── my-project/            ← Your content
    ├── stories/
    ├── worldbuilding/
    ├── documents/
    └── notes/
```

## Usage

**Important: Run commands from the parent directory!**

```bash
cd ..   # Go to parent containing both toolkit and project

# Index your content
./fairy-toolkit/index.py --rebuild

# Search
./fairy-toolkit/search.py "character discovers secret"
./fairy-toolkit/search.py -t story "adventure"
./fairy-toolkit/search.py -c CharacterName "dialogue"
./fairy-toolkit/search.py -i   # Interactive mode
```

## Task Runner

[`just`](https://github.com/casey/just) is the recommended way to run common tasks. Run from the `fairy-toolkit/` directory:

```bash
cd fairy-toolkit/

# Project status overview
just status

# Rebuild index after adding/editing content
just update

# Semantic search (fairy project)
just search "fairy flight modes"

# Search Claude conversation dump
just search-claude "acculturation nurse Aisling"

# Interactive search REPL
just search-i

# Catalog fairy-related Claude conversations
just catalog
just catalog-detail   # with artifact info and match reasons

# List stories by timeline or era
just timeline
just era academy

# Find stories missing frontmatter
just missing

# Run consistency checks (frontmatter + stale refs)
just check

# Index management
just index           # Full rebuild (fairy project)
just index-inc       # Incremental (changed files only)
just index-claude    # Rebuild Claude dump index
just index-status    # Show index info
```

Run `just --list` to see all available recipes.

## Using with Claude Code

**Run Claude Code from the parent directory:**

```bash
cd /path/to/parent/
claude
```

This lets Claude see both the toolkit docs (CLAUDE.md) and your content. The CLAUDE.md explains the workflow and available tools.

## Content Structure

Put your writing in the content directory:

```
my-project/
├── stories/           # Narrative content (.md, .txt)
├── worldbuilding/     # Lore, characters, reference docs
├── documents/         # Other documents
├── notes/             # Working notes, drafts
└── vector_db/         # Created by index.py (gitignored)
```

Content is auto-tagged by directory (`stories/` → type:story, etc.)

## Dependencies

**NixOS / Nix:** Dependencies are handled automatically via nix-shell shebangs in each script. Just run the scripts directly - nix will fetch:
- lancedb
- sentence-transformers
- pyyaml
- pyarrow

**Other systems:** You'll need lancedb, sentence-transformers, pyyaml, and pyarrow. Install via your package manager of choice.

**Note:** ChromaDB is blacklisted (leaks telemetry). We use LanceDB instead.

## Configuration

The toolkit reads `config.yaml` for the default content location:

```yaml
content_root: ../my-project
```

Override with `--root`:
```bash
./fairy-toolkit/search.py --root ../other-project "query"
```

## Documentation

- `CLAUDE.md` - Full usage reference (also symlinked to parent)
- `docs/vector_search.md` - Architecture and implementation details
- `docs/worldbuilding-workflow.md` - Extracting characters/locations from stories
