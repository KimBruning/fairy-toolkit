# Creative Writing Toolkit

Semantic search and indexing tools for creative writing projects. Find stories, worldbuilding docs, and notes by meaning rather than keywords.

## Installation

```bash
# Clone the toolkit
git clone https://github.com/yourname/fairy-toolkit.git
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

Uses nix-shell for automatic dependency management:
- chromadb
- sentence-transformers
- pyyaml

Or install manually:
```bash
pip install chromadb sentence-transformers pyyaml
```

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
