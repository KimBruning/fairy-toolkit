# Creative Writing Toolkit

**Built for NixOS** - scripts use nix-shell shebangs for automatic dependency management.

**IMPORTANT: Run Claude Code from the PARENT directory, not from inside fairy-toolkit/ or your content directory!**

```bash
cd /path/to/parent-containing-both/
claude
```

This ensures Claude sees both the toolkit and your content.

## Directory Structure

```
parent/
├── CLAUDE.md              ← You are here (symlinked from toolkit)
├── fairy-toolkit/         ← Search/index tools
│   ├── search.py
│   ├── index.py
│   └── config.yaml        ← Points to content directory
└── my-project/            ← Your content (name from config.yaml)
    ├── stories/
    ├── worldbuilding/
    ├── documents/
    ├── notes/
    └── vector_db/         ← Created by index.py
```

## Quick Reference

```bash
# Search (from parent dir)
./fairy-toolkit/search.py "character conflict"
./fairy-toolkit/search.py -t story "adventure"
./fairy-toolkit/search.py -c CharacterName "dialogue"
./fairy-toolkit/search.py -i                    # interactive mode

# Index/rebuild (after adding content)
./fairy-toolkit/index.py --rebuild
./fairy-toolkit/index.py --status

# Setup new project (first time)
./fairy-toolkit/setup.py --dirname my-project
```

## Search Filters

| Flag | Description | Example |
|------|-------------|---------|
| `-t, --type` | Content type | story, worldbuilding, document, note |
| `-c, --char` | Character name | -c Gwyneth |
| `--topic` | Topic keyword | --topic magic |
| `-n` | Number of results | -n 10 |

## Workflow

1. **Find something**: `./fairy-toolkit/search.py "what you're looking for"`
2. **Read/edit**: Open the file path from search results
3. **Update worldbuilding**: If story establishes new canon, update docs
4. **Re-index**: After adding/changing content, run `./fairy-toolkit/index.py`

## Content Structure

The toolkit expects these directories in your content project:

- `stories/` - Narrative content (.md, .txt)
- `worldbuilding/` - Lore, characters, systems
- `documents/` - Reference materials
- `notes/` - Working notes, drafts

Files are auto-tagged by directory (stories/ → type:story, etc.)

## Configuration

Edit `fairy-toolkit/config.yaml` to change the default content directory:

```yaml
content_root: ../my-project
```

Or use `--root` to override:
```bash
./fairy-toolkit/search.py --root ../other-project "query"
```
