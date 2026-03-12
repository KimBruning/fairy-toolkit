# fairy-toolkit justfile
# Run tasks with: just <recipe>

# Default content root
root := "../fairy_project"
claude_dump := "../claude_2026-03-11_dump"

# === On Update workflow ===

# Full update: rebuild index after story changes
update: index
    @echo ""
    @echo "Index rebuilt. Don't forget:"
    @echo "  - Tag new stories (just missing)"
    @echo "  - Update worldbuilding (spells, characters, locations)"
    @echo "  - Update The List"
    @echo "  - git commit"

# === Search & Browse ===

# Semantic search (fairy project)
search query:
    ./search.py "{{query}}"

# Search Claude conversation dump
search-claude query:
    ./search.py --root {{claude_dump}} "{{query}}"

# Interactive search REPL
search-i:
    ./search.py -i

# Catalog fairy-related Claude conversations
catalog:
    ./catalog-claude.py --fairy

# Catalog with artifact details
catalog-detail:
    ./catalog-claude.py --fairy --show-artifacts --why

# List stories by timeline
timeline:
    ./list-stories.py --sort fairy_age

# List stories missing frontmatter
missing:
    ./list-stories.py --missing

# List stories filtered by era
era value:
    ./list-stories.py --filter era={{value}} --sort fairy_age

# === Indexing ===

# Rebuild vector index from scratch
index:
    ./index.py --rebuild

# Incremental index (only changed files)
index-inc:
    ./index.py

# Show index status
index-status:
    ./index.py --status

# Rebuild Claude dump index
index-claude:
    ./index.py --root {{claude_dump}} --rebuild

# === Artifacts ===

# List artifacts in a Claude conversation
artifacts file:
    ./extract-artifacts.py "{{file}}"

# Show full artifact content (optionally filtered)
artifacts-full file +search='':
    ./extract-artifacts.py "{{file}}" --full {{search}}

# Search artifacts across all Claude conversations
artifacts-search +terms:
    ./extract-artifacts.py {{claude_dump}} -s {{terms}}

# List stories awaiting review
review:
    @echo "=== Stories for Review ==="
    @ls -1 {{root}}/snippets/stories-for-review/ 2>/dev/null || echo "(none)"

# === Consistency Checks ===

# Run all checks
check: check-frontmatter check-refs
    @echo ""
    @echo "All checks done."

# Stories without frontmatter
check-frontmatter:
    @echo "=== Stories missing frontmatter ==="
    @./list-stories.py --missing

# Check for broken internal path references in docs
check-refs:
    @echo "=== Checking path references ==="
    @cd {{root}} && grep -rn 'worldbuilding/[A-Z]' CLAUDE.md notes/ 2>/dev/null && echo "^ These may be stale after reorg" || echo "No stale top-level worldbuilding refs found."

# === Status Dashboard ===

# Quick project status overview
status:
    @echo "=== Git Status ==="
    @cd {{root}} && git status --short
    @echo ""
    @echo "=== Fairy Project Index ==="
    @./index.py --status 2>/dev/null | head -3 || echo "(run 'just index' to build)"
    @echo ""
    @echo "=== Claude Dump Index ==="
    @./index.py --root {{claude_dump}} --status 2>/dev/null | head -3 || echo "(run 'just index-claude' to build)"
    @echo ""
    @echo "=== Stories without frontmatter ==="
    @./list-stories.py --missing 2>/dev/null | tail -3
