# fairy-toolkit justfile
# Run tasks with: just <recipe>

# Default content root
root := "../fairy_project"

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

# Semantic search
search query:
    ./search.py "{{query}}"

# Interactive search REPL
search-i:
    ./search.py -i

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
    @echo "=== Index ==="
    @./index.py --status 2>/dev/null || echo "(run 'just index' to build)"
    @echo ""
    @echo "=== Stories without frontmatter ==="
    @./list-stories.py --missing 2>/dev/null | tail -3
